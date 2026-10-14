"""Numerical checks for the unbounded, continuous-feedback pendulum policy."""

import unittest

import numpy as np
from scipy.linalg import expm

from core.pendulum_policy_gradient import (
    DEFAULT_REWARD_SOURCE,
    PendulumConfig,
    RewardError,
    compile_reward,
    evaluate_parameters,
    evaluate_rollouts,
    goal_mask,
    sample_initial_states,
    train_policy_gradient,
)


ACTION_COST = """def reward(state, action, next_state):
    return -action**2
"""


class PendulumPolicyGradientTests(unittest.TestCase):
    def test_exact_cancellation_matches_linear_system_without_angle_or_torque_clipping(self):
        config = PendulumConfig(b=0.03, dt=0.02, horizon=120)
        starts = np.array([[7.2, 4.0], [-4.5, -3.0], [0.3, -0.2]])
        weights = np.array([config.b, -1.0])
        reward = compile_reward(ACTION_COST, config)
        batch = evaluate_rollouts(weights, starts, config, reward)

        matrix = np.array([[0.0, 1.0], [-1.0, -1.0]])
        times = np.arange(config.horizon + 1) * config.dt
        expected = np.stack([starts @ expm(matrix * t).T for t in times], axis=1)
        np.testing.assert_allclose(batch["states"], expected, atol=2e-6, rtol=0)
        np.testing.assert_array_equal(batch["states"][:, 0], starts)
        theta, omega = batch["states"][:, :-1].transpose(2, 0, 1)
        expected_actions = -np.sin(theta) + config.b * omega - theta - omega
        np.testing.assert_allclose(batch["actions"], expected_actions, atol=1e-12)
        self.assertGreater(np.max(np.abs(batch["actions"])), 10.0)
        self.assertGreater(batch["states"][0, 1, 0], np.pi)

    def test_entering_reward_neighborhood_does_not_freeze_or_latch_reward(self):
        config = PendulumConfig(b=0.03, dt=0.04, horizon=30,
                                theta_tolerance=0.05, omega_tolerance=0.05)
        starts = np.array([[0.0, 0.03]])
        # Gravity cancels, while positive velocity feedback makes the small
        # initial state leave the reward neighborhood during observation.
        weights = [config.b + 4.0, -1.0]
        reward = compile_reward(DEFAULT_REWARD_SOURCE, config)
        batch = evaluate_rollouts(weights, starts, config, reward)
        matrix = np.array([[0.0, 1.0], [-1.0, 3.0]])
        expected = np.stack([
            starts @ expm(matrix * t).T
            for t in np.arange(config.horizon + 1) * config.dt
        ], axis=1)
        np.testing.assert_allclose(batch["states"], expected, atol=3e-6, rtol=0)
        self.assertFalse(np.array_equal(batch["states"][0, 1], starts[0]))
        self.assertGreater(np.abs(batch["states"][0, -1, 1]), config.omega_tolerance)
        self.assertEqual(batch["rewards"][0, 0], 1.0)
        self.assertEqual(batch["rewards"][0, -1], 0.0)
        self.assertGreater(np.abs(batch["actions"][0, -1]), 0.0)

    def test_discounted_returns_and_reward_arguments_use_each_real_transition(self):
        config = PendulumConfig(dt=0.03, horizon=5, discount=0.6)
        source = """def reward(state, action, next_state):
    return (2*state[..., 0] - 3*state[..., 1] + 0.1*action
            + 5*next_state[..., 0] + 7*next_state[..., 1])
"""
        reward = compile_reward(source, config)
        batch = evaluate_rollouts([0.2, -0.7], [[0.4, -0.1], [-0.3, 0.2]],
                                  config, reward)
        expected_rewards = (2 * batch["states"][:, :-1, 0]
                            - 3 * batch["states"][:, :-1, 1]
                            + 0.1 * batch["actions"]
                            + 5 * batch["states"][:, 1:, 0]
                            + 7 * batch["states"][:, 1:, 1])
        np.testing.assert_allclose(batch["rewards"], expected_rewards)
        for k in range(config.horizon):
            expected_return = expected_rewards[:, k:] @ (
                config.discount ** np.arange(config.horizon - k))
            np.testing.assert_allclose(batch["returns"][:, k], expected_return)
        np.testing.assert_allclose(batch["return"], batch["returns"][:, 0])
        self.assertAlmostEqual(batch["score"], np.mean(batch["return"]))

    def test_scalar_reward_broadcasts_across_monte_carlo_batch(self):
        config = PendulumConfig(horizon=4, discount=0.7)
        reward = compile_reward("def reward(state, action, next_state):\n    return 2.5\n", config)
        batch = evaluate_rollouts([0.0, 0.0], [[0.2, 0.1], [-0.2, -0.3]],
                                  config, reward)
        np.testing.assert_array_equal(batch["rewards"], np.full((2, 4), 2.5))
        np.testing.assert_allclose(batch["return"], 2.5 * sum(0.7**k for k in range(4)))

    def test_reward_is_called_once_per_step_on_whole_transition_batches(self):
        config = PendulumConfig(dt=0.03, horizon=5)
        starts = np.array([[0.4, 0.1], [-0.7, 0.2], [0.2, -0.3], [-0.2, -0.1]])
        calls = []

        def reward(state, action, next_state):
            calls.append((state.copy(), action.copy(), next_state.copy()))
            return np.sum(state**2 - next_state**2, axis=-1) - action**2

        for passive in [False, True]:
            calls.clear()
            batch = evaluate_rollouts([0.2, -0.7], starts, config, reward, passive=passive)
            self.assertEqual(len(calls), config.horizon)
            for k, (state, action, next_state) in enumerate(calls):
                self.assertEqual(state.shape, (4, 2))
                self.assertEqual(action.shape, (4,))
                np.testing.assert_array_equal(state, batch["states"][:, k])
                np.testing.assert_array_equal(action, batch["actions"][:, k])
                np.testing.assert_array_equal(next_state, batch["states"][:, k + 1])
                if passive:
                    np.testing.assert_array_equal(action, np.zeros(4))

        calls.clear()
        pairs = np.array([[[0.2, -0.7], [-0.3, -0.4]], [[0.1, -0.9], [0.4, -0.2]]])
        scores = evaluate_parameters(pairs, starts, config, reward, chunk_size=3)
        self.assertEqual(scores.shape, (2, 2))
        self.assertEqual(len(calls), 2 * config.horizon)
        for k, (state, action, next_state) in enumerate(calls):
            n_policies = 3 if k < config.horizon else 1
            self.assertEqual(state.shape, (n_policies, 4, 2))
            self.assertEqual(next_state.shape, state.shape)
            self.assertEqual(action.shape, (n_policies, 4))
            if k % config.horizon == 0:
                np.testing.assert_array_equal(state, np.broadcast_to(starts, state.shape))
            else:
                np.testing.assert_array_equal(state, calls[k - 1][2])

    def test_reward_shape_must_preserve_both_policy_and_episode_axes(self):
        config = PendulumConfig(horizon=2)
        # Valid for a single episode batch, but loses the episode axis when
        # the landscape supplies a batch of policies and episodes together.
        reward = compile_reward("""def reward(state, action, next_state):
    return np.ones(state.shape[0])
""", config)
        with self.assertRaisesRegex(RewardError, r"shape \(2, 3\); got \(2,\)"):
            evaluate_parameters([[0.2, -0.7], [0.1, -0.4]],
                                [[0.4, 0.1], [-0.7, 0.2], [0.2, -0.3]], config, reward)

    def test_sampling_is_seeded_and_respects_displayed_initial_state_box(self):
        config = PendulumConfig(theta_width=0.7, omega_width=0.4)
        starts = sample_initial_states(config, 64, seed=13)
        np.testing.assert_array_equal(starts, sample_initial_states(config, 64, seed=13))
        self.assertEqual(starts.shape, (64, 2))
        self.assertTrue(np.isfinite(starts).all())
        self.assertTrue(np.all(np.abs(starts) <= [config.theta_width, config.omega_width]))
        self.assertFalse(np.array_equal(starts, sample_initial_states(config, 64, seed=14)))

    def test_default_reward_and_capture_match_numpy_isclose_at_configured_origin_boundaries(self):
        config = PendulumConfig(theta_tolerance=0.02, omega_tolerance=0.03)
        reward = compile_reward(DEFAULT_REWARD_SOURCE, config)
        theta = np.array([
            0.0, config.theta_tolerance, np.nextafter(config.theta_tolerance, np.inf),
            -config.theta_tolerance, np.nextafter(-config.theta_tolerance, -np.inf),
            0.0, 0.0, 2 * np.pi,
        ]).reshape(2, 4)
        omega = np.array([
            0.0, config.omega_tolerance, 0.0, -config.omega_tolerance, 0.0,
            np.nextafter(config.omega_tolerance, np.inf),
            np.nextafter(-config.omega_tolerance, -np.inf), 0.0,
        ]).reshape(2, 4)
        expected = (np.isclose(theta, 0.0, atol=config.theta_tolerance, rtol=0.0)
                    & np.isclose(omega, 0.0, atol=config.omega_tolerance, rtol=0.0))
        np.testing.assert_array_equal(expected, [[True, True, False, True],
                                                [False, False, False, False]])
        next_states = np.stack([theta, omega], axis=-1)
        np.testing.assert_array_equal(reward(np.zeros_like(next_states), np.zeros_like(theta), next_states),
                                       expected.astype(float))
        np.testing.assert_array_equal(goal_mask(np.stack([theta, omega], axis=-1), config),
                                       expected)

        # Recompiling the same editable source picks up both new tolerances;
        # the earlier callable remains bound to its original configuration.
        wider = PendulumConfig(theta_tolerance=0.04, omega_tolerance=0.06)
        wider_reward = compile_reward(DEFAULT_REWARD_SOURCE, wider)
        probe_theta, probe_omega = np.array([0.03, 0.01]), np.array([0.01, 0.04])
        probe_states = np.column_stack([probe_theta, probe_omega])
        np.testing.assert_array_equal(reward(np.zeros_like(probe_states), np.zeros(2), probe_states), [0, 0])
        np.testing.assert_array_equal(wider_reward(np.zeros_like(probe_states), np.zeros(2), probe_states), [1, 1])
        np.testing.assert_array_equal(goal_mask(np.column_stack([probe_theta, probe_omega]), wider),
                                       [True, True])

    def test_landscape_shape_and_chunking_match_independent_monte_carlo_rollouts(self):
        config = PendulumConfig(horizon=9, dt=0.02, discount=0.83)
        reward = compile_reward("""def reward(state, action, next_state):
    return (np.sum(state**2 - next_state**2, axis=-1) - action**2)
""", config)
        starts = np.array([[0.4, 0.1], [-0.7, 0.2], [0.2, -0.3]])
        pairs = np.array([
            [[0.03, -1.0], [0.1, -0.4], [-0.2, 0.5]],
            [[0.7, -0.6], [-0.3, -0.2], [0.4, 0.1]],
        ])
        expected = np.array([
            evaluate_rollouts(weights, starts, config, reward)["score"]
            for weights in pairs.reshape(-1, 2)
        ]).reshape(2, 3)
        for chunk_size in [1, 4, 256]:
            scores = evaluate_parameters(pairs, starts, config, reward, chunk_size=chunk_size)
            self.assertEqual(scores.shape, (2, 3))
            np.testing.assert_allclose(scores, expected, atol=1e-12, rtol=1e-12)

    def test_common_batch_central_difference_matches_analytic_action_cost_gradient(self):
        config = PendulumConfig(horizon=1, theta_width=0.7, omega_width=0.5)
        reward = compile_reward(ACTION_COST, config)
        initial = np.array([0.3, -0.7])
        epsilon, learning_rate = 0.02, 0.1
        result = train_policy_gradient(
            initial, config, reward, iterations=3, batch_size=12, seed=17,
            epsilon=epsilon, learning_rate=learning_rate,
        )
        snapshots = result["snapshots"]
        self.assertEqual(len(snapshots), 4)
        np.testing.assert_array_equal(snapshots[0]["weights"], initial)
        np.testing.assert_array_equal(initial, [0.3, -0.7])

        for index, snapshot in enumerate(snapshots):
            with self.subTest(iteration=index):
                weights = snapshot["weights"]
                theta, omega = snapshot["batch_starts"].T
                features = np.column_stack([omega, np.sin(theta)])
                actions = features @ weights - theta - omega
                analytic_gradient = -2 * np.mean(actions[:, None] * features, axis=0)
                np.testing.assert_allclose(snapshot["gradient"], analytic_gradient,
                                           atol=1e-12, rtol=1e-12)
                perturbations = epsilon * np.eye(2)
                plus = evaluate_parameters(weights + perturbations,
                                           snapshot["batch_starts"], config, reward)
                minus = evaluate_parameters(weights - perturbations,
                                            snapshot["batch_starts"], config, reward)
                np.testing.assert_array_equal(snapshot["plus_scores"], plus)
                np.testing.assert_array_equal(snapshot["minus_scores"], minus)
                np.testing.assert_allclose(snapshot["gradient"],
                                           (plus - minus) / (2 * epsilon), atol=1e-12)
                # The plotted trajectories must be the very samples used for
                # each finite-difference return, in +e1, +e2, -e1, -e2 order.
                self.assertEqual(len(snapshot["perturbation_rollouts"]), 4)
                for probe, batch, score in zip(
                    snapshot["perturbed_weights"], snapshot["perturbation_rollouts"],
                    np.r_[snapshot["plus_scores"], snapshot["minus_scores"]],
                ):
                    np.testing.assert_array_equal(batch["states"][:, 0], snapshot["batch_starts"])
                    independent = evaluate_rollouts(probe, snapshot["batch_starts"], config, reward)
                    np.testing.assert_array_equal(batch["states"], independent["states"])
                    np.testing.assert_array_equal(batch["return"], independent["return"])
                    self.assertEqual(score, batch["score"])
                self.assertEqual(snapshot["rollout_count"], 6 * 12 * (index + 1))
                if index < len(snapshots) - 1:
                    np.testing.assert_allclose(snapshots[index + 1]["weights"],
                                               weights + learning_rate * analytic_gradient)

    def test_seeded_snapshot_prefix_is_preserved_and_uses_its_own_weights(self):
        config = PendulumConfig(horizon=8, theta_width=0.8, omega_width=0.6)
        reward = compile_reward(ACTION_COST, config)
        initial = np.array([0.2, -0.4])
        arguments = dict(batch_size=10, seed=29, epsilon=0.015, learning_rate=0.03)
        full = train_policy_gradient(initial, config, reward, iterations=3, **arguments)
        prefix = train_policy_gradient(initial, config, reward, iterations=1, **arguments)
        for index in [0, 1]:
            for key in ["weights", "gradient", "batch_starts", "plus_scores", "minus_scores"]:
                np.testing.assert_array_equal(full["snapshots"][index][key],
                                               prefix["snapshots"][index][key])
        for snapshot in full["snapshots"]:
            stored = snapshot["rollout"]
            independent = evaluate_rollouts(snapshot["weights"], stored["states"][:, 0],
                                            config, reward)
            np.testing.assert_array_equal(stored["states"], independent["states"])
            np.testing.assert_array_equal(stored["actions"], independent["actions"])
            self.assertEqual(snapshot["batch_score"], independent["score"])
            evaluation = evaluate_parameters(snapshot["weights"], full["evaluation_starts"],
                                             config, reward)
            self.assertAlmostEqual(snapshot["evaluation_score"], float(evaluation), places=12)
        full["snapshots"][-1]["weights"][0] = 9999.0
        np.testing.assert_array_equal(full["snapshots"][0]["weights"], initial)

    def test_sparse_default_reward_can_have_exactly_flat_finite_differences(self):
        config = PendulumConfig(horizon=1, dt=0.02)
        reward = compile_reward(DEFAULT_REWARD_SOURCE, config)
        starts = np.array([[0.6, 0.2], [-0.7, -0.1], [0.3, -0.4]])
        weights = np.array([0.1, -0.3])
        perturbations = 0.02 * np.eye(2)
        plus = evaluate_parameters(weights + perturbations, starts, config, reward)
        minus = evaluate_parameters(weights - perturbations, starts, config, reward)
        np.testing.assert_array_equal(plus, [0.0, 0.0])
        np.testing.assert_array_equal(minus, [0.0, 0.0])
        np.testing.assert_array_equal((plus - minus) / 0.04, [0.0, 0.0])

    def test_invalid_reward_code_reports_context_instead_of_silent_fallback(self):
        config = PendulumConfig(horizon=2)
        sources = {
            "syntax": "def reward(state, action, next_state)\n    return 0\n",
            "missing_function": "def another_function(state, action, next_state):\n    return 0\n",
            "signature": "def reward(theta):\n    return theta\n",
            "old_signature": "def reward(theta, omega, u, t):\n    return 0\n",
            "runtime": "def reward(state, action, next_state):\n    return missing_variable\n",
            "nan": "def reward(state, action, next_state):\n    return np.nan + 0*action\n",
            "infinity": "def reward(state, action, next_state):\n    return np.inf + 0*action\n",
            "shape": "def reward(state, action, next_state):\n    return np.zeros((2, 2))\n",
            "not_vectorized": "def reward(state, action, next_state):\n    return 1.0 if next_state[..., 0] > 0 else 0.0\n",
        }
        for kind, source in sources.items():
            with self.subTest(kind=kind), self.assertRaisesRegex(RewardError, "(?i)reward"):
                reward = compile_reward(source, config)
                evaluate_rollouts([0.1, -0.4], [[0.2, 0.1], [-0.3, 0.2]], config, reward)

    def test_runtime_reward_failure_after_successful_compile_includes_original_error_and_time(self):
        config = PendulumConfig(horizon=2)
        source = """def reward(state, action, next_state):
    if np.any(np.abs(state[..., 0]) > 0.1):
        raise ZeroDivisionError("custom reward failed away from origin")
    return 1.0
"""
        reward = compile_reward(source, config)
        with self.assertRaisesRegex(RewardError, r"reward failed at t=.*ZeroDivisionError.*away from origin"):
            evaluate_rollouts([0.1, -0.4], [[0.2, 0.1]], config, reward)


if __name__ == "__main__":
    unittest.main()
