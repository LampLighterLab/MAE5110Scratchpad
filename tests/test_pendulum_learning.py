"""Numerical checks for the upright, discounted-reward pendulum examples."""

import unittest

import numpy as np
from scipy.integrate import quad

from pendulum_learning import (
    Pendulum,
    _interpolate,
    _interpolation,
    collect_policy_data,
    discount_sum,
    fit_value,
    fit_linear_policy,
    in_target,
    initial_state_limits,
    initial_states,
    predict_value,
    policy_action,
    policy_features,
    rollout,
    step,
    train_actor_critic,
    value_policy_weights,
    value_iteration,
    vi_action,
    vi_rollout,
    vi_policy_rollout,
)


class PendulumLearningTests(unittest.TestCase):
    def test_weighted_policy_fit_solves_weighted_objective_and_validates_weights(self):
        model = Pendulum(torque_limit=10)
        states = np.array([[0.1], [0.2], [2.0]])
        actions = np.array([-0.3, -0.6, 0.1])
        data = {"states": states, "actions": actions, "batch": None}
        weights = np.array([1.0, 2.0, 0.005])
        fit = fit_linear_policy(model, None, data=data, sample_weights=weights)
        expected = np.sum(weights * states[:, 0] * actions) / np.sum(weights * states[:, 0]**2)
        np.testing.assert_allclose(fit["coefficients"], [expected])
        residual = states[:, 0] * expected - actions
        self.assertAlmostEqual(fit["weighted_rmse"], np.sqrt(np.average(residual**2, weights=weights)))
        scaled = fit_linear_policy(model, None, data=data, sample_weights=100 * weights)
        np.testing.assert_allclose(scaled["coefficients"], fit["coefficients"])
        dropped = fit_linear_policy(model, None, data=data, sample_weights=[1, 1, 0])
        np.testing.assert_allclose(dropped["coefficients"], [-3])
        uniform = fit_linear_policy(model, None, data=data, sample_weights=np.ones(3))
        unweighted = fit_linear_policy(model, None, data=data)
        np.testing.assert_array_equal(uniform["coefficients"], unweighted["coefficients"])
        for invalid in [[0, 0, 0], [-1, 1, 1], [1, np.nan, 1], [1, np.inf, 1], [1, 1]]:
            with self.assertRaises(ValueError):
                fit_linear_policy(model, None, data=data, sample_weights=invalid)

    def test_value_weighting_improves_default_local_capture(self):
        for kind, nt, nw, horizon in [("overdamped", 1601, 1, 100),
                                      ("undamped", 241, 161, 180)]:
            with self.subTest(kind=kind):
                model = Pendulum(kind=kind, horizon=horizon, torque_cost=0.1,
                                 damping=0.03 if kind == "undamped" else 0,
                                 target_theta=np.pi / nt, target_omega=3 / max(nw - 1, 1))
                solution = value_iteration(model, n_theta=nt, n_omega=nw,
                                           n_actions=11 if kind == "overdamped" else 15)
                data = collect_policy_data(model, solution, source="grid")
                weights = value_policy_weights(data["values"], power=16)
                weighted = fit_linear_policy(model, solution, data=data, sample_weights=weights)
                uniform = fit_linear_policy(model, solution, data=data)
                starts = initial_states(model, 256, seed=41)
                weighted_batch = rollout(model, weighted["coefficients"], starts)
                uniform_batch = rollout(model, uniform["coefficients"], starts)
                self.assertGreaterEqual(weighted_batch["success"].mean(), 0.95)
                self.assertLessEqual(uniform_batch["success"].mean(), 0.05)
                self.assertGreater(weighted_batch["return"].mean(), uniform_batch["return"].mean() + 10)

    def test_value_weights_handle_negative_values_and_zero_initialization(self):
        np.testing.assert_allclose(value_policy_weights([-1, 0, 2, 10], power=2), [0, 0, .04, 1])
        np.testing.assert_allclose(value_policy_weights([-10, 0, 20, 100], power=2), [0, 0, .04, 1])
        np.testing.assert_array_equal(value_policy_weights([-1, 0, 2], power=0), 1)
        np.testing.assert_array_equal(value_policy_weights([0, 0, 0], power=16), 1)
        np.testing.assert_array_equal(value_policy_weights([-3, -1, 0], power=16), 1)
        for invalid in [-1, np.inf, np.nan]:
            with self.assertRaises(ValueError):
                value_policy_weights([0, 1], power=invalid)
        for invalid in [[], [np.inf], [np.nan], [[1, 2]]]:
            with self.assertRaises(ValueError):
                value_policy_weights(invalid)

    def test_grid_imitation_covers_displayed_states_and_selected_actions(self):
        for kind in ["overdamped", "undamped"]:
            with self.subTest(kind=kind):
                model = Pendulum(kind=kind, horizon=8, torque_cost=0.1)
                solution = value_iteration(model, n_theta=81, n_omega=41)
                names = (["theta", "sin_theta", "sin_2theta"] if model.dimension == 1
                         else ["theta", "omega", "sin_theta", "energy_pump",
                               "omega_cos_theta", "omega_cubed"])
                for iteration in [0, 3, 8]:
                    data = collect_policy_data(model, solution, remaining=iteration, source="grid")
                    self.assertEqual(data["source"], "grid")
                    self.assertEqual(len(data["states"]), solution["policy"][iteration].size)
                    np.testing.assert_array_equal(data["actions"], solution["policy"][iteration].ravel())
                    np.testing.assert_array_equal(data["values"], solution["values"][iteration].ravel())
                    self.assertLess(data["states"][:, 0].min(), -3)
                    self.assertGreater(data["states"][:, 0].max(), 3)
                    if model.dimension == 2:
                        np.testing.assert_array_equal(data["states"][:, 1].min(), -model.omega_limit)
                        np.testing.assert_array_equal(data["states"][:, 1].max(), model.omega_limit)
                    fit = fit_linear_policy(model, solution, data=data, feature_names=names)
                    design = policy_features(model, data["states"], names)
                    residual = design @ fit["coefficients"] - data["actions"]
                    # The unregularized fit must solve the actual grid least-squares problem.
                    np.testing.assert_allclose(design.T @ residual, 0, atol=1e-8)
                    self.assertAlmostEqual(fit["raw_rmse"], np.sqrt(np.mean(residual**2)))
                    self.assertLessEqual(fit["rmse"], fit["raw_rmse"] + 1e-12)
                    self.assertLess(fit["condition"], 100)
                    if iteration == 0:
                        np.testing.assert_array_equal(fit["coefficients"], 0)
                with self.assertRaises(ValueError):
                    collect_policy_data(model, solution, source="unknown")
                with self.assertRaises(ValueError):
                    collect_policy_data(model, solution, source="grid", remaining=9)

    def test_snapshot_imitation_uses_only_selected_iteration_and_zero_initialization(self):
        for kind in ["overdamped", "undamped"]:
            with self.subTest(kind=kind):
                model = Pendulum(kind=kind, horizon=10, torque_cost=0.1)
                solution = value_iteration(model, n_theta=81, n_omega=41)
                names = (["theta", "sin_theta"] if model.dimension == 1 else
                         ["theta", "omega", "sin_theta", "energy_pump"])
                for iteration in [0, 4, model.horizon]:
                    data = collect_policy_data(model, solution, n_samples=160,
                                               remaining=iteration)
                    np.testing.assert_array_equal(
                        data["actions"], vi_action(model, solution, data["states"], iteration))
                    fit = fit_linear_policy(model, solution, remaining=iteration,
                                            data=data, feature_names=names)
                    self.assertEqual(fit["iteration"], iteration)
                    if iteration == 0:
                        np.testing.assert_array_equal(data["actions"], 0)
                        np.testing.assert_array_equal(fit["coefficients"], 0)
                        np.testing.assert_array_equal(fit["predictions"], 0)
                    if iteration == 4:
                        # Later Bellman results must not leak into an earlier fit.
                        altered = {**solution, "values": solution["values"].copy(),
                                   "policy": solution["policy"].copy()}
                        altered["values"][iteration:] = 12345
                        altered["policy"][iteration + 1:] = model.torque_limit
                        repeated = collect_policy_data(model, altered, n_samples=160,
                                                       remaining=iteration)
                        np.testing.assert_array_equal(repeated["states"], data["states"])
                        np.testing.assert_array_equal(repeated["actions"], data["actions"])
                with self.assertRaises(ValueError):
                    fit_linear_policy(model, solution, remaining=0, data=data)
                for invalid in [-1, 1.5, model.horizon + 1]:
                    with self.assertRaises(ValueError):
                        vi_policy_rollout(model, solution, initial_states(model, 2), invalid)

    def test_snapshot_rollout_keeps_policy_fixed_through_observation_horizon(self):
        model = Pendulum(horizon=12, torque_cost=0.1)
        solution = value_iteration(model, n_theta=101)
        for iteration in [0, 3, 12]:
            batch = vi_policy_rollout(model, solution, [[0.1], [0.3], [1.0]], iteration)
            self.assertEqual(batch["actions"].shape, (3, model.horizon))
            expected = vi_action(model, solution, batch["states"][:, :-1], iteration)
            np.testing.assert_array_equal(batch["actions"], expected)

    def test_viscous_damping_dissipates_energy_with_and_without_control(self):
        model = Pendulum(kind="undamped", dt=1e-5, damping=0.03)
        states = np.array([[0.4, 0.8], [-0.7, -0.5], [1.2, -0.3]])
        for action in [0.0, 0.1]:
            following = step(model, states, action)
            energy = 0.5 * states[:, 1]**2 + np.cos(states[:, 0])
            next_energy = 0.5 * following[:, 1]**2 + np.cos(following[:, 0])
            expected = states[:, 1] * action - model.damping * states[:, 1]**2
            np.testing.assert_allclose((next_energy - energy) / model.dt,
                                       expected, atol=1e-6)
        model = Pendulum(kind="undamped", damping=0.03, horizon=180)
        batch = rollout(model, [0.0, 0.0], [[1.5, 0.4]])
        theta, omega = batch["states"][0].T
        energy = 0.5 * omega**2 + np.cos(theta)
        self.assertTrue(np.all(np.diff(energy) <= 1e-7))
        self.assertLess(energy[-1], energy[0] - 0.1)
        for invalid in [-0.1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                Pendulum(damping=invalid)

    def test_selected_features_include_exact_energy_pumping_terms(self):
        model = Pendulum(kind="undamped")
        states = np.array([[0.0, 0.0], [1.1, -0.8], [-2.0, 1.3]])
        names = ("energy_pump", "omega_cos_theta", "omega_cubed", "omega",
                 "sin_theta", "cos_theta")
        features = policy_features(model, states, names)
        np.testing.assert_allclose(features[:, 0],
                                   0.5 * features[:, 2] + features[:, 1] - features[:, 3])
        np.testing.assert_allclose(features[:, 4], np.sin(states[:, 0]))
        np.testing.assert_allclose(features[:, 5], np.cos(states[:, 0]))
        np.testing.assert_allclose(policy_features(model, states, ["omega", "theta"]), states[:, ::-1])
        np.testing.assert_allclose(policy_action(model, [], states, []), 0)
        with self.assertRaises(ValueError):
            policy_features(Pendulum(), [[0.2]], ["omega"])
        # The same energy feature is used when executing a controller.
        batch = rollout(model, [-0.7], states, feature_names=["energy_pump"])
        theta, omega = batch["states"][:, :-1].transpose(2, 0, 1)
        expected = -0.7 * omega * (0.5 * omega**2 + np.cos(theta) - 1)
        np.testing.assert_allclose(batch["mean"], expected)
        np.testing.assert_allclose(batch["actions"][batch["alive"]],
                                   np.clip(expected, -model.torque_limit, model.torque_limit)[batch["alive"]])

    def test_least_squares_fits_selected_basis_and_handles_dependent_or_empty_features(self):
        model = Pendulum(kind="undamped")
        rng = np.random.default_rng(31)
        states = rng.uniform([-np.pi, -1.2], [np.pi, 1.2], size=(500, 2))
        theta, omega = states.T
        actions = -0.03 * theta + 0.04 * np.sin(2 * theta) - 0.07 * omega * np.cos(theta)
        data = {"states": states, "actions": actions, "batch": None}
        fit = fit_linear_policy(model, None, data=data,
                                feature_names=["theta", "sin_2theta", "omega_cos_theta"])
        np.testing.assert_allclose(fit["coefficients"], [-0.03, 0.04, -0.07], atol=1e-12)
        np.testing.assert_allclose(fit["predictions"], actions, atol=1e-12)
        data["actions"] = -0.05 * omega * (0.5 * omega**2 + np.cos(theta) - 1)
        fit = fit_linear_policy(model, None, data=data,
                                feature_names=["omega", "omega_cos_theta", "omega_cubed", "energy_pump"])
        self.assertEqual(fit["rank"], 3)
        np.testing.assert_allclose(fit["predictions"], data["actions"], atol=1e-12)
        empty = fit_linear_policy(model, None, data=data, feature_names=[])
        self.assertEqual(empty["rank"], 0)
        self.assertEqual(empty["coefficients"].shape, (0,))
        np.testing.assert_array_equal(empty["predictions"], 0)

    def test_single_grid_cell_capture_is_centered_at_upright(self):
        for kind, n_theta, n_omega in [("overdamped", 1601, 81),
                                       ("undamped", 241, 161)]:
            with self.subTest(kind=kind):
                model = Pendulum(kind=kind, horizon=1,
                                 target_theta=np.pi / n_theta,
                                 target_omega=3.0 / (n_omega - 1))
                solution = value_iteration(model, n_theta=n_theta, n_omega=n_omega)
                if model.dimension == 1:
                    states = solution["theta"][:, None]
                    limits = np.array([model.target_theta])
                else:
                    theta, omega = np.meshgrid(solution["theta"], solution["omega"], indexing="ij")
                    states = np.stack([theta, omega], axis=-1)
                    limits = np.array([model.target_theta, model.target_omega])
                self.assertEqual(int(solution["target_mask"].sum()), 1)
                np.testing.assert_allclose(states[solution["target_mask"]], 0.0, atol=1e-14)
                np.testing.assert_allclose(np.diff(solution["theta"]), 2 * model.target_theta)
                if model.dimension == 2:
                    np.testing.assert_allclose(np.diff(solution["omega"]), 2 * model.target_omega)
                inside = 0.9 * limits
                outside = 1.1 * limits[:, None] * np.eye(model.dimension)
                self.assertTrue(in_target(model, inside))
                self.assertFalse(in_target(model, outside).any())
                starts = np.vstack([inside, outside[0]])
                batch = rollout(model, np.zeros(model.dimension), starts)
                np.testing.assert_array_equal(batch["success"], [True, False])
                np.testing.assert_allclose(batch["return"], [1.0, 0.0])

    def test_centered_periodic_grid_interpolates_nodes_and_seam(self):
        for n_theta in [40, 41]:
            model = Pendulum(kind="undamped", horizon=1)
            solution = value_iteration(model, n_theta=n_theta, n_omega=9)
            theta, omega = np.meshgrid(solution["theta"], solution["omega"], indexing="ij")
            states = np.stack([theta, omega], axis=-1)
            values = np.cos(theta) + np.sin(theta) + 0.5 * omega
            for turns in [-2, 0, 3]:
                shifted = states.copy()
                shifted[..., 0] += turns * 2 * np.pi
                indices = _interpolation(model, solution["theta"], solution["omega"], shifted)
                np.testing.assert_allclose(_interpolate(values, indices), values, atol=1e-12)
            seam = np.array([[-np.pi, 0.3], [np.pi, 0.3]])
            indices = _interpolation(model, solution["theta"], solution["omega"], seam)
            predicted = _interpolate(values, indices)
            np.testing.assert_allclose(predicted[0], predicted[1], atol=1e-12)
            expected = np.interp(-np.pi, solution["theta"],
                                 np.cos(solution["theta"]) + np.sin(solution["theta"]),
                                 period=2 * np.pi) + 0.15
            np.testing.assert_allclose(predicted, expected, atol=1e-12)

    def test_unforced_upright_equilibrium_is_unstable(self):
        angles = np.array([0.2, -0.2])
        overdamped = step(Pendulum(), angles[:, None], np.zeros(2))
        self.assertTrue(np.all(np.abs(overdamped[:, 0]) > np.abs(angles)))
        undamped = step(Pendulum(kind="undamped"),
                        np.column_stack((angles, np.zeros(2))), np.zeros(2))
        self.assertTrue(np.all(undamped[:, 1] * angles > 0))
        self.assertTrue(np.all(np.abs(undamped[:, 0]) > np.abs(angles)))

    def test_local_start_sampling_excludes_goal_and_handles_weak_actuators(self):
        for kind, target in [("overdamped", 0.06), ("undamped", 0.12)]:
            for torque_limit in [0.0, 0.01, 0.15, 0.35, 0.65]:
                with self.subTest(kind=kind, torque_limit=torque_limit):
                    model = Pendulum(kind=kind, target_theta=target,
                                     torque_limit=torque_limit)
                    starts = initial_states(model, n=128, seed=19)
                    limits = initial_state_limits(model)
                    self.assertEqual(starts.shape, (128, model.dimension))
                    self.assertTrue(np.isfinite(starts).all())
                    self.assertTrue(np.all(np.abs(starts) <= limits))
                    self.assertFalse(in_target(model, starts).any())
                    np.testing.assert_array_equal(starts,
                                                   initial_states(model, n=128, seed=19))

    def test_geometric_reward_boundary_and_discount_endpoints(self):
        for gamma in [0.5, 0.98, 1.0]:
            model = Pendulum(discount=gamma)
            horizons = np.arange(5)
            expected = [sum(gamma**t for t in range(h)) for h in horizons]
            np.testing.assert_allclose(discount_sum(model, horizons), expected)
        for invalid in [-0.1, 0.0, 1.1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                Pendulum(discount=invalid)
        for invalid in [-0.1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                Pendulum(torque_cost=invalid)

    def test_captured_goal_earns_reward_through_horizon_and_torque_is_applied(self):
        model = Pendulum(horizon=4, discount=0.5, torque_cost=2.0)
        batch = rollout(model, [-20.0], [[0.0], [0.061], [2.0]])
        penalty = model.torque_cost * model.torque_limit**2
        expected_rewards = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [1.0 - penalty, 1.0, 1.0, 1.0],
            [-penalty, -penalty, -penalty, -penalty],
        ])
        np.testing.assert_array_equal(batch["success"], [True, True, False])
        np.testing.assert_allclose(batch["rewards"], expected_rewards)
        np.testing.assert_allclose(batch["return"], [1.875, 1.875 - penalty,
                                                   -1.875 * penalty])
        np.testing.assert_allclose(batch["return"],
                                   batch["goal_return"] + batch["torque_return"])
        np.testing.assert_array_equal(batch["actions"][~batch["alive"]], 0.0)
        self.assertTrue(np.all(np.abs(batch["latent"][batch["alive"]]) >
                               model.torque_limit))
        self.assertTrue(np.isnan(batch["hit_time"][-1]))

    def test_last_transition_reward_and_timeout_have_no_elapsed_time_penalty(self):
        model = Pendulum(horizon=1, discount=0.7, torque_cost=0.0)
        batch = rollout(model, [-2.0], [[0.061], [1.0], [0.0]])
        np.testing.assert_array_equal(batch["success"], [True, False, True])
        np.testing.assert_allclose(batch["return"], [1.0, 0.0, 1.0])
        np.testing.assert_allclose(batch["returns"][:, -1], batch["rewards"][:, -1])
        self.assertEqual(batch["hit_time"][0], model.dt)
        self.assertTrue(np.isnan(batch["hit_time"][1]))
        self.assertEqual(batch["hit_time"][2], 0.0)

    def test_discounted_mc_returns_and_seeded_reproducibility(self):
        model = Pendulum(horizon=40, discount=0.8, torque_cost=0.1)
        starts = np.array([[0.0], [0.05], [0.1], [0.3], [1.0], [2.0]])
        batch = rollout(model, [-4.0], starts, seed=11, exploration=0.9)
        repeated = rollout(model, [-4.0], starts, seed=11, exploration=0.9)
        np.testing.assert_array_equal(batch["states"], repeated["states"])
        for t in [0, 1, 17, 39]:
            powers = model.discount**np.arange(model.horizon - t)
            expected = batch["rewards"][:, t:] @ powers
            np.testing.assert_allclose(batch["returns"][:, t], expected, atol=1e-12)
        np.testing.assert_allclose(batch["return"], batch["returns"][:, 0])
        active = batch["alive"]
        self.assertTrue(np.any(np.abs(batch["latent"][active]) > model.torque_limit))
        np.testing.assert_allclose(batch["actions"][active],
                                   np.clip(batch["latent"][active],
                                           -model.torque_limit, model.torque_limit))
        # The Gaussian score needs the unsaturated mean and latent sample.
        np.testing.assert_allclose(batch["mean"], -4.0 * batch["states"][:, :-1, 0])

    def test_earlier_capture_increases_reward_without_a_time_penalty(self):
        model = Pendulum(horizon=100, discount=0.98, torque_cost=0.0)
        batch = rollout(model, [-10.0], [[0.1], [0.2], [0.3]])
        self.assertTrue(batch["success"].all())
        arrival_steps = np.rint(batch["hit_time"] / model.dt).astype(int)
        expected = np.array([
            sum(model.discount**t for t in range(m - 1, model.horizon))
            for m in arrival_steps
        ])
        np.testing.assert_allclose(batch["return"], expected)
        self.assertTrue(np.all(np.diff(arrival_steps) > 0))
        self.assertTrue(np.all(np.diff(batch["return"]) < 0))

    def test_undamped_unforced_orbit_does_not_reward_crossing_at_speed(self):
        model = Pendulum(kind="undamped", horizon=180, target_theta=0.12,
                         torque_cost=0.0)
        # Energy above the upright separatrix produces full rotations, with
        # nonzero speed whenever the orbit crosses the upright angle.
        batch = rollout(model, [0.0, 0.0], [[0.25, 0.9]])
        self.assertFalse(batch["success"][0])
        self.assertEqual(batch["return"][0], 0.0)
        theta, omega = batch["states"][0].T
        energy = 0.5 * omega**2 + np.cos(theta)
        np.testing.assert_allclose(energy, energy[0], atol=2e-5, rtol=0)
        self.assertLess(np.min(np.abs(theta)), model.target_theta)

    def test_vi_and_greedy_actions_maximize_immediate_plus_discounted_reward(self):
        model = Pendulum(horizon=8, discount=0.7, torque_cost=3.0)
        solution = value_iteration(model, n_theta=101, n_actions=5)
        theta = solution["theta"]
        states = theta[~solution["target_mask"], None]
        actions = solution["actions"]
        following = step(model, states[:, None, :], actions[None, :])
        arrived = in_target(model, following)
        for h in [1, 3, 8]:
            continuation = np.interp(following[..., 0], theta,
                                     solution["values"][h - 1], period=2 * np.pi)
            goal_value = sum(model.discount**t for t in range(h - 1))
            continuation = np.where(arrived, goal_value, continuation)
            q = (arrived.astype(float) - model.torque_cost * actions**2 +
                 model.discount * continuation)
            best = q.argmax(axis=1)
            np.testing.assert_allclose(solution["values"][h, ~solution["target_mask"]],
                                       q[np.arange(len(states)), best], atol=2e-6)
            expected_actions = actions[best]
            np.testing.assert_allclose(vi_action(model, solution, states, h),
                                       expected_actions)
            np.testing.assert_allclose(solution["policy"][h, ~solution["target_mask"]],
                                       expected_actions)
            np.testing.assert_allclose(solution["values"][h, solution["target_mask"]],
                                       sum(model.discount**t for t in range(h)))

    def test_local_upright_vi_reaches_goal_at_analytic_bang_bang_time(self):
        model = Pendulum(torque_cost=0.0)
        solution = value_iteration(model, n_theta=321)
        starts = np.array([[0.08], [0.12], [0.18], [0.25], [0.3]])
        self.assertTrue(np.all(starts[:, 0] < np.arcsin(model.torque_limit)))
        batch = vi_rollout(model, solution, starts)
        continuous_times = np.array([
            quad(lambda theta: 1 / (model.torque_limit - np.sin(theta)),
                 model.target_theta, theta)[0]
            for theta in starts[:, 0]
        ])
        self.assertTrue(batch["success"].all())
        errors = batch["hit_time"] - continuous_times
        self.assertTrue(np.all(errors >= -1e-6))
        self.assertTrue(np.all(errors <= model.dt + 1e-6))

    def test_overdamped_torque_cannot_capture_outside_the_upright_basin(self):
        model = Pendulum(torque_cost=0.0)
        boundary = np.arcsin(model.torque_limit)
        starts = [[boundary + 0.02], [-boundary - 0.02], [1.0], [-1.0]]
        batch = rollout(model, [-100.0], starts)
        self.assertFalse(batch["success"].any())
        np.testing.assert_array_equal(batch["return"], 0.0)
        self.assertTrue(np.all(np.abs(batch["states"][..., 0]) > boundary))

    def test_critic_can_predict_negative_rewards_and_has_positive_goal_boundary(self):
        model = Pendulum(horizon=1, discount=0.8, torque_cost=5.0)
        states = np.linspace(0.5, 2.0, 64)[:, None]
        batch = rollout(model, [20.0], states)
        critic = fit_value(model, batch)
        predictions = predict_value(model, critic, states, 1)
        penalty = model.torque_cost * model.torque_limit**2
        self.assertTrue(np.all(predictions < 0))
        self.assertTrue(np.all(predictions >= -penalty - 1e-12))
        np.testing.assert_allclose(batch["return"], -penalty)
        np.testing.assert_allclose(predict_value(model, critic, [[0.0]], 1), [1.0])
        np.testing.assert_allclose(predict_value(model, critic, states, 0), 0.0)
        captured = rollout(model, [-0.4], [[0.0], [0.01]])
        captured_critic = fit_value(model, captured)
        self.assertEqual(captured_critic["n_samples"], 0)
        np.testing.assert_allclose(
            predict_value(model, captured_critic, [[0.0], [0.01]], 1), [1.0, 1.0])

    def test_default_actor_critic_improves_reward_from_random_initialization(self):
        for kind, horizon, target_theta, rho in [("overdamped", 100, np.pi / 1601, 0.1),
                                                  ("undamped", 180, np.pi / 241, 0.1)]:
            with self.subTest(kind=kind):
                model = Pendulum(kind=kind, horizon=horizon, target_theta=target_theta,
                                 target_omega=3.0 / 160, discount=0.98, torque_cost=rho)
                result = train_actor_critic(model, iterations=30, seed=7)
                initial = result["snapshots"][0]
                final = result["snapshots"][-1]
                expected = np.random.default_rng(7).uniform(-0.15, 0.15,
                                                            size=model.dimension)
                np.testing.assert_array_equal(initial["coefficients"], expected)
                self.assertGreater(final["metrics"]["mean_return"],
                                   initial["metrics"]["mean_return"] + 1.0)
                if kind == "undamped":
                    self.assertLess(final["coefficients"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
