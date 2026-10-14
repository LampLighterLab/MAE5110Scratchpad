"""End-to-end numerical contracts for the shared learning interface."""

from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix

from core import learning_algorithms as algorithms
from core import learning_models as models
from core import learning_views as views
from core import walker_q_iteration as walker


def two_state_model():
    # Stay or switch. State 1 earns reward; there is no terminal state.
    return models.LearningModel("mdp", np.array([[0.], [1.]]), np.arange(2),
                                csr_matrix([[1., 0, 0], [0, 1., 0], [0, 1., 0], [1., 0, 0]]),
                                {"target_state": 1}, ("State",), "Action", (np.arange(2),))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = [models.build_model("mdp"),
                      models.build_model("overdamped", {"angles": 21, "actions": 5}),
                      models.build_model("pendulum", {"angles": 15, "velocities": 9, "actions": 5}),
                      models.build_model("walker", {"states": 31, "actions": 7})]

    def test_all_models_are_stochastic_with_finite_current_state_rewards(self):
        for model in self.models:
            with self.subTest(model=model.kind):
                assert_allclose(np.asarray(model.matrix.sum(axis=1)).ravel(), 1)
                self.assertTrue(np.all(model.matrix.data >= 0))
                reward, initial = models.prepare_tables(model, models.REWARDS[model.kind])
                self.assertEqual(reward.shape, model.shape)
                self.assertTrue(np.all(np.isfinite(reward)))
                assert_array_equal(initial, 0)
                assert_allclose(model.features(4).sum(axis=1), 1)
                assert_allclose(model.features(4, model.states), model.features(4))

    def test_walker_capture_maps_to_real_zero_and_keeps_assignment_dynamics(self):
        model = self.models[-1]
        grid = model.context["grid"]
        assert_array_equal(model.matrix.toarray(), grid.matrix.toarray())
        for row in np.flatnonzero(grid.status.ravel() == 1):
            self.assertEqual(model.matrix[row, 0], 1)
        assert_array_equal(model.matrix[:model.shape[1], 0].toarray(), 1)

    def test_fitted_bellman_updates_stay_within_discounted_reward_bounds(self):
        for model in self.models:
            with self.subTest(model=model.kind):
                reward, initial = models.prepare_tables(model, models.REWARDS[model.kind])
                settings = algorithms.Settings(approximation=True, features=16, iterations=200)
                result = algorithms.train(model, reward, initial, settings)
                lower, upper = min(0, reward.min() / (1 - settings.discount)), max(0, reward.max() / (1 - settings.discount))
                self.assertGreaterEqual(result.final.q.min(), lower - 1e-9)
                self.assertLessEqual(result.final.q.max(), upper + 1e-9)
                self.assertLess(result.changes[-1], result.changes[1] * 1e-3)

    def test_editors_use_physical_current_state_action_and_parameter_dictionary(self):
        model = self.models[2]
        reward, initial = models.prepare_tables(model,
            'def reward(state, action, params):\n    return state[..., 0] + action + params["bonus"]',
            'def initial_q(state, action, params):\n    return params["discount"]',
            params={"bonus": 7}, discount=0.9)
        assert_allclose(reward, model.states[:, :1] + model.actions[None, :] + 7)
        assert_allclose(initial, 0.9)
        with self.assertRaises(walker.EditorError):
            models.prepare_tables(model, 'def reward(state, action, params):\n    return np.nan')

    def test_all_learning_combinations_use_the_same_interface(self):
        for model in self.models:
            reward, initial = models.prepare_tables(model, models.REWARDS[model.kind])
            for method in ("planning", "rl"):
                for approximate in (False, True):
                    for policy in ("greedy", "fit", "iteration"):
                        with self.subTest(model=model.kind, method=method, approximate=approximate, policy=policy):
                            settings = algorithms.Settings(method=method, policy=policy, approximation=approximate,
                                iterations=2, samples=8, features=4, policy_features=4, evaluation_sweeps=2)
                            result = algorithms.train(model, reward, initial, settings)
                            self.assertTrue(np.isfinite(result.final.q).all())
                            assert_allclose(result.final.policy.sum(axis=1), 1)
                            self.assertEqual(result.final.iteration, 2)
                            self.assertEqual(result.final.visits.sum(), 16 if method == "rl" else 0)
                            assert_array_equal(result.snapshots[0].visits, 0)


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.model = two_state_model()
        self.reward, self.initial = models.prepare_tables(self.model, models.REWARDS["mdp"])

    def test_value_iteration_matches_hand_computed_optimum(self):
        result = algorithms.train(self.model, self.reward, self.initial,
                                   algorithms.Settings(discount=0.9, iterations=300))
        assert_allclose(result.final.q, [[8.1, 9.0], [10.0, 9.1]], atol=1e-10)
        assert_allclose(result.final.value, [9, 10], atol=1e-10)

    def test_classical_policy_iteration_reaches_same_optimum(self):
        result = algorithms.train(self.model, self.reward, self.initial,
                                   algorithms.Settings(discount=0.9, iterations=4, policy="iteration"))
        assert_allclose(result.final.q, [[8.1, 9.0], [10.0, 9.1]], atol=1e-10)
        self.assertLess(result.residuals[-1], 1e-10)
        evaluated = algorithms.evaluate_policy(self.model, self.reward, result.final.policy, 0.9)
        assert_allclose(evaluated, [9, 10])

    def test_sampled_update_is_one_actual_td_update_not_a_full_backup(self):
        settings = algorithms.Settings(method="rl", iterations=1, samples=1, learning_rate=0.2,
                                       discount=0.9, seed=4)
        initial = np.full(self.model.shape, 2.)
        result = algorithms.train(self.model, self.reward, initial, settings)
        state, action = np.argwhere(result.final.visits)[0]
        expected = initial.copy()
        expected[state, action] += 0.2 * (self.reward[state, action] + 0.9 * 2 - 2)
        assert_allclose(result.final.q, expected)
        assert_array_equal(result.snapshots[0].q, initial)

    def test_rl_reproducibility_and_learning_progress(self):
        settings = algorithms.Settings(method="rl", iterations=30, samples=100, discount=0.9, epsilon=0.5)
        first = algorithms.train(self.model, self.reward, self.initial, settings)
        second = algorithms.train(self.model, self.reward, self.initial, settings)
        assert_array_equal(first.final.q, second.final.q)
        assert_array_equal(first.final.visits, second.final.visits)
        self.assertLess(first.residuals[-1], first.residuals[0] * 0.01)

    def test_policy_fit_changes_deployment_not_value_learning(self):
        model = models.build_model("mdp")
        reward, initial = models.prepare_tables(model, models.REWARDS["mdp"])
        settings = algorithms.Settings(iterations=100, policy_features=2)
        greedy = algorithms.train(model, reward, initial, settings)
        fitted = algorithms.train(model, reward, initial, replace(settings, policy="fit"))
        assert_array_equal(greedy.final.q, fitted.final.q)
        expected = model.features(2) @ fitted.final.policy_weights
        assert_array_equal(fitted.final.policy.argmax(axis=1), expected.argmax(axis=1))
        self.assertLess(fitted.policy_parameters, np.prod(model.shape))
        self.assertTrue(np.any(fitted.final.policy != greedy.final.policy))

    def test_approximation_is_used_during_learning_and_shares_values(self):
        model = models.build_model("mdp")
        reward, initial = models.prepare_tables(model, models.REWARDS["mdp"])
        result = algorithms.train(model, reward, initial,
                                   algorithms.Settings(approximation=True, features=2, iterations=10))
        assert_allclose(result.final.q[0], result.final.q[2])
        assert_allclose(result.final.q, model.features(2) @ result.final.q_weights)
        self.assertEqual(result.critic_parameters, 2 * model.shape[1])

    def test_terminal_has_zero_bootstrap_even_with_optimistic_initialization(self):
        model = two_state_model()
        model.matrix = csr_matrix(np.tile([0., 0, 1.], (4, 1)))
        result = algorithms.train(model, self.reward, np.full((2, 2), 1000.), algorithms.Settings(iterations=1))
        assert_array_equal(result.final.q, self.reward)

    def test_rollout_changes_with_selected_policy_and_keeps_learning_label(self):
        model = models.build_model("overdamped", {"angles": 21, "actions": 5})
        for sign in (-1, 1):
            reward, initial = models.prepare_tables(model, f'def reward(state, action, params):\n    return {sign} * action')
            result = algorithms.train(model, reward, initial, algorithms.Settings(iterations=20, policy="fit", policy_features=4))
            trace = views.physical_rollout(model, result.final, [0.5], fitted=True, steps=4)
            assert_allclose(trace["actions"], sign * model.actions[-1])
            player = views.animation_html(model, trace, f"Iteration 20 · sign {sign}")
            payload = json.loads(player.split("const data=", 1)[1].split(",canvas=", 1)[0])
            self.assertEqual(payload["label"], f"Iteration 20 · sign {sign}")
            self.assertNotIn("__DATA__", player)


class NotebookReactivityTests(unittest.TestCase):
    def test_reward_and_snapshot_changes_reach_the_fitted_walker_rollout(self):
        import matplotlib.pyplot as plt
        import nb_learning_lab

        ui = lambda value: SimpleNamespace(value=value)
        controls = {"iterations": 1, "discount": 0.95, "samples": 8}
        overrides = {"model_choice": ui("walker"), "learning_mode": ui("planning"),
                     "approximate_value": ui(False), "policy_choice": ui("fit"),
                     "training_controls": ui(controls), "training_defaults": controls,
                     "initial_editor": ui(None), "extra_parameters": ui("{}")}
        for sign, iteration in ((1, 1), (-1, 1), (1, 0)):
            with self.subTest(reward_sign=sign, iteration=iteration):
                _, defs = nb_learning_lab.app.run(defs=overrides | {
                    "reward_editor": ui(f"def reward(state, action, params):\n    return {sign} * action"),
                    "iteration": ui(iteration),
                })
                self.assertIsNone(defs["experiment_error"])
                self.assertEqual(defs["snapshot"].iteration, iteration)
                expected = defs["model"].actions[-1 if sign == 1 and iteration == 1 else 0]
                self.assertAlmostEqual(defs["policy_trace"]["actions"][0], expected)
                plt.close("all")


if __name__ == "__main__":
    unittest.main()
