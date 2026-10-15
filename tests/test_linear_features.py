"""Full feature-vector contracts, representability, and notebook deployment."""

import json
from types import SimpleNamespace
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from core import learning_algorithms as algorithms
from core import learning_models as models
from core import learning_views as views
from core import linear_features


def expression_source(expression):
    return f'def features(state, parameters):\n    return np.array([{expression}], dtype=float)\n'


class FeatureVectorTests(unittest.TestCase):
    def test_suggestions_generate_complete_valid_vectors_for_all_models(self):
        small = {'mdp': {'states': 5}, 'overdamped': {'angles': 11, 'actions': 3},
                 'pendulum': {'angles': 9, 'velocities': 5, 'actions': 3},
                 'walker': {'states': 21, 'actions': 5}, 'slip': {'states': 11, 'actions': 5}}
        for kind, settings in small.items():
            with self.subTest(kind=kind):
                model = models.build_model(kind, settings)
                catalog = linear_features.suggestions(kind)
                self.assertLessEqual(len(catalog), 6)
                for selected in (None, (), tuple(catalog)):
                    basis = linear_features.build_basis(model, linear_features.source_for(kind, selected))
                    self.assertTrue(np.isfinite(basis.grid).all())
                    assert_allclose(basis.evaluate(model.states[:2]), basis.grid[:2])
                    if selected == ():
                        self.assertEqual(basis.grid.shape, (len(model.states), 0))
                if kind == 'mdp':
                    basis = linear_features.build_basis(model, linear_features.source_for(kind, ('one_hot',)))
                    assert_array_equal(basis.grid, np.eye(len(model.states)))

    def test_array_order_units_and_entire_vector_are_preserved(self):
        model = models.build_model('pendulum', {'angles': 9, 'velocities': 5, 'actions': 3})
        source = '''def features(state, parameters):
    theta, omega = state
    return np.array([omega**2, 1., theta*omega, np.sin(theta), theta])
'''
        basis = linear_features.build_basis(model, source)
        points = np.array([[.3, 2.], [-.7, -.4]])
        expected = np.column_stack([points[:, 1]**2, np.ones(2), points[:, 0]*points[:, 1],
                                    np.sin(points[:, 0]), points[:, 0]])
        assert_allclose(basis.evaluate(points), expected)
        self.assertEqual(basis.labels, ('φ[0]', 'φ[1]', 'φ[2]', 'φ[3]', 'φ[4]'))
        # User code has exactly one column; no bias or suggested terms are added.
        single = linear_features.build_basis(model, expression_source('np.linalg.norm(state)'))
        self.assertEqual(single.grid.shape[1], 1)
        assert_allclose(single.evaluate(points)[:, 0], np.linalg.norm(points, axis=1))

    def test_scalar_function_uses_copied_parameters_and_arbitrary_numpy(self):
        model = models.build_model('slip', {'states': 11, 'actions': 5})
        parameters = {'scale': 2., 'nested': {'count': 0}}
        source = '''def features(state, parameters):
    assert np.ndim(state) == 0
    parameters["nested"]["count"] += 1
    speed = np.sqrt(np.maximum(0, 2 * parameters["total_energy"] * (1-state) / parameters["mass"]))
    return np.array([speed, np.tanh(state / parameters["scale"]), 3.0])
'''
        basis = linear_features.build_basis(model, source, parameters)
        self.assertEqual(parameters['nested']['count'], 0)
        assert_allclose(basis.grid[:, 0], np.sqrt(2*1860/80*(1-model.states[:, 0])))
        parameters['scale'] = 100.
        assert_allclose(basis.evaluate([[.37]])[0], [np.sqrt(2*1860/80*.63), np.tanh(.37/2), 3])

    def test_invalid_arrays_get_clear_errors(self):
        model = models.build_model('overdamped', {'angles': 11, 'actions': 3})
        bad_sources = (
            '', 'def features(:',
            'def other(state, parameters): return np.array([])',
            'def features(state): return np.array([])',
            'def features(state, parameters): return {"old contract": state}',
            'def features(state, parameters): return [1., state]',
            'def features(state, parameters): return np.array(state)',
            'def features(state, parameters): return np.array([[1., state]])',
            'def features(state, parameters): return np.array([np.nan])',
            'def features(state, parameters): return np.array([1j])',
            'def features(state, parameters): return np.zeros(1 if state < 0 else 2)',
            'def features(state, parameters): return np.random.random(3)',
        )
        for source in bad_sources:
            with self.subTest(source=source), self.assertRaisesRegex(linear_features.FeatureError, 'Q features'):
                linear_features.build_basis(model, source)
        two_state = models.build_model('pendulum', {'angles': 9, 'velocities': 5, 'actions': 3})
        with self.assertRaisesRegex(linear_features.FeatureError, 'read-only'):
            linear_features.build_basis(two_state,
                'def features(state, parameters):\n    state[:] = 0\n    return np.array([1.])')

    def test_polynomial_array_recovers_known_coefficients(self):
        model = models.build_model('slip', {'states': 11, 'actions': 5})
        reward, initial, _ = models.prepare_tables(model,
            'def reward(state, action, next_state, params):\n    return 2 - 3*state + 4*state**2 + action')
        result = algorithms.train(model, reward, initial,
            algorithms.Settings(approximation=True, feature_source=expression_source('1., state, state**2'),
                                iterations=1, discount=0, ridge=1e-12))
        assert_allclose(result.final.q_weights, np.vstack([2+model.actions, np.full(5, -3), np.full(5, 4)]), atol=1e-9)
        assert_allclose(result.final.q, reward, atol=1e-10)

    def test_one_hot_array_recovers_tabular_values_with_or_without_redundant_bias(self):
        model = models.build_model('mdp', {'states': 5})
        reward, initial, edges = models.prepare_tables(model, models.REWARDS['mdp'])
        table = algorithms.train(model, reward, initial, algorithms.Settings(iterations=30))
        for selected in (('one_hot',), ('bias', 'one_hot')):
            result = algorithms.train(model, reward, initial,
                algorithms.Settings(approximation=True, feature_source=linear_features.source_for('mdp', selected),
                                    iterations=30, ridge=1e-12), transition_rewards=edges)
            assert_allclose(result.final.q, table.final.q, atol=1e-9)

    def test_sampled_td_updates_exactly_the_returned_vector(self):
        model = models.build_model('mdp', {'states': 3, 'actions': 2})
        settings = algorithms.Settings(method='rl', approximation=True, feature_source=expression_source('1., state'),
            iterations=1, samples=1, learning_rate=.2, discount=0, seed=4)
        result = algorithms.train(model, np.full(model.shape, 2.), np.zeros(model.shape), settings)
        state, action = np.argwhere(result.final.visits)[0]
        phi = np.array([1., float(state)])
        weights = np.zeros((2, 2))
        weights[:, action] = .2 * 2 * phi / (phi @ phi)
        assert_allclose(result.final.q_weights, weights)
        assert_allclose(result.final.q, result.final.feature_basis.grid @ weights)

    def test_empty_arrays_give_zero_scores_in_planning_and_rl(self):
        model = models.build_model('mdp')
        reward, initial, edges = models.prepare_tables(model, models.REWARDS['mdp'])
        for method in ('planning', 'rl'):
            result = algorithms.train(model, reward, initial,
                algorithms.Settings(method=method, approximation=True, policy='fit',
                                    feature_source=expression_source(''), policy_feature_source=expression_source(''),
                                    iterations=2, samples=4), transition_rewards=edges)
            self.assertEqual(result.critic_parameters, 0)
            self.assertEqual(result.policy_parameters, 0)
            assert_array_equal(result.final.q, 0)
            assert_array_equal(result.final.policy.argmax(axis=1), 0)

    def test_array_features_support_all_learning_modes(self):
        model = models.build_model('overdamped', {'angles': 31, 'actions': 3})
        source = expression_source('np.sin(parameters["frequency"] * state)')
        reward, initial, edges = models.prepare_tables(model,
            'def reward(state, action, next_state, params):\n    return 2*np.sin(params["frequency"]*state)*action',
            params={'frequency': 3})
        for method in ('planning', 'rl'):
            for policy in ('greedy', 'fit', 'iteration'):
                with self.subTest(method=method, policy=policy):
                    result = algorithms.train(model, reward, initial,
                        algorithms.Settings(method=method, policy=policy, approximation=True,
                            feature_source=source, policy_feature_source=source,
                            discount=0, iterations=1, samples=16, ridge=1e-12),
                        feature_parameters={'frequency': 3}, transition_rewards=edges)
                    assert_allclose(result.final.q, result.final.feature_basis.grid @ result.final.q_weights)
                    if method == 'planning':
                        assert_allclose(result.final.q_weights[0], 2*model.actions, atol=1e-10)
                    if policy == 'fit':
                        scores = result.final.policy_feature_basis.grid @ result.final.policy_weights
                        assert_array_equal(scores.argmax(axis=1), result.final.policy.argmax(axis=1))

    def test_policy_array_changes_left_right_behavior_without_changing_q(self):
        model = models.build_model('overdamped', {'angles': 41, 'actions': 3})
        reward, initial, _ = models.prepare_tables(model,
            'def reward(state, action, next_state, params):\n    return state * action')
        experiments, actions = [], []
        for expression in ('np.cos(state)', 'state'):
            result = algorithms.train(model, reward, initial,
                algorithms.Settings(approximation=True, policy='fit', feature_source=expression_source('state'),
                    policy_feature_source=expression_source(expression), iterations=1, discount=0))
            experiments.append(result)
            actions.append([views.physical_rollout(model, result.final, [start], fitted=True, steps=1)['actions'][0]
                            for start in (-.5, .5)])
            figure = views.feature_figure(model, result.final, 0)
            self.assertEqual(figure.axes[0].lines[0].get_label(), 'φ[0]')
            self.assertEqual(figure.axes[2].lines[0].get_label(), 'ψ[0]')
        assert_array_equal(experiments[0].final.q, experiments[1].final.q)
        self.assertEqual(actions[0][0], actions[0][1])
        assert_allclose(actions[1], [-model.actions[-1], model.actions[-1]])

    def test_inactive_source_does_not_block_tabular_learning(self):
        model = models.build_model('mdp')
        reward, initial, _ = models.prepare_tables(model, models.REWARDS['mdp'])
        result = algorithms.train(model, reward, initial,
            algorithms.Settings(feature_source='invalid code', policy_feature_source='invalid code', iterations=1))
        self.assertIsNone(result.final.feature_basis)
        assert_allclose(result.final.q, reward)


class FeatureEditorTests(unittest.TestCase):
    def run_notebook(self, overrides):
        import matplotlib.pyplot as plt
        import nb_learning_lab
        ui = lambda value: SimpleNamespace(value=value)
        settings = {'iterations': 1, 'discount': 0., 'samples': 4}
        _, defs = nb_learning_lab.app.run(defs={
            'model_choice': ui('overdamped'), 'approximate_value': ui(True),
            'policy_choice': ui('fit'), 'learning_mode': ui('planning'),
            'training_controls': ui(settings), 'training_defaults': settings,
        } | overrides)
        self.assertIsNone(defs['experiment_error'])
        plt.close('all')
        return defs

    def test_suggestion_checkboxes_populate_independent_complete_editors(self):
        ui = lambda value: SimpleNamespace(value=value)
        q, policy = {'cos_theta': True}, {'theta': True, 'bias': True}
        defs = self.run_notebook({'q_suggestions': ui(q), 'q_suggestion_defaults': q,
                                  'policy_suggestions': ui(policy), 'policy_suggestion_defaults': policy})
        self.assertEqual(defs['settings'].feature_source, defs['q_feature_default'])
        self.assertEqual(defs['settings'].policy_feature_source, defs['policy_feature_default'])
        self.assertIn('np.cos(theta)', defs['q_feature_default'])
        self.assertEqual(defs['snapshot'].q_weights.shape[0], 1)
        self.assertEqual(defs['snapshot'].policy_weights.shape[0], 2)
        assert_allclose(defs['snapshot'].feature_basis.grid[:, 0], np.cos(defs['model'].states[:, 0]))

    def test_user_array_replaces_generated_code_without_hidden_columns(self):
        ui = lambda value: SimpleNamespace(value=value)
        code = expression_source('np.sin(state), 2.0')
        defs = self.run_notebook({'q_feature_editor': ui(code),
                                  'q_feature_default': linear_features.source_for('overdamped', ('cos_theta',)),
                                  'policy_feature_editor': ui(expression_source('state')),
                                  'policy_feature_default': linear_features.source_for('overdamped')})
        self.assertEqual(defs['snapshot'].q_weights.shape[0], 2)
        assert_allclose(defs['snapshot'].feature_basis.grid[:, 1], 2.)
        self.assertEqual(defs['snapshot'].policy_weights.shape[0], 1)
        # Changing parameters or the other checklist must not recreate this editor.
        import nb_learning_lab
        nb_learning_lab.app._maybe_initialize()
        cells = [cell._cell for _, cell in nb_learning_lab.app._cell_manager.valid_cells()]
        q_cell = next(cell for cell in cells if 'q_feature_editor' in cell.defs)
        self.assertFalse(q_cell.refs & {'policy_suggestions', 'available_params', 'model'})

    def test_parameter_edits_reach_array_features_and_physical_animation(self):
        ui = lambda value: SimpleNamespace(value=value)
        source = expression_source('np.sin(parameters["frequency"] * state)')
        rollout = {'state_0': 1.3, 'steps': 2, 'seed': 0}
        saved = []
        for frequency in (1, 3):
            defs = self.run_notebook({
                'q_feature_editor': ui(source), 'q_feature_default': source,
                'policy_feature_editor': ui(source), 'policy_feature_default': source,
                'rollout_controls': ui(rollout), 'rollout_defaults': rollout,
                'reward_editor': ui('def reward(state, action, next_state, params):\n    return np.sin(params["frequency"]*state)*action'),
                'extra_parameters': ui(json.dumps({'frequency': frequency})), 'initial_editor': ui(None),
            })
            self.assertAlmostEqual(defs['policy_trace']['actions'][0],
                                   np.sign(np.sin(frequency * 1.3)) * defs['model'].actions[-1])
            saved.append(defs['snapshot'])
        assert_allclose(saved[0].feature_basis.evaluate([[1.3]]), [[np.sin(1.3)]])
        assert_allclose(saved[1].feature_basis.evaluate([[1.3]]), [[np.sin(3*1.3)]])


if __name__ == '__main__':
    unittest.main()
