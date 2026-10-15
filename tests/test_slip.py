"""Physical invariants and learning/rollout integration for the SLIP model."""

import json
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from core import learning_algorithms as algorithms
from core import learning_models as models
from core import learning_views as views
from core import slip


class SLIPTests(unittest.TestCase):
    def test_apex_reduction_reconstructs_the_paper_energy(self):
        p = slip.PAPER_PARAMS
        self.assertEqual(p, dict(mass=80., stiffness=8200., resting_length=1., gravity=9.81, total_energy=1860.))
        for s in (0, .2, .6, 1):
            apex = slip.apex_state(s, p)
            self.assertEqual(apex[3], 0)
            self.assertAlmostEqual(slip.apex_coordinate(apex, p), s)
            self.assertAlmostEqual(p['mass'] * (p['gravity'] * apex[1] + apex[2]**2 / 2), p['total_energy'])

    def test_hop_preserves_energy_and_has_fixed_foot_during_stance(self):
        p = slip.PAPER_PARAMS
        apex = slip.apex_state(.6, p)
        result = slip.step(apex, np.pi / 6, p, record=True)
        self.assertEqual(result['status'], slip.RETURN)
        self.assertEqual(result['state'][3], 0)
        trace = result['trace']
        stance = trace[:, 7] == 1
        self.assertTrue(np.any(stance) and np.any(~stance))
        assert_allclose(trace[stance, 5], trace[stance, 5][0])
        assert_array_equal(trace[stance, 6], 0)
        leg_length = np.linalg.norm(trace[:, 1:3] - trace[:, 5:7], axis=1)
        spring_energy = np.where(stance, .5 * p['stiffness'] * (p['resting_length'] - leg_length)**2, 0)
        energy = p['mass'] * (p['gravity'] * trace[:, 2] + np.sum(trace[:, 3:5]**2, axis=1) / 2) + spring_energy
        assert_allclose(energy, p['total_energy'], rtol=1e-8)
        self.assertTrue(np.all(np.diff(trace[:, 0]) >= 0))
        refined = slip.step(apex, np.pi / 6, p, max_step=.005, rtol=1e-11, atol=1e-13)
        assert_allclose(result['state'], refined['state'], atol=1e-8, rtol=1e-8)

    def test_vertical_hop_does_not_count_as_reversal(self):
        result = slip.step(slip.apex_state(1, slip.PAPER_PARAMS), 0, slip.PAPER_PARAMS)
        self.assertEqual(result['status'], slip.RETURN)
        self.assertAlmostEqual(result['next_state'], 1, places=9)

    def test_infeasible_and_failed_returns_do_not_become_apex_zero(self):
        p = slip.PAPER_PARAMS
        for state, angle, expected in ((.1, 0, slip.INFEASIBLE), (0, .5, slip.FALL), (.6, np.pi/2, slip.FALL)):
            result = slip.step(slip.apex_state(state, p), angle, p)
            self.assertEqual(result['status'], expected)
            self.assertTrue(np.isnan(result['next_state']))
        model = models.build_model('slip', {'states': 21, 'actions': 17})
        failed = model.context['status'].ravel() != slip.RETURN
        assert_allclose(model.matrix[failed, model.terminal].toarray(), 1)
        assert_array_equal(model.matrix[failed, :model.terminal].toarray(), 0)
        successful = ~failed
        interpolated = np.asarray(model.matrix @ np.r_[model.states[:, 0], 0]).ravel()
        assert_allclose(interpolated[successful], model.context['next_states'].ravel()[successful], atol=1e-8)

    def test_selected_learning_snapshot_controls_physical_animation(self):
        model = models.build_model('slip', {'states': 31, 'actions': 21})
        traces = []
        for sign in (-1, 1):
            reward, initial, edges = models.prepare_tables(model,
                f'def reward(state, action, next_state, params):\n    return {sign} * action')
            for approximate, policy in ((False, 'greedy'), (True, 'greedy'), (False, 'fit')):
                settings = algorithms.Settings(discount=0, iterations=1, approximation=approximate, policy=policy)
                experiment = algorithms.train(model, reward, initial, settings, transition_rewards=edges)
                trace = views.physical_rollout(model, experiment.final, [.6], fitted=policy == 'fit', steps=3)
                expected = model.actions[0 if sign < 0 else -1]
                self.assertAlmostEqual(trace['actions'][0], expected)
                assert_allclose(trace['trace'][:, 8], expected)
                player = views.animation_html(model, trace, f'SLIP iteration 1 sign {sign}')
                payload = json.loads(player.split('const data=', 1)[1].split(',canvas=', 1)[0])
                assert_allclose(payload['frames'], trace['trace'])
                self.assertIn(f'sign {sign}', payload['label'])
                traces.append(trace)
        self.assertNotEqual(traces[0]['trace'].shape, traces[-1]['trace'].shape)

    def test_policy_observes_continuous_next_apex(self):
        decisions = []
        def choose(state):
            decisions.append(state)
            return np.pi/6
        trace = slip.rollout(.6, choose, slip.PAPER_PARAMS, steps=2)
        self.assertEqual(len(decisions), 2)
        self.assertAlmostEqual(decisions[1], trace['states'][1, 0])
        self.assertNotAlmostEqual(decisions[1] * 100, round(decisions[1] * 100), places=5)
        self.assertGreater(trace['trace'][-1, 1], 0)


if __name__ == '__main__':
    unittest.main()
