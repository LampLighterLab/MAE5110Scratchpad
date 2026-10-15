"""Physical-event checks and independent finite-MDP checks for the walker lesson."""

import unittest
from unittest.mock import patch
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from scipy.sparse import csr_matrix

from core import walker_q_iteration as core
from core import walker_rollout as playback
from core import walker_stopping as stopping


class WalkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = stopping.controller_params()
        cls.region = stopping.capture_region()
        cls.states = np.linspace(0, np.sqrt(19.62), 101)
        cls.actions = np.linspace(np.pi / 8, np.pi / 7, 17)
        cls.grid = core.build_grid(cls.states, cls.actions, region=cls.region)

    def test_numeric_return_matches_independent_energy_map(self):
        returning = self.grid.status == stopping.RETURN
        exact = core.return_map(self.states[:, None], self.actions[None, :])
        np.testing.assert_allclose(self.grid.next_speed[returning], exact[returning], rtol=2e-8, atol=2e-8)
        self.assertGreater(returning.sum(), 500)

    def test_capture_states_settle_under_bounded_ankle_controller(self):
        capture = self.grid.status == stopping.CAPTURE
        points = self.grid.transition_states[:, capture][:, ::7]
        self.assertGreater(points.shape[1], 50)
        status, final, _ = stopping.simulate_balance(points, self.params)
        np.testing.assert_array_equal(status, stopping.CAPTURE)
        self.assertTrue(np.all(stopping.has_settled(final, self.params)))
        # Includes captures away from theta=0, not a section-speed threshold.
        self.assertTrue(np.any(np.abs(points[0]) > 0.1))
        torque = stopping.compute_ankle_torque(points, self.params)
        self.assertTrue(np.all(torque >= self.params['torque_min']))
        self.assertTrue(np.all(torque <= self.params['torque_max']))

    def test_capture_classification_survives_timestep_refinement(self):
        refined = stopping.simulate_transitions(self.states[::5, None], self.actions[None, ::4],
            self.region, self.params | {'timestep': self.params['timestep'] / 2})
        np.testing.assert_array_equal(refined['status'], self.grid.status[::5, ::4])

    def test_sparse_rows_and_terminal_semantics_for_both_interpolators(self):
        for kind in ['nearest', 'linear']:
            grid = self.grid if kind == 'nearest' else core.build_grid(self.states, self.actions, interpolation=kind)
            np.testing.assert_allclose(np.asarray(grid.matrix.sum(axis=1)).ravel(), 1)
            self.assertTrue(np.all(grid.matrix.data >= 0))
            self.assertLessEqual(np.max(np.diff(grid.matrix.indptr)), 1 if kind == 'nearest' else 2)
            terminal_weight = grid.matrix[:, grid.terminal].toarray().ravel().reshape(grid.shape)
            np.testing.assert_allclose(terminal_weight, grid.done)
            # All captures map to the actual standing node, which self-loops.
            standing_weight = grid.matrix[:, 0].toarray().ravel().reshape(grid.shape)
            np.testing.assert_allclose(standing_weight, grid.status == stopping.CAPTURE)
            np.testing.assert_array_equal(grid.next_speed[grid.status == stopping.CAPTURE], 0)
            np.testing.assert_array_equal(standing_weight[0], 1)
            # Returning rows never acquire a false capture by rounding.
            self.assertEqual(grid.matrix[:, np.flatnonzero(grid.source_goal)[1:]].nnz, 0)

    def test_grid_exits_are_terminal_not_clamped(self):
        transitions = stopping.simulate_transitions(self.states[:, None], self.actions[None, :],
                                                    self.region, self.params)
        transitions['status'][-1, 0] = stopping.RETURN
        transitions['states'][1, -1, 0] = self.states[-1] + 0.1
        with patch.object(stopping, 'simulate_transitions', return_value=transitions):
            grid = core.build_grid(self.states, self.actions)
        outside = grid.status == 4
        self.assertTrue(np.any(outside))
        terminal_weight = grid.matrix[:, grid.terminal].toarray().ravel().reshape(grid.shape)
        self.assertTrue(np.all(terminal_weight[outside] == 1))
        np.testing.assert_allclose(core.successor_speeds(grid)[outside], transitions['states'][1][outside])

    def test_reward_gets_physical_next_speed_and_capture_zero(self):
        reward, _ = core.prepare_tables(self.grid,
            'def reward(state, action, next_state, params):\n    return np.where(np.isfinite(next_state), next_state, -5)',
            core.DEFAULT_INITIAL_SOURCE, discount=.9)
        captured = self.grid.status == stopping.CAPTURE
        returning = self.grid.status == stopping.RETURN
        np.testing.assert_array_equal(reward[captured], 0)
        np.testing.assert_allclose(reward[returning], self.grid.transition_states[1][returning])
        np.testing.assert_array_equal(reward[self.grid.done], -5)

    def test_sparse_success_reward_propagates_by_decision_count(self):
        reward, q0 = core.prepare_tables(self.grid, core.DEFAULT_REWARD_SOURCE,
                                         core.DEFAULT_INITIAL_SOURCE, discount=.95)
        result = core.q_iteration(self.grid, reward, q0)
        self.assertTrue(result.converged)
        self.assertGreater(result.updates, 1)
        np.testing.assert_allclose(result.q_at(1), reward)
        # Independent scalar state/action backup, with optimistic initialization.
        optimistic = np.full(self.grid.shape, 8.0)
        oracle = optimistic.copy()
        for _ in range(10):
            new = np.zeros_like(oracle)
            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    r = float(self.grid.states[i] == 0)
                    row = self.grid.matrix.getrow(i * self.grid.shape[1] + j)
                    continuation = sum(weight * oracle[node].max() for node, weight in
                                       zip(row.indices, row.data) if node != self.grid.terminal)
                    new[i, j] = r + .95 * continuation
            oracle = new
        actual = core.q_iteration(self.grid, reward, optimistic)
        np.testing.assert_allclose(actual.q_at(10), oracle)
        np.testing.assert_allclose(actual.final_q, result.final_q, atol=2e-7)
        np.testing.assert_allclose(actual.final_q[0], 1 / (1 - .95), atol=2e-7)
        np.testing.assert_array_equal(reward[0], 1)
        np.testing.assert_array_equal(reward[1:], 0)

    def test_checkpoint_playback_and_residual_meaning(self):
        reward = np.where(self.states[:, None] == 0, np.ones(self.grid.shape), -.03)
        q0 = np.ones(self.grid.shape) * 10
        result = core.q_iteration(self.grid, reward, q0, max_updates=8,
                                  history_bytes=2 * q0.nbytes)
        self.assertGreater(result.checkpoint_stride, 1)
        q = result.q_at(0)
        for k in range(result.updates + 1):
            np.testing.assert_array_equal(result.q_at(k), q)
            expected = result.backup(q)
            self.assertAlmostEqual(result.residuals[k], np.max(np.abs(expected - q)))
            q = expected
        capped = core.q_iteration(self.grid, reward, q0, max_updates=0)
        self.assertFalse(capped.converged)
        self.assertEqual(capped.updates, 0)

    def test_editable_rewards_initializations_and_errors(self):
        r, q = core.prepare_tables(self.grid, 'def reward(state, action, next_state, params): return -2',
                                   'def initial_q(omega, alpha): return omega**2 + alpha', discount=.9)
        expected = self.states[:, None]**2 + self.actions[None, :]
        np.testing.assert_allclose(q, expected)
        np.testing.assert_array_equal(r, -2)
        for source in ['def reward(:', 'def reward(*args): return np.nan',
                       'def reward(*args): return np.ones((3, 4))',
                       'def reward(*args): raise RuntimeError("example")']:
            with self.subTest(source=source), self.assertRaises(core.EditorError):
                core.prepare_tables(self.grid, source, core.DEFAULT_INITIAL_SOURCE, discount=.9)
        with self.assertRaises(ValueError):
            core.q_iteration(self.grid, r, q, discount=1)

    def test_reward_receives_current_scalar_state_action_and_parameter_dictionary(self):
        source = '''def reward(state, action, next_state, params):
    assert isinstance(params, dict)
    assert "goal" not in params and "failed" not in params and "next_state" not in params
    assert params["discount"] == 0.9
    value = -0.5 * params["mass"] * (params["length"] * state)**2
    params["mass"] = 999  # Editing the local copy cannot change the model.
    return value - params["action_cost"] * action**2
'''
        r, _ = core.prepare_tables(self.grid, source, core.DEFAULT_INITIAL_SOURCE,
                                  discount=.9, params={'action_cost': 2.0})
        expected = -.5 * self.states[:, None]**2 - 2 * self.actions[None, :]**2
        np.testing.assert_allclose(r, expected)
        self.assertEqual(self.grid.params['mass'], 1)
        # Failure gets this same current-state reward, with no hidden override.
        np.testing.assert_allclose(r[self.grid.done], expected[self.grid.done])

    def test_rollout_uses_current_policy_and_preserves_continuous_speeds(self):
        q = np.zeros(self.grid.shape)
        q[:, -1] = 1
        rollout = playback.rollout_policy(self.grid, q, 4.1, self.region)
        self.assertGreaterEqual(len(rollout.decisions), 2)
        self.assertTrue(all(d['alpha'] == self.actions[-1] for d in rollout.decisions))
        expected_speed = float(core.return_map(4.1, self.actions[-1]))
        self.assertAlmostEqual(rollout.decisions[1]['speed'], expected_speed, places=7)
        self.assertGreater(np.min(np.abs(self.states - expected_speed)), 1e-5)
        other = playback.rollout_policy(self.grid, -q, 4.1, self.region)
        self.assertEqual(other.decisions[0]['alpha'], self.actions[0])

    def test_speed_reward_predicts_deceleration_in_grid_and_physical_rollout(self):
        reward, q0 = core.prepare_tables(self.grid,
            'def reward(state, action, next_state, params): return state',
            core.DEFAULT_INITIAL_SOURCE, discount=.95)
        result = core.q_iteration(self.grid, reward, q0)
        index = 80
        physical = playback.rollout_policy(self.grid, result.final_q, self.states[index], self.region)
        speeds = playback.section_speeds(physical)
        self.assertEqual(physical.outcome, 'Standing (settled)')
        self.assertTrue(np.all(np.diff(speeds) < 0))
        self.assertTrue(all(d['alpha'] == self.actions[0] for d in physical.decisions))
        grid_speeds, discounted_return = [], 0.
        for k in range(20):
            grid_speeds.append(self.states[index])
            if index == 0:
                break
            action = result.final_q[index].argmax()
            discounted_return += .95**k * self.states[index]
            index = self.grid.matrix.getrow(index * len(self.actions) + action).indices[0]
            self.assertNotEqual(index, self.grid.terminal)
        self.assertEqual(index, 0)
        self.assertAlmostEqual(discounted_return, result.final_q[80].max())
        np.testing.assert_allclose(speeds, grid_speeds, atol=2 * np.diff(self.states)[0])
        returning = self.grid.status == stopping.RETURN
        self.assertTrue(np.all((self.grid.next_speed - self.states[:, None])[returning] < 0))

    def test_animation_keeps_world_hip_continuous_at_footstrike(self):
        rollout = playback.rollout_policy(self.grid, np.zeros(self.grid.shape), 4.1, self.region)
        trace = rollout.trace
        feet, counts = playback.world_foot_positions(trace, self.params['length'])
        hips = feet + self.params['length'] * np.column_stack([np.sin(trace[:, 1]), np.cos(trace[:, 1])])
        impacts = np.flatnonzero(np.diff(counts)) + 1
        self.assertEqual(len(impacts), rollout.footstrikes)
        self.assertGreater(len(impacts), 1)
        np.testing.assert_allclose(hips[impacts], hips[impacts - 1], atol=1e-12)
        np.testing.assert_allclose(feet[:, 1], -np.tan(self.params['incline']) * feet[:, 0], atol=1e-8)
        np.testing.assert_array_equal(trace[trace[:, 5] == 1, 3], 0)
        self.assertEqual(rollout.outcome, 'Standing (settled)')
        self.assertTrue(stopping.has_settled(trace[-1, 1:3], self.params))
        frames = np.asarray(playback.animation_data(rollout, self.params)['frames'])
        self.assertTrue(np.all(np.isfinite(frames)))
        self.assertTrue(np.all(np.diff(frames[:, 0]) > 0))
        self.assertEqual(frames[-1, 0], trace[-1, 0])

    def test_rollout_standing_starts_failure_and_limits(self):
        q = np.zeros(self.grid.shape)
        standing = playback.rollout_policy(self.grid, q, 0, self.region)
        self.assertEqual(standing.decisions, [])
        self.assertEqual(standing.outcome, 'Standing (settled)')
        np.testing.assert_array_equal(playback.section_speeds(standing), [0.])
        self.assertEqual(len(playback.animation_data(standing, self.params)['frames']), 1)
        balancing = playback.rollout_policy(self.grid, q, .1, self.region)
        self.assertEqual(balancing.decisions, [])
        self.assertTrue(np.all(balancing.trace[:, 5] == 0))
        self.assertEqual(balancing.outcome, 'Standing (settled)')
        np.testing.assert_array_equal(playback.section_speeds(balancing), [.1, 0.])
        limited = playback.rollout_policy(self.grid, q, 4.1, self.region, max_decisions=1)
        self.assertEqual(limited.outcome, 'Decision limit reached')
        self.assertEqual(len(limited.decisions), 1)
        np.testing.assert_allclose(playback.section_speeds(limited), [4.1, core.return_map(4.1, self.actions[0])])
        timed = playback.rollout_policy(self.grid, q, 4.1, self.region, max_time=.001)
        self.assertEqual(timed.outcome, 'Walking time limit reached')
        np.testing.assert_array_equal(playback.section_speeds(timed), [4.1])
        i, j = np.argwhere(self.grid.status == stopping.FAILURE)[0]
        q[:, j] = 1
        failed = playback.rollout_policy(self.grid, q, self.states[i], self.region)
        self.assertEqual(failed.outcome, 'Step failed (reversal or ground contact)')
        self.assertIsNone(failed.capture_time)
        np.testing.assert_array_equal(playback.section_speeds(failed), [self.states[i]])

    @unittest.skipUnless(shutil.which('node'), 'Node is needed to exercise the animation player')
    def test_animation_replays_when_the_learned_policy_changes(self):
        reward, q0 = core.prepare_tables(self.grid,
            'def reward(state, action, next_state, params): return action',
            core.DEFAULT_INITIAL_SOURCE, discount=.95)
        learned = core.q_iteration(self.grid, reward, q0)
        rollouts = [playback.rollout_policy(self.grid, q, 4.1, self.region)
                    for q in (q0, learned.final_q)]
        self.assertNotEqual(rollouts[0].decisions[0]['alpha'], rollouts[1].decisions[0]['alpha'])
        players = []
        for k, rollout in zip((0, learned.updates), rollouts):
            label = f'Value iteration · sweep {k} · greedy policy'
            players.append({'html': playback.animation_html(rollout, self.params, policy_label=label),
                            'label': label, 'alpha': f"{rollout.decisions[0]['alpha']:.4f}",
                            'autoplay': True, 'duration': float(rollout.trace[-1, 0])})
        players.append(players[-1] | {'html': playback.animation_html(rollouts[-1], self.params,
            policy_label=label, autoplay=False), 'autoplay': False})
        standing = playback.rollout_policy(self.grid, learned.final_q, 0, self.region)
        players.append({'html': playback.animation_html(standing, self.params, policy_label=label),
                        'label': label, 'autoplay': True, 'duration': 0})
        harness = Path(__file__).with_name('walker_player_check.cjs')
        result = subprocess.run([shutil.which('node'), str(harness)],
                                input=json.dumps(players), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class SampledLearningTests(unittest.TestCase):
    def setUp(self):
        # Standing self-loops. From state 1, action 0 captures and action 1 fails.
        self.successors = np.array([0, 0, 0, 2])
        self.grid = core.TransitionGrid(
            states=np.array([0., 1.]), actions=np.array([.4, .45]),
            next_speed=np.array([[0., 0.], [0., np.nan]]),
            status=np.array([[1, 1], [1, 2]]), source_goal=np.array([True, False]),
            transition_states=np.zeros((2, 2, 2)), impacts=np.zeros((2, 2)),
            matrix=csr_matrix((np.ones(4), (np.arange(4), self.successors)), shape=(4, 3)),
            interpolation='nearest', build_seconds=0., params={},
        )

    def test_sampled_updates_use_greedy_bootstrap_and_exact_replay(self):
        reward = np.array([[.2, .4], [.8, 1.6]])
        initial = np.array([[2., 5.], [3., 7.]])
        result = core.q_learning(self.grid, reward, initial, discount=.7,
            learning_rate=.35, epsilon=1., seed=4, max_updates=9,
            samples_per_iteration=7, episode_length=5, history_bytes=2 * initial.nbytes)
        self.assertGreater(result.checkpoint_stride, 1)
        oracle = initial.copy()
        for sample, (row, value) in enumerate(zip(result.sample_rows, result.sample_values)):
            i, j = divmod(int(row), 2)
            successor = self.successors[row]
            continuation = 0. if successor == 2 else oracle[successor].max()
            oracle[i, j] += .35 * (reward[i, j] + .7 * continuation - oracle[i, j])
            self.assertAlmostEqual(value, oracle[i, j])
            if (sample + 1) % 7 == 0:
                k = (sample + 1) // 7
                np.testing.assert_array_equal(result.q_at(k), oracle)
                self.assertAlmostEqual(result.residuals[k], np.max(np.abs(result.backup(oracle) - oracle)))
        np.testing.assert_array_equal(result.visit_counts.ravel(), np.bincount(result.sample_rows, minlength=4))
        self.assertTrue(np.all(result.visit_counts > 0))
        np.testing.assert_array_equal(result.q_at(0), initial)
        # A single sample changes exactly one entry, not an entire Bellman table.
        one = core.q_learning(self.grid, reward, initial, max_updates=1, samples_per_iteration=1)
        self.assertEqual(np.count_nonzero(one.final_q != initial), 1)
        zero = core.q_learning(self.grid, reward, initial, max_updates=0)
        np.testing.assert_array_equal(zero.final_q, initial)
        self.assertEqual(zero.sample_rows.size, 0)

    def test_epsilon_controls_exploration_and_seed_reproduces_history(self):
        initial = np.tile([-100., 10.], (2, 1))
        kwargs = dict(discount=.5, max_updates=2, samples_per_iteration=32, seed=6)
        greedy = core.q_learning(self.grid, np.zeros((2, 2)), initial, epsilon=0., **kwargs)
        self.assertTrue(np.all(greedy.sample_rows % 2 == 1))
        explore = core.q_learning(self.grid, np.zeros((2, 2)), initial, epsilon=1., **kwargs)
        self.assertTrue(np.any(explore.sample_rows % 2 == 0))
        repeat = core.q_learning(self.grid, np.zeros((2, 2)), initial, epsilon=1., **kwargs)
        np.testing.assert_array_equal(repeat.sample_rows, explore.sample_rows)
        np.testing.assert_array_equal(repeat.sample_values, explore.sample_values)

    def test_standing_reward_and_time_limit_bootstrap_match_known_solution(self):
        reward = np.array([[1., 1.], [0., 0.]])
        # Reset every sample; truncation must not erase the standing self-loop's value.
        result = core.q_learning(self.grid, reward, np.zeros((2, 2)), discount=.5,
            epsilon=.6, learning_rate=.3, max_updates=200, samples_per_iteration=64,
            episode_length=1, tolerance=1e-7)
        self.assertTrue(result.converged)
        np.testing.assert_allclose(result.final_q, [[2., 2.], [1., 0.]], atol=2e-7)

    def test_linear_successors_are_sampled_rather_than_averaged(self):
        self.grid.matrix = csr_matrix(np.tile([.25, 0., .75], (4, 1)))
        # Each target is either 0 (failure) or .5 (state 0), never the mean .125.
        result = core.q_learning(self.grid, np.zeros((2, 2)), np.ones((2, 2)),
            discount=.5, epsilon=1., learning_rate=1., max_updates=1,
            samples_per_iteration=1, seed=0)
        self.assertIn(result.sample_values[0], (0., .5))
        # Many independent single updates verify the transition weights as well.
        values = [core.q_learning(self.grid, np.zeros((2, 2)), np.ones((2, 2)),
            discount=.5, learning_rate=1., max_updates=1, samples_per_iteration=1,
            seed=seed).sample_values[0] for seed in range(500)]
        self.assertAlmostEqual(np.mean(np.asarray(values) == .5), .25, delta=.06)


if __name__ == '__main__':
    unittest.main()
