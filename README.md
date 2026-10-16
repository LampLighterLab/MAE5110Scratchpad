# Dynamics, control, and learning notebooks

Interactive [marimo](https://marimo.io/) lessons on dynamics, control, and learning.
Runnable notebooks are the `nb_*.py` files at the repo root.

## Setup

From this directory:

```sh
uv sync
```

## Choose a lesson

Start with `uv run marimo run nb_learning_lab.py` to choose a model and compare
value iteration with sampled RL, tabular with approximated values, and greedy
actions with a fitted policy or policy iteration. Reward code and parameters
are editable; plots and animations follow the selected learning iteration.
Models include a simple MDP, two pendulums, the inverted-pendulum walker, and
SLIP with parameters from [Beyond Basins of Attraction](https://arxiv.org/abs/1806.08081).
Editable rewards use `reward(state, action, next_state, params)` throughout.
Define each complete Q or policy feature vector in `features(state, parameters)`,
returning a one-dimensional NumPy array for one physical state. A few suggested
features per model can populate the editor; then edit the code freely and inspect
its learned coefficients and contributions.

| What to explore | Run |
| --- | --- |
| Reward design: random-policy samples, editable variables, reward overlays and reward histograms | `uv run marimo run nb_reward_design.py` |
| Inverted pendulum reward design: position and velocity errors, two rewards and overlaid reward histograms | `uv run marimo run nb_reward_design_inverted.py` |
| Compare models, value representations, and policy-learning choices in one GUI | `uv run marimo run nb_learning_lab.py` |
| Stable limit cycles and attraction from different initial conditions | `uv run marimo run nb_van_der_pol.py` |
| How feedback linearization changes a pendulum's phase portrait | `uv run marimo run nb_pendulum_feedback_linearization.py` |
| Markov transitions, absorbing states, actions, and discounted rewards | `uv run marimo run nb_markov_process.py` |
| Value iteration in one state dimension; fitting a simple policy and comparing actor–critic learning | `uv run marimo run nb_overdamped_pendulum.py` |
| Value iteration in angle–velocity space; torque limits and policy approximation | `uv run marimo run nb_undamped_pendulum.py` |
| Learning feedback gains with Monte Carlo finite differences and custom rewards | `uv run marimo run nb_pendulum_policy_gradient.py` |
| Poincaré maps, value iteration versus ε-greedy Q-learning, and animated walker policies | `uv run marimo run nb_walker_q_iteration.py` |

Use `marimo edit` instead of `marimo run` to edit notebook cells.

## Repository layout

- `nb_*.py` — interactive lessons.
- `core/` — reusable dynamics, controllers, and learning code; model data in `data/`.
- `tests/` — numerical and notebook checks.
- `plots/` — exported figures.

## Checks

```sh
uv run python -m unittest discover -s tests -v
uv run marimo check --strict nb_*.py
```

To compare suggested features by policy return and physical rollout success:

```sh
uv run python benchmarks/check_learning_features.py --output /tmp/features.json
```

The [feature audit](benchmarks/FEATURE_AUDIT.md) records which examples work and
which fail; passing the numerical tests alone does not establish successful learning.
