# Dynamics, control, and learning notebooks

Interactive [marimo](https://marimo.io/) lessons on dynamics, control, and learning.
Runnable notebooks are the `nb_*.py` files at the repo root.

## Setup

From this directory:

```sh
uv sync
```

## Choose a lesson

| What to explore | Run |
| --- | --- |
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
