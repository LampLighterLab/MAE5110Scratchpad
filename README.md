# Legged robots notebooks

Interactive [marimo](https://marimo.io/) notebooks for illustrating simple
models of walking and related dynamical systems.

## Setup

Install the environment with uv:

```sh
uv sync
```

Open the Van der Pol oscillator notebook:

```sh
uv run marimo edit van_der_pol.py
```

Use the **μ** slider to change the oscillator's nonlinear damping. The large,
equal-axis phase portrait keeps a fixed scale for direct comparison. Additional
controls adjust the vector-field overlay, animation speed, and automatic
resampling of states. Separate sliders set how many randomly placed points begin
on, inside, and outside the limit cycle. A reverse-time toggle flips both the
vector field and the animated trajectories.

Open the Markov-process notebook:

```sh
uv run marimo edit markov_process.py
```

It visualizes a random deterministic path blended with uniform transitions,
the resulting transition matrix, and the state distribution over time. Controls
can change the number of states and impose a terminal state. An optional reward
view adds a transition-reward matrix to the same process and shows the expected
discounted return accumulated through the current time step, together with a
return-over-time plot and the finite-horizon value function over all states.
An actions toggle turns the same model into a simple MDP: each selectable action
has its own persistent transition path and reward matrix.
