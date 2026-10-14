import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    from scipy.integrate import solve_ivp

    return Line2D, mo, np, plt, solve_ivp


@app.cell
def _(mo):
    damping = mo.ui.slider(
        0.0, 0.15, step=0.01, value=0.03, debounce=True,
        label="Viscous damping b", show_value=True,
    )
    duration = mo.ui.slider(
        5, 40, step=5, value=20, debounce=True,
        label="Simulation duration", show_value=True,
    )
    mo.hstack([damping, duration], justify="start", gap=3)
    return damping, duration


@app.cell
def _(np, solve_ivp):
    def linearizing_torque(state, damping_coefficient):
        """Unbounded torque: cancel gravity/damping, then use v = -theta - omega."""
        theta, omega = np.asarray(state)[..., 0], np.asarray(state)[..., 1]
        return -np.sin(theta) + damping_coefficient * omega - theta - omega

    def pendulum_rhs(_time, state, damping_coefficient, controlled=False):
        theta, omega = state
        torque = linearizing_torque(state, damping_coefficient) if controlled else 0.0
        return np.array([omega, np.sin(theta) - damping_coefficient * omega + torque])

    def simulate_pendulum(starts, times, damping_coefficient, controlled=False):
        """Continuous feedback with unwrapped angles; no clipping or capture rule."""
        trajectories = []
        for state in starts:
            solution = solve_ivp(
                pendulum_rhs, (times[0], times[-1]), state,
                args=(damping_coefficient, controlled), t_eval=times,
                rtol=1e-9, atol=1e-11, max_step=0.05,
            )
            if not solution.success:
                raise RuntimeError(solution.message)
            trajectories.append(solution.y.T)
        return np.array(trajectories)

    return linearizing_torque, pendulum_rhs, simulate_pendulum


@app.cell
def _(damping, duration, np, simulate_pendulum):
    _theta, _omega = np.meshgrid(
        [-2.5, -1.0, -0.25, 0.25, 1.0, 2.5], [-1.2, 0.0, 1.2],
    )
    initial_states = np.column_stack([_theta.ravel(), _omega.ravel()])
    times = np.linspace(0.0, duration.value, 1201)
    passive_trajectories = simulate_pendulum(initial_states, times, damping.value)
    controlled_trajectories = simulate_pendulum(
        initial_states, times, damping.value, controlled=True,
    )
    return controlled_trajectories, initial_states, passive_trajectories, times


@app.cell
def _(Line2D, controlled_trajectories, damping, np, passive_trajectories, plt):
    def _draw_orbits(axis, trajectories, color, width, alpha, order, starts=False):
        for _orbit in trajectories:
            _angle = (_orbit[:, 0] + np.pi) % (2 * np.pi) - np.pi
            _angle[np.flatnonzero(np.abs(np.diff(_angle)) > np.pi) + 1] = np.nan
            axis.plot(_angle, _orbit[:, 1], color=color, lw=width, alpha=alpha, zorder=order)
            if starts:
                axis.scatter(_angle[0], _orbit[0, 1], color=color, s=14, zorder=order + 1)
            _index = len(_orbit) // 8
            if np.isfinite(_angle[_index:_index + 5]).all():
                axis.annotate(
                    "", xy=(_angle[_index + 4], _orbit[_index + 4, 1]),
                    xytext=(_angle[_index], _orbit[_index, 1]),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": width},
                    zorder=order,
                )

    phase_portrait, _axes = plt.subplots(1, 2, figsize=(13, 5.4), layout="constrained")
    for _axis in _axes:
        _axis.set(
            xlim=(-np.pi, np.pi), ylim=(-3, 3),
            xlabel=r"Angle from upright $\theta$ [rad]",
            ylabel=r"Angular velocity $\dot\theta$",
        )
        _axis.set_xticks(
            [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"],
        )
        _axis.axhline(0, color="#d1d5db", lw=0.7)
        _axis.axvline(0, color="#d1d5db", lw=0.7)
        _axis.scatter(0, 0, marker="+", color="black", s=55, zorder=5)
        _axis.grid(alpha=0.15)
    _draw_orbits(_axes[0], passive_trajectories, "#9ca3af", 1.2, 0.85, 2, starts=True)
    _draw_orbits(_axes[1], passive_trajectories, "#9ca3af", 0.9, 0.3, 1)
    _draw_orbits(_axes[1], controlled_trajectories, "#e67700", 1.5, 0.85, 3, starts=True)
    _axes[0].set_title(
        "Passive pendulum\n" + rf"$\ddot\theta=\sin\theta-{damping.value:.2f}\dot\theta$",
    )
    _axes[1].set_title(
        "Feedback linearization + unit state gains\n" + r"$\ddot\theta=-\theta-\dot\theta$",
    )
    _axes[1].legend(handles=[
        Line2D([], [], color="#9ca3af", label="Passive"),
        Line2D([], [], color="#e67700", label="Feedback linearized"),
    ], loc="upper right")
    phase_portrait
    return (phase_portrait,)


if __name__ == "__main__":
    app.run()
