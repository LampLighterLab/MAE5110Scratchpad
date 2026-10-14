import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from core import pendulum_policy_gradient as pg

    return Line2D, Rectangle, mo, np, pg, plt


@app.cell
def _(mo):
    mo.md(r"""
    $\ddot\theta=\sin\theta-b\dot\theta+u,\qquad
    u=w_1\dot\theta+w_2\sin\theta-\theta-\dot\theta$

    $K=[-1,-1]\text{ (fixed)},\qquad
    (w_1,w_2)_{\rm cancellation}=(b,-1)$
    """)
    return


@app.cell
def _(mo):
    initial_parameters = mo.ui.dictionary({
        "w1": mo.ui.number(step=0.05, value=0.5, label="Starting w₁"),
        "w2": mo.ui.number(step=0.05, value=-0.5, label="Starting w₂"),
    }).form(submit_button_label="Set starting parameters and restart")
    initial_parameters
    return (initial_parameters,)


@app.cell
def _(mo):
    _fields = {
        "batch_size": mo.ui.slider(steps=[8, 16, 32, 64], value=32, label="MC rollouts per probe", show_value=True),
        "epsilon": mo.ui.number(0.005, 1, step=0.005, value=0.15, label="Finite-difference ε"),
        "learning_rate": mo.ui.number(0.001, 0.5, step=0.005, value=0.05, label="Ascent step size η"),
        "iterations": mo.ui.slider(0, 20, value=8, step=1, label="Gradient updates", show_value=True),
        "seed": mo.ui.number(0, 9999, value=0, step=1, label="Random seed"),
    }
    _template = mo.vstack([
        mo.hstack([mo.md("{batch_size}"), mo.md("{epsilon}"), mo.md("{learning_rate}")], justify="start", gap=3),
        mo.hstack([mo.md("{iterations}"), mo.md("{seed}")], justify="start", gap=3),
    ])
    experiment_controls = mo.ui.batch(_template, _fields).form(
        submit_button_label="Run / restart experiment",
    )
    experiment_controls
    return (experiment_controls,)


@app.cell
def _(mo, pg):
    physics_controls = mo.ui.dictionary({
        "b": mo.ui.slider(0.0, 0.15, step=0.01, value=0.03, show_value=True, debounce=True, label="Damping b"),
        "duration": mo.ui.slider(4, 16, step=2, value=10, show_value=True, debounce=True, label="Rollout duration"),
        "discount": mo.ui.slider(0.90, 1.0, step=0.005, value=0.98, show_value=True, debounce=True, label="Discount γ per step"),
        "theta_tolerance": mo.ui.number(0.005, 0.3, step=0.005, value=0.05, label="np.isclose: θ absolute tolerance"),
        "omega_tolerance": mo.ui.number(0.005, 0.3, step=0.005, value=0.05, label="np.isclose: θ̇ absolute tolerance"),
    }).form(submit_button_label="Apply pendulum settings and restart")
    reward_editor = mo.ui.code_editor(
        value=pg.DEFAULT_REWARD_SOURCE, language="python", min_height=150,
        label=("reward(state, action, next_state): states (..., 2) = [θ, θ̇]; "
               "action and reward (...). Available: np, theta_atol, omega_atol."),
    ).form(submit_button_label="Apply Python reward and restart")
    mo.accordion({"Reward Python and pendulum settings": mo.vstack([reward_editor, physics_controls])})
    return physics_controls, reward_editor


@app.cell
def _(pg, physics_controls, reward_editor):
    _settings = physics_controls.value or {
        "b": 0.03, "duration": 10, "discount": 0.98,
        "theta_tolerance": 0.05, "omega_tolerance": 0.05,
    }
    config = pg.PendulumConfig(
        b=float(_settings["b"]), dt=0.04,
        horizon=int(round(float(_settings["duration"]) / 0.04)),
        discount=float(_settings["discount"]),
        theta_tolerance=float(_settings["theta_tolerance"]),
        omega_tolerance=float(_settings["omega_tolerance"]),
    )
    reward_source = pg.DEFAULT_REWARD_SOURCE if reward_editor.value is None else reward_editor.value
    reward_function = None
    reward_error = None
    try:
        reward_function = pg.compile_reward(reward_source, config)
    except pg.RewardError as _error:
        reward_error = str(_error)
    return config, reward_error, reward_function, reward_source


@app.cell
def _(config, experiment_controls, initial_parameters, mo, pg, reward_error, reward_function):
    _initial = initial_parameters.value or {"w1": 0.5, "w2": -0.5}
    _settings = experiment_controls.value or {
        "batch_size": 32,
        "epsilon": 0.15, "learning_rate": 0.05, "iterations": 8, "seed": 0,
    }
    experiment = None
    experiment_error = reward_error
    if reward_function is not None:
        try:
            experiment = pg.train_policy_gradient(
                [_initial["w1"], _initial["w2"]], config, reward_function,
                iterations=int(_settings["iterations"]), batch_size=int(_settings["batch_size"]),
                seed=int(_settings["seed"]), epsilon=float(_settings["epsilon"]),
                learning_rate=float(_settings["learning_rate"]),
            )
        except (pg.RewardError, ValueError) as _error:
            experiment_error = str(_error)
    if experiment_error:
        mo.output.append(mo.callout(experiment_error, kind="danger"))
    return experiment, experiment_error


@app.cell
def _(experiment, mo):
    mo.stop(experiment is None)
    _restart_on_experiment_change = experiment
    get_iteration, set_iteration = mo.state(0, allow_self_loops=True)
    return get_iteration, set_iteration


@app.cell
def _(experiment, get_iteration, mo, set_iteration):
    mo.stop(experiment is None)
    _last = len(experiment["snapshots"]) - 1
    iteration = mo.ui.slider(
        0, _last, value=get_iteration(), step=1, show_value=True,
        full_width=True, debounce=True, include_input=True,
        label="Policy-gradient update (0 = initial guess)", on_change=set_iteration,
    )
    first = mo.ui.button(label="First", disabled=get_iteration() == 0, on_click=lambda _: set_iteration(0))
    previous = mo.ui.button(label="← Previous", disabled=get_iteration() == 0,
                            on_click=lambda _: set_iteration(lambda i: max(0, i - 1)))
    next_update = mo.ui.button(label="Next update →", disabled=get_iteration() == _last,
                               on_click=lambda _: set_iteration(lambda i: min(_last, i + 1)))
    last = mo.ui.button(label="Last", disabled=get_iteration() == _last,
                       on_click=lambda _: set_iteration(_last))
    mo.vstack([iteration, mo.hstack([first, previous, next_update, last], justify="start")])
    return first, iteration, last, next_update, previous


@app.cell
def _(mo):
    show_return_heatmap = mo.ui.switch(value=True, label="Show return heatmap")
    show_return_heatmap
    return (show_return_heatmap,)


@app.cell
def _(config, experiment, mo, np):
    mo.stop(experiment is None)
    _points = np.concatenate([
        np.array([s["weights"] for s in experiment["snapshots"]]),
        np.concatenate([s["perturbed_weights"] for s in experiment["snapshots"]]),
        np.array([s["weights"] + s["learning_rate"] * s["gradient"]
                  for s in experiment["snapshots"]]),
    ])
    _lower = np.minimum([-1.5, -2.0], _points.min(axis=0) - 0.3)
    _upper = np.maximum([1.5, 0.5], _points.max(axis=0) + 0.3)
    _lower = np.minimum(_lower, config.cancellation_weights - 0.4)
    _upper = np.maximum(_upper, config.cancellation_weights + 0.4)
    w1_axis = np.linspace(_lower[0], _upper[0], 33)
    w2_axis = np.linspace(_lower[1], _upper[1], 33)
    return w1_axis, w2_axis


@app.cell
def _(config, experiment, mo, np, pg, reward_function, show_return_heatmap, w1_axis, w2_axis):
    mo.stop(experiment is None)
    parameter_returns = None
    landscape_error = None
    if show_return_heatmap.value:
        _w1, _w2 = np.meshgrid(w1_axis, w2_axis)
        try:
            parameter_returns = pg.evaluate_parameters(
                np.stack([_w1, _w2], axis=-1), experiment["snapshots"][0]["batch_starts"],
                config, reward_function,
            )
        except (pg.RewardError, ValueError) as _error:
            landscape_error = str(_error)
    return landscape_error, parameter_returns


@app.cell
def _(config, experiment, mo, pg):
    mo.stop(experiment is None)
    passive_rollouts = pg.passive_rollouts(experiment["snapshots"][0]["batch_starts"], config)
    return (passive_rollouts,)


@app.cell
def _(experiment, iteration, mo):
    mo.stop(experiment is None)
    selected_snapshot = experiment["snapshots"][int(iteration.value)]
    return (selected_snapshot,)


@app.cell
def _(Line2D, Rectangle, config, experiment, landscape_error, mo, np, parameter_returns,
      passive_rollouts, plt, selected_snapshot, show_return_heatmap, w1_axis, w2_axis):
    mo.stop(experiment is None)
    _snapshot = selected_snapshot
    _iteration = _snapshot["iteration"]
    dashboard, _axes = plt.subplots(1, 2, figsize=(14, 5.8), layout="constrained")
    _probe_colors = ["#e67700", "#1864ab", "#9c36b5", "#2b8a3e"]
    _probe_labels = ["+ε e₁", "+ε e₂", "−ε e₁", "−ε e₂"]

    def _orbits(_batch, _color, _passive=False):
        for _states in _batch["states"]:
            _theta = _states[:, 0]
            _axes[0].plot(_theta, _states[:, 1], color=_color,
                          lw=0.9 if _passive else 1.1, alpha=0.25 if _passive else 0.45,
                          zorder=1 if _passive else 3)
            _arrow = len(_states) // 5
            if np.isfinite(_theta[_arrow:_arrow + 4]).all():
                _axes[0].annotate("", xy=(_theta[_arrow + 3], _states[_arrow + 3, 1]),
                                  xytext=(_theta[_arrow], _states[_arrow, 1]),
                                  arrowprops={"arrowstyle": "->", "color": _color, "lw": 1},
                                  zorder=1 if _passive else 3)

    _orbits(passive_rollouts, "#9ca3af", _passive=True)
    for _batch, _color in zip(_snapshot["perturbation_rollouts"], _probe_colors):
        _orbits(_batch, _color)
    _axes[0].add_patch(Rectangle(
        (-config.theta_tolerance, -config.omega_tolerance),
        2 * config.theta_tolerance, 2 * config.omega_tolerance,
        facecolor="#2b8a3e", edgecolor="#2b8a3e", alpha=0.15, zorder=2,
    ))
    _axes[0].scatter(0, 0, marker="+", color="black", s=55, zorder=5)
    _axes[0].set(
        xlim=(-np.pi, np.pi), ylim=(-3, 3),
        xlabel=r"Unwrapped angle from upright $\theta$ [rad]", ylabel=r"Angular velocity $\dot\theta$",
        title=f"Perturbation trajectories · update {_iteration}\nw₁ = {_snapshot['weights'][0]:.3f}, w₂ = {_snapshot['weights'][1]:.3f}",
    )
    _axes[0].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                       [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    _axes[0].legend(handles=[
        Line2D([], [], color="#9ca3af", label="Passive"),
    ] + [Line2D([], [], color=_color, label=_label)
         for _color, _label in zip(_probe_colors, _probe_labels)], loc="upper right", fontsize=8)
    _axes[0].grid(alpha=0.15)

    _has_heatmap = show_return_heatmap.value and parameter_returns is not None
    _foreground = "white" if _has_heatmap else "#343a40"
    if _has_heatmap:
        _image = _axes[1].pcolormesh(w1_axis, w2_axis, parameter_returns,
                                    shading="auto", cmap="viridis")
        dashboard.colorbar(_image, ax=_axes[1], label="Mean discounted return (same MC starts)")
    elif landscape_error:
        _axes[1].text(0.5, 0.9, landscape_error, transform=_axes[1].transAxes,
                      ha="center", va="top", wrap=True, fontsize=9)
    _path = np.array([s["weights"] for s in experiment["snapshots"][:_iteration + 1]])
    _axes[1].plot(_path[:, 0], _path[:, 1], "o-", color=_foreground, lw=1.5, ms=4,
                  markeredgecolor="#343a40", label="Ascent path", zorder=4)
    _axes[1].scatter(*experiment["initial_weights"], marker="X", s=90,
                     color="#343a40", edgecolor="white", linewidth=0.8,
                     label="Starting parameters", zorder=6)
    _axes[1].scatter(*config.cancellation_weights, marker="*", s=190, color="#ffec99",
                     edgecolor="black", linewidth=1, label="Exact cancellation (b, −1)", zorder=6)
    _weights = _snapshot["weights"]
    _probes = _snapshot["perturbed_weights"]
    _probe_returns = np.concatenate([_snapshot["plus_scores"], _snapshot["minus_scores"]])
    _axes[1].scatter(
        _probes[:, 0], _probes[:, 1], c=_probe_colors, s=85,
        edgecolor="white" if _has_heatmap else "#343a40", linewidth=0.8,
        label="Probes (colors match trajectories)", zorder=5,
    )
    for _point, _score, _label, _color, _offset, _ha, _va in zip(
        _probes, _probe_returns, _probe_labels, _probe_colors,
        [(10, 0), (0, 12), (-10, 0), (0, -12)],
        ["left", "center", "right", "center"],
        ["center", "bottom", "center", "top"],
    ):
        _axes[1].annotate(
            f"{_label}\nJ = {_score:.3f}", xy=_point, xytext=_offset,
            textcoords="offset points", ha=_ha, va=_va, fontsize=8, color=_color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
            zorder=8,
        )
    _axes[1].scatter(*_weights, color="#343a40", marker="D", s=45, edgecolor="white",
                     linewidth=1.2, label=f"Current · J = {_snapshot['batch_score']:.3f}", zorder=7)
    _proposed = _weights + _snapshot["learning_rate"] * _snapshot["gradient"]
    _axes[1].annotate("", xy=_proposed, xytext=_weights,
                      arrowprops={"arrowstyle": "-|>", "color": _foreground, "lw": 2}, zorder=6)
    _axes[1].axvline(1 + config.b, color=_foreground, ls=":", lw=0.8, alpha=0.6)
    _axes[1].axhline(0, color=_foreground, ls=":", lw=0.8, alpha=0.6)
    _axes[1].set(xlim=(w1_axis[0], w1_axis[-1]), ylim=(w2_axis[0], w2_axis[-1]),
                  xlabel=r"$w_1$ · velocity cancellation", ylabel=r"$w_2$ · gravity cancellation",
                  title=f"Parameter space · update {_iteration}\nFinite-difference estimate and proposed step")
    _handles, _labels = _axes[1].get_legend_handles_labels()
    _axes[1].legend(
        handles=_handles + [Line2D([], [], color=_foreground, lw=2, marker=">", label="Gradient ascent η∇J")],
        loc="upper left", fontsize=8,
    )
    dashboard
    return (dashboard,)


@app.cell
def _(experiment, mo, np, selected_snapshot):
    mo.stop(experiment is None)
    _snapshot = selected_snapshot
    mo.ui.table([
        {"Parameter": _name,
         "Current": float(_snapshot["weights"][_i]),
         "J(w + εeᵢ)": float(_snapshot["plus_scores"][_i]),
         "J(w − εeᵢ)": float(_snapshot["minus_scores"][_i]),
         "Gradient estimate": float(_snapshot["gradient"][_i]),
         "Gradient SE": float(_snapshot["gradient_stderr"][_i]),
         "η × gradient": float(_snapshot["learning_rate"] * _snapshot["gradient"][_i])}
        for _i, _name in enumerate(["w₁", "w₂"])
    ], selection=None, pagination=False, show_column_summaries=False, show_data_types=False,
       show_search=False, show_download=False,
       label=(f"Update {_snapshot['iteration']} · mean MC return {_snapshot['batch_score']:.3f} · "
              f"held-out mean {_snapshot['evaluation_score']:.3f} · "
              f"target hits {100 * np.mean(_snapshot['rollout']['success']):.1f}% · "
              f"{len(_snapshot['batch_starts'])} shared starts per probe"))
    return


if __name__ == "__main__":
    app.run()
