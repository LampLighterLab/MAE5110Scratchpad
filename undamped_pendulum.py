import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    import numpy as np

    import pendulum_learning as learning

    return Line2D, Rectangle, learning, mo, np, plt


@app.cell
def _(mo):
    physics_controls = mo.ui.dictionary(
        {
            "torque": mo.ui.slider(
                start=0.15, stop=0.60, step=0.05, value=0.35,
                label="Maximum torque", show_value=True, debounce=True,
            ),
            "torque_cost": mo.ui.slider(
                start=0.0, stop=5.0, step=0.1, value=0.1,
                label="Torque-squared penalty ρ", show_value=True,
                include_input=True, debounce=True,
            ),
            "damping": mo.ui.slider(
                start=0.0, stop=0.15, step=0.01, value=0.03,
                label="Passive damping b", show_value=True, debounce=True,
            ),
            "horizon": mo.ui.slider(
                start=100, stop=240, step=20, value=180,
                label="Episode cutoff (steps)", show_value=True, debounce=True,
            ),
            "discount": mo.ui.slider(
                start=0.90, stop=0.995, step=0.005, value=0.98,
                label="Discount factor γ per step", show_value=True,
                include_input=True, debounce=True,
            ),
        },
        label="Upright pendulum (θ = 0): model and objective",
    )
    physics_controls
    return (physics_controls,)


@app.cell
def _(learning, np, physics_controls):
    _settings = physics_controls.value or {
        "torque": 0.35, "torque_cost": 0.1, "horizon": 180, "discount": 0.98,
    }
    grid_theta_count, grid_omega_count = 241, 161
    _omega_limit = 3.0
    model = learning.Pendulum(
        kind="undamped", dt=0.08,
        torque_limit=float(_settings["torque"]),
        horizon=int(_settings["horizon"]),
        discount=float(_settings["discount"]), goal_reward=1.0,
        torque_cost=float(_settings["torque_cost"]),
        damping=float(_settings.get("damping", 0.03)),
        target_theta=np.pi / grid_theta_count,
        target_omega=_omega_limit / (grid_omega_count - 1), omega_limit=_omega_limit,
    )
    _bounds = learning.value_bounds(model, model.horizon)
    minimum_return = float(_bounds[0])
    maximum_return = float(_bounds[1])
    phase_start_limits = learning.initial_state_limits(model)
    return grid_omega_count, grid_theta_count, maximum_return, minimum_return, model, phase_start_limits


@app.cell
def _(Line2D, Rectangle, learning, model, np, plt):
    def full_angle_axis(axis):
        axis.set_xlim(-np.pi, np.pi)
        axis.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    def state_map(axis, theta, omega, values, title, *, cmap, vmin, vmax):
        """Identical axes and color ranges make snapshots comparable."""
        _image = axis.pcolormesh(
            theta, omega, np.asarray(values).T, shading="auto",
            cmap=cmap, vmin=vmin, vmax=vmax,
        )
        axis.add_patch(Rectangle(
            (-model.target_theta, -model.target_omega),
            2 * model.target_theta, 2 * model.target_omega,
            edgecolor="black", facecolor="none", linewidth=1.2,
        ))
        axis.set(
            xlabel=r"Angle $\theta$ (rad; 0 = upright)", ylabel=r"Velocity $\omega$ (rad / time unit)",
            title=title, xlim=(-np.pi, np.pi),
            ylim=(-model.omega_limit, model.omega_limit),
        )
        full_angle_axis(axis)
        return _image

    def phase_rollout(controller, initial_states):
        _starts = np.asarray(initial_states, dtype=float)
        _states = np.empty((len(_starts), model.horizon + 1, 2))
        _states[:, 0] = _starts
        _alive = np.zeros((len(_starts), model.horizon), dtype=bool)
        _active = ~learning.in_target(model, _starts)
        for _time in range(model.horizon):
            _alive[:, _time] = _active
            _torque = np.where(_active, controller(_states[:, _time]), 0.0)
            _next = learning.step(model, _states[:, _time], _torque)
            _states[:, _time + 1] = np.where(
                _active[:, None], _next, _states[:, _time],
            )
            _active &= ~learning.in_target(model, _states[:, _time + 1])
        return {"states": _states, "alive": _alive, "success": ~_active}

    def phase_portrait(axis, trajectories, title, controlled=None):
        _groups = [(trajectories, "#9ca3af", .6, .9, 1)]
        if controlled is not None:
            _groups.append((controlled, "#ea580c", .8, 1.2, 3))
        for _orbits, _color, _alpha, _width, _zorder in _groups:
            for _index, _states in enumerate(_orbits["states"]):
                if _zorder == 3:
                    _color = "#e67700" if _orbits["success"][_index] else "#1864ab"
                _end = int(np.sum(_orbits["alive"][_index])) + 1
                _path = _states[:_end]
                _theta = _path[:, 0].copy()
                _theta[np.flatnonzero(np.abs(np.diff(_theta)) > np.pi) + 1] = np.nan
                axis.plot(
                    _theta, _path[:, 1], color=_color, linewidth=_width,
                    alpha=_alpha, zorder=_zorder,
                    label=(
                        ("Passive (u = 0)" if _zorder == 1 else "Current policy")
                        if _index == 0 else None
                    ),
                )
                axis.scatter(
                    _path[0, 0], _path[0, 1], s=14, color=_color,
                    edgecolor="white", linewidth=.4, zorder=_zorder + 1,
                )
                if len(_path) > 8:
                    _visible = np.flatnonzero(
                        (np.abs(_theta) < .9 * np.pi)
                        & (np.abs(_path[:, 1]) < .9 * model.omega_limit)
                    )
                    _visible = _visible[(_visible >= 3) & (_visible < len(_path) - 3)]
                    _arrow = (
                        int(_visible[len(_visible) // 2]) if len(_visible)
                        else min(max(3, int(.18 * (len(_path) - 1))), len(_path) - 4)
                    )
                    if np.all(np.isfinite(_theta[_arrow - 2:_arrow + 3])):
                        axis.annotate(
                            "", xy=(_theta[_arrow + 2], _path[_arrow + 2, 1]),
                            xytext=(_theta[_arrow - 2], _path[_arrow - 2, 1]),
                            arrowprops={"arrowstyle": "-|>", "color": _color, "lw": _width},
                            zorder=_zorder,
                        )
        axis.add_patch(Rectangle(
            (-model.target_theta, -model.target_omega),
            2 * model.target_theta, 2 * model.target_omega,
            edgecolor="black", facecolor="#d1d5db", alpha=.25, linewidth=1,
        ))
        axis.set(
            xlabel=r"Angle $\theta$ (rad; 0 = upright)", ylabel=r"Velocity $\omega$ (rad / time unit)",
            title=title, xlim=(-np.pi, np.pi),
            ylim=(-model.omega_limit, model.omega_limit),
        )
        full_angle_axis(axis)
        axis.grid(alpha=.15)
        _handles = [Line2D([], [], color="#9ca3af", label="Passive (u = 0)")]
        if controlled is not None:
            _handles.extend([
                Line2D([], [], color="#e67700", label="Reached target"),
                Line2D([], [], color="#1864ab", label="Not reached by cutoff"),
            ])
        axis.legend(handles=_handles, loc="upper center", fontsize=8, ncol=2)

    def plot_rollouts(rollouts, labels, passive, initial_index=0):
        _figure, _axes = plt.subplots(1, 3, figsize=(15, 3.9), constrained_layout=True)
        _colors = ["#9ca3af", "#0284c7", "#ea580c"]
        phase_portrait(_axes[2], passive, "Fitted policy orbits", controlled=rollouts[-1])
        for _index, (_rollout, _label, _color) in enumerate(zip(rollouts, labels, _colors)):
            if _index:
                _color = "#e67700" if _rollout["success"][initial_index] else "#1864ab"
            _end = int(np.sum(_rollout["alive"][initial_index])) + 1
            _states = np.asarray(_rollout["states"])[initial_index, :_end]
            _time = np.arange(len(_states)) * model.dt
            _theta = _states[:, 0].copy()
            _jumps = np.flatnonzero(np.abs(np.diff(_theta)) > np.pi) + 1
            _theta[_jumps] = np.nan
            _axes[0].plot(_time, _theta, label=_label, color=_color, linewidth=1.7)
            _axes[1].plot(_time, _states[:, 1], color=_color, linewidth=1.7)
            if _index < len(rollouts) - 1:
                _axes[2].plot(
                    _theta, _states[:, 1], color="#9ca3af", linewidth=1.2,
                    alpha=.7, linestyle=["--", ":", "-."][_index], zorder=2,
                )
        _axes[0].axhspan(-model.target_theta, model.target_theta, color="#16a34a", alpha=.12)
        _axes[1].axhspan(-model.target_omega, model.target_omega, color="#16a34a", alpha=.12)
        _axes[0].set(xlabel="Normalized time", ylabel=r"$\theta$ (rad; 0 = upright)", ylim=(-np.pi, np.pi))
        _axes[1].set(xlabel="Normalized time", ylabel=r"$\omega$ (rad / time unit)", ylim=(-3, 3))
        for _axis in _axes:
            _axis.grid(alpha=.15)
        _axes[0].legend(fontsize=8)
        return _figure

    return full_angle_axis, phase_portrait, phase_rollout, plot_rollouts, state_map


@app.cell
def _(learning, model, np, phase_rollout):
    _theta, _omega = np.meshgrid(
        np.linspace(-np.pi, np.pi, 8, endpoint=False) + np.pi / 8,
        [-1.2, 0.0, 1.2], indexing="ij",
    )
    phase_initial_states = np.vstack([
        np.column_stack([_theta.ravel(), _omega.ravel()]),
        learning.initial_states(model, n=8, seed=27),
    ])
    passive_orbits = phase_rollout(lambda states: np.zeros(len(states)), phase_initial_states)
    return passive_orbits, phase_initial_states


@app.cell
def _(grid_omega_count, grid_theta_count, learning, model):
    vi = learning.value_iteration(
        model, n_theta=grid_theta_count, n_omega=grid_omega_count, n_actions=15,
    )
    return (vi,)


@app.cell
def _(mo, model):
    _reset_on_model_change = model
    get_bellman_step, set_bellman_step = mo.state(
        0, allow_self_loops=True,
    )
    return get_bellman_step, set_bellman_step


@app.cell
def _(get_bellman_step, mo, model, set_bellman_step):
    bellman_step = mo.ui.slider(
        start=0, stop=model.horizon, step=1, value=get_bellman_step(),
        label="Bellman iteration (0 = zero initialization)", show_value=True,
        full_width=True, debounce=True, include_input=True,
        on_change=set_bellman_step,
    )
    bellman_first = mo.ui.button(
        label="First", disabled=get_bellman_step() == 0,
        on_click=lambda _: set_bellman_step(0),
    )
    bellman_previous = mo.ui.button(
        label="← Previous", disabled=get_bellman_step() == 0,
        on_click=lambda _: set_bellman_step(lambda i: max(0, i - 1)),
    )
    bellman_next = mo.ui.button(
        label="Next →", disabled=get_bellman_step() == model.horizon,
        on_click=lambda _: set_bellman_step(lambda i: min(model.horizon, i + 1)),
    )
    bellman_last = mo.ui.button(
        label="Last", disabled=get_bellman_step() == model.horizon,
        on_click=lambda _: set_bellman_step(model.horizon),
    )
    mo.vstack([
        bellman_step,
        mo.hstack([bellman_first, bellman_previous, bellman_next, bellman_last], justify="start"),
    ])
    return bellman_first, bellman_last, bellman_next, bellman_previous, bellman_step


@app.cell
def _(
    bellman_step, imitation_map, learning, maximum_return, minimum_return, model,
    passive_orbits, phase_initial_states, phase_portrait, phase_rollout,
    plt, state_map, vi,
):
    _h = int(bellman_step.value)
    _controlled = phase_rollout(
        lambda states: learning.vi_action(model, vi, states, remaining=_h),
        phase_initial_states,
    )
    _figure, _axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    _value_image = state_map(
        _axes[0, 0], vi["theta"], vi["omega"], vi["values"][_h],
        f"Value after {_h} backups ({_h * model.dt:.1f} time units)",
        cmap="viridis", vmin=minimum_return, vmax=maximum_return,
    )
    _policy_image = state_map(
        _axes[1, 0], vi["theta"], vi["omega"], vi["policy"][_h],
        f"Optimal action with {_h} steps remaining",
        cmap="coolwarm", vmin=-model.torque_limit, vmax=model.torque_limit,
    )
    phase_portrait(
        _axes[0, 1], passive_orbits, f"Controlled phase portrait (fixed h = {_h})",
        controlled=_controlled,
    )
    _fitted_image = state_map(
        _axes[1, 1], vi["theta"], vi["omega"], imitation_map,
        f"Fitted policy · iteration {_h} · u = clip(φᵀk)",
        cmap="coolwarm", vmin=-model.torque_limit, vmax=model.torque_limit,
    )
    _figure.colorbar(_fitted_image, ax=_axes[1, 1], label="Torque")
    _figure.colorbar(_value_image, ax=_axes[0, 0], label="Expected discounted reward")
    _figure.colorbar(_policy_image, ax=_axes[1, 0], label="Torque")
    _figure
    return


@app.cell
def _(learning, mo):
    policy_fit_source = mo.ui.dropdown(
        options={"Full state grid": "grid", "Local optimal rollouts": "rollouts"},
        value="Full state grid", label="Policy fitting data",
    )
    policy_basis = mo.ui.multiselect(
        options=learning.policy_feature_options(2), value=["θ", "θ̇"],
        label="Fitted policy features φ(θ, θ̇) — empty = zero torque", full_width=True,
    )
    policy_weight_power = mo.ui.slider(
        steps=[0, 1, 2, 4, 8, 16, 32], value=16, debounce=True, show_value=True,
        label="Value-weight exponent p (0 = uniform, 1 = positive value)",
    )
    mo.vstack([policy_fit_source, policy_basis, policy_weight_power])
    return policy_basis, policy_fit_source, policy_weight_power


@app.cell
def _(bellman_step, learning, model, policy_fit_source, vi):
    demonstration_data = learning.collect_policy_data(
        model, vi, n_samples=4000, remaining=bellman_step.value,
        source=policy_fit_source.value,
    )
    return (demonstration_data,)


@app.cell
def _(np, vi):
    _theta_mesh, _omega_mesh = np.meshgrid(vi["theta"], vi["omega"], indexing="ij")
    plot_states = np.column_stack([_theta_mesh.ravel(), _omega_mesh.ravel()])
    grid_shape = _theta_mesh.shape
    return grid_shape, plot_states


@app.cell
def _(demonstration_data, grid_shape, learning, model, plot_states, policy_basis,
      policy_weight_power, vi):
    _weights = learning.value_policy_weights(
        demonstration_data["values"], power=policy_weight_power.value,
    )
    imitation = learning.fit_linear_policy(
        model, vi, data=demonstration_data, feature_names=policy_basis.value,
        sample_weights=_weights,
    )
    imitation_map = learning.policy_action(
        model, imitation["coefficients"], plot_states, imitation["feature_names"],
    ).reshape(grid_shape)
    imitation_map[vi["target_mask"]] = 0
    return imitation, imitation_map


@app.cell
def _(imitation, imitation_map, learning, model, passive_orbits, phase_initial_states,
      phase_portrait, phase_rollout, plt, state_map, vi):
    _h = imitation["iteration"]
    _figure, _axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    _image = state_map(
        _axes[0, 0], vi["theta"], vi["omega"], imitation_map,
        f"Fitted policy · iteration {_h} · u = clip(φᵀk)",
        cmap="coolwarm", vmin=-model.torque_limit, vmax=model.torque_limit,
    )
    _error = imitation_map - vi["policy"][_h]
    _error_image = state_map(
        _axes[1, 0], vi["theta"], vi["omega"], _error,
        "Fitted torque − Bellman torque",
        cmap="PuOr", vmin=-2 * model.torque_limit, vmax=2 * model.torque_limit,
    )
    _controlled = phase_rollout(
        lambda states: learning.policy_action(
            model, imitation["coefficients"], states, imitation["feature_names"]),
        phase_initial_states,
    )
    phase_portrait(_axes[0, 1], passive_orbits, "Passive phase portrait (u = 0)")
    phase_portrait(_axes[1, 1], passive_orbits, f"Fitted policy orbits · iteration {_h}", controlled=_controlled)
    _figure.colorbar(_image, ax=_axes[0, 0], label="Torque")
    _figure.colorbar(_error_image, ax=_axes[1, 0], label="Torque error")
    _figure
    return


@app.cell
def _(imitation, learning, mo):
    _labels = {name: label for label, name in learning.policy_feature_options(2).items()}
    mo.ui.table([
        {"Feature": _labels[name], "Coefficient": float(weight)}
        for name, weight in zip(imitation["feature_names"], imitation["coefficients"])
    ], selection=None, pagination=False, show_column_summaries=False,
       label=(f"Iteration {imitation['iteration']} · {len(imitation['states'])} samples · "
              f"weighted LS RMSE {imitation['weighted_rmse']:.4f} · "
              f"unweighted clipped RMSE {imitation['rmse']:.4f} · "
              f"rank {imitation['rank']}/{len(imitation['feature_names'])}"))
    return


@app.cell
def _(mo):
    show_actor_critic = mo.ui.checkbox(value=False, label="Show actor–critic experiment")
    show_actor_critic
    return (show_actor_critic,)


@app.cell
def _(
    mo, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    training_controls = mo.ui.dictionary(
        {
            "seed": mo.ui.number(start=0, stop=9999, step=1, value=7, label="Random seed"),
            "iterations": mo.ui.slider(
                start=5, stop=50, step=5, value=30,
                label="Actor updates", show_value=True,
            ),
            "batch_size": mo.ui.dropdown(
                options={"64 episodes": 64, "128 episodes": 128, "256 episodes": 256},
                value="128 episodes", label="Episodes per batch",
            ),
        },
        label="Train again from random feedback",
    ).form(submit_button_label="Run seeded actor–critic experiment")
    training_controls
    return (training_controls,)


@app.cell
def _(
    learning, model, training_controls, mo, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    _settings = training_controls.value or {"seed": 7, "iterations": 30, "batch_size": 128}
    training = learning.train_actor_critic(
        model, iterations=int(_settings["iterations"]),
        batch_size=int(_settings["batch_size"]), seed=int(_settings["seed"]),
    )
    return (training,)


@app.cell
def _(
    learning, maximum_return, minimum_return, model, np, plt, training, mo, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    _batch = training["initial_batch"]
    _mask = _batch["alive"]
    _states = _batch["states"][:, :-1][_mask]
    _remaining = _batch["remaining"][_mask]
    _returns = _batch["returns"][_mask]
    _fitted = learning.predict_value(
        model, training["snapshots"][0]["critic"], _states, _remaining,
    )
    _sample = np.random.default_rng(0).choice(
        len(_states), size=min(2200, len(_states)), replace=False,
    )
    _figure, _axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    _cloud = _axes[0].scatter(
        _states[_sample, 0], _states[_sample, 1], c=_returns[_sample],
        s=5, alpha=.5, cmap="viridis", vmin=minimum_return, vmax=maximum_return,
    )
    _axes[0].set(
        title="Data gathered by the initial random actor",
        xlabel=r"$\theta$ (rad)", ylabel=r"$\omega$ (rad / time unit)",
        xlim=(-np.pi, np.pi), ylim=(-model.omega_limit, model.omega_limit),
    )
    _figure.colorbar(_cloud, ax=_axes[0], label="Monte Carlo discounted return")
    _axes[1].scatter(_returns[_sample], _fitted[_sample], s=5, alpha=.2, color="#7c3aed")
    _axes[1].plot(
        [minimum_return, maximum_return], [minimum_return, maximum_return], "k--", lw=1,
    )
    _axes[1].set(
        title="First critic: predictions on its training batch",
        xlabel="Monte Carlo discounted return", ylabel="Fitted discounted return",
        xlim=(minimum_return, maximum_return), ylim=(minimum_return, maximum_return),
    )
    _figure
    return


@app.cell
def _(
    mo, training, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    get_training_step, set_training_step = mo.state(
        len(training["snapshots"]) - 1, allow_self_loops=True,
    )
    return get_training_step, set_training_step


@app.cell
def _(
    get_training_step, mo, set_training_step, training, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    _last = len(training["snapshots"]) - 1
    training_step = mo.ui.slider(
        start=0, stop=_last, step=1, value=get_training_step(),
        label="Actor–critic iteration (0 = random actor)", show_value=True,
        full_width=True, debounce=True, include_input=True,
        on_change=set_training_step,
    )
    learning_first = mo.ui.button(
        label="First", disabled=get_training_step() == 0,
        on_click=lambda _: set_training_step(0),
    )
    learning_previous = mo.ui.button(
        label="← Previous", disabled=get_training_step() == 0,
        on_click=lambda _: set_training_step(lambda i: max(0, i - 1)),
    )
    learning_next = mo.ui.button(
        label="Next →", disabled=get_training_step() == _last,
        on_click=lambda _: set_training_step(lambda i: min(_last, i + 1)),
    )
    learning_last = mo.ui.button(
        label="Last", disabled=get_training_step() == _last,
        on_click=lambda _: set_training_step(_last),
    )
    mo.vstack([
        training_step,
        mo.hstack([learning_first, learning_previous, learning_next, learning_last], justify="start"),
    ])
    return learning_first, learning_last, learning_next, learning_previous, training_step


@app.cell
def _(
    grid_shape, learning, model, plot_states, training, training_step, vi, mo, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    selected_snapshot = training["snapshots"][int(training_step.value)]
    learned_value_map = learning.predict_value(
        model, selected_snapshot["critic"], plot_states, remaining=model.horizon,
    ).reshape(grid_shape)
    learned_policy_map = learning.policy_action(
        model, selected_snapshot["coefficients"], plot_states,
    ).reshape(grid_shape)
    learned_policy_map[vi["target_mask"]] = 0
    return learned_policy_map, learned_value_map, selected_snapshot


@app.cell
def _(
    learned_policy_map, learned_value_map, learning, maximum_return, minimum_return, model,
    passive_orbits, phase_initial_states, phase_portrait, phase_rollout, plt, selected_snapshot,
    state_map, training_step, vi, mo, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    _iteration = int(training_step.value)
    _controlled = phase_rollout(
        lambda states: learning.policy_action(model, selected_snapshot["coefficients"], states),
        phase_initial_states,
    )
    _figure, _axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    _critic_image = state_map(
        _axes[0, 0], vi["theta"], vi["omega"], learned_value_map,
        f"Fitted stochastic-policy value, iteration {_iteration}",
        cmap="viridis", vmin=minimum_return, vmax=maximum_return,
    )
    _actor_image = state_map(
        _axes[1, 0], vi["theta"], vi["omega"], learned_policy_map,
        f"Deterministic actor torque, iteration {_iteration}",
        cmap="coolwarm", vmin=-model.torque_limit, vmax=model.torque_limit,
    )
    phase_portrait(_axes[0, 1], passive_orbits, "Passive phase portrait (u = 0)")
    phase_portrait(
        _axes[1, 1], passive_orbits, f"Actor phase portrait (iteration {_iteration})",
        controlled=_controlled,
    )
    _figure.colorbar(_critic_image, ax=_axes[0, 0], label="Expected discounted reward")
    _figure.colorbar(_actor_image, ax=_axes[1, 0], label="Torque")
    _figure
    return


@app.cell
def _(imitation, learning, model, vi):
    evaluation_states = learning.initial_states(model, 256, seed=41)
    vi_evaluation = learning.vi_policy_rollout(model, vi, evaluation_states, imitation["iteration"])
    imitation_evaluation = learning.rollout(
        model, imitation["coefficients"], evaluation_states, feature_names=imitation["feature_names"],
    )
    return evaluation_states, imitation_evaluation, vi_evaluation


@app.cell
def _(
    imitation_evaluation, maximum_return, minimum_return, np, plt, training, training_step,
    vi_evaluation, mo, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    _history = training["history"]
    _steps = np.arange(len(_history))
    _return = np.array([_row["mean_return"] for _row in _history])
    _hit = np.array([_row["success_rate"] for _row in _history])
    _figure, _axes = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
    _axes[0].plot(_steps, _return, color="#0284c7", label="Deterministic actor")
    _axes[0].plot(
        _steps, [_row["mean_return_stochastic"] for _row in _history],
        color="#0284c7", linestyle="--", alpha=.6, label="Exploring training batch",
    )
    _axes[0].axhline(np.mean(vi_evaluation["return"]), color="#475569", label="Bellman rollout")
    _axes[0].axhline(np.mean(imitation_evaluation["return"]), color="#7c3aed", label="Least squares")
    _padding = 0.03 * (maximum_return - minimum_return)
    _axes[0].set(
        ylabel="Mean discounted return",
        ylim=(minimum_return - _padding, maximum_return + _padding),
    )
    _axes[0].legend(fontsize=7)
    _axes[1].plot(_steps, 100 * _hit, color="#0284c7")
    _axes[1].plot(
        _steps, [100 * _row["success_rate_stochastic"] for _row in _history],
        color="#0284c7", linestyle="--", alpha=.6,
    )
    _axes[1].set(ylabel="Target-hit fraction (%)", ylim=(-2, 102))
    _axes[2].plot(_steps, [_row["critic_rmse"] for _row in _history], color="#d97706")
    _axes[2].set(ylabel="Critic in-sample RMSE (reward units)", ylim=(0, None))
    for _axis in _axes:
        _axis.axvline(training_step.value, color="black", linestyle=":", alpha=.5)
        _axis.set(xlabel="Actor updates", xlim=(0, len(_history) - 1))
        _axis.grid(alpha=.15)
    _figure
    return


@app.cell
def _(imitation, imitation_evaluation, mo, np, vi_evaluation):
    _rows = []
    for _label, _batch in [
        (f"Bellman snapshot · iteration {imitation['iteration']}", vi_evaluation),
        (f"Fitted policy · iteration {imitation['iteration']}", imitation_evaluation),
    ]:
        _rows.append({
            "Controller": _label,
            "Mean discounted return": round(float(np.mean(_batch["return"])), 3),
            "Goal reward (+)": round(float(np.mean(_batch["goal_return"])), 3),
            "Torque reward (−)": round(float(np.mean(_batch["torque_return"])), 3),
            "Target hits (%)": round(100 * float(np.mean(_batch["success"])), 1),
            "Mean time to hit / cutoff": round(float(np.mean(_batch["elapsed_time"])), 3),
        })
    mo.ui.table(
        _rows, selection=None, show_column_summaries=False, show_data_types=False,
        show_search=False, show_download=False,
        label="256 identical held-out starts, without exploration; normalized time units",
    )
    return


@app.cell
def _(mo, model, phase_start_limits):
    _theta_span = max(float(phase_start_limits[0]) - model.target_theta, 0.02)
    _rest = model.target_theta + .5 * _theta_span
    _outward = model.target_theta + .2 * _theta_span
    _opposite = -(model.target_theta + .8 * _theta_span)
    _rest_label = f"At rest: θ = {_rest:.3f}, ω = 0"
    trajectory_case = mo.ui.dropdown(
        options={
            _rest_label: [_rest, 0.0],
            "Moving away from upright": [_outward, .4 * float(phase_start_limits[1])],
            "Opposite side, moving toward upright": [_opposite, .5 * float(phase_start_limits[1])],
        },
        value=_rest_label,
        label="Common initial condition (θ = 0 upright)",
    )
    trajectory_case
    return (trajectory_case,)


@app.cell
def _(np, phase_rollout, trajectory_case):
    comparison_passive_orbits = phase_rollout(
        lambda states: np.zeros(len(states)),
        np.asarray(trajectory_case.value)[None, :],
    )
    return (comparison_passive_orbits,)


@app.cell
def _(
    comparison_passive_orbits, imitation, learning, model, np, plot_rollouts,
    trajectory_case, vi,
):
    _start = np.asarray(trajectory_case.value)[None, :]
    _rollouts = [
        learning.rollout(model, [0.0, 0.0], _start),
        learning.vi_policy_rollout(model, vi, _start, imitation["iteration"]),
        learning.rollout(model, imitation["coefficients"], _start,
                         feature_names=imitation["feature_names"]),
    ]
    _labels = ["Passive", "Bellman", "Fitted policy"]
    plot_rollouts(_rollouts, _labels, comparison_passive_orbits)
    return


if __name__ == "__main__":
    app.run()
