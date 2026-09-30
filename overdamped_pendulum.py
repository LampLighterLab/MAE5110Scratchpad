import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    from dataclasses import replace

    import pendulum_learning as pl

    return Line2D, mo, np, pl, plt, replace


@app.cell
def _(mo):
    torque = mo.ui.slider(
        0.15, 0.65, step=0.05, value=0.35, debounce=True,
        label=r"Torque limit $u_{\max}$", show_value=True,
    )
    planning_horizon = mo.ui.slider(
        60, 160, step=20, value=100, debounce=True,
        label="Time horizon (control steps)", show_value=True,
    )
    torque_penalty = mo.ui.slider(
        0.0, 5.0, step=0.1, value=0.1, debounce=True,
        label=r"Torque-penalty weight $\rho$", show_value=True,
    )
    discount_factor = mo.ui.slider(
        0.90, 0.995, step=0.005, value=0.98, debounce=True,
        label=r"Discount per control step $\gamma$", show_value=True, include_input=True,
    )
    mo.vstack([
        mo.hstack([torque, torque_penalty], justify="start", gap=3),
        mo.hstack([discount_factor, planning_horizon], justify="start", gap=3),
    ])
    return discount_factor, planning_horizon, torque, torque_penalty


@app.cell
def _(discount_factor, np, pl, planning_horizon, torque, torque_penalty):
    _n_theta = 1601
    plant = pl.Pendulum(
        kind="overdamped", dt=0.08, torque_limit=torque.value,
        horizon=planning_horizon.value, target_theta=np.pi / _n_theta,
        torque_cost=torque_penalty.value,
        discount=discount_factor.value, goal_reward=1.0,
    )
    vi = pl.value_iteration(plant, n_theta=_n_theta, n_actions=11)
    theta_grid = vi["theta"]
    angle_window = np.pi
    display_theta = theta_grid[:, None]
    return angle_window, display_theta, plant, theta_grid, vi


@app.cell
def _(Line2D, angle_window, np, pl, plant):
    _limit = pl.initial_state_limits(plant)[0]
    _angles = np.linspace(max(2 * plant.target_theta, 0.1 * _limit), _limit, 6)
    _global_angles = np.linspace(-np.pi, np.pi, 24, endpoint=False) + np.pi / 24
    orbit_starts = np.unique(np.concatenate((_global_angles, -_angles, _angles)))[:, None]
    passive_orbits = pl.rollout(plant, [0.0], orbit_starts)

    def full_angle_axis(axis):
        axis.set_xlim(-np.pi, np.pi)
        axis.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                        [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    def plot_phase_portrait(axis, title, controlled=None):
        layers = [(passive_orbits, "#9ca3af", "Passive", 2)]
        if controlled is not None:
            layers.append((controlled, "#e67700", "Current policy", 3))
        for batch, color, label, order in layers:
            for index, states in enumerate(batch["states"]):
                if order == 3:
                    color = "#e67700" if batch["success"][index] else "#1864ab"
                length = int(batch["alive"][index].sum())
                theta = states[:length, 0].copy()
                velocity = np.sin(theta) + batch["actions"][index, :length]
                if not length:
                    continue
                axis.scatter(theta[0], velocity[0], s=15, color=color, zorder=order + 2,
                             edgecolor="white", linewidth=0.5)
                jumps = np.flatnonzero(np.abs(np.diff(theta)) > np.pi) + 1
                theta[jumps] = np.nan
                velocity[jumps] = np.nan
                axis.plot(theta, velocity, color=color, lw=1.2, alpha=0.75,
                          zorder=order, label=label if index == 0 else None)
                visible = np.flatnonzero(np.isfinite(theta) & (np.abs(theta) < angle_window))
                for fraction in (0.25,):
                    if len(visible) < 4:
                        continue
                    start = int(visible[int(fraction * (len(visible) - 1))])
                    stop = min(start + 3, length - 1)
                    if stop > start and np.isfinite(theta[start:stop + 1]).all():
                        axis.annotate("", xy=(theta[stop], velocity[stop]),
                                      xytext=(theta[start], velocity[start]),
                                      arrowprops=dict(arrowstyle="->", color=color, lw=1.3),
                                      zorder=order + 1)
        axis.axvspan(-plant.target_theta, plant.target_theta, color="#2b8a3e", alpha=0.12)
        axis.axhline(0, color="#adb5bd", lw=0.8)
        axis.scatter([0], [0], marker="+", color="black", s=45, zorder=5)
        limit = 1 + plant.torque_limit + 0.05
        axis.set(title=title, xlabel=r"Angle from upright $\theta$ [rad]",
                 ylabel=r"Instantaneous rate $\dot\theta=\sin\theta+u$",
                 xlim=(-angle_window, angle_window), ylim=(-limit, limit))
        handles = [Line2D([], [], color="#9ca3af", label="Passive")]
        if controlled is not None:
            handles.extend([
                Line2D([], [], color="#e67700", label="Reached target"),
                Line2D([], [], color="#1864ab", label="Not reached by cutoff"),
            ])
        axis.legend(handles=handles, loc="upper right", fontsize=8)
        full_angle_axis(axis)
        axis.grid(alpha=0.2)

    return full_angle_axis, orbit_starts, passive_orbits, plot_phase_portrait


@app.cell
def _(mo, plant):
    _reset_on_model_change = plant
    get_bellman_sweep, set_bellman_sweep = mo.state(
        0, allow_self_loops=True,
    )
    return get_bellman_sweep, set_bellman_sweep


@app.cell
def _(get_bellman_sweep, mo, plant, set_bellman_sweep):
    bellman_sweep = mo.ui.slider(
        0, plant.horizon, step=1, value=get_bellman_sweep(),
        label="Bellman iteration (0 = zero initialization)", show_value=True,
        full_width=True, include_input=True, debounce=True,
        on_change=set_bellman_sweep,
    )
    bellman_first = mo.ui.button(
        label="First", disabled=get_bellman_sweep() == 0,
        on_click=lambda _: set_bellman_sweep(0),
    )
    bellman_previous = mo.ui.button(
        label="← Previous", disabled=get_bellman_sweep() == 0,
        on_click=lambda _: set_bellman_sweep(lambda i: max(0, i - 1)),
    )
    bellman_next = mo.ui.button(
        label="Next →", disabled=get_bellman_sweep() == plant.horizon,
        on_click=lambda _: set_bellman_sweep(lambda i: min(plant.horizon, i + 1)),
    )
    bellman_last = mo.ui.button(
        label="Last", disabled=get_bellman_sweep() == plant.horizon,
        on_click=lambda _: set_bellman_sweep(plant.horizon),
    )
    mo.vstack([
        bellman_sweep,
        mo.hstack([bellman_first, bellman_previous, bellman_next, bellman_last], justify="start"),
    ])
    return bellman_first, bellman_last, bellman_next, bellman_previous, bellman_sweep


@app.cell
def _(angle_window, bellman_sweep, full_angle_axis, orbit_starts, pl, plant, plot_fitted_policy, plot_phase_portrait, plt, replace, theta_grid, vi):
    _h = bellman_sweep.value
    _passive_return = (pl.rollout(replace(plant, horizon=_h), [0.0], theta_grid[:, None])["return"]
                       if _h else 0 * theta_grid)
    _lower, _upper = pl.value_bounds(plant, plant.horizon)
    _controlled = pl.vi_policy_rollout(plant, vi, orbit_starts, _h)
    _fig, _ax = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    _ax[0, 0].plot(theta_grid, vi["values"][_h], color="#1864ab", lw=2,
                label=f"Value iteration: {_h} sweeps")
    _ax[0, 0].plot(theta_grid, _passive_return, "--",
                color="#9ca3af", label="Passive pendulum: zero torque")
    _ax[0, 0].set(title=f"Value iteration · h = {_h}",
               xlabel=r"Angle $\theta$ [rad]", ylabel="Discounted value V",
               ylim=(_lower - 0.02 * _upper, 1.03 * _upper))
    _ax[0, 0].legend(fontsize=9)
    _ax[1, 0].step(theta_grid, vi["policy"][_h], where="mid", color="#1864ab")
    _ax[1, 0].set(title=f"Current policy · h = {_h}",
               xlabel=r"Angle $\theta$ [rad]", ylabel=r"Greedy torque $u$",
               ylim=(-1.15 * plant.torque_limit, 1.15 * plant.torque_limit))
    plot_phase_portrait(_ax[0, 1], f"Current-policy orbits · fixed h = {_h}", _controlled)
    plot_fitted_policy(_ax[1, 1])
    for _axis in _ax[:, 0]:
        _axis.axvspan(-plant.target_theta, plant.target_theta, color="#2b8a3e",
                     alpha=0.15)
        _axis.grid(alpha=0.2)
        _axis.set(xlim=(-angle_window, angle_window), xlabel=r"Angle from upright $\theta$ [rad]")
        full_angle_axis(_axis)
    _fig
    return


@app.cell
def _(mo, pl):
    policy_fit_source = mo.ui.dropdown(
        options={"Full state grid": "grid", "Local optimal rollouts": "rollouts"},
        value="Full state grid", label="Policy fitting data",
    )
    policy_basis = mo.ui.multiselect(
        options=pl.policy_feature_options(1), value=["θ"],
        label="Fitted policy features φ(θ) — empty = zero torque", full_width=True,
    )
    policy_weight_power = mo.ui.slider(
        steps=[0, 1, 2, 4, 8, 16, 32], value=16, debounce=True, show_value=True,
        label="Value-weight exponent p (0 = uniform, 1 = positive value)",
    )
    mo.vstack([policy_fit_source, policy_basis, policy_weight_power])
    return policy_basis, policy_fit_source, policy_weight_power


@app.cell
def _(bellman_sweep, pl, plant, policy_fit_source, vi):
    demonstration_data = pl.collect_policy_data(
        plant, vi, n_samples=3000, remaining=bellman_sweep.value,
        source=policy_fit_source.value,
    )
    return (demonstration_data,)


@app.cell
def _(demonstration_data, pl, plant, policy_basis, policy_weight_power, vi):
    _weights = pl.value_policy_weights(
        demonstration_data["values"], power=policy_weight_power.value,
    )
    imitation = pl.fit_linear_policy(
        plant, vi, data=demonstration_data, feature_names=policy_basis.value,
        sample_weights=_weights,
    )
    return (imitation,)


@app.cell
def _(angle_window, display_theta, full_angle_axis, imitation, np, pl, plant, theta_grid, vi):
    _unclipped = pl.policy_features(plant, display_theta, imitation["feature_names"]) @ imitation["coefficients"]
    fitted_torque = np.where(np.abs(display_theta[:, 0]) <= plant.target_theta, 0,
                            np.clip(_unclipped, -plant.torque_limit, plant.torque_limit))

    def plot_fitted_policy(_ax):
        _stride = max(1, len(imitation["actions"]) // 1200)
        _h = imitation["iteration"]
        _ax.scatter(imitation["states"][::_stride, 0], imitation["actions"][::_stride],
                    s=7, alpha=0.15, color="#1864ab", label=f"State / action at iteration {_h}")
        _ax.plot(theta_grid, vi["policy"][_h], color="#1864ab", alpha=0.5,
                 label=f"Bellman policy · iteration {_h}")
        _ax.plot(display_theta[:, 0], _unclipped, "--", color="#e67700", lw=1,
                 label="Unclipped least-squares fit")
        _ax.plot(display_theta[:, 0], fitted_torque,
                 color="#e67700", lw=2, label="Executed fitted policy, clipped")
        _ax.axvspan(-plant.target_theta, plant.target_theta, color="#2b8a3e", alpha=0.12)
        _ax.set(title=f"Fitted policy · iteration {_h} · u = clip(φᵀk)",
                xlabel=r"Angle from upright $\theta$ [rad]", ylabel="Torque u",
                xlim=(-angle_window, angle_window),
                ylim=(-1.6 * plant.torque_limit, 1.6 * plant.torque_limit))
        _ax.grid(alpha=0.2)
        _ax.legend(loc="upper right", fontsize=9)
        full_angle_axis(_ax)
    return fitted_torque, plot_fitted_policy


@app.cell
def _(fitted_torque, full_angle_axis, imitation, orbit_starts,
      pl, plant, plot_fitted_policy, plot_phase_portrait, plt, theta_grid, vi):
    _fig, _axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    _h = imitation["iteration"]
    plot_fitted_policy(_axes[0, 0])
    _axes[1, 0].plot(theta_grid, fitted_torque - vi["policy"][_h], color="#e67700")
    _axes[1, 0].set(title="Fitted torque − Bellman torque", xlabel=r"Angle from upright $\theta$ [rad]",
                    ylabel="Torque error", ylim=(-2 * plant.torque_limit, 2 * plant.torque_limit))
    full_angle_axis(_axes[1, 0])
    _axes[1, 0].grid(alpha=0.2)
    _controlled = pl.rollout(plant, imitation["coefficients"], orbit_starts,
                             feature_names=imitation["feature_names"])
    plot_phase_portrait(_axes[0, 1], "Passive phase portrait · upright origin")
    plot_phase_portrait(_axes[1, 1], f"Fitted policy orbits · iteration {_h}", _controlled)
    _fig
    return


@app.cell
def _(imitation, mo, pl):
    _labels = {name: label for label, name in pl.policy_feature_options(1).items()}
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
    learning_steps = mo.ui.slider(
        5, 60, step=5, value=30, debounce=True, show_value=True,
        label="Actor updates",
    )
    episodes_per_batch = mo.ui.slider(
        steps=[64, 128, 256], value=128, label="Rollouts per batch", show_value=True,
    )
    random_seed = mo.ui.number(start=0, stop=9999, step=1, value=7, label="Random seed")
    mo.hstack([learning_steps, episodes_per_batch, random_seed], justify="start", gap=3)
    return episodes_per_batch, learning_steps, random_seed


@app.cell
def _(
    episodes_per_batch, learning_steps, pl, plant, random_seed, mo, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    learning = pl.train_actor_critic(
        plant, iterations=learning_steps.value,
        batch_size=episodes_per_batch.value, seed=int(random_seed.value),
    )
    initial_experience = learning["initial_batch"]
    return initial_experience, learning


@app.cell
def _(
    full_angle_axis, initial_experience, learning, np, pl, plant, plt, mo, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    _batch = initial_experience
    _initial = learning["snapshots"][0]
    _fig, _ax = plt.subplots(1, 3, figsize=(14, 3.5), layout="constrained")
    _t = np.arange(plant.horizon + 1) * plant.dt
    for _j in range(min(12, len(_batch["states"]))):
        _length = int(_batch["alive"][_j].sum())
        _ax[0].plot(_t[:_length + 1], _batch["states"][_j, :_length + 1, 0],
                    color="#9ca3af", alpha=0.65)
    _ax[0].axhspan(-plant.target_theta, plant.target_theta, color="#2b8a3e", alpha=0.15)
    _ax[0].set(xlabel="Time", ylabel=r"Angle $\theta$ [rad]", title="Initial random policy: 12 episodes",
               ylim=(-np.pi, np.pi))
    _mask = _batch["alive"]
    _stride = max(1, int(_mask.sum()) // 1500)
    _ax[1].scatter(_batch["states"][:, :-1, 0][_mask][::_stride],
                   _batch["actions"][_mask][::_stride], s=4, alpha=0.2, color="#6741d9")
    _ax[1].set(xlabel=r"Angle $\theta$ [rad]", ylabel="Applied torque", title="Exploration around linear feedback")
    _starts = _batch["states"][:, 0]
    _prediction = pl.predict_value(plant, _initial["critic"], _starts, plant.horizon)
    _order = np.argsort(_starts[:, 0])
    _ax[2].scatter(_starts[:, 0], _batch["return"], s=14, alpha=0.5, color="#6741d9", label="MC episode return")
    _ax[2].plot(_starts[_order, 0], _prediction[_order], color="#c92a2a", lw=2, label="Fitted initial critic")
    _lower, _upper = pl.value_bounds(plant, plant.horizon)
    _ax[2].set(xlabel=r"Initial angle $\theta_0$", ylabel="Discounted return", title="Critic slice: full horizon remains",
               ylim=(_lower - 0.02 * _upper, 1.04 * _upper))
    _ax[2].legend(fontsize=8)
    full_angle_axis(_ax[1])
    full_angle_axis(_ax[2])
    for _axis in _ax:
        _axis.grid(alpha=0.2)
    _fig
    return


@app.cell
def _(
    learning, mo, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    get_learning_iteration, set_learning_iteration = mo.state(
        len(learning["snapshots"]) - 1, allow_self_loops=True,
    )
    return get_learning_iteration, set_learning_iteration


@app.cell
def _(
    get_learning_iteration, learning, mo, set_learning_iteration, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    _last = len(learning["snapshots"]) - 1
    learning_iteration = mo.ui.slider(
        0, _last, value=get_learning_iteration(),
        step=1, show_value=True, full_width=True, include_input=True, debounce=True,
        label="Actor–critic iteration (0 = random initialization)",
        on_change=set_learning_iteration,
    )
    learning_first = mo.ui.button(
        label="First", disabled=get_learning_iteration() == 0,
        on_click=lambda _: set_learning_iteration(0),
    )
    learning_previous = mo.ui.button(
        label="← Previous", disabled=get_learning_iteration() == 0,
        on_click=lambda _: set_learning_iteration(lambda i: max(0, i - 1)),
    )
    learning_next = mo.ui.button(
        label="Next →", disabled=get_learning_iteration() == _last,
        on_click=lambda _: set_learning_iteration(lambda i: min(_last, i + 1)),
    )
    learning_last = mo.ui.button(
        label="Last", disabled=get_learning_iteration() == _last,
        on_click=lambda _: set_learning_iteration(_last),
    )
    mo.vstack([
        learning_iteration,
        mo.hstack([learning_first, learning_previous, learning_next, learning_last], justify="start"),
    ])
    return learning_first, learning_iteration, learning_last, learning_next, learning_previous


@app.cell
def _(
    learning, learning_iteration, mo, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    selected_snapshot = learning["snapshots"][learning_iteration.value]
    return (selected_snapshot,)


@app.cell
def _(
    angle_window, display_theta, full_angle_axis, imitation, learning, np, orbit_starts, pl, plant,
    plot_phase_portrait, plt, selected_snapshot, theta_grid, vi, mo, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    _fig, _ax = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    _initial = learning["snapshots"][0]
    for _snapshot, _label, _color in [
        (_initial, "Random initial policy", "#868e96"),
        (selected_snapshot, f"Iteration {selected_snapshot['iteration']}", "#6741d9"),
    ]:
        _critic = pl.predict_value(plant, _snapshot["critic"], display_theta, plant.horizon)
        _ax[0, 0].plot(display_theta[:, 0], _critic, color=_color, label=_label)
        _u = np.clip(display_theta @ _snapshot["coefficients"], -plant.torque_limit, plant.torque_limit)
        _u[pl.in_target(plant, display_theta)] = 0
        _ax[1, 0].plot(display_theta[:, 0], _u, color=_color, label=_label)
    _ax[0, 0].plot(theta_grid, vi["values"][-1], "--", color="#1864ab", label="Optimal grid value")
    _training_limit = pl.initial_state_limits(plant)[0]
    _ax[0, 0].axvspan(-angle_window, -_training_limit, color="#868e96", alpha=0.12)
    _ax[0, 0].axvspan(_training_limit, angle_window, color="#868e96", alpha=0.12)
    _lower, _upper = pl.value_bounds(plant, plant.horizon)
    _ax[0, 0].set(title=f"Exploratory value critic · iteration {selected_snapshot['iteration']}",
               xlabel=r"Angle $\theta$", ylabel="Discounted value V",
               ylim=(_lower - 0.02 * _upper, 1.04 * _upper))
    _ax[0, 0].legend(fontsize=8)
    _imitation_torque = np.where(pl.in_target(plant, display_theta), 0,
                                pl.policy_action(plant, imitation["coefficients"], display_theta,
                                                 imitation["feature_names"]))
    _ax[1, 0].plot(display_theta[:, 0], _imitation_torque,
                "--", color="#e67700", label="Least-squares imitation")
    _ax[1, 0].set(title=f"Current policy · k = {selected_snapshot['coefficients'][0]:.3f}",
               xlabel=r"Angle $\theta$", ylabel="Torque u",
               ylim=(-1.15 * plant.torque_limit, 1.15 * plant.torque_limit))
    _ax[1, 0].legend(fontsize=8)
    _controlled = pl.rollout(plant, selected_snapshot["coefficients"], orbit_starts)
    plot_phase_portrait(_ax[0, 1], "Passive phase portrait · upright origin")
    plot_phase_portrait(_ax[1, 1],
                        f"Current-policy orbits · iteration {selected_snapshot['iteration']}", _controlled)
    for _axis in _ax[:, 0]:
        _axis.grid(alpha=0.2)
        _axis.set(xlim=(-angle_window, angle_window), xlabel=r"Angle from upright $\theta$ [rad]")
        full_angle_axis(_axis)
    _fig
    return


@app.cell
def _(
    learning, pl, plant, plt, selected_snapshot, mo, show_actor_critic,
):
    mo.stop(not show_actor_critic.value)
    _fig, _ax = plt.subplots(figsize=(11, 3.4), layout="constrained")
    _lower, _upper = pl.value_bounds(plant, plant.horizon)
    _ax.plot([_h["mean_return"] for _h in learning["history"]], color="#6741d9", label="Fixed-start deterministic evaluation")
    _ax.plot([_h["mean_return_stochastic"] for _h in learning["history"]], color="#6741d9", alpha=0.4,
                label="Fresh exploratory batches")
    _ax.axvline(selected_snapshot["iteration"], color="black", ls=":", lw=1)
    _ax.set(title="Learning curve", xlabel="Actor updates", ylabel="Mean discounted return", ylim=(_lower - 0.02 * _upper, 1.04 * _upper))
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.2)
    _fig
    return


@app.cell
def _(imitation, np, pl, plant, vi):
    comparison_starts = np.linspace(-np.pi, np.pi, 161)[:, None]
    comparison_rollouts = {
        "Passive": pl.rollout(plant, [0.0], comparison_starts),
        "Bellman snapshot": pl.vi_policy_rollout(plant, vi, comparison_starts, imitation["iteration"]),
        "Fitted policy": pl.rollout(plant, imitation["coefficients"], comparison_starts,
                                     feature_names=imitation["feature_names"]),
    }
    return comparison_rollouts, comparison_starts


@app.cell
def _(comparison_rollouts, comparison_starts, full_angle_axis, mo, np, plt):
    _fig, _ax = plt.subplots(figsize=(11, 3.4), layout="constrained")
    _rows = []
    for (_label, _batch), _color in zip(comparison_rollouts.items(), ["#9ca3af", "#1864ab", "#e67700"]):
        _ax.plot(comparison_starts[:, 0], _batch["return"], label=_label, color=_color)
        _fail = ~_batch["success"]
        _ax.scatter(comparison_starts[_fail, 0], _batch["return"][_fail], marker="x", color=_color, s=20)
        _rows.append({
            "Controller": _label,
            "Mean discounted return": round(float(np.mean(_batch["return"])), 3),
            "Goal reward contribution": round(float(np.mean(_batch["goal_return"])), 3),
            "Torque contribution (negative)": round(float(np.mean(_batch["torque_return"])), 4),
            "Mean elapsed time (capped)": round(float(np.mean(_batch["elapsed_time"])), 3),
            "Reached target (%)": round(100 * float(np.mean(_batch["success"])), 1),
            "Timed out": int(_fail.sum()),
        })
    _ax.set(xlabel=r"Initial angle $\theta_0$", ylabel="Discounted return (higher is better)",
            title=f"Same {len(comparison_starts)} initial states, no exploration")
    full_angle_axis(_ax)
    _ax.grid(alpha=0.2)
    _ax.legend(fontsize=9)
    mo.vstack([
        _fig,
        mo.ui.table(_rows, selection=None, pagination=False, show_column_summaries=False),
    ])
    return


@app.cell
def _(mo, np, plant):
    demonstration_angle = mo.ui.slider(
        -3.1, 3.1, step=0.01,
        value=round(0.65 * np.arcsin(min(plant.torque_limit, 1.0)), 2),
        show_value=True, include_input=True,
        label="Initial angle from upright for trajectory comparison", debounce=True,
    )
    demonstration_time = mo.ui.slider(
        0, plant.horizon, step=1, value=0, show_value=True,
        label="Playback step", full_width=True,
    )
    mo.vstack([demonstration_angle, demonstration_time])
    return demonstration_angle, demonstration_time


@app.cell
def _(demonstration_angle, imitation, np, pl, plant, vi):
    _start = np.array([[demonstration_angle.value]])
    demonstration_rollouts = {
        "Passive": pl.rollout(plant, [0.0], _start),
        "Bellman snapshot": pl.vi_policy_rollout(plant, vi, _start, imitation["iteration"]),
        "Fitted policy": pl.rollout(plant, imitation["coefficients"], _start,
                                     feature_names=imitation["feature_names"]),
    }
    return (demonstration_rollouts,)


@app.cell
def _(demonstration_rollouts, demonstration_time, np, plant, plt):
    _fig, _ax = plt.subplots(1, 4, figsize=(16, 3.5), layout="constrained")
    _t = np.arange(plant.horizon + 1) * plant.dt
    for (_name, _batch), _color in zip(demonstration_rollouts.items(), ["#9ca3af", "#1864ab", "#e67700"]):
        if _name != "Passive":
            _color = "#e67700" if _batch["success"][0] else "#1864ab"
        _length = int(_batch["alive"][0].sum())
        _end = min(demonstration_time.value, _length)
        _angle = _batch["states"][0, _end, 0]
        _ax[0].plot([0, np.sin(_angle)], [0, np.cos(_angle)], "o-", lw=2, alpha=0.75, color=_color, label=_name)
        _ax[1].plot(_t[:_length + 1], _batch["states"][0, :_length + 1, 0], color=_color, label=_name)
        _ax[2].step(_t[:_length], _batch["actions"][0, :_length], where="post", color=_color)
        _ax[3].step(_t[1:], _batch["rewards"][0], where="pre", color=_color)
    _ax[0].set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal", title=f"Upright θ = 0 · t = {demonstration_time.value * plant.dt:.2f}")
    _ax[0].axis("off")
    _ax[0].legend(fontsize=8, loc="upper right")
    _ax[1].axhspan(-plant.target_theta, plant.target_theta, alpha=0.12, color="#2b8a3e")
    _ax[1].set(xlabel="Time", ylabel=r"Angle $\theta$", title="Motion until target capture",
               ylim=(-np.pi, np.pi))
    _ax[2].set(xlabel="Time", ylabel="Torque u", title="Control history")
    _ax[3].set(xlabel="Time", ylabel=r"Reward $r_t$", title="Reward continues after capture")
    for _axis in _ax[1:]:
        _axis.axvline(demonstration_time.value * plant.dt, color="black", ls=":", lw=1)
        _axis.grid(alpha=0.2)
    _fig
    return


if __name__ == "__main__":
    app.run()
