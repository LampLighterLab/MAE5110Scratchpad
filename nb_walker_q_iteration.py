import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    from core import walker_q_iteration as core
    from core import walker_stopping as stopping
    from core import walker_rollout as playback
    return BoundaryNorm, ListedColormap, Patch, core, mo, np, playback, plt, stopping


@app.cell
def _(mo):
    mo.md(r"""
    # Foot placement: learning on a Poincaré section

    **One state:** forward crossing speed $\omega=\dot\theta$ at $\theta=0$.
    **One action:** next half inter-leg angle $\alpha$, held through touchdown.
    The ankle is a fixed stopping policy: it stays off until the state enters
    its capture region, then balances to rest with the swing leg clear.

    Every capture maps to **$\omega=0$, upright standing**, which is absorbing.
    The default sparse reward is **+1 while the current state is zero, and
    zero otherwise**. There is no hand-tuned target-speed band. Discounting
    favors reaching standing in fewer decisions.
    Switch between full value-iteration sweeps and sampled ε-greedy Q-learning.
    Each run opens at its final iteration; scrub backward to inspect learning.
    Expand the settings below to change the grid, reward, initialization, or training.
    """)
    return


@app.cell
def _(mo):
    learning_mode = mo.ui.radio({
        "Value iteration": "value_iteration",
        "RL · ε-greedy Q-learning": "q_learning",
    }, value="Value iteration", inline=True, label="Learning method")
    learning_mode
    return (learning_mode,)


@app.cell
def _(ankle_view, dynamics_view, grid_setup_view, grid_summary, inspection_controls, learning_settings_view, mo, reward_editors_view, reward_params_view):
    mo.accordion({"Dynamics, rewards, initialization, and learning settings": mo.vstack([
        grid_setup_view, grid_summary, inspection_controls, dynamics_view, ankle_view,
        reward_params_view, reward_editors_view, learning_settings_view,
    ])})
    return


@app.cell
def _(mo):
    grid_controls = mo.ui.dictionary({
        "states": mo.ui.slider(steps=[51, 101, 201, 301, 401, 801], value=301, label="State grid points", show_value=True),
        "actions": mo.ui.slider(steps=[17, 33, 65, 81, 129], value=81, label="Action grid points", show_value=True),
        "speed_scale": mo.ui.number(0.2, 2.0, step=0.1, value=1.0, label="Speed upper bound / √(2g/ℓ)"),
        "slope": mo.ui.number(-0.2, 0.3, step=0.01, value=0.06, label="Downhill slope β (rad)"),
        "interpolation": mo.ui.dropdown(["nearest", "linear"], value="nearest", label="Successor lookup"),
    }).form(submit_button_label="Build / rebuild transition matrix")
    grid_setup_view = mo.vstack([
        grid_controls,
        mo.md(r"Assignment defaults: $g=9.81$, $m=\ell=1$, $\alpha\in[\pi/8,\pi/7]$. "
              "Changing slope recomputes the ankle capture region and may take longer. "
              "Reward edits and iteration playback reuse this matrix."),
    ])
    return grid_controls, grid_setup_view


@app.cell
def _(core, grid_controls, mo, np, stopping):
    _settings = grid_controls.value or {
        "states": 301, "actions": 81, "speed_scale": 1.0,
        "slope": 0.06, "interpolation": "nearest",
    }
    model = core.Walker(slope=float(_settings["slope"]))
    params = stopping.controller_params(model.slope)
    with mo.status.spinner(title="Loading or computing the ankle controller's capture region…"):
        region = stopping.capture_region(model.slope)
    with mo.status.spinner(title="Gridding section → impact → capture or next section…"):
        grid = core.build_grid(
            np.linspace(0, float(_settings["speed_scale"]) * np.sqrt(2 * model.gravity / model.length), int(_settings["states"])),
            np.linspace(np.pi / 8, np.pi / 7, int(_settings["actions"])),
            model, str(_settings["interpolation"]), region=region,
        )
    section_capture_limit = float(np.interp(0, region["angles"], region["upper"]))
    grid_summary = mo.md(
        f"**{grid.shape[0]:,} states × {grid.shape[1]:,} actions** · "
        f"{grid.build_seconds:.3f} s to build · "
        f"sparse P: {grid.matrix.shape[0]:,} × {grid.matrix.shape[1]:,}, "
        f"{grid.matrix.nnz:,} nonzeros.  \n"
        f"At θ = 0, the ankle controller can take over up to approximately "
        f"**ω = {section_capture_limit:.3f} rad/s**. It can also capture away from the section."
    )
    return grid, grid_summary, model, params, region, section_capture_limit


@app.cell
def _(grid, mo):
    selected_state = mo.ui.slider(0, len(grid.states) - 1, value=int(0.8 * (len(grid.states) - 1)), step=1, show_value=True, debounce=True, label="Inspect state index i")
    selected_action = mo.ui.slider(0, len(grid.actions) - 1, value=len(grid.actions) // 2, step=1, show_value=True, debounce=True, label="Inspect action index j")
    inspection_controls = mo.hstack([selected_state, selected_action], justify="start", gap=3)
    return inspection_controls, selected_action, selected_state


@app.cell
def _(plt):
    def heatmap(ax, grid, values, title, *, cmap="viridis", vmin=None, vmax=None):
        colors = plt.get_cmap(cmap).copy()
        colors.set_bad("#e5e7eb")
        image = ax.pcolormesh(grid.actions, grid.states, values, shading="nearest",
                              cmap=colors, vmin=vmin, vmax=vmax, rasterized=True)
        ax.set(xlabel="Action α (rad)", ylabel="Section speed ω (rad/s)", title=title)
        return image
    return (heatmap,)


@app.cell
def _(BoundaryNorm, ListedColormap, Patch, grid, heatmap, mo, np, plt, selected_action, selected_state):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    _image = heatmap(_axes[0], grid, grid.next_speed, "Next state F(ω, α); 0 = standing", vmin=0, vmax=grid.states[-1])
    _fig.colorbar(_image, ax=_axes[0], label="ω next (rad/s)")
    _colors = ["#3979b7", "#24937d", "#c95d56", "#dda844", "#8173aa"]
    _names = ["Next section", "Capture → standing (ω=0)", "Failed step", "Timeout", "Grid exit"]
    _axes[1].pcolormesh(grid.actions, grid.states, grid.status, shading="nearest",
                       cmap=ListedColormap(_colors), norm=BoundaryNorm(np.arange(-0.5, 5.5), 5))
    _axes[1].set(xlabel="Action α (rad)", ylabel="Section speed ω (rad/s)", title="Transition outcome before rounding")
    _axes[1].legend(handles=[Patch(color=c, label=n) for c, n in zip(_colors, _names)], loc="upper left", fontsize=8)
    for _ax in _axes:
        _ax.plot(grid.actions[selected_action.value], grid.states[selected_state.value], "o", ms=7, mfc="none", mec="black", mew=1.5)
    dynamics_view = mo.vstack([mo.md("### Gridded dynamics\nCapture maps to speed 0 after ankle balancing. Grey denotes failure, timeout, or a grid exit."), _fig])
    return (dynamics_view,)


@app.cell
def _(grid, mo, np, params, plt, region, selected_action, selected_state, stopping):
    _omega = grid.states[selected_state.value]
    _alpha = grid.actions[selected_action.value]
    selected_transition = stopping.simulate_transitions(_omega, _alpha, region, params, record=True)
    _trace = np.asarray(selected_transition["trace"])
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    _axes[0].fill_between(region["angles"], region["lower"], region["upper"], color="#24937d", alpha=0.22, label="Ankle capture region")
    _axes[0].plot(_trace[:, 1], _trace[:, 2], ".-", ms=2, color="#3979b7", label="Passive stance / impact")
    _axes[0].axvline(0, color="0.5", lw=1, ls="--", label="θ = 0 section")
    _axes[0].plot(0, _omega, "o", color="black", ms=5)
    _axes[1].plot(_trace[:, 0], _trace[:, 3] / (params["mass"] * params["gravity"] * params["length"]), color="#3979b7", label="Passive walking")
    if int(selected_transition["status"]) == stopping.CAPTURE:
        _status, _final, _balance_trace = stopping.simulate_balance(selected_transition["states"], params, record=True)
        _balance = np.asarray(_balance_trace)
        _axes[0].plot(_balance[:, 1], _balance[:, 2], color="#24937d", label="Ankle feedback to rest")
        _axes[1].plot(_balance[:, 0] + _trace[-1, 0], _balance[:, 3] / (params["mass"] * params["gravity"] * params["length"]), color="#24937d", label="Ankle feedback")
    _axes[0].set(xlabel="θ (rad)", ylabel="θ̇ (rad/s)", xlim=(-0.65, 0.65), ylim=(-1, max(5, _omega + 0.5)), title=f"Selected transition: ω={_omega:.3f}, α={_alpha:.4f}")
    _axes[1].axhline(-0.1, color="0.4", ls="--", lw=1)
    _axes[1].axhline(0.05, color="0.4", ls="--", lw=1)
    _axes[1].set(xlabel="Time (s)", ylabel="τ / (mgℓ)", title="Ankle activates only after capture")
    for _ax in _axes:
        _ax.legend(fontsize=8)
    ankle_view = mo.accordion({"Inspect the ankle controller and selected transition": mo.vstack([
        mo.md(r"""
        $\tau=mg\ell\,\mathrm{clip}(-\sin\theta-\theta-2\dot\theta/\sqrt{g/\ell},-0.1,0.05)$.
        These are the assignment's $k_p=1$, $k_d=2$ gains. The unsaturated law
        cancels gravity; the implemented controller respects the torque limits.
        The green region is the empirical basin of this **bounded** controller,
        not merely the set where its instantaneous torque is unsaturated.
        Entry latches balancing on and removes the swing leg from contact.

        The default region comes from the assignment's 481 × 801 rollout grid,
        with its successful intervals narrowed by half a velocity cell.
        Numerical settling uses the assignment's internal $10^{-4}$ criterion;
        that is a convergence check for the balancer, not a chosen stopping-speed reward band.
        """), _fig,
    ])})
    return ankle_view, selected_transition


@app.cell
def _(mo, params, solver_controls):
    # Add reward-specific parameters to this dictionary when needed.
    _discount = (solver_controls.value or {"discount": 0.95})["discount"]
    reward_params = params | {"discount": float(_discount)}
    _rows = "\n".join(f"| `{key}` | `{value!r}` |" for key, value in reward_params.items())
    reward_params_view = mo.md("### Reward parameters\n\n"
          "These values are passed as the `params` dictionary to your reward. "
          "`action` supplies the chosen α; `angle_of_attack` below is the model's nominal default.\n\n"
          "| Parameter | Value |\n| :--- | ---: |\n" + _rows)
    return reward_params, reward_params_view


@app.cell
def _(core, mo):
    _explanation = mo.md(r"""
    ### Reward and initialization

    The reward interface is `reward(state, action, params)`. `state` is the
    **current scalar Poincaré speed**, and `action` is $\alpha$ in radians.
    Both are arrays of shape `(states, actions)`; return a scalar or a
    broadcastable reward array. `params` is an ordinary dictionary containing
    the model/controller parameters and `discount`. For example:
    `-0.5 * params["mass"] * (params["length"] * state)**2` penalizes kinetic energy.
    No outcome labels or next-state placeholders are passed to the reward.

    The `reward_params` cell supplies the dictionary; add any custom reward
    parameters there when editing the notebook. Model keys include `gravity`,
    `length`, `mass`, `incline`, `torque_min`, `torque_max`, `position_gain`,
    `velocity_gain`, and the controller's numerical settings.

    Capture leads to the physical standing state, speed 0. Standing self-loops,
    so the default reward continues there and $V^*(0)=1/(1-\gamma)$.
    Rewards are evaluated **before** the transition: a walking action that
    leads to standing first earns its standing reward on the next decision.
    Failures receive the current-state reward and have zero continuation.

    Initialization still uses `initial_q(omega, alpha)`. A function of `omega`
    alone broadcasts a $V_0$ across actions. All grid states, including standing,
    use your initialization. Available editor globals: `np`, `discount`.
    """)
    reward_editor = mo.ui.code_editor(value=core.DEFAULT_REWARD_SOURCE, language="python", min_height=160, label="reward(state, action, params)").form(submit_button_label="Apply reward and restart learning")
    initial_editor = mo.ui.code_editor(value=core.DEFAULT_INITIAL_SOURCE, language="python", min_height=180, label="initial_q(omega, alpha)").form(submit_button_label="Apply initialization and restart learning")
    reward_editors_view = mo.vstack([_explanation, mo.hstack([reward_editor, initial_editor], widths="equal", gap=2)])
    return initial_editor, reward_editor, reward_editors_view


@app.cell
def _(mo):
    solver_controls = mo.ui.dictionary({
        "discount": mo.ui.slider(0.0, 0.995, step=0.005, value=0.95, label="Discount γ per section decision", show_value=True),
        "max_updates": mo.ui.number(0, 5000, step=10, value=1000, label="Maximum learning iterations (sweeps / RL batches)"),
        "tolerance_power": mo.ui.slider(3, 12, value=8, step=1, label="Residual tolerance: 10^(-x)", show_value=True),
    }).form(submit_button_label="Run / restart learning")
    rl_controls = mo.ui.dictionary({
        "epsilon": mo.ui.slider(0.0, 1.0, step=0.05, value=0.3, label="Exploration probability ε", show_value=True),
        "learning_rate": mo.ui.slider(0.01, 1.0, step=0.01, value=0.2, label="Learning rate η", show_value=True),
        "samples_per_iteration": mo.ui.slider(steps=[64, 128, 256, 512, 1024], value=512, label="Sampled transitions per batch", show_value=True),
        "episode_length": mo.ui.number(1, 256, step=1, value=32, label="Episode step limit"),
        "seed": mo.ui.number(0, 1000000, step=1, value=0, label="Random seed"),
    }).form(submit_button_label="Apply exploration settings and restart learning")
    return rl_controls, solver_controls


@app.cell
def _(learning_mode, mo, rl_controls, solver_controls):
    _items = [mo.md("### Learning settings"), solver_controls]
    if learning_mode.value == "q_learning":
        _items += [rl_controls, mo.md(r"""
        Each RL iteration is a batch of **sequential sampled transitions**.
        With probability ε choose a uniform action; otherwise choose a maximizing
        action, breaking ties randomly. Update only the visited pair:

        $$Q(s,a)\leftarrow Q(s,a)+\eta[r(s,a)+\gamma\max_b Q(s',b)-Q(s,a)].$$

        Episodes start uniformly over grid states, reset after failure or the
        step limit, and keep collecting reward while standing. Time-limit resets
        retain the bootstrap. With linear successor lookup, sample its weights
        as probabilities in the gridded MDP. The fixed learning rate can leave
        fluctuations in that case. Full Bellman residuals are diagnostics only.
        """)]
    learning_settings_view = mo.vstack(_items)
    return (learning_settings_view,)


@app.cell
def _(core, grid, initial_editor, learning_mode, mo, reward_editor, reward_params, rl_controls, solver_controls):
    _settings = solver_controls.value or {"discount": 0.95, "max_updates": 1000, "tolerance_power": 8}
    experiment = None
    experiment_error = None
    try:
        reward_table, initial_table = core.prepare_tables(
            grid,
            core.DEFAULT_REWARD_SOURCE if reward_editor.value is None else reward_editor.value,
            core.DEFAULT_INITIAL_SOURCE if initial_editor.value is None else initial_editor.value,
            discount=float(_settings["discount"]), params=reward_params,
        )
        _kwargs = dict(discount=float(_settings["discount"]),
                       tolerance=10.0 ** (-int(_settings["tolerance_power"])),
                       max_updates=int(_settings["max_updates"]))
        if learning_mode.value == "q_learning":
            _rl = rl_controls.value or {"epsilon": 0.3, "learning_rate": 0.2,
                                       "samples_per_iteration": 512, "episode_length": 32, "seed": 0}
            with mo.status.spinner(title="Learning from ε-greedy sampled trajectories…"):
                experiment = core.q_learning(grid, reward_table, initial_table, **_kwargs,
                    epsilon=float(_rl["epsilon"]), learning_rate=float(_rl["learning_rate"]),
                    samples_per_iteration=int(_rl["samples_per_iteration"]),
                    episode_length=int(_rl["episode_length"]), seed=int(_rl["seed"]))
        else:
            experiment = core.q_iteration(grid, reward_table, initial_table, **_kwargs)
    except (core.EditorError, ValueError) as _error:
        experiment_error = str(_error)
    if experiment_error:
        mo.output.append(mo.callout(experiment_error, kind="danger"))
    return experiment, experiment_error


@app.cell
def _(experiment, mo, np):
    mo.stop(experiment is None)
    _status = "Converged" if experiment.converged else "Update limit reached; not yet converged"
    _units = "batches" if experiment.method == "q_learning" else "sweeps"
    _sampling = ""
    if experiment.method == "q_learning":
        _sampling = (f"{len(experiment.sample_rows):,} sampled transitions · "
                     f"{100 * np.count_nonzero(experiment.visit_counts) / experiment.visit_counts.size:.1f}% of state–action pairs visited.  \n")
    mo.md(
        f"**{_status}** · {experiment.updates:,} {_units} · {experiment.seconds[-1]:.3f} s  \n"
        + _sampling +
        f"Final Bellman residual: **{experiment.residuals[-1]:.3g}** · "
        f"certified grid-Q error bound: **{experiment.residuals[-1] / (1 - experiment.discount):.3g}**.  \n"
        "The bound concerns this discretized MDP; it does not bound capture-region or state-grid error."
    )
    return


@app.cell
def _(experiment, mo):
    mo.stop(experiment is None)
    _restart_on_experiment_change = experiment
    get_iteration, set_iteration = mo.state(experiment.updates, allow_self_loops=True)
    return get_iteration, set_iteration


@app.cell
def _(experiment, get_iteration, mo, set_iteration):
    mo.stop(experiment is None)
    _last = experiment.updates
    iteration = mo.ui.slider(0, max(1, _last), value=min(get_iteration(), _last), step=1,
                             disabled=_last == 0, show_value=True, include_input=True,
                             debounce=True, full_width=True, label=f"Learning {experiment.iteration_unit} k (0 = initialization)", on_change=set_iteration)
    first = mo.ui.button(label="First", disabled=get_iteration() == 0, on_click=lambda _: set_iteration(0))
    previous = mo.ui.button(label="← Previous", disabled=get_iteration() == 0, on_click=lambda _: set_iteration(lambda k: max(0, k - 1)))
    next_sweep = mo.ui.button(label="Next →", disabled=get_iteration() == _last, on_click=lambda _: set_iteration(lambda k: min(_last, k + 1)))
    last = mo.ui.button(label="Last", disabled=get_iteration() == _last, on_click=lambda _: set_iteration(_last))
    mo.vstack([iteration, mo.hstack([first, previous, next_sweep, last], justify="start")])
    return first, iteration, last, next_sweep, previous


@app.cell
def _(experiment, iteration, mo, np):
    mo.stop(experiment is None)
    sweep = min(int(iteration.value), experiment.updates)
    q_current = experiment.q_at(sweep)
    v_current = np.max(q_current, axis=1)
    greedy_indices = np.argmax(q_current, axis=1)
    return greedy_indices, q_current, sweep, v_current


@app.cell
def _(experiment, mo):
    mo.stop(experiment is None)
    q_limits = (
        min(float(q.min()) for q in experiment.checkpoints.values()),
        max(float(q.max()) for q in experiment.checkpoints.values()),
    )
    if experiment.method == "q_learning" and experiment.sample_values.size:
        q_limits = (min(q_limits[0], float(experiment.sample_values.min())),
                    max(q_limits[1], float(experiment.sample_values.max())))
    if q_limits[0] == q_limits[1]:
        q_limits = (q_limits[0] - 0.5, q_limits[1] + 0.5)
    return (q_limits,)


@app.cell
def _(mo):
    show_greedy_overlay = mo.ui.switch(value=False, label="Show greedy actions on the Q heatmap (markers)")
    show_greedy_overlay
    return (show_greedy_overlay,)


@app.cell
def _(experiment, greedy_indices, grid, heatmap, mo, np, plt, q_current, q_limits, section_capture_limit, selected_state, show_greedy_overlay, sweep, v_current):
    mo.stop(experiment is None)
    _fig, _axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    _stage = f"{experiment.iteration_unit} {sweep}"
    _im = heatmap(_axes[0, 0], grid, q_current, f"Q at {_stage}", vmin=q_limits[0], vmax=q_limits[1])
    _fig.colorbar(_im, ax=_axes[0, 0], label="Discounted return")
    if show_greedy_overlay.value:
        _walking = ~grid.source_goal
        _axes[0, 0].scatter(grid.actions[greedy_indices][_walking], grid.states[_walking],
                            s=10, facecolors="white", edgecolors="#263445", linewidths=0.4,
                            label="Greedy action")
        _axes[0, 0].legend(fontsize=8)
    _axes[0, 0].axhline(grid.states[selected_state.value], color="black", ls=":", lw=1)
    _axes[0, 1].plot(grid.states, v_current, color="#3979b7", lw=2, label=f"V at {_stage}")
    _axes[0, 1].plot(grid.states, experiment.final_q.max(axis=1), color="#24937d", ls="--", label="Last computed V")
    _axes[0, 1].set(xlabel="Section speed ω (rad/s)", ylabel="V = max Q", title="State value")
    _axes[0, 1].legend(fontsize=9)
    _policy = np.where(grid.source_goal, np.nan, grid.actions[greedy_indices])
    _final_policy = np.where(grid.source_goal, np.nan, grid.actions[np.argmax(experiment.final_q, axis=1)])
    _axes[1, 0].plot(grid.states, _policy, color="#3979b7", lw=1.5, label=f"Greedy α at {_stage}")
    _axes[1, 0].plot(grid.states, _final_policy, color="#24937d", ls="--", label="Last computed policy")
    _axes[1, 0].axvspan(0, min(section_capture_limit, grid.states[-1]), alpha=0.15, color="#24937d", label="Already in ankle capture region")
    _axes[1, 0].set(xlabel="Section speed ω (rad/s)", ylabel="α (rad)", title="Greedy policy (first action wins ties)")
    _axes[1, 0].legend(fontsize=8)
    _k = np.arange(len(experiment.residuals))
    _floor = min(experiment.tolerance * 0.01, 1e-14)
    _axes[1, 1].semilogy(_k, np.maximum(experiment.residuals, _floor), color="#3979b7", label="||TQ − Q||∞")
    if experiment.method == "value_iteration":
        _envelope = experiment.residuals[0] * experiment.discount**_k
        _axes[1, 1].semilogy(_k, np.maximum(_envelope, _floor), color="0.6", ls=":", label="γ^k × initial residual bound")
    _axes[1, 1].axhline(experiment.tolerance, color="#c95d56", lw=1, ls="--", label="Stopping tolerance")
    _axes[1, 1].axvline(sweep, color="black", ls=":", lw=1)
    _axes[1, 1].set(xlabel=f"Learning {experiment.iteration_unit} k", ylabel="Bellman residual", title="Convergence (zeros plotted at the floor)")
    _axes[1, 1].legend(fontsize=8)
    mo.vstack([_fig, mo.md(
        f"At **{_stage}**: residual **{experiment.residuals[sweep]:.3g}**, "
        f"grid-Q error bound **{experiment.residuals[sweep] / (1 - experiment.discount):.3g}**, "
        f"distance to last computed Q **{np.max(np.abs(q_current - experiment.final_q)):.3g}**. "
        "Q colors use fixed limits throughout this run. "
        "**V(ω) is the return from starting at speed ω; it is not a speed trajectory.** "
        "In RL, incomplete visitation can make Q and the greedy policy irregular. "
        "The optional markers show selected actions without connecting different states."
    )])
    return


@app.cell
def _(grid, mo):
    rollout_controls = mo.ui.dictionary({
        "initial_speed": mo.ui.number(0, float(grid.states[-1]), step=0.05,
                                     value=float(0.8 * grid.states[-1]), label="Starting section speed (rad/s)"),
        "max_decisions": mo.ui.number(1, 30, step=1, value=15, label="Maximum foot-placement decisions"),
    }).form(submit_button_label="Apply rollout settings")
    autoplay_rollout = mo.ui.switch(value=True, label="Auto-play and repeat the updated policy")
    return autoplay_rollout, rollout_controls


@app.cell
def _(autoplay_rollout, experiment, get_iteration, mo, rollout_controls, set_iteration):
    mo.stop(experiment is None)
    _last = experiment.updates
    animation_iteration = mo.ui.slider(0, max(1, _last), value=min(get_iteration(), _last), step=1,
        disabled=_last == 0, show_value=True, include_input=True, debounce=True, full_width=True,
        label=f"Animate learning {experiment.iteration_unit} (same selection as Q-space)", on_change=set_iteration)
    mo.vstack([
        mo.md("### Animate the current policy\n"
              "The walker uses the greedy policy at the selected learning iteration. "
              "This slider and the Q-space slider stay in sync. Changing the iteration, "
              "learning method, or applied reward rebuilds and replays the rollout. "
              "Different Q values can still choose the same actions and produce the same motion."),
        animation_iteration, autoplay_rollout,
        mo.accordion({"Rollout starting state and duration": rollout_controls}),
    ])
    return (animation_iteration,)


@app.cell
def _(autoplay_rollout, experiment, grid, mo, playback, q_current, region, rollout_controls, sweep):
    mo.stop(experiment is None)
    _settings = rollout_controls.value or {"initial_speed": 0.8 * grid.states[-1], "max_decisions": 15}
    policy_rollout = playback.rollout_policy(
        grid, q_current, float(_settings["initial_speed"]), region,
        max_decisions=int(_settings["max_decisions"]),
    )
    _capture = "" if policy_rollout.capture_time is None else f" · ankle activation at {policy_rollout.capture_time:.2f} s"
    _actions = ", ".join(f'{d["alpha"]:.4f}' for d in policy_rollout.decisions) or "No foot placement needed"
    _method = "Value iteration" if experiment.method == "value_iteration" else "ε-greedy Q-learning"
    _label = f"{_method} · {experiment.iteration_unit} {sweep} / {experiment.updates} · greedy policy"
    mo.vstack([
        mo.md(f"**Policy at {experiment.iteration_unit} {sweep}** · {policy_rollout.outcome} · "
              f"{policy_rollout.footstrikes} footstrikes · {policy_rollout.trace[-1, 0]:.2f} s{_capture}"),
        mo.iframe(playback.animation_html(policy_rollout, grid.params,
                  policy_label=_label, autoplay=autoplay_rollout.value), height="530px"),
        mo.md(f"Chosen α values (rad): **{_actions}**.  \n"
              "This is a continuous physical rollout: speeds are not snapped to grid nodes. "
              "At each section crossing, the same nearest/linear lookup selects a greedy action. "
              "Foot positions advance at impact, and the fixed ankle controller is simulated "
              "through settling. Playback stops at failure, a grid exit, or the decision/20 s limit."),
    ])
    return (policy_rollout,)


@app.cell
def _(grid, mo, np, playback, plt, policy_rollout, stopping):
    _speeds = playback.section_speeds(policy_rollout)
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 3), layout="constrained")
    _axes[0].plot(np.arange(len(_speeds)), _speeds, "o-", color="#3979b7")
    _axes[0].set(xlabel="Section decision k (capture maps to standing)",
                 ylabel="Section speed ω (rad/s)", title="Speeds along the animated rollout")
    _decisions = policy_rollout.decisions
    _axes[1].plot(np.arange(len(_decisions)), [d["alpha"] for d in _decisions], "o-", color="#24937d")
    _axes[1].set(xlabel="Section decision k", ylabel="Chosen α (rad)", title="Actions from the selected Q table",
                 ylim=(grid.actions[0] - 0.002, grid.actions[-1] + 0.002))
    for _ax in _axes:
        _ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    _items = [_fig]
    _returning = grid.status == stopping.RETURN
    if np.any(_returning) and np.all((grid.next_speed - grid.states[:, None])[_returning] < 0):
        _items.append(mo.callout(mo.md(
            "**This grid has no speed-increasing returning action.** "
            "The ankle stays off while walking and automatically balances after capture. "
            "At the assignment defaults (slope 0.06 rad, α ∈ [π/8, π/7]), "
            "collision losses exceed the downhill energy gained at every allowed angle. "
            "`return state` rewards keeping section speed high for as many decisions as possible, "
            "but sustained walking is unavailable with those defaults. "
            "It also rewards speed per decision, rather than average forward speed per second."
        ), kind="info"))
    _items.append(mo.md("The final zero denotes successful capture followed by settling; "
                        "failures and timeouts are not drawn as standing. "
                        "The continuous balancing motion is shown in the player."))
    mo.vstack(_items)
    return


@app.cell
def _(experiment, grid, heatmap, mo, np, plt, q_current, selected_action, selected_state, sweep):
    mo.stop(experiment is None)
    _i, _j = int(selected_state.value), int(selected_action.value)
    _continuation = grid.continuation(q_current)[_i, _j]
    _reward = experiment.reward[_i, _j]
    _backup = _reward + experiment.discount * _continuation
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 3.8), layout="constrained")
    _im = heatmap(_axes[0], grid, experiment.reward, "Immediate reward R(ω, α)")
    _fig.colorbar(_im, ax=_axes[0])
    _axes[1].plot(grid.actions, q_current[_i], color="#3979b7", label=f"Q at {experiment.iteration_unit} {sweep}")
    _axes[1].plot(grid.actions, experiment.backup(q_current)[_i], ls="--", color="#24937d", label="Full Bellman target TQ")
    _axes[1].axvline(grid.actions[_j], color="0.5", lw=1, ls=":")
    _axes[1].set(xlabel="α (rad)", ylabel="Q", title=f"Action slice at ω = {grid.states[_i]:.3f} rad/s")
    _axes[1].legend(fontsize=8)
    _row = grid.matrix.getrow(_i * len(grid.actions) + _j)
    _successors = ", ".join(
        f"{'failure' if int(c) == grid.terminal else f'ω={grid.states[c]:.4f}'}: {w:.4f}"
        for c, w in zip(_row.indices, _row.data)
    )
    mo.accordion({"Inspect the reward and one Bellman target": mo.vstack([mo.md(f"ω = **{grid.states[_i]:.4f}**, α = **{grid.actions[_j]:.5f}**. "
                     f"Successor weights: {_successors}.  \n"
                     f"**TQ = {_reward:.5g} + {experiment.discount:.3g} × {_continuation:.5g} = {_backup:.5g}**, "
                     f"compared with current Q = {q_current[_i, _j]:.5g}. "
                     "For RL, this is the expected target; each update samples one successor and moves partway toward its target."), _fig])})
    return


@app.cell
def _(grid, mo, plt, selected_action):
    _j = int(selected_action.value)
    _matrix = grid.matrix[_j::len(grid.actions)]
    _fig, _ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    _ax.spy(_matrix, markersize=1.5, aspect="auto")
    _ax.axvline(grid.terminal - 0.5, color="#c95d56", ls="--", lw=1)
    _ax.set(xlabel="Successor index (0 = standing; last = failure)", ylabel="Current state index", title=f"P for action α={grid.actions[_j]:.5f}")
    mo.accordion({"Sparse transition matrix and numerical conventions": mo.vstack([
        mo.md(r"""
        Flatten `(i, j)` in state-major order: row `i * A + j` of `grid.matrix`
        contains the successor weights. Column 0 is the physical standing state.
        One extra failure column has zero continuation. All rows sum to one.

        $$V_k(i)=\max_j Q_k(i,j),\qquad
        Q_{k+1}=R+\gamma\,\mathrm{reshape}\!\left(P[V_k;0]\right).$$

        Each sweep is synchronous. Nearest lookup has one nonzero per row;
        linear interpolation has at most two. Returning transitions use only
        walking grid nodes, so rounding cannot invent a successful capture.
        Captured trajectories map to state 0, including those already in the
        capture region. State 0 self-loops. Values outside the state grid use
        the failure column. Near the capture boundary, interpolation uses the
        nearest available walking node. Outcomes are classified before gridding.

        Rewards use the current scalar state and action, never outcome flags.
        Standing remains in the Q table and receives reward on every decision.
        The discount is per discrete decision, not per second or footstrike;
        capture can happen before touchdown. The balancer's settling trajectory
        is folded into the transition to standing, without intermediate rewards.
        Failure actions remain available and have zero continuation.

        Changing the reward or initialization reruns learning using this cached map.
        Value iteration uses full sparse Bellman sweeps; ε-greedy RL updates only
        the sampled state–action pair and bootstraps from the greedy successor value.
        Playback uses checkpoints and reconstructs intermediate sweeps for value
        iteration, or replays the actual sampled update log for RL. The residual bound is
        $\|Q_k-Q^*\|_\infty\leq\|TQ_k-Q_k\|_\infty/(1-\gamma)$.
        Tighten state/action grids to check discretization sensitivity separately.

        Model and controller sources: `../../CodeAssignments/assignments/assignment_2.md`,
        `walking_control.py`, the completed `instructor/inverted_pendulum_walker.py`,
        and `assignment_2_report.md`. Local copies keep this notebook independent
        of the assignment's unimplemented public model stub.
        """), _fig,
    ])})
    return


if __name__ == "__main__":
    app.run()
