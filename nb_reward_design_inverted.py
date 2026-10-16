import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from core import reward_design as rd
    from core import reward_design_views as rdv
    return mo, np, plt, rd, rdv


@app.cell
def _(mo):
    mo.md(r"""
    # Reward design: upright and at rest

    The inverted pendulum has two states:
    $\dot\theta=\omega$, $\dot\omega=\sin\theta-b\omega+u$.
    The goal is **$(\theta,\omega)=(0,0)$**: upright with zero angular velocity.
    Angles are measured from upright in radians and wrapped to $[-\pi,\pi)$.

    Collect random-policy trajectories, then edit two error variables and their rewards.
    Each error histogram shows its reward curve on the right axis. The bottom plot
    overlays the two distributions of **actual per-step rewards** using shared bins.
    Both channels use the same samples; editing expressions reuses the fixed data.
    """)
    return


@app.cell
def _(mo, rdv):
    rollout_controls = mo.ui.dictionary({
        **rdv.initial_distribution_fields("inverted"),
        "episodes": mo.ui.slider(steps=[16, 32, 64, 128, 256], value=128,
                                  label="Episodes", show_value=True),
        "steps": mo.ui.slider(steps=[100, 300, 600, 1200], value=600,
                               label="Steps per episode", show_value=True),
        "torque_limit": mo.ui.number(0, 5, step=0.1, value=1.0, label="Torque limit"),
        "walk_scale": mo.ui.number(0, 2, step=0.05, value=0.15,
                                    label="Random-walk increment SD / torque limit"),
        "damping": mo.ui.number(0, 1, step=0.01, value=0.03, label="Damping b"),
        "seed": mo.ui.number(0, 999999, step=1, value=0, label="Random seed"),
    }).form(submit_button_label="Collect rollout data")
    return (rollout_controls,)


@app.cell
def _(rd, rollout_controls):
    _settings = rollout_controls.value or {
        "episodes": 128, "steps": 600, "torque_limit": 1.0,
        "walk_scale": 0.15, "damping": 0.03, "seed": 0,
    }
    rollout_data = rd.collect_rollouts(kind="inverted", **_settings)
    return (rollout_data,)


@app.cell
def _(mo):
    preview_count = mo.ui.slider(1, 256, step=1, value=12, show_value=True,
                                 debounce=True, label="Animate first N episodes (up to batch size)")
    return (preview_count,)


@app.cell
def _(mo, preview_count, rdv, rollout_controls, rollout_data):
    rollout_figure = rdv.rollout_preview(rollout_data)
    _animation = mo.iframe(rdv.pendulum_animation_html(rollout_data, preview_count.value), height="460px")
    mo.accordion({"Rollout settings": mo.vstack([
        mo.md(r"""
        Choose **Uniform** and drag the angle and velocity bounds, **Normal** and set
        each coordinate's mean and standard deviation, or **Single state** using the
        means as the starting state. Angle and velocity are sampled independently.
        Only the selected distribution's parameters apply. Normal angles are sampled
        on the real line and wrapped for display. Press **Collect rollout data** to apply.

        Torque starts at zero. Each step adds Gaussian noise and clips to the torque limit.
        Torque is held for $\Delta t=0.04$ and integrated with RK4.
        There is no velocity clipping or goal termination. Change the seed for new data.
        The state-space plot shows all saved trajectories in translucent grey, with
        all initial states marked by red circles. The star marks upright at rest.
        Play the animation to see the first N physical pendulums, overlaid at a shared pivot.
        """),
        rollout_controls, rollout_figure, preview_count, _animation,
    ])})
    return (rollout_figure,)


@app.cell
def _(mo):
    get_position_preset, set_position_preset = mo.state(("np.exp(-0.5 * (x / 0.25)**2)", 0))
    get_position_variable, set_position_variable = mo.state(("theta", 0))

    def _energy_example(_):
        set_position_variable(lambda old: ("0.5 * omega**2 + np.cos(theta) - 1", old[1] + 1))
        set_position_preset(lambda old: ("np.exp(-0.5 * (x / 0.25)**2)", old[1] + 1))

    def _position_example(_):
        set_position_variable(lambda old: ("theta", old[1] + 1))
        set_position_preset(lambda old: ("np.exp(-0.5 * (x / 0.25)**2)", old[1] + 1))

    energy_example_button = mo.ui.button(label="Example: energy error (E − 1)", on_click=_energy_example)
    position_example_button = mo.ui.button(label="Restore position example", on_click=_position_example)
    position_squared_button = mo.ui.button(
        label="Reset: squared error (−x²)",
        on_click=lambda _: set_position_preset(lambda old: ("-x**2", old[1] + 1)),
    )
    position_exponential_button = mo.ui.button(
        label="Reset: squared exponential",
        on_click=lambda _: set_position_preset(
            lambda old: ("np.exp(-0.5 * (x / 0.25)**2)", old[1] + 1)),
    )
    return (energy_example_button, get_position_preset, get_position_variable,
            position_example_button, position_exponential_button, position_squared_button)


@app.cell
def _(get_position_preset, mo):
    position_default_source = get_position_preset()[0]
    position_reward = mo.ui.text(value=position_default_source,
                                  full_width=True, label="Position reward: r₁(x) =").form(
        submit_button_label="Apply position reward")
    return position_default_source, position_reward


@app.cell
def _(mo):
    get_velocity_preset, set_velocity_preset = mo.state(("np.exp(-0.5 * (x / 0.5)**2)", 0))
    velocity_squared_button = mo.ui.button(
        label="Reset: squared error (−x²)",
        on_click=lambda _: set_velocity_preset(lambda old: ("-x**2", old[1] + 1)),
    )
    velocity_exponential_button = mo.ui.button(
        label="Reset: squared exponential",
        on_click=lambda _: set_velocity_preset(
            lambda old: ("np.exp(-0.5 * (x / 0.5)**2)", old[1] + 1)),
    )
    return get_velocity_preset, velocity_exponential_button, velocity_squared_button


@app.cell
def _(get_velocity_preset, mo):
    velocity_default_source = get_velocity_preset()[0]
    velocity_reward = mo.ui.text(value=velocity_default_source,
                                  full_width=True, label="Velocity reward: r₂(x) =").form(
        submit_button_label="Apply velocity reward")
    return velocity_default_source, velocity_reward


@app.cell
def _(get_position_variable, mo):
    position_variable_default = get_position_variable()[0]
    position_variable = mo.ui.text(value=position_variable_default, full_width=True,
                                    label="Variable 1: x =").form(
        submit_button_label="Apply variable 1")
    return position_variable, position_variable_default


@app.cell
def _(mo):
    velocity_variable = mo.ui.text(value="omega", full_width=True,
                                    label="Velocity variable: x =").form(
        submit_button_label="Apply velocity variable")
    bins_control = mo.ui.slider(10, 150, step=5, value=60, show_value=True,
                                debounce=True, label="Histogram bins")
    reward_axis_min = mo.ui.number(stop=0, step=1, value=-10,
                                   label="Reward-axis minimum (maximum fixed at 1.1)")
    return bins_control, reward_axis_min, velocity_variable


@app.cell
def _(bins_control, energy_example_button, mo, position_example_button, position_exponential_button, position_reward,
      position_squared_button, position_variable, reward_axis_min, velocity_exponential_button,
      velocity_reward, velocity_squared_button, velocity_variable):
    mo.vstack([
        mo.md(r"""
        **Edit the variables.** Available in both channels: `theta`, `omega`,
        `theta_unwrapped`, `error` (angle error from upright, equal to `theta`),
        `u` (applied torque), `t` (time within episode), and `np`.
        Each sample is the state **after** a torque step, paired with that torque.
        Try `theta**2`, `omega**2`, or `0.5 * omega**2 + np.cos(theta) - 1`
        (normalized energy error relative to upright at rest).

        **Edit the rewards.** Each reward uses `x` from its own variable expression.
        Try `-x**2`, `1.0 * (np.abs(x) < 0.1)`, or a constant `1`.
        Use `**` for powers and `np.where(condition, a, b)` for piecewise rewards.
        """),
        mo.md(r"""
        **Energy example:** $E=\frac12\omega^2+\cos\theta$, with desired energy
        $E^*=E(0,0)=1$. The button sets variable 1 to $x=E-1$ and its reward to
        $r(x)=\exp[-\frac12(x/0.25)^2]$. Reward is 1 at the desired energy.
        Matching energy alone also rewards moving states with $E=1$; it does not
        uniquely identify the upright equilibrium at rest.
        """),
        mo.hstack([energy_example_button, position_example_button], justify="start"),
        mo.hstack([
            mo.vstack([position_variable,
                       mo.hstack([position_squared_button, position_exponential_button], justify="start"),
                       position_reward]),
            mo.vstack([velocity_variable,
                       mo.hstack([velocity_squared_button, velocity_exponential_button], justify="start"),
                       velocity_reward]),
        ], widths="equal", align="start"),
        bins_control, reward_axis_min,
    ])
    return


@app.cell
def _(mo, position_default_source, position_reward, position_variable, position_variable_default, rd, rollout_data,
      velocity_default_source, velocity_reward, velocity_variable):
    _namespace = rd.sample_namespace(rollout_data)
    channels = []
    for _label, _variable, _reward, _default_variable, _default_reward, _color in [
        ("Position", position_variable, position_reward, position_variable_default,
         position_default_source, "#0284c7"),
        ("Velocity", velocity_variable, velocity_reward, "omega",
         velocity_default_source, "#e67700"),
    ]:
        _source = _default_variable if _variable.value is None else _variable.value
        if _source.strip() == "0.5 * omega**2 + np.cos(theta) - 1":
            _label = "Energy"
        _reward_source = _default_reward if _reward.value is None else _reward.value
        _channel = {"label": _label, "variable": _source, "reward_source": _reward_source,
                    "color": _color, "samples": None, "rewards": None}
        _stage = "variable"
        try:
            _channel["samples"] = rd.evaluate_expression(
                _source, _namespace, len(_namespace["theta"]))
            _stage = "reward"
            _channel["rewards"] = rd.evaluate_expression(
                _reward_source, {"x": _channel["samples"]}, len(_namespace["theta"]))
        except ValueError as _exc:
            mo.output.append(mo.callout(f"{_label} {_stage}: {_exc}", kind="danger"))
        channels.append(_channel)
    return (channels,)


@app.cell
def _(bins_control, channels, mo, np, plt, rd, reward_axis_min, rollout_data):
    reward_figure = plt.figure(figsize=(14, 9), layout="constrained")
    _layout = reward_figure.add_gridspec(2, 2)
    _error_axes = [reward_figure.add_subplot(_layout[0, 0]),
                   reward_figure.add_subplot(_layout[0, 1])]
    _reward_axis = reward_figure.add_subplot(_layout[1, :])
    for _axis, _channel in zip(_error_axes, channels):
        _x, _r = _channel["samples"], _channel["rewards"]
        _color = _channel["color"]
        _axis.set(title=f"{_channel['label']} error and reward",
                  xlabel=f"x = {_channel['variable']}", ylabel="Sample count")
        _axis.grid(axis="y", alpha=0.15)
        if _x is None:
            _axis.text(0.5, 0.5, "Apply a valid variable expression",
                        ha="center", transform=_axis.transAxes)
            continue
        _axis.hist(_x, bins=bins_control.value, color=_color, alpha=0.35,
                    edgecolor="white", linewidth=0.4)
        _axis.axvline(0, color="#111827", linewidth=1.8, linestyle="--", zorder=5)
        if _r is None:
            continue
        _overlay = _axis.twinx()
        _grid = np.linspace(float(_x.min()), float(_x.max()), 1200)
        try:
            _curve = rd.evaluate_expression(_channel["reward_source"], {"x": _grid}, len(_grid))
            _overlay.plot(_grid, _curve, color=_color, lw=2, label="Reward curve r(x)")
        except ValueError:
            mo.output.append(mo.callout(
                f"{_channel['label']} curve undefined between samples; showing sampled rewards.",
                kind="warn"))
        _indices = np.linspace(0, len(_x) - 1, min(1500, len(_x)), dtype=int)
        _overlay.scatter(_x[_indices], _r[_indices], color=_color, s=5, alpha=0.18,
                          label="Evaluated samples (up to 1,500)")
        _overlay.set_ylabel("Per-step reward r(x)", color=_color)
        _overlay.set_ylim(reward_axis_min.value, 1.1)
        _overlay.axhline(0, color="#111827", linewidth=1.2, linestyle="--", alpha=0.7)
        _overlay.tick_params(axis="y", colors=_color)
        _overlay.legend(loc="upper right", fontsize=8)

    _valid = [_channel for _channel in channels if _channel["rewards"] is not None]
    if _valid:
        # Identical edges make the overlaid counts directly comparable.
        _edges = np.histogram_bin_edges(
            np.concatenate([_channel["rewards"] for _channel in _valid]), bins=bins_control.value)
        for _channel in _valid:
            _reward_axis.hist(_channel["rewards"], bins=_edges, histtype="stepfilled",
                              color=_channel["color"], alpha=0.3,
                              label=f"{_channel['label']} reward")
            _reward_axis.hist(_channel["rewards"], bins=_edges, histtype="step",
                              color=_channel["color"], linewidth=1.5)
        _reward_axis.legend()
    else:
        _reward_axis.text(0.5, 0.5, "Apply valid reward expressions",
                          ha="center", transform=_reward_axis.transAxes)
    _reward_axis.set(xlabel="Per-step reward", ylabel="Sample count",
                      title="Actual reward distributions · same samples and shared bin edges")
    _reward_axis.grid(axis="y", alpha=0.15)
    _reward_axis.axvline(0, color="#111827", linewidth=1.8, linestyle="--", zorder=5)
    mo.output.append(reward_figure)
    _episodes, _steps = rollout_data["actions"].shape
    mo.output.append(mo.md(
        f"**{_episodes * _steps:,} samples per channel** · {_episodes} episodes × {_steps} steps · "
        f"{_steps * rollout_data['dt']:g} time units per episode. "
        "All samples contribute to the histograms; dots show a subset for readability. "
        "Rewards are undiscounted per-step values. High position reward and high velocity "
        "reward can occur at different times: these marginal histograms alone do not "
        "show how often both states are near zero together."
    ))
    if _valid:
        mo.output.append(mo.ui.table([
            {"Reward": _channel["label"],
             "Mean": float(np.mean(_channel["rewards"])),
             "Standard deviation": float(np.std(_channel["rewards"])),
             "Minimum": float(np.min(_channel["rewards"])),
             "Maximum": float(np.max(_channel["rewards"])),
             "Zero rewards (%)": float(100 * np.mean(_channel["rewards"] == 0))}
            for _channel in _valid
        ], selection=None, pagination=False, show_column_summaries=False,
            show_data_types=False, show_search=False, show_download=False))
    return (reward_figure,)


if __name__ == "__main__":
    app.run()
