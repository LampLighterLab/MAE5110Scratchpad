"""Refresh the offline presentation: python -m slides.build_double_integrator_assets."""
from functools import cache
from io import StringIO
from inspect import getsource
from itertools import product
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import double_integrator_figures as plots
from core import double_integrator_tracking as tracking


def svg(figure):
    with StringIO() as buf:
        figure.savefig(buf, format="svg", metadata={"Date": None})
        return buf.getvalue()[buf.getvalue().index("<svg"):]


def plot_frame(figure, limits):
    """Keep the notebook's axes, with series and interactive legends supplied by the deck."""
    axes = figure.axes
    for index, (ax, ylim) in enumerate(zip(axes, limits, strict=True)):
        ax.set_ylim(ylim)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        for artist in [*ax.lines, *ax.collections]:
            if artist.get_gid() != "passive-field":
                artist.remove()
        ax.axhline(0, color=plots.INK, lw=.8, alpha=.32)
        ax.plot([], [], gid=f"series-{index}", zorder=3)
    picture = svg(figure)
    # Matplotlib's SVG coordinates are points with their origin at the top left.
    width, height = figure.get_size_inches() * 72
    panels = []
    for ax in axes:
        left, bottom, w, h = ax.get_position().bounds
        panels.append(dict(left=left*width, top=(1-bottom-h)*height, width=w*width,
                           height=h*height, xlim=ax.get_xlim(), ylim=ax.get_ylim()))
    plt.close(figure)
    return dict(svg=picture, panels=panels)


def numbers(values):
    return np.round(values, 12).tolist()


def phase_comparison_frame(state_limits):
    figure = plt.figure(figsize=(13, 4.9), layout="constrained")
    grid = figure.add_gridspec(2, 2, width_ratios=[1.3, 1])
    phase = figure.add_subplot(grid[:, 0])
    position = figure.add_subplot(grid[0, 1])
    velocity = figure.add_subplot(grid[1, 1], sharex=position)
    labels = [r"Position $x_t$ (m)", r"Velocity $\dot{x}_t$ (m/s)"]
    phase.set(xlabel=labels[0], ylabel=labels[1],
              xlim=state_limits[0], title="State space", aspect="equal")
    phase.set_xticks(np.arange(0., 3.01, .5))
    phase.set_yticks(np.arange(-1., 2.01, .5))
    for axis, label in zip([position, velocity], labels):
        axis.set(xlim=(-.13, 4.13), xticks=range(5), ylabel=label)
    position.set_title("Time traces")
    position.tick_params(labelbottom=False)
    velocity.set_xlabel("Time (s)")
    for axis in figure.axes:
        axis.spines[["top", "right"]].set_visible(False)
    # Passive dynamics are (v, 0). Use one common scale so speed is visible.
    x, v = np.meshgrid(np.arange(0., 3.01, .25), np.arange(-1., 2.01, .25))
    moving = v != 0
    phase.quiver(x[moving], v[moving], v[moving], np.zeros(moving.sum()),
                 angles="xy", scale_units="xy", scale=9, pivot="middle",
                 color="#c5cbcd", width=.0025, headlength=4.5, gid="passive-field")
    return plot_frame(figure, [state_limits[1], *state_limits])


def rollout_comparison_frame(state_limits, acceleration_limits, title):
    figure = plt.figure(figsize=(13, 5.5), layout="constrained")
    grid = figure.add_gridspec(3, 2, width_ratios=[1, 1.25])
    phase = figure.add_subplot(grid[:, 0])
    position = figure.add_subplot(grid[0, 1])
    velocity = figure.add_subplot(grid[1, 1], sharex=position)
    acceleration = figure.add_subplot(grid[2, 1], sharex=position)
    phase.set(xlabel="Position (m)", ylabel="Velocity (m/s)", aspect="equal",
              xlim=state_limits[0], title="State space")
    position.set_title(title)
    for axis, ylabel in [(position, "Position (m)"), (velocity, "Velocity (m/s)"),
                         (acceleration, "Acceleration (m/s²)")]:
        axis.set(xlim=(-.08, 4.08), ylabel=ylabel, xticks=range(5))
        axis.tick_params(labelbottom=axis is acceleration)
    acceleration.set_xlabel("Time (s)")
    for axis in figure.axes:
        axis.grid(alpha=.16)
        axis.spines[["top", "right"]].set_visible(False)
    return plot_frame(figure, [state_limits[1], *state_limits, acceleration_limits])


def build():
    mpl.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "mae5110-tracking"})
    baseline = tracking.reference_motion(2, .5)
    example = tracking.optimize(baseline, 1.25)
    state_limits = [(-.25, 3.25), (-1.25, 2.25)]
    accel = (-3.35, 3.35)
    frames = {
        "phaseComparison": phase_comparison_frame(state_limits),
        "forcing": plot_frame(plots.forcing_plot(baseline), [(-3.2, 3.2), accel]),
        "prediction": plot_frame(plots.prediction_plot(baseline, example), state_limits),
        "rollout": rollout_comparison_frame(state_limits, accel, "Open-loop time traces"),
        "feedback": rollout_comparison_frame(state_limits, (-8.2, 8.2), "Tracking with feedback"),
        "feedbackWide": rollout_comparison_frame(state_limits, (-12, 12), "Tracking with feedback"),
        "optimized": rollout_comparison_frame(state_limits, accel, "Planned motion"),
        "residual": plot_frame(plots.residual_plot(baseline, example), [(-1.65, 1.65), accel]),
    }
    for frame in frames.values():
        for index in range(len(frame["panels"])):
            assert f'<g id="series-{index}"/>' in frame["svg"]

    @cache
    def model(amplitude, dt):
        return tracking.reference_motion(amplitude, dt)

    time_steps = [.1, .25, .5, 1]
    gains = {}
    for dt in time_steps:
        m = model(0, dt)
        gains[dt] = numbers(tracking.tracking_gain(m.A, m.B))

    curves = {}
    cases = {}
    for amp_index, dt, limit_index in product(range(21), time_steps, range(1, 13)):
        amplitude, limit = amp_index / 10, limit_index / 4
        m = model(amplitude, dt)
        curve_key = f"{amplitude:g}/{dt:g}"
        if curve_key not in curves:
            curves[curve_key] = dict(time=numbers(m.curve_time), p=numbers(m.curve[:, 0]),
                                     v=numbers(m.curve[:, 1]))
        clipped = np.clip(m.feedforward, -limit, limit)
        open_loop = tracking.rollout(m.A, m.B, m.reference[0], clipped)
        plans = {}
        for running_cost, terminal in product([False, True], repeat=2):
            states, actions = tracking.solve_tracking_knots(m, limit, running_cost=running_cost,
                                                           terminal=terminal)
            plans[f"{int(running_cost)}/{int(terminal)}"] = dict(states=numbers(states),
                                                                 actions=numbers(actions))
        passive = m.reference[:-1] @ m.A.T
        corrected = passive + clipped[:, None]*m.B
        cases[f"{amplitude:g}/{dt:g}/{limit:g}"] = dict(
            curveKey=curve_key, time=numbers(m.time), ref=numbers(m.reference), forcing=numbers(m.forcing),
            required=numbers(m.feedforward), clipped=numbers(clipped),
            passive=numbers(passive), corrected=numbers(corrected), open=numbers(open_loop),
            plans=plans, limit=limit, steps=len(clipped),
            defects=[numbers(tracking.defects(m.A, m.B, s, a))
                     for s, a in [(m.reference, clipped), (open_loop, clipped)]],
        )

    path = Path(__file__).with_name("double_integrator_tracking.html")
    start = '<script id="cases" type="application/json">'
    before, after = path.read_text().split(start)
    _, after = after.split("</script>", 1)
    payload = json.dumps(dict(cases=cases, curves=curves, frames=frames, gains=gains, timeSteps=time_steps,
                              solverSource=getsource(tracking.solve_tracking_knots)),
                         separators=(",", ":"), allow_nan=False)
    path.write_text(before + start + payload + "</script>" + after)
    print(f"Embedded {len(cases)} slider settings in {path.name}")


if __name__ == "__main__":
    build()
