"""Figures shared by the marimo lesson and the standalone slide deck."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from core.double_integrator_tracking import defects, qp_matrices

INK = "#233845"
BLUE = "#006e9c"
ORANGE = "#bc531f"
TEAL = "#08785e"
GRAY = "#667781"
STATE_LABELS = ["Position (m)", "Velocity (m/s)"]


def panels(labels, width=11):
    fig, axes = plt.subplots(1, len(labels), figsize=(width, 3.65), layout="constrained")
    for ax, label in zip(axes, labels):
        ax.set(xlabel="Time (s)", ylabel=label, xticks=range(5))
        ax.grid(alpha=.16)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-.13, 4.13)
        ax.axhline(0, color=INK, lw=.8, alpha=.32)
    return fig, axes


def ref(ax, model, j):
    ax.plot(model.curve_time, model.curve[:, j], "--", color=BLUE, lw=2, label="Reference")
    ax.plot(model.time, model.reference[:, j], "o", color=BLUE, markersize=4.5, markerfacecolor="white")


def step(ax, model, values, **style):
    # Include the right endpoint for the last held interval.
    ax.step(model.time, np.r_[values, values[-1]], where="post", **style)


def reference_plot(model):
    fig, axes = panels(STATE_LABELS)
    passive = model.reference[0] + np.outer(model.time, [model.reference[0, 1], 0.])
    for j, ax in enumerate(axes):
        ax.plot(model.time, passive[:, j],
                color=GRAY, lw=2, label="Passive motion")
        ref(ax, model, j)
    axes[0].legend(frameon=False)
    return fig


def forcing_plot(model, k=0):
    """Read acceleration at the intersection of required and predicted Δv."""
    fig, (inverse, action) = plt.subplots(1, 2, figsize=(11, 3.45), layout="constrained")
    dt = model.time[k+1] - model.time[k]
    required_dv = -model.forcing[k, 1]
    recovered = model.feedforward[k]
    inputs = np.array([-3.2, 3.2])
    for ax in (inverse, action):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.16)
        ax.set_ylim(*inputs)
    inverse.set(xlabel="Acceleration (m/s²)", ylabel="Velocity change (m/s)",
                title="One-step velocity change", xlim=inputs)
    inverse.plot(inputs, dt*inputs, color=INK, lw=2, label="Predicted  ΔT a")
    inverse.axhline(required_dv, color=BLUE, ls="--", lw=1.7, label="Required velocity change")
    inverse.vlines(recovered, inputs[0], required_dv, color=TEAL, ls=":", lw=1.5)
    inverse.plot(recovered, required_dv, "o", color=TEAL)
    inverse.legend(frameon=False, fontsize=9)
    action.set(xlabel="Time (s)", ylabel="Acceleration (m/s²)",
               title="Recovered action  āₖ", xlim=(-.13, 4.13), xticks=range(5))
    step(action, model, model.feedforward, color=TEAL, lw=2)
    action.plot(model.time[k:k+2], [recovered]*2, color=TEAL, lw=4)
    action.plot(model.time[k:k+2].mean(), recovered, "o", color=TEAL)
    return fig


def prediction_plot(model, plan):
    fig, axes = panels(STATE_LABELS)
    time = model.time[1:]
    passive = model.reference[:-1] @ model.A.T
    corrected = passive + plan.clipped[:, None] * model.B
    for j, ax in enumerate(axes):
        ax.plot(model.curve_time, model.curve[:, j], "--", lw=1.25, color=BLUE)
        ax.scatter(time, model.reference[1:, j], s=94, facecolors="none",
                   edgecolors=BLUE, linewidths=1.5, label="Next reference knot", zorder=3)
        ax.vlines(time, passive[:, j], corrected[:, j], color=TEAL, lw=3, alpha=.8)
        ax.scatter(time, passive[:, j], marker="x", color=ORANGE, s=45, label="Passive prediction", zorder=4)
        ax.scatter(time, corrected[:, j], color=TEAL, s=18, label="After clipped input", zorder=5)
    axes[0].legend(frameon=True, fontsize=9)
    return fig


def rollout_plot(model, plan, optimized=False):
    fig, axes = panels([*STATE_LABELS, "Acceleration (m/s²)"], width=12)
    for j, ax in enumerate(axes[:2]):
        ref(ax, model, j)
        ax.plot(model.time, plan.open_loop[:, j], ".-", color=ORANGE,
                label="Clipped rollout", lw=1.8)
        if optimized:
            ax.plot(model.time, plan.states[:, j], ".-", color=TEAL, label="Joint optimum", lw=1.9)
    ax = axes[2]
    step(ax, model, model.feedforward, color=BLUE, ls="--", lw=1.6, label="Feedforward")
    step(ax, model, plan.clipped, color=ORANGE, lw=2, label="Clipped")
    if optimized:
        step(ax, model, plan.actions, color=TEAL, lw=1.9, label="Optimized")
    for sign in [-1, 1]:
        ax.axhline(sign*plan.limit, color=INK, ls=":", alpha=.65, label="Input limits" if sign==1 else None)
    for ax in (axes[0], axes[2]):
        ax.legend(fontsize=8.5, framealpha=.9)
    return fig


def residual_plot(model, plan):
    fig, axes = panels(["Position defect (m)", "Velocity defect (m/s)"])
    pairs = [
        (model.reference, plan.clipped, "Reference + clipped inputs", ORANGE, "s", 8),
        (plan.open_loop, plan.clipped, "Rollout + clipped inputs", INK, "o", 9),
        (plan.states, plan.actions, "Planned states + inputs", TEAL, "+", 7),
    ]
    for states, actions, label, color, marker, size in pairs:
        residual = defects(model.A, model.B, states, actions)
        for j, ax in enumerate(axes):
            ax.plot(model.time[:-1], residual[:, j], marker, markerfacecolor="none",
                    color=color, markersize=size, label=label)
    # A pair of feasible trajectories can coincide at roundoff. Keep zero legible.
    for ax in axes:
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -.04), max(hi, .04))
        ax.set_xlabel("Interval start (s)")
    axes[0].legend(fontsize=8.5, framealpha=.9)
    return fig


def matrix_plot(model, steps=3):
    _, _, H, W, D, _ = qp_matrices(model, steps)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.45), layout="constrained")
    inputs = [f"a{k}" for k in range(steps)]
    variables = [label for k in range(steps) for label in (f"p{k+1}", f"v{k+1}", f"a{k}")]
    states = [label for k in range(steps) for label in (f"p{k+1}", f"v{k+1}")]
    for ax, title, matrix, rows, cols in zip(
        axes, ["Condensed cost  H", "Joint cost  W", "Dynamics equalities  D"],
        [H, W.toarray(), D.toarray()], [inputs, variables, states], [inputs, variables, variables]
    ):
        ax.imshow(abs(matrix) > 1e-12, interpolation="nearest",
                  cmap=ListedColormap(["#f0f2ef", BLUE]), vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(f"{matrix.shape[0]} × {matrix.shape[1]}")
        if steps == 3:
            ax.set_xticks(range(len(cols)), cols, fontsize=8)
            ax.set_yticks(range(len(rows)), rows, fontsize=8)
            for row, col in zip(*np.nonzero(abs(matrix) > 1e-12)):
                ax.text(col, row, f"{matrix[row,col]:.2g}", ha="center", va="center",
                        color="white", fontsize=8)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
    return fig
