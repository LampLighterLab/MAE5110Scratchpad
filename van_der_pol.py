import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import re

    import marimo as mo
    import numpy as np
    from scipy.integrate import solve_ivp

    return json, mo, np, re, solve_ivp


@app.cell
def _(mo):
    mo.md(r"""
    # Van der Pol oscillator

    The Van der Pol equation

    \[
    \ddot{x} - \mu(1-x^2)\dot{x} + x = 0
    \]

    is a simple model of a **self-sustaining oscillator**. For
    $\mu > 0$, small motions are amplified while large motions are damped,
    producing a stable limit cycle—a useful motif for rhythmic locomotion.
    The plot axes stay fixed as $\mu$ changes so the geometry can be compared
    directly.
    """)
    return


@app.cell
def _(mo):
    mu = mo.ui.slider(
        start=0.0, stop=4.0, step=0.1, value=1.0,
        label=r"Nonlinearity $\mu$", show_value=True, full_width=True,
    )
    animate_points = mo.ui.switch(value=True, label="Animate tracked states")
    show_vector_field = mo.ui.switch(value=True, label="Show vector field")
    reverse_time = mo.ui.switch(value=False, label="Reverse time")
    auto_resample = mo.ui.switch(value=True, label="Auto-resample states")
    resample_clock = mo.ui.refresh(
        options=["2s", "5s", "10s"], default_interval="5s",
        label="Resampling interval",
    )
    animation_speed = mo.ui.slider(
        start=0.25, stop=2.0, step=0.25, value=1.0,
        label="Animation speed", show_value=True,
    )
    on_cycle_count = mo.ui.slider(
        start=0, stop=8, step=1, value=1,
        label="Points on limit cycle", show_value=True,
    )
    outside_count = mo.ui.slider(
        start=0, stop=8, step=1, value=1,
        label="Points outside", show_value=True,
    )
    inside_count = mo.ui.slider(
        start=0, stop=8, step=1, value=1,
        label="Points inside", show_value=True,
    )

    mo.vstack([
        mu,
        mo.hstack(
            [animate_points, show_vector_field, reverse_time, auto_resample],
            justify="space-between", gap=2,
        ),
        mo.hstack(
            [on_cycle_count, outside_count, inside_count],
            justify="space-between", gap=2,
        ),
        mo.hstack([animation_speed, resample_clock], justify="space-between", gap=2),
    ], gap=1.2)
    return (
        animate_points,
        animation_speed,
        auto_resample,
        inside_count,
        mu,
        on_cycle_count,
        outside_count,
        resample_clock,
        reverse_time,
        show_vector_field,
    )


@app.cell
def _(mu, np, solve_ivp):
    mu_value = mu.value

    def van_der_pol(_time, state):
        position, velocity = state
        acceleration = mu_value * (1.0 - position**2) * velocity - position
        return velocity, acceleration

    _settling_time = np.linspace(0.0, 80.0, 4001)
    _settled = solve_ivp(
        van_der_pol, (_settling_time[0], _settling_time[-1]), (2.0, 0.0),
        t_eval=_settling_time, method="DOP853", rtol=1e-8, atol=1e-10,
        max_step=0.05,
    )
    limit_cycle = _settled.y[:, _settled.t >= 60.0]

    # Shared by every mu value, making geometric comparisons meaningful.
    position_limits = (-4.0, 4.0)
    velocity_limits = (-8.0, 8.0)
    return limit_cycle, mu_value, position_limits, van_der_pol, velocity_limits


@app.cell
def _(
    auto_resample,
    inside_count,
    limit_cycle,
    np,
    on_cycle_count,
    outside_count,
    re,
    resample_clock,
    reverse_time,
    solve_ivp,
    van_der_pol,
):
    # Timer ticks keep this seed fixed when auto-resampling is switched off.
    _refresh_key = resample_clock.value if auto_resample.value else "paused"
    _seed = abs(hash((_refresh_key, 5110))) % (2**32)
    _rng = np.random.default_rng(_seed)

    _interval_match = re.match(r"([0-9.]+)s", resample_clock.value)
    animation_horizon = (
        float(_interval_match.group(1)) if _interval_match is not None else 5.0
    )
    _animation_time = np.linspace(0.0, animation_horizon, 181)
    _cycle_count = limit_cycle.shape[1]
    _tracked_initial_states = []
    tracked_categories = []
    for _ in range(on_cycle_count.value):
        _tracked_initial_states.append(
            limit_cycle[:, _rng.integers(0, _cycle_count)]
        )
        tracked_categories.append(0)
    for _ in range(outside_count.value):
        # Sample well outside the cycle, rejecting candidates that would begin
        # beyond the fixed viewing window.
        for _attempt in range(100):
            _outside_candidate = (
                _rng.uniform(1.35, 1.60)
                * limit_cycle[:, _rng.integers(0, _cycle_count)]
            )
            if (
                abs(_outside_candidate[0]) <= 3.7
                and abs(_outside_candidate[1]) <= 7.5
            ):
                break
        _tracked_initial_states.append(_outside_candidate)
        tracked_categories.append(1)
    for _ in range(inside_count.value):
        _tracked_initial_states.append(
            _rng.uniform(0.25, 0.60)
            * limit_cycle[:, _rng.integers(0, _cycle_count)]
        )
        tracked_categories.append(2)

    tracked_trajectories = []
    _time_direction = -1.0 if reverse_time.value else 1.0

    def _directed_dynamics(_time, state):
        _derivative = van_der_pol(_time, state)
        return (
            _time_direction * _derivative[0],
            _time_direction * _derivative[1],
        )

    for _tracked_state in _tracked_initial_states:
        _tracked = solve_ivp(
            _directed_dynamics, (_animation_time[0], _animation_time[-1]),
            _tracked_state, t_eval=_animation_time, method="DOP853",
            rtol=2e-8, atol=1e-10, max_step=0.03,
        )
        tracked_trajectories.append(_tracked.y)
    return animation_horizon, tracked_categories, tracked_trajectories


@app.cell
def _(
    animate_points,
    animation_horizon,
    animation_speed,
    json,
    limit_cycle,
    mo,
    mu_value,
    np,
    position_limits,
    reverse_time,
    show_vector_field,
    tracked_categories,
    tracked_trajectories,
    van_der_pol,
    velocity_limits,
):
    # 40 pixels per state-space unit on both axes: a true equal-aspect plot.
    _left, _top, _plot_width, _plot_height = 245, 72, 320, 640

    def _map_x(values):
        return _left + (np.asarray(values) - position_limits[0]) * _plot_width / (
            position_limits[1] - position_limits[0]
        )

    def _map_y(values):
        return _top + _plot_height - (
            np.asarray(values) - velocity_limits[0]
        ) * _plot_height / (velocity_limits[1] - velocity_limits[0])

    def _svg_path(x_values, y_values):
        _xs = _map_x(x_values)
        _ys = _map_y(y_values)
        return " ".join(
            ("M" if _index == 0 else "L") + f"{_x:.2f},{_y:.2f}"
            for _index, (_x, _y) in enumerate(zip(_xs, _ys))
        )

    _cycle_path = _svg_path(limit_cycle[0], limit_cycle[1])
    _vector_arrows = ""
    if show_vector_field.value:
        _field_x = np.linspace(-3.5, 3.5, 9)
        _field_y = np.linspace(-7.0, 7.0, 17)
        _arrow_parts = []
        for _x_value in _field_x:
            for _y_value in _field_y:
                _dx, _dy = van_der_pol(0.0, (_x_value, _y_value))
                if reverse_time.value:
                    _dx, _dy = -_dx, -_dy
                _magnitude = np.hypot(_dx, _dy)
                if _magnitude < 1e-10:
                    continue
                # Normalize direction so the field remains legible near fast regions.
                _half_length = 0.22
                _ux = _half_length * _dx / _magnitude
                _uy = _half_length * _dy / _magnitude
                _arrow_parts.append(
                    f'<line class="field-arrow" x1="{_map_x(_x_value - _ux):.2f}" '
                    f'y1="{_map_y(_y_value - _uy):.2f}" '
                    f'x2="{_map_x(_x_value + _ux):.2f}" '
                    f'y2="{_map_y(_y_value + _uy):.2f}" marker-end="url(#arrowhead)"/>'
                )
        _vector_arrows = "".join(_arrow_parts)

    _track_colors = ["#dc2626", "#d97706", "#7c3aed"]
    _track_labels = ["on limit cycle", "outside", "inside"]
    _track_paths = "".join(
        '<path class="tracked-path" stroke="' + _color + '" d="'
        + _svg_path(_track[0], _track[1])
        + '" />'
        for _track, _color in (
            (_track, _track_colors[_category])
            for _track, _category in zip(tracked_trajectories, tracked_categories)
        )
    )
    _track_points = "".join(
        f'<circle class="tracked-point" data-index="{_index}" '
        f'fill="{_color}" r="4.5" />'
        for _index, _color in enumerate(
            _track_colors[_category] for _category in tracked_categories
        )
    )
    _x_grid = "".join(
        f'<line class="{("axis" if _tick == 0 else "grid")}" '
        f'x1="{_map_x(_tick):.2f}" x2="{_map_x(_tick):.2f}" '
        f'y1="{_top}" y2="{_top + _plot_height}"/>'
        f'<text x="{_map_x(_tick):.2f}" y="735" '
        f'text-anchor="middle">{_tick:g}</text>'
        for _tick in [-4, -2, 0, 2, 4]
    )
    _y_grid = "".join(
        f'<line class="{("axis" if _tick == 0 else "grid")}" '
        f'x1="{_left}" x2="{_left + _plot_width}" '
        f'y1="{_map_y(_tick):.2f}" y2="{_map_y(_tick):.2f}"/>'
        f'<text x="{_left - 12}" y="{_map_y(_tick) + 4:.2f}" '
        f'text-anchor="end">{_tick:g}</text>'
        for _tick in [-8, -4, 0, 4, 8]
    )
    _pixel_tracks = [
        list(zip(
            _map_x(_track[0]).round(2),
            _map_y(_track[1]).round(2),
        ))
        for _track in tracked_trajectories
    ]
    _track_json = json.dumps(_pixel_tracks)
    _legend = "".join(
        f'<circle cx="{235 + 145 * _index}" cy="50" r="4.5" '
        f'fill="{_color}" stroke="#ffffff" stroke-width="1.5"/>'
        f'<text x="{245 + 145 * _index}" y="54">{_label} '
        f'({tracked_categories.count(_index)})</text>'
        for _index, (_label, _color) in enumerate(
            zip(_track_labels, _track_colors)
        )
    )

    _template = """
    <html><head><style>
      body { margin: 0; background: #ffffff; color: #111827; font-family: ui-sans-serif, system-ui, sans-serif; }
      svg { width: min(100%, 810px); height: auto; margin: 0 auto; display: block; }
      text { fill: #111827; font-size: 13px; }
      .title { font-size: 16px; font-weight: 650; }
      .axis-label { font-size: 14px; font-weight: 550; }
      .frame { fill: #ffffff; stroke: #6b7280; stroke-width: 1; }
      .grid { stroke: #d1d5db; stroke-width: 1; }
      .axis { stroke: #4b5563; stroke-width: 1.2; }
      .cycle-path { fill: none; stroke: #dc2626; stroke-width: 2.6; }
      .field-arrow { stroke: #64748b; stroke-width: 1.15; opacity: .48; }
      .tracked-path { fill: none; stroke-width: 1.3; opacity: .58; stroke-dasharray: 4 4; }
      .tracked-point { stroke: #ffffff; stroke-width: 1.5; }
    </style></head><body>
    <svg viewBox="0 0 810 780" role="img" aria-label="Animated Van der Pol phase portrait with equal axes">
      <title>Van der Pol oscillator phase portrait</title>
      <desc>Fixed equal-scale axes with a limit cycle, vector field, and three animated states.</desc>
      <defs>
        <clipPath id="phase-clip"><rect x="245" y="72" width="320" height="640"/></clipPath>
        <marker id="arrowhead" markerWidth="5" markerHeight="5" refX="4.2" refY="2.5" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L5,2.5 L0,5 Z" fill="#64748b" opacity=".48"/>
        </marker>
      </defs>
      <text class="title" x="405" y="24" text-anchor="middle">Phase portrait · μ = __MU____DIRECTION__</text>
      __LEGEND__
      <rect class="frame" x="245" y="72" width="320" height="640"/>
      __X_GRID____Y_GRID__
      <g clip-path="url(#phase-clip)">
        __VECTOR_ARROWS__<path class="cycle-path" d="__CYCLE_PATH__"/>
        __TRACK_PATHS____TRACK_POINTS__
      </g>
      <text class="axis-label" x="405" y="768" text-anchor="middle">position x</text>
      <text class="axis-label" transform="translate(172 392) rotate(-90)" text-anchor="middle">velocity dx/dt</text>
    </svg>
    <script>
      (() => {
        const tracks = __TRACK_JSON__;
        const points = Array.from(document.querySelectorAll('.tracked-point'));
        const duration = __DURATION__ * 1000 / __SPEED__;
        const running = __RUNNING__;
        const start = performance.now();
        function place(fraction) {
          tracks.forEach((track, k) => {
            const scaled = fraction * (track.length - 1);
            const i = Math.floor(scaled);
            const j = Math.min(i + 1, track.length - 1);
            const alpha = scaled - i;
            const x = track[i][0] * (1 - alpha) + track[j][0] * alpha;
            const y = track[i][1] * (1 - alpha) + track[j][1] * alpha;
            points[k].setAttribute('cx', x); points[k].setAttribute('cy', y);
          });
        }
        function frame(now) {
          place(((now - start) % duration) / duration);
          if (running) requestAnimationFrame(frame);
        }
        if (running) requestAnimationFrame(frame); else place(0);
      })();
    </script></body></html>
    """
    _html = (
        _template.replace("__MU__", f"{mu_value:.1f}")
        .replace("__DIRECTION__", " · reverse time" if reverse_time.value else "")
        .replace("__LEGEND__", _legend)
        .replace("__VECTOR_ARROWS__", _vector_arrows)
        .replace("__CYCLE_PATH__", _cycle_path)
        .replace("__TRACK_PATHS__", _track_paths)
        .replace("__TRACK_POINTS__", _track_points)
        .replace("__X_GRID__", _x_grid)
        .replace("__Y_GRID__", _y_grid)
        .replace("__TRACK_JSON__", _track_json)
        .replace("__DURATION__", str(animation_horizon))
        .replace("__SPEED__", str(animation_speed.value))
        .replace("__RUNNING__", str(animate_points.value).lower())
    )
    mo.iframe(_html, height="790px")
    return


@app.cell
def _(mo, mu_value):
    behavior = (
        "simple harmonic motion" if mu_value == 0
        else "increasingly sharp relaxation oscillations" if mu_value >= 3
        else "a smooth attracting limit cycle"
    )
    mo.callout(
        mo.md(
            rf"At $\mu={mu_value:.1f}$, the model shows **{behavior}**. "
            r"The lightly shaded arrows show the local phase-space flow; "
            r"the colored points compare states on, outside, and inside the cycle."
        ),
        kind="info",
    )
    return


if __name__ == "__main__":
    app.run()
