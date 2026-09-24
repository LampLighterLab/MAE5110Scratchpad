import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Markov processes: transition dynamics

    A Markov process moves among a finite collection of states. Its transition
    matrix uses the convention

    \[
    P_{ij} = \Pr(X_{t+1}=j \mid X_t=i),
    \qquad \sum_j P_{ij}=1.
    \]

    This example draws a random ordering of the states and builds a deterministic
    path $D$ through that ordering. The **transition concentration** $c$ blends
    the path with the uniform transition matrix $U$:

    \[
    P(c)=cD+(1-c)U.
    \]

    Thus $c=1$ follows the path exactly, wrapping from the last state to the
    first, while $c=0$ assigns equal probability to every *other* state. There
    are no self-transitions unless a terminal state is enabled; its row is then
    overridden by $P_{kk}=1$ and $P_{kj}=0$ for $j\ne k$.

    Turn on **Enable rewards** to attach a reward $R_{ij}$ to each possible
    transition. The same time control then shows the expected discounted return
    accumulated from step $0$ through step $t-1$.

    Turn on **Enable actions** to obtain an MDP. Each action has its own
    transition and reward matrices, $P^{(a)}$ and $R^{(a)}$. The selected action
    is used at every step, making the resulting state distribution, return, and
    value function easy to compare across actions.
    """)
    return


@app.cell
def _(mo):
    node_count = mo.ui.slider(
        start=3,
        stop=10,
        step=1,
        value=6,
        label="Number of states",
        show_value=True,
    )
    enable_actions = mo.ui.switch(value=False, label="Enable actions")
    action_count = mo.ui.slider(
        start=2,
        stop=4,
        step=1,
        value=2,
        label="Number of actions",
        show_value=True,
    )
    regenerate = mo.ui.run_button(label="Generate new random paths")
    return action_count, enable_actions, node_count, regenerate


@app.cell
def _(action_count, node_count, np, regenerate):
    regenerate.value
    number_of_states = node_count.value
    number_of_actions = action_count.value
    _rng = np.random.default_rng()
    action_path_orders = [
        _rng.permutation(number_of_states).tolist()
        for _action in range(number_of_actions)
    ]
    return action_path_orders, number_of_actions, number_of_states


@app.cell
def _(
    action_count,
    action_path_orders,
    enable_actions,
    mo,
    node_count,
    number_of_actions,
    number_of_states,
    regenerate,
):
    concentration = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=1.0,
        label=r"Transition concentration $c$",
        show_value=True,
        full_width=True,
    )
    initial_state = mo.ui.dropdown(
        options={f"State {state}": state for state in range(number_of_states)},
        value=f"State {action_path_orders[0][0]}",
        label="Initial state",
    )
    terminal_enabled = mo.ui.switch(value=False, label="Enable terminal state")
    terminal_state = mo.ui.dropdown(
        options={f"State {state}": state for state in range(number_of_states)},
        value=f"State {action_path_orders[0][-1]}",
        label="Terminal state",
    )
    selected_action = mo.ui.dropdown(
        options={
            f"Action {action}": action for action in range(number_of_actions)
        },
        value="Action 0",
        label="Selected action",
    )
    enable_rewards = mo.ui.switch(value=False, label="Enable rewards")
    discount_factor = mo.ui.slider(
        start=0.0,
        stop=0.99,
        step=0.01,
        value=0.90,
        label=r"Discount factor $\gamma$",
        show_value=True,
    )
    time_step = mo.ui.slider(
        start=0,
        stop=30,
        step=1,
        value=0,
        label=r"Time step $t$",
        show_value=True,
        include_input=True,
        full_width=True,
    )

    mo.vstack(
        [
            concentration,
            mo.hstack(
                [node_count, initial_state, regenerate],
                justify="space-between",
                gap=2,
            ),
            mo.hstack(
                [terminal_enabled, terminal_state],
                justify="space-between",
                gap=2,
            ),
            mo.hstack(
                [enable_rewards, discount_factor],
                justify="space-between",
                gap=2,
            ),
            mo.hstack(
                [enable_actions, action_count, selected_action],
                justify="space-between",
                gap=2,
            ),
            time_step,
        ],
        gap=1.2,
    )
    return (
        concentration,
        discount_factor,
        enable_rewards,
        initial_state,
        selected_action,
        terminal_enabled,
        terminal_state,
        time_step,
    )


@app.cell
def _(action_path_orders, enable_actions, selected_action):
    active_action = selected_action.value if enable_actions.value else 0
    path_order = action_path_orders[active_action]
    return active_action, path_order


@app.cell
def _(
    action_path_orders,
    active_action,
    concentration,
    np,
    number_of_actions,
    number_of_states,
    terminal_enabled,
    terminal_state,
):
    uniform_matrix = np.full(
        (number_of_states, number_of_states), 1.0 / (number_of_states - 1)
    )
    np.fill_diagonal(uniform_matrix, 0.0)
    deterministic_matrices = np.zeros(
        (number_of_actions, number_of_states, number_of_states)
    )
    transition_matrices = np.zeros_like(deterministic_matrices)
    for _action, _path_order in enumerate(action_path_orders):
        for _index, _state in enumerate(_path_order):
            _next_state = (
                _path_order[_index + 1]
                if _index + 1 < number_of_states
                else _path_order[0]
            )
            deterministic_matrices[_action, _state, _next_state] = 1.0

        if terminal_enabled.value:
            deterministic_matrices[_action, terminal_state.value, :] = 0.0
            deterministic_matrices[
                _action, terminal_state.value, terminal_state.value
            ] = 1.0

        transition_matrices[_action] = (
            concentration.value * deterministic_matrices[_action]
            + (1.0 - concentration.value) * uniform_matrix
        )
        if terminal_enabled.value:
            transition_matrices[_action, terminal_state.value, :] = 0.0
            transition_matrices[
                _action, terminal_state.value, terminal_state.value
            ] = 1.0

    deterministic_matrix = deterministic_matrices[active_action]
    transition_matrix = transition_matrices[active_action]
    return deterministic_matrix, transition_matrix


@app.cell
def _(initial_state, np, number_of_states, time_step, transition_matrix):
    initial_distribution = np.zeros(number_of_states)
    initial_distribution[initial_state.value] = 1.0
    state_distribution = initial_distribution @ np.linalg.matrix_power(
        transition_matrix, time_step.value
    )
    return (state_distribution,)


@app.cell
def _(
    active_action,
    deterministic_matrix,
    discount_factor,
    enable_actions,
    enable_rewards,
    expected_immediate_reward,
    expected_return_history,
    expected_total_return,
    finite_horizon_values,
    initial_state,
    mo,
    np,
    number_of_states,
    path_order,
    return_horizons,
    reward_matrix,
    state_distribution,
    terminal_enabled,
    terminal_state,
    time_step,
    transition_matrix,
):
    _spacing = 620 / (number_of_states - 1)
    _node_radius = min(27.0, 0.34 * _spacing)
    _x_positions = {
        _state: 50 + _spacing * _index
        for _index, _state in enumerate(range(number_of_states))
    }
    _node_y = 180
    _max_probability = max(float(transition_matrix.max()), 1e-12)
    _action_suffix = (
        f" under action A{active_action}" if enable_actions.value else ""
    )
    _action_superscript = (
        f"<sup>(A{active_action})</sup>" if enable_actions.value else ""
    )

    _edge_records = []
    for _source in range(number_of_states):
        for _target in range(number_of_states):
            _probability = float(transition_matrix[_source, _target])
            if _probability < 1e-10:
                continue
            _edge_records.append((_probability, _source, _target))
    _edge_records.sort()

    _edges = []
    for _probability, _source, _target in _edge_records:
        _source_x = _x_positions[_source]
        _target_x = _x_positions[_target]
        _opacity = 0.12 + 0.70 * _probability
        _width = 1.6 + 3.0 * _probability
        _preferred = deterministic_matrix[_source, _target] > 0.5
        _edge_class = "edge preferred-edge" if _preferred else "edge"

        if _source == _target:
            _path = (
                f"M {_source_x - 0.48 * _node_radius:.1f},"
                f"{_node_y - 0.88 * _node_radius:.1f} "
                f"C {_source_x - 2.0 * _node_radius:.1f},"
                f"{_node_y - 3.25 * _node_radius:.1f} "
                f"{_source_x + 2.0 * _node_radius:.1f},"
                f"{_node_y - 3.25 * _node_radius:.1f} "
                f"{_source_x + 0.48 * _node_radius:.1f},"
                f"{_node_y - 0.88 * _node_radius:.1f}"
            )
        elif abs(_target_x - _source_x) <= 1.05 * _spacing:
            _direction = 1 if _target_x > _source_x else -1
            _line_y = _node_y - 8 if _direction > 0 else _node_y + 8
            _path = (
                f"M {_source_x + (_node_radius + 3) * _direction:.1f},{_line_y} "
                f"L {_target_x - (_node_radius + 3) * _direction:.1f},{_line_y}"
            )
        else:
            _direction = 1 if _target_x > _source_x else -1
            # Forward edges use the upper routing lane and reverse edges the
            # lower lane, keeping long transitions clear of intermediate nodes.
            _arc_side = -1 if _direction > 0 else 1
            _start_x = _source_x + 0.72 * _node_radius * _direction
            _end_x = _target_x - 0.72 * _node_radius * _direction
            _start_y = _node_y + 0.72 * _node_radius * _arc_side
            _end_y = _start_y
            _middle_x = 0.5 * (_source_x + _target_x)
            _control_y = 34 if _arc_side < 0 else 326
            _path = (
                f"M {_start_x:.1f},{_start_y:.1f} "
                f"Q {_middle_x:.1f},{_control_y} {_end_x:.1f},{_end_y:.1f}"
            )
        _edges.append(
            f'<path class="{_edge_class}" d="{_path}" '
            f'style="opacity:{_opacity:.3f};stroke-width:{_width:.2f}" '
            f'marker-end="url(#arrowhead)"/>'
        )

    _nodes = []
    for _state in range(number_of_states):
        _probability = float(state_distribution[_state])
        _lightness = 98.0 - 47.0 * np.sqrt(_probability)
        _is_initial = _state == initial_state.value
        _is_terminal = terminal_enabled.value and _state == terminal_state.value
        if _is_terminal:
            _nodes.append(
                f'<circle class="terminal-ring" cx="{_x_positions[_state]:.1f}" '
                f'cy="{_node_y}" r="{_node_radius + 4:.1f}"/>'
            )
        _nodes.append(
            f'<circle class="node" cx="{_x_positions[_state]:.1f}" cy="{_node_y}" '
            f'r="{_node_radius:.1f}" fill="hsl(28 94% {_lightness:.1f}%)" '
            f'stroke="{("#ea580c" if _is_initial else "#475569")}" '
            f'stroke-width="{(3 if _is_initial else 1.5)}"/>'
            f'<text class="node-label" x="{_x_positions[_state]:.1f}" y="{_node_y - 3}" '
            f'text-anchor="middle">S{_state}</text>'
            f'<text class="probability-label" x="{_x_positions[_state]:.1f}" '
            f'y="{_node_y + 13:.1f}" '
            f'text-anchor="middle">{_probability:.3f}</text>'
        )

    if terminal_enabled.value:
        _terminal_position = path_order.index(terminal_state.value)
        _displayed_path = path_order[: _terminal_position + 1]
        _path_label = " → ".join(f"S{_state}" for _state in _displayed_path)
        _path_label += f" → S{terminal_state.value} → ⋯"
    else:
        _path_label = " → ".join(f"S{_state}" for _state in path_order)
        _path_label += f" → S{path_order[0]} → ⋯"

    _header_cells = "".join(
        f'<th scope="col">S{_state}</th>' for _state in range(number_of_states)
    )
    _matrix_rows = []
    for _row in range(number_of_states):
        _cells = []
        for _column in range(number_of_states):
            _value = float(transition_matrix[_row, _column])
            _lightness = 98.0 - 48.0 * (_value / _max_probability)
            _text_color = "#ffffff" if _lightness < 64 else "#111827"
            _cells.append(
                f'<td style="background:hsl(214 78% {_lightness:.1f}%);'
                f'color:{_text_color}">{_value:.2f}</td>'
            )
        _matrix_rows.append(
            f'<tr><th scope="row">S{_row}</th>{"".join(_cells)}</tr>'
        )

    _distribution_text = "[" + ", ".join(
        f"{_probability:.3f}" for _probability in state_distribution
    ) + "]"

    _reward_section = ""
    if enable_rewards.value:
        _reward_header_cells = "".join(
            f'<th scope="col">S{_state}</th>'
            for _state in range(number_of_states)
        )
        _reward_rows = []
        for _row in range(number_of_states):
            _reward_cells = []
            for _column in range(number_of_states):
                if transition_matrix[_row, _column] <= 1e-12:
                    _reward_cells.append('<td class="inactive">—</td>')
                    continue
                _reward = float(reward_matrix[_row, _column])
                _reward_class = (
                    "positive"
                    if _reward > 0
                    else "negative"
                    if _reward < 0
                    else "zero"
                )
                _reward_cells.append(
                    f'<td class="{_reward_class}">{_reward:+.0f}</td>'
                )
            _reward_rows.append(
                f'<tr><th scope="row">S{_row}</th>{"".join(_reward_cells)}'
                f'<td class="expected">{expected_immediate_reward[_row]:+.2f}</td></tr>'
            )

        _chart_width = 560.0
        _chart_height = 245.0
        _left = 54.0
        _right = 18.0
        _top = 20.0
        _bottom = 42.0
        _plot_width = _chart_width - _left - _right
        _plot_height = _chart_height - _top - _bottom

        _return_min = min(0.0, float(np.min(expected_return_history)))
        _return_max = max(0.0, float(np.max(expected_return_history)))
        _return_span = _return_max - _return_min
        if _return_span < 1e-9:
            _return_min -= 1.0
            _return_max += 1.0
        else:
            _return_padding = 0.12 * _return_span
            _return_min -= _return_padding
            _return_max += _return_padding
        _return_span = _return_max - _return_min
        _horizon_max = max(1, int(return_horizons[-1]))

        def _return_x(_horizon):
            return _left + _plot_width * _horizon / _horizon_max

        def _return_y(_value):
            return _top + _plot_height * (_return_max - _value) / _return_span

        _return_points = " ".join(
            f"{_return_x(_horizon):.1f},{_return_y(_value):.1f}"
            for _horizon, _value in zip(
                return_horizons, expected_return_history
            )
        )
        _return_y_ticks = []
        for _tick_index in range(5):
            _tick_value = _return_min + _tick_index * _return_span / 4
            _tick_y = _return_y(_tick_value)
            _return_y_ticks.append(
                f'<line class="chart-grid" x1="{_left}" y1="{_tick_y:.1f}" '
                f'x2="{_chart_width - _right}" y2="{_tick_y:.1f}"/>'
                f'<text class="chart-tick" x="{_left - 7}" y="{_tick_y + 4:.1f}" '
                f'text-anchor="end">{_tick_value:+.1f}</text>'
            )
        _return_x_ticks = []
        _tick_horizons = sorted(
            set(
                int(round(_index * _horizon_max / 4))
                for _index in range(5)
            )
        )
        for _tick_horizon in _tick_horizons:
            _tick_x = _return_x(_tick_horizon)
            _return_x_ticks.append(
                f'<line class="chart-tick-mark" x1="{_tick_x:.1f}" '
                f'y1="{_top + _plot_height}" x2="{_tick_x:.1f}" '
                f'y2="{_top + _plot_height + 5}"/>'
                f'<text class="chart-tick" x="{_tick_x:.1f}" '
                f'y="{_top + _plot_height + 19}" '
                f'text-anchor="middle">{_tick_horizon}</text>'
            )
        _return_chart = f"""
        <svg class="reward-chart" viewBox="0 0 {_chart_width:.0f} {_chart_height:.0f}"
             role="img" aria-label="Expected cumulative return through the current horizon">
          <title>Expected cumulative return from state {initial_state.value}</title>
          {"".join(_return_y_ticks)}
          {"".join(_return_x_ticks)}
          <line class="chart-axis" x1="{_left}" y1="{_top}" x2="{_left}" y2="{_top + _plot_height}"/>
          <line class="chart-axis" x1="{_left}" y1="{_top + _plot_height}" x2="{_chart_width - _right}" y2="{_top + _plot_height}"/>
          <polyline class="return-line" points="{_return_points}"/>
          <circle class="return-point" cx="{_return_x(return_horizons[-1]):.1f}"
                  cy="{_return_y(expected_return_history[-1]):.1f}" r="4.5"/>
          <text class="chart-value" x="{_return_x(return_horizons[-1]) - 5:.1f}"
                y="{_return_y(expected_return_history[-1]) - 9:.1f}"
                text-anchor="end">{expected_total_return:+.2f}</text>
          <text class="axis-title" x="{_left + _plot_width / 2:.1f}" y="{_chart_height - 3}" text-anchor="middle">horizon h (transitions)</text>
          <text class="axis-title" transform="translate(13 {_top + _plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">expected return</text>
        </svg>
        """

        _value_min = min(0.0, float(np.min(finite_horizon_values)))
        _value_max = max(0.0, float(np.max(finite_horizon_values)))
        _value_span = _value_max - _value_min
        if _value_span < 1e-9:
            _value_min -= 1.0
            _value_max += 1.0
        else:
            _value_padding = 0.12 * _value_span
            _value_min -= _value_padding
            _value_max += _value_padding
        _value_span = _value_max - _value_min

        def _value_y(_value):
            return _top + _plot_height * (_value_max - _value) / _value_span

        _zero_y = _value_y(0.0)
        _bar_slot = _plot_width / number_of_states
        _bar_width = min(36.0, 0.62 * _bar_slot)
        _value_bars = []
        for _state, _value in enumerate(finite_horizon_values):
            _bar_x = _left + (_state + 0.5) * _bar_slot - _bar_width / 2
            _value_end_y = _value_y(float(_value))
            _bar_y = min(_zero_y, _value_end_y)
            _bar_height = max(1.2, abs(_zero_y - _value_end_y))
            _bar_class = "value-bar selected" if _state == initial_state.value else "value-bar"
            _label_y = _value_end_y - 6 if _value >= 0 else _value_end_y + 14
            _value_bars.append(
                f'<rect class="{_bar_class}" x="{_bar_x:.1f}" y="{_bar_y:.1f}" '
                f'width="{_bar_width:.1f}" height="{_bar_height:.1f}"/>'
                f'<text class="chart-value" x="{_bar_x + _bar_width / 2:.1f}" '
                f'y="{_label_y:.1f}" text-anchor="middle">{_value:+.1f}</text>'
                f'<text class="chart-tick" x="{_bar_x + _bar_width / 2:.1f}" '
                f'y="{_top + _plot_height + 18}" text-anchor="middle">S{_state}</text>'
            )
        _value_chart = f"""
        <svg class="reward-chart" viewBox="0 0 {_chart_width:.0f} {_chart_height:.0f}"
             role="img" aria-label="Finite-horizon value function for every starting state">
          <title>Value function at horizon {time_step.value}</title>
          <line class="chart-grid zero-line" x1="{_left}" y1="{_zero_y:.1f}" x2="{_chart_width - _right}" y2="{_zero_y:.1f}"/>
          <line class="chart-axis" x1="{_left}" y1="{_top}" x2="{_left}" y2="{_top + _plot_height}"/>
          <line class="chart-axis" x1="{_left}" y1="{_top + _plot_height}" x2="{_chart_width - _right}" y2="{_top + _plot_height}"/>
          {"".join(_value_bars)}
          <text class="axis-title" x="{_left + _plot_width / 2:.1f}" y="{_chart_height - 3}" text-anchor="middle">starting state s</text>
          <text class="axis-title" transform="translate(13 {_top + _plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">V<tspan baseline-shift="sub">{time_step.value}</tspan>(s)</text>
        </svg>
        """
        _reward_section = f"""
        <section class="reward-section">
          <div class="reward-heading">
            <div>
              <h2>Transition reward matrix R{_action_superscript}</h2>
              <p class="subtitle">Rows are current states; columns are next states. A dash marks an impossible transition.</p>
            </div>
            <div class="return-summary">
              <span>Expected discounted return through t = {time_step.value}</span>
              <strong>{expected_total_return:+.3f}</strong>
              <small>&gamma; = {discount_factor.value:.2f}</small>
            </div>
          </div>
          <table aria-label="Transition reward matrix">
            <thead><tr><th scope="col">from\\to</th>{_reward_header_cells}<th scope="col">E[R | S<sub>i</sub>]</th></tr></thead>
            <tbody>{"".join(_reward_rows)}</tbody>
          </table>
          <p class="note">Return shown is E[&Sigma;<sup>t-1</sup><sub>k=0</sub> &gamma;<sup>k</sup>R<sub>X<sub>k</sub>,X<sub>k+1</sub></sub> | X<sub>0</sub> = S<sub>{initial_state.value}</sub>].</p>
          <div class="reward-charts">
            <figure>
              <figcaption>Expected return from selected state S{initial_state.value}{_action_suffix}</figcaption>
              {_return_chart}
            </figure>
            <figure>
              <figcaption>Finite-horizon value function at t = {time_step.value}</figcaption>
              {_value_chart}
            </figure>
          </div>
        </section>
        """

    _html = f"""
    <html><head><style>
      * {{ box-sizing: border-box; }}
      body {{ margin: 0; padding: 10px 14px; background: #ffffff; color: #111827;
              font-family: ui-sans-serif, system-ui, sans-serif; }}
      .layout {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(420px, 1fr);
                 gap: 28px; align-items: start; max-width: 1280px; margin: 0 auto; }}
      h2 {{ margin: 0 0 8px; font-size: 17px; font-weight: 650; }}
      .subtitle {{ margin: 0 0 8px; color: #475569; font-size: 13px; }}
      svg {{ width: 100%; height: auto; display: block; }}
      svg text {{ fill: #111827; font-family: ui-sans-serif, system-ui, sans-serif; }}
      .edge {{ fill: none; stroke: #64748b; }}
      .preferred-edge {{ stroke: #2563eb; }}
      .node-label {{ font-size: 14px; font-weight: 650; }}
      .terminal-ring {{ fill: none; stroke: #b91c1c; stroke-width: 2; }}
      .probability-label {{ font-size: 10px; font-variant-numeric: tabular-nums; fill: #334155; }}
      .path-label {{ font-size: 13px; fill: #334155; }}
      table {{ border-collapse: collapse; width: 100%; table-layout: fixed;
               font-variant-numeric: tabular-nums; }}
      th, td {{ border: 1px solid #cbd5e1; text-align: center; padding: 8px 3px; font-size: 11px; }}
      th {{ background: #f8fafc; color: #334155; font-weight: 650; }}
      .distribution {{ grid-column: 1 / -1; border-top: 1px solid #cbd5e1;
                       padding-top: 12px; font-size: 14px; }}
      .distribution code {{ margin-left: 8px; color: #9a3412; font-size: 13px;
                            font-variant-numeric: tabular-nums; }}
      .note {{ color: #475569; font-size: 12px; margin-top: 8px; }}
      .reward-section {{ grid-column: 1 / -1; border-top: 1px solid #cbd5e1;
                         padding-top: 14px; }}
      .reward-heading {{ display: flex; justify-content: space-between; align-items: start;
                         gap: 24px; }}
      .reward-section table {{ max-width: 940px; }}
      .reward-section td.positive {{ background: #dcfce7; color: #166534; }}
      .reward-section td.negative {{ background: #fee2e2; color: #991b1b; }}
      .reward-section td.zero {{ background: #f1f5f9; color: #475569; }}
      .reward-section td.inactive {{ background: #ffffff; color: #94a3b8; }}
      .reward-section td.expected {{ background: #eff6ff; color: #1e3a8a; font-weight: 650; }}
      .return-summary {{ min-width: 250px; padding: 10px 14px; border: 1px solid #fdba74;
                         border-radius: 8px; background: #fff7ed; text-align: right; }}
      .return-summary span, .return-summary small {{ display: block; color: #7c2d12; }}
      .return-summary strong {{ display: block; color: #9a3412; font-size: 25px;
                                font-variant-numeric: tabular-nums; }}
      .reward-charts {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
                        gap: 28px; margin-top: 20px; }}
      figure {{ margin: 0; min-width: 0; }}
      figcaption {{ margin-bottom: 4px; color: #334155; font-size: 13px; font-weight: 650; }}
      .reward-chart {{ width: 100%; height: auto; display: block; }}
      .chart-axis {{ stroke: #64748b; stroke-width: 1; }}
      .chart-grid {{ stroke: #e2e8f0; stroke-width: 1; }}
      .zero-line {{ stroke: #94a3b8; }}
      .chart-tick-mark {{ stroke: #64748b; stroke-width: 1; }}
      .chart-tick {{ fill: #475569; font-size: 10px; font-variant-numeric: tabular-nums; }}
      .axis-title {{ fill: #334155; font-size: 11px; }}
      .chart-value {{ fill: #111827; font-size: 10px; font-weight: 650;
                      font-variant-numeric: tabular-nums; }}
      .return-line {{ fill: none; stroke: #2563eb; stroke-width: 2.5; }}
      .return-point {{ fill: #ea580c; stroke: #ffffff; stroke-width: 1.5; }}
      .value-bar {{ fill: #2563eb; }}
      .value-bar.selected {{ fill: #ea580c; }}
      @media (max-width: 850px) {{
        .layout {{ grid-template-columns: 1fr; }}
        .distribution {{ grid-column: 1; }}
        .reward-section {{ grid-column: 1; }}
        .reward-heading {{ display: block; }}
        .return-summary {{ margin-bottom: 12px; text-align: left; }}
        .reward-charts {{ grid-template-columns: 1fr; }}
      }}
    </style></head><body>
      <div class="layout">
        <section>
          <h2>State-transition graph</h2>
          <p class="subtitle">Node shading and labels show Pr(X<sub>{time_step.value}</sub> = S<sub>i</sub>){_action_suffix}.</p>
          <svg viewBox="0 0 720 350" role="img" aria-label="Markov state-transition graph and current state distribution">
            <title>State-transition graph at time {time_step.value}</title>
            <defs>
              <marker id="arrowhead" markerWidth="5" markerHeight="5" refX="4.4" refY="2.5"
                      orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L5,2.5 L0,5 Z" fill="#64748b"/>
              </marker>
            </defs>
            <text class="path-label" x="360" y="24" text-anchor="middle">
              {f'action A{active_action} path' if enable_actions.value else 'deterministic path'}: {_path_label}
            </text>
            {"".join(_edges)}
            {"".join(_nodes)}
          </svg>
        </section>
        <section>
          <h2>Transition matrix P{_action_superscript}</h2>
          <p class="subtitle">Rows are current states; columns are next states.</p>
          <table aria-label="Transition probability matrix">
            <thead><tr><th scope="col">from\\to</th>{_header_cells}</tr></thead>
            <tbody>{"".join(_matrix_rows)}</tbody>
          </table>
          <p class="note">Every row sums to 1. Darker cells have higher transition probability.</p>
        </section>
        <div class="distribution">
          <strong>State distribution at t = {time_step.value}:</strong>
          <code>{_distribution_text}</code>
        </div>
        {_reward_section}
      </div>
    </body></html>
    """
    mo.iframe(_html, height="1450px" if enable_rewards.value else "700px")
    return


@app.cell
def _(
    active_action,
    concentration,
    discount_factor,
    enable_actions,
    enable_rewards,
    expected_total_return,
    mo,
    terminal_enabled,
    terminal_state,
    time_step,
):
    _description = (
        "deterministic motion along the sampled path"
        if concentration.value == 1.0
        else "uniform transitions to every state"
        if concentration.value == 0.0
        else "a mixture of directed and diffuse transitions"
    )
    _terminal_description = (
        f" State {terminal_state.value} is terminal."
        if terminal_enabled.value
        else ""
    )
    _reward_description = (
        rf" With rewards enabled, the expected discounted return accumulated "
        rf"through $t={time_step.value}$ is **{expected_total_return:+.3f}** "
        rf"at $\gamma={discount_factor.value:.2f}$."
        if enable_rewards.value
        else ""
    )
    _action_description = (
        rf" Action $A_{{{active_action}}}$ is applied at every step, so the "
        rf"view uses $P^{{(A_{active_action})}}$"
        + (
            rf" and $R^{{(A_{active_action})}}$."
            if enable_rewards.value
            else "."
        )
        if enable_actions.value
        else ""
    )
    _callout_text = (
        rf"At $c={concentration.value:.2f}$ the process has **{_description}**. "
        + _terminal_description
        + rf" Move the time slider backward or forward to inspect $p_t=p_0P^t$; "
        + rf"the current view is $t={time_step.value}$."
        + _action_description
        + _reward_description
    )
    mo.callout(
        mo.md(_callout_text),
        kind="info",
    )
    return


@app.cell
def _(action_path_orders, np, number_of_actions, number_of_states):
    # Changing the sampled paths also samples a new, persistent reward model.
    _path_signature = tuple(tuple(_path) for _path in action_path_orders)
    _reward_rng = np.random.default_rng()
    reward_matrices = _reward_rng.integers(
        -5,
        6,
        size=(number_of_actions, number_of_states, number_of_states),
    ).astype(float)
    for _action in range(number_of_actions):
        np.fill_diagonal(reward_matrices[_action], 0.0)
    return (reward_matrices,)


@app.cell
def _(active_action, reward_matrices):
    reward_matrix = reward_matrices[active_action]
    return (reward_matrix,)


@app.cell
def _(
    discount_factor,
    initial_state,
    np,
    number_of_states,
    reward_matrix,
    time_step,
    transition_matrix,
):
    expected_immediate_reward = np.sum(
        transition_matrix * reward_matrix, axis=1
    )
    finite_horizon_values = np.zeros(number_of_states)
    _return_history = [0.0]
    for _step in range(time_step.value):
        finite_horizon_values = (
            expected_immediate_reward
            + discount_factor.value
            * transition_matrix
            @ finite_horizon_values
        )
        _return_history.append(
            float(finite_horizon_values[initial_state.value])
        )
    return_horizons = np.arange(time_step.value + 1)
    expected_return_history = np.asarray(_return_history)
    expected_total_return = float(
        finite_horizon_values[initial_state.value]
    )
    return (
        expected_immediate_reward,
        expected_return_history,
        expected_total_return,
        finite_horizon_values,
        return_horizons,
    )


if __name__ == "__main__":
    app.run()
