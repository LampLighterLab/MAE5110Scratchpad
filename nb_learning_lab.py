import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import marimo as mo
    import numpy as np
    from core import learning_models as models
    from core import learning_algorithms as algorithms
    from core import learning_views as views
    return algorithms, json, mo, models, np, views


@app.cell
def _(mo):
    mo.md("""
    # Learning lab

    Choose a model, then change how its controller learns. Compare the value
    function, chosen actions, convergence, and a rollout at any saved iteration.
    Settings and reward code are folded away below; each run opens at its final iteration.
    """)
    return


@app.cell
def _(mo, models):
    model_choice = mo.ui.dropdown(models.MODEL_LABELS, value="Simple MDP", label="1 · Model", full_width=True)
    model_choice
    return (model_choice,)


@app.cell
def _(mo):
    learning_mode = mo.ui.radio({"Value iteration / model-based learning": "planning",
                                 "RL · sampled transitions": "rl"},
                                value="Value iteration / model-based learning", label="2 · Learning", inline=True)
    approximate_value = mo.ui.switch(value=False, label="3 · Approximate Q instead of a table")
    policy_choice = mo.ui.dropdown({"Greedy argmax of action values": "greedy",
                                    "Fit a parameterized policy to greedy actions": "fit",
                                    "Policy iteration · evaluate, then improve": "iteration"},
                                   value="Greedy argmax of action values", label="4 · Policy", full_width=True)
    mo.vstack([learning_mode, approximate_value, policy_choice])
    return approximate_value, policy_choice, learning_mode


@app.cell
def _(approximate_value, policy_choice, learning_mode, mo, model_choice):
    _policy, _approx, _rl = policy_choice.value, approximate_value.value, learning_mode.value == "rl"
    _method = ("Sampled policy iteration" if _rl else ("Approximate policy iteration" if _approx else "Classical policy iteration")) if _policy == "iteration" else (
        "ε-greedy Q-learning" if _rl else "Q-value iteration")
    _description = {
        "greedy": "Actions maximize Q. For a state value V, the action scores are r(s,a) + γ E[V(s′)]. RL explores with ε-greedy actions.",
        "fit": "Learn Q, then fit a separate feature-based classifier to its greedy action labels. Rollouts use the fitted policy; the fit can lose performance even with accurate values.",
        "iteration": "Hold the policy fixed while evaluating it, then replace it with greedy actions. Tabular planning solves policy evaluation exactly; RL and approximated values use partial evaluation.",
    }[_policy]
    _basis = ("For this MDP, groups of state IDs share parameters; fewer groups force aliasing." if model_choice.value == "mdp" else
              "The approximator uses normalized radial basis functions, with separate output weights for each action. "
              "Full backups fit local weighted averages; RL trains the weights with sampled TD updates.")
    mo.callout(mo.md(f"**{_method} · {'approximated' if _approx else 'tabular'} Q**  \n{_description}  \n"
                     + (_basis if _approx else "")), kind="info")
    return


@app.cell
def _(mo, model_choice, models):
    model_defaults = models.DEFAULTS[model_choice.value].copy()
    _labels = {
        "states": "Number of states", "actions": "Number of actions", "angles": "Angle grid points",
        "velocities": "Velocity grid points", "concentration": "Transition concentration (0–1)",
        "seed": "Model random seed", "torque_limit": "Maximum torque", "dt": "Decision timestep",
        "damping": "Damping", "omega_limit": "Velocity grid limit", "speed_limit": "Section-speed grid limit",
        "slope": "Downhill slope (rad)", "alpha_min": "Minimum α (rad)", "alpha_max": "Maximum α (rad)",
    }
    model_controls = mo.ui.dictionary({
        _key: mo.ui.number(value=_value, step=1 if isinstance(_value, int) else 0.01, label=_labels[_key])
        for _key, _value in model_defaults.items()
    }).form(submit_button_label="Apply model parameters and rebuild dynamics")
    mo.accordion({"Model parameters and grid": mo.vstack([
        model_controls, mo.md("Pendulum angle is measured from upright (θ = 0). "
                              "The IPW uses the assignment’s fixed ankle-feedback capture region. "
                              "Changing its slope recomputes that region and can take longer."),
    ])})
    return model_controls, model_defaults


@app.cell
def _(mo, model_choice, model_controls, model_defaults, models):
    model_error = None
    model = None
    try:
        with mo.status.spinner(title="Building the model’s transition matrix…"):
            model = models.build_model(model_choice.value, model_controls.value or model_defaults)
    except (ValueError, TypeError) as _error:
        model_error = str(_error)
    mo.stop(model_error is not None, mo.callout(mo.md(f"Model settings: {model_error}"), kind="danger"))
    mo.md(f"**{model.shape[0]:,} states × {model.shape[1]:,} actions** · "
          f"{model.matrix.nnz:,} nonzero transition weights. Reward edits reuse this matrix.")
    return model, model_error


@app.cell
def _(json, mo, model_choice, models):
    _extra = {"mdp": {},
              "overdamped": {"torque_cost": 0.05},
              "pendulum": {"torque_cost": 0.05, "speed_cost": 0.05}, "walker": {}}[model_choice.value]
    extra_parameters = mo.ui.code_editor(value=json.dumps(_extra, indent=2), language="json", min_height=90,
                                         label="Reward parameters (JSON dictionary)").form(submit_button_label="Apply reward parameters")
    reward_editor = mo.ui.code_editor(value=models.REWARDS[model_choice.value], language="python", min_height=150,
                                      label="reward(state, action, params)").form(submit_button_label="Apply reward and restart learning")
    initial_editor = mo.ui.code_editor(value=models.INITIAL_SOURCE, language="python", min_height=120,
                                       label="initial_q(state, action, params)").form(submit_button_label="Apply initial Q and restart learning")
    return extra_parameters, initial_editor, reward_editor


@app.cell
def _(mo, model_choice):
    training_defaults = {
        "discount": 0.95, "iterations": 200, "samples": 256, "epsilon": 0.3,
        "learning_rate": 0.15, "features": 4 if model_choice.value == "mdp" else 16, "policy_features": 4 if model_choice.value == "mdp" else 16, "evaluation_sweeps": 20,
        "episode_length": 32, "seed": 0,
    }
    training_controls = mo.ui.dictionary({
        "discount": mo.ui.slider(0, 0.995, step=0.005, value=0.95, label="Discount γ", show_value=True),
        "iterations": mo.ui.number(0, 2000, step=10, value=200, label="Learning iterations"),
        "samples": mo.ui.number(1, 4096, step=64, value=256, label="RL samples per iteration"),
        "epsilon": mo.ui.slider(0, 1, step=0.05, value=0.3, label="RL exploration ε", show_value=True),
        "learning_rate": mo.ui.slider(0.01, 1, step=0.01, value=0.15, label="Critic learning rate", show_value=True),
        "features": mo.ui.slider(2, 64, step=1, value=training_defaults["features"], label="Value approximator features / MDP groups", show_value=True),
        "policy_features": mo.ui.slider(2, 64, step=1, value=training_defaults["policy_features"], label="Fitted policy features / MDP groups", show_value=True),
        "evaluation_sweeps": mo.ui.number(1, 500, step=1, value=20, label="Approximate policy evaluation sweeps (model-based)"),
        "episode_length": mo.ui.number(1, 1000, step=1, value=32, label="RL episode step limit"),
        "seed": mo.ui.number(0, 1000000, step=1, value=0, label="Learning random seed"),
    }).form(submit_button_label="Apply learning settings and restart")
    mo.accordion({"Learning settings": mo.vstack([
        training_controls,
        mo.md("RL settings apply to sampled learning. Feature counts apply when the corresponding "
              "approximator or fitted policy is enabled. Policy-evaluation sweeps apply only to "
              "model-based policy iteration with approximated values."),
    ])})
    return training_controls, training_defaults


@app.cell
def _(extra_parameters, json, mo, model, training_controls, training_defaults):
    parameter_error = None
    reward_parameters = {}
    try:
        reward_parameters = json.loads(extra_parameters.value) if extra_parameters.value else {}
        if not isinstance(reward_parameters, dict):
            raise ValueError("Use a JSON object, for example {\"torque_cost\": 0.05}.")
        _reserved = (set(model.params) - {"target_state", "torque_cost", "speed_cost"}) | {"discount"}
        if _reserved & set(reward_parameters):
            raise ValueError("Change model parameters in Model settings, and discount in Learning settings.")
    except (ValueError, TypeError) as _error:
        parameter_error = str(_error)
    _discount = (training_controls.value or training_defaults)["discount"]
    available_params = model.params | reward_parameters | {"discount": _discount}
    parameters_view = mo.ui.table([{"Parameter": k, "Value": str(v)} for k, v in available_params.items()],
                                  selection=None, pagination=False, show_column_summaries=False)
    return available_params, parameter_error, parameters_view, reward_parameters


@app.cell
def _(extra_parameters, initial_editor, mo, model, parameters_view, reward_editor):
    _state = ("`state[..., 0]` is θ and `state[..., 1]` is ω." if model.kind == "pendulum" else
              {"mdp": "`state` is the current numeric state ID.", "overdamped": "`state` is the current angle θ.",
               "walker": "`state` is the current scalar section speed; zero is upright standing."}[model.kind])
    mo.accordion({"Reward design, available parameters, and initial values": mo.vstack([
        mo.md(f"{_state} `action` is the current action. Inputs are NumPy arrays over the state–action grid; "
              "return a scalar or a broadcastable array. `np` is available. `params` is an ordinary dictionary. "
              "Rewards use the current state; failure has zero future value."),
        mo.md("**Parameters available to reward and initialization**"), parameters_view,
        extra_parameters, reward_editor, initial_editor,
    ])})
    return


@app.cell
def _(algorithms, approximate_value, policy_choice, initial_editor, learning_mode, mo, model,
      models, parameter_error, reward_editor, reward_parameters, training_controls, training_defaults):
    experiment = None
    experiment_error = parameter_error
    if experiment_error is None:
        try:
            settings = algorithms.Settings(
                method=learning_mode.value, approximation=approximate_value.value, policy=policy_choice.value,
                **(training_controls.value or training_defaults),
            )
            _reward, _initial = models.prepare_tables(model,
                                                       models.REWARDS[model.kind] if reward_editor.value is None else reward_editor.value,
                                                       models.INITIAL_SOURCE if initial_editor.value is None else initial_editor.value,
                                                       params=reward_parameters, discount=settings.discount)
            with mo.status.spinner(title=f"Running {settings.label}…"):
                experiment = algorithms.train(model, _reward, _initial, settings)
        except Exception as _error:
            experiment_error = f"{type(_error).__name__}: {_error}"
    mo.stop(experiment_error is not None, mo.callout(mo.md(f"Experiment: {experiment_error}"), kind="danger"))
    return experiment, experiment_error, settings


@app.cell
def _(experiment, mo):
    mo.stop(experiment is None)
    iteration = mo.ui.slider(steps=[s.iteration for s in experiment.snapshots] if len(experiment.snapshots) > 1 else [0, 1],
                             value=experiment.final.iteration, disabled=len(experiment.snapshots) == 1,
                             label="Learning snapshot (final by default)", show_value=True, full_width=True, debounce=True)
    iteration
    return (iteration,)


@app.cell
def _(experiment, iteration, mo, model):
    mo.stop(experiment is None)
    snapshot = next(s for s in experiment.snapshots if s.iteration == iteration.value)
    _work = (f"{snapshot.iteration * experiment.settings.samples:,} sampled transitions" if experiment.settings.method == "rl" else
             f"{snapshot.iteration:,} policy improvements (each includes evaluation)" if experiment.settings.policy == "iteration" else
             f"{snapshot.iteration:,} full matrix sweeps")
    _coverage = (f" · {(snapshot.visits > 0).mean():.1%} of state–action pairs visited" if experiment.settings.method == "rl" else "")
    _approx = (f"{experiment.critic_parameters:,} critic weights / {model.shape[0] * model.shape[1]:,} Q entries")
    _fit = (f"Greedy-label agreement: {(snapshot.policy.argmax(axis=1) == snapshot.q.argmax(axis=1)).mean():.1%}. "
            if experiment.settings.policy == "fit" else "")
    mo.md(f"**{experiment.settings.label} · iteration {snapshot.iteration} / {experiment.final.iteration}**  \n"
          f"{_work}{_coverage} · {_approx} · {experiment.policy_parameters:,} policy weights.  \n"
          f"{_fit}Run time: {experiment.seconds[-1]:.2f} s. A small change in Q need not mean a small Bellman residual, "
          "especially with limited features or unvisited states.")
    return (snapshot,)


@app.cell
def _(mo, model):
    velocity_slice = mo.ui.slider(0, len(model.axes[1]) - 1 if len(model.axes) > 1 else 1,
                                  value=len(model.axes[1]) // 2 if len(model.axes) > 1 else 0,
                                  step=1, label="Velocity slice for state–action heatmaps", show_value=True, debounce=True)
    velocity_slice if len(model.axes) > 1 else mo.md("")
    return (velocity_slice,)


@app.cell
def _(experiment, mo, model, snapshot, velocity_slice, views):
    mo.stop(experiment is None)
    views.learning_figure(model, experiment, snapshot, int(velocity_slice.value))
    return


@app.cell
def _(algorithms, experiment, mo, model, np, snapshot, velocity_slice, views):
    mo.stop(experiment is None)
    evaluated_values = algorithms.evaluate_policy(model, experiment.reward, snapshot.policy, experiment.settings.discount)
    mo.accordion({"Reward, policy probabilities, exploration coverage, and value accuracy": mo.vstack([
        views.diagnostics_figure(model, experiment, snapshot, int(velocity_slice.value)),
        mo.md(f"Independent evaluation on this grid: mean policy return **{np.mean(evaluated_values):.4f}** "
              f"from uniformly distributed starts. Maximum gap between max Q and this policy’s return: "
              f"**{np.max(np.abs(evaluated_values - snapshot.value)):.4f}**.  \n"
              "Interpolation weights define the sampled gridded MDP. Physical rollouts below also expose discretization errors. "
              "Approximation can leave an error floor; off-policy function approximation can also diverge."),
    ])})
    return (evaluated_values,)


@app.cell
def _(mo, model):
    _start = {"mdp": [0], "overdamped": [0.5], "pendulum": [0.5, 0.0],
              "walker": [0.8 * model.axes[0][-1]]}[model.kind]
    rollout_defaults = {"state_0": _start[0], "steps": 15 if model.kind == "walker" else 100, "seed": 0}
    _fields = {"state_0": mo.ui.number(float(model.axes[0][0]), float(model.axes[0][-1]),
                                       step=1 if model.kind == "mdp" else 0.05,
                                       value=_start[0], label=f"Initial {model.state_labels[0]}"),
               "steps": mo.ui.number(1, 500, step=1, value=rollout_defaults["steps"], label="Rollout decisions"),
               "seed": mo.ui.number(0, 1000000, step=1, value=0, label="Rollout seed")}
    if len(model.axes) > 1:
        _fields["state_1"] = mo.ui.number(float(model.axes[1][0]), float(model.axes[1][-1]), step=0.05, value=0.0,
                                          label=f"Initial {model.state_labels[1]}")
        rollout_defaults["state_1"] = 0.0
    rollout_controls = mo.ui.dictionary(_fields).form(submit_button_label="Update rollout")
    mo.accordion({"Rollout initial state, duration, and random seed": rollout_controls})
    return rollout_controls, rollout_defaults


@app.cell
def _(experiment, mo, model, rollout_controls, rollout_defaults, snapshot, views):
    mo.stop(experiment is None)
    _controls = rollout_controls.value or rollout_defaults
    _start = [_controls["state_0"]] + ([_controls["state_1"]] if len(model.axes) > 1 else [])
    policy_trace = views.physical_rollout(model, snapshot, _start, fitted=experiment.settings.policy == "fit",
                                         steps=int(_controls["steps"]), seed=int(_controls["seed"]))
    _label = f"{experiment.settings.label} · iteration {snapshot.iteration} · {'fitted policy' if experiment.settings.policy == 'fit' else 'greedy policy'}"
    mo.vstack([
        mo.md(f"### Selected-policy rollout\n\n{policy_trace['outcome']}. "
              "The animation and action trace use the learning snapshot selected above."),
        mo.iframe(views.animation_html(model, policy_trace, _label), height="530px" if model.kind == "walker" else "420px"),
        views.rollout_figure(model, policy_trace),
    ])
    return (policy_trace,)


if __name__ == "__main__":
    app.run()
