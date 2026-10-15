"""Linked plots and physical policy rollouts for the learning lab."""

import json

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from . import learning_algorithms as algorithms
from . import pendulum_learning as pendulum
from . import walker_rollout as walker
from . import slip


def _q_map(ax, model, table, title, omega_index=0, **kwargs):
    values = table if len(model.axes) == 1 else table.reshape(
        len(model.axes[0]), len(model.axes[1]), -1)[:, omega_index, :]
    image = ax.pcolormesh(model.actions, model.axes[0], values, shading="nearest", **kwargs)
    ax.set(xlabel=model.action_label, ylabel=model.state_labels[0], title=title)
    if len(model.axes) > 1:
        ax.set_title(f"{title} · ω = {model.axes[1][omega_index]:.2f}")
    return image


def _state_map(fig, ax, model, values, title, **kwargs):
    if len(model.axes) == 1:
        ax.plot(model.axes[0], values, color="#2563a4", linewidth=2)
        ax.set(xlabel=model.state_labels[0], title=title)
        ax.grid(alpha=0.2)
    else:
        image = ax.pcolormesh(*model.axes, values.reshape(len(model.axes[0]), -1).T,
                              shading="nearest", **kwargs)
        ax.set(xlabel=model.state_labels[0], ylabel=model.state_labels[1], title=title)
        fig.colorbar(image, ax=ax)


def learning_figure(model, experiment, snapshot, omega_index=0):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    # Fixed across this run; no rescaling as the learning slider moves.
    low = min(float(s.q.min()) for s in experiment.snapshots)
    high = max(float(s.q.max()) for s in experiment.snapshots)
    if high <= low:
        high = low + 1
    image = _q_map(axes[0, 0], model, snapshot.q, "Q(state, action)", omega_index,
                   cmap="viridis", vmin=low, vmax=high, rasterized=True)
    fig.colorbar(image, ax=axes[0, 0])
    _state_map(fig, axes[0, 1], model, snapshot.value, "Value estimate · max Q",
               cmap="viridis", vmin=low, vmax=high)
    if len(model.axes) == 1:
        padding = 0.04 * (high - low)
        axes[0, 1].set_ylim(low - padding, high + padding)
    _state_map(fig, axes[1, 0], model, snapshot.policy @ model.actions,
               "Fitted policy action" if experiment.settings.policy == "fit" else "Greedy action",
               cmap="coolwarm", vmin=model.actions.min(), vmax=model.actions.max())
    if len(model.axes) == 1:
        axes[1, 0].set_ylim(model.actions.min() - 0.05 * np.ptp(model.actions),
                           model.actions.max() + 0.05 * np.ptp(model.actions))
    ax = axes[1, 1]
    k = np.arange(len(experiment.residuals))
    ax.semilogy(k, np.maximum(experiment.residuals, 1e-12), label="Bellman optimality residual")
    ax.semilogy(k[1:], np.maximum(experiment.changes[1:], 1e-12), label="Change in Q", alpha=0.7)
    ax.axvline(snapshot.iteration, color="#d77919", linewidth=2)
    xlabel = ("Sample batches" if experiment.settings.method == "rl" else
              "Policy improvements" if experiment.settings.policy == "iteration" else "Full sweeps")
    ax.set(xlabel=xlabel,
           ylabel="Maximum absolute error", title="Learning progress · smaller is better")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    plt.close(fig)
    return fig


def diagnostics_figure(model, experiment, snapshot, omega_index=0):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), layout="constrained")
    for ax, values, title in zip(axes[:2], (experiment.reward, snapshot.policy),
                                  ("Expected transition reward", "Policy action probabilities")):
        image = _q_map(ax, model, values, title, omega_index, cmap="viridis", rasterized=True)
        fig.colorbar(image, ax=ax)
    if experiment.settings.method == "rl":
        image = _q_map(axes[2], model, np.log1p(snapshot.visits), "log(1 + sample visits)", omega_index,
                       cmap="magma", vmin=0, rasterized=True)
        fig.colorbar(image, ax=axes[2])
    else:
        _state_map(fig, axes[2], model, np.asarray(model.matrix[:, model.terminal].toarray()).reshape(model.shape).mean(axis=1),
                   "Mean terminal probability", vmin=0, vmax=1)
    plt.close(fig)
    return fig


def feature_figure(model, snapshot, action_index, omega_index=0):
    """Plot explicit features and their weighted contributions for one action."""
    active = [("Q", snapshot.feature_basis, snapshot.q_weights),
              ("Policy score", snapshot.policy_feature_basis, snapshot.policy_weights)]
    active = [item for item in active if item[2] is not None]
    fig, axes = plt.subplots(len(active), 2, figsize=(12, 3 * len(active)), squeeze=False, layout="constrained")
    points = model.states if len(model.axes) == 1 else np.column_stack([
        model.axes[0], np.full(len(model.axes[0]), model.axes[1][omega_index])])
    for (left, right), (title, basis, weights) in zip(axes, active):
        phi = basis.evaluate(points)
        names = basis.labels
        contributions = phi * weights[:, action_index]
        for i, name in enumerate(names):
            left.plot(points[:, 0], phi[:, i], label=name, alpha=.8)
            right.plot(points[:, 0], contributions[:, i], label=name, alpha=.7)
        right.plot(points[:, 0], contributions.sum(axis=1), color="black", lw=2, label="Sum")
        left.set_title(f"{title} features")
        right.set_title(f"Weighted contributions · {model.action_label} = {model.actions[action_index]:.3g}")
        for ax in (left, right):
            ax.set_xlabel(model.state_labels[0])
            ax.grid(alpha=.2)
            if names or ax is right:
                ax.legend(fontsize=8, ncol=min(4, max(1, len(names))))
    plt.close(fig)
    return fig


def policy_at(model, policy, state):
    """Interpolate the selected snapshot's distribution at a physical state."""
    state = np.atleast_1d(state)
    if model.kind == "walker":
        grid = model.context["grid"]
        nodes = np.flatnonzero(~grid.source_goal)
        upper = int(np.clip(np.searchsorted(grid.states[nodes], state[0]), 0, len(nodes) - 1))
        lo, hi = nodes[max(0, upper - 1)], nodes[upper]
        weight = 0 if lo == hi else np.clip((state[0] - grid.states[lo]) / (grid.states[hi] - grid.states[lo]), 0, 1)
        result = (1 - weight) * policy[lo] + weight * policy[hi]
    elif model.kind == "mdp":
        result = policy[int(state[0])]
    elif model.kind == "slip":
        nodes = model.axes[0]
        hi = int(np.clip(np.searchsorted(nodes, state[0]), 1, len(nodes) - 1))
        lo = hi - 1
        weight = np.clip((state[0] - nodes[lo]) / (nodes[hi] - nodes[lo]), 0, 1)
        result = (1 - weight) * policy[lo] + weight * policy[hi]
    else:
        theta = model.axes[0]
        ti = float((state[0] - theta[0]) % (2 * np.pi) * len(theta) / (2 * np.pi))
        lo, weight = int(np.floor(ti)) % len(theta), ti - np.floor(ti)
        hi = (lo + 1) % len(theta)
        if len(model.axes) == 1:
            result = (1 - weight) * policy[lo] + weight * policy[hi]
        else:
            omega = model.axes[1]
            wi = float(np.clip((state[1] - omega[0]) / (omega[1] - omega[0]), 0, len(omega) - 1))
            w0 = min(int(wi), len(omega) - 2)
            ww = wi - w0
            grid = policy.reshape(len(theta), len(omega), -1)
            result = ((1 - weight) * ((1 - ww) * grid[lo, w0] + ww * grid[lo, w0 + 1])
                      + weight * ((1 - ww) * grid[hi, w0] + ww * grid[hi, w0 + 1]))
    return result / result.sum()


def physical_rollout(model, snapshot, start, *, fitted=False, steps=150, seed=0):
    """Continuous physics for pendulums, IPW and SLIP; discrete transitions for MDP.

    Tabular policies interpolate Q before argmax. Approximators evaluate their
    saved feature definitions at the physical state. Fixed seeds make MDP comparisons fair.
    """
    def choose(state):
        if fitted:
            scores = snapshot.policy_feature_basis.evaluate(np.atleast_2d(state)) @ snapshot.policy_weights
            return int(np.argmax(scores))
        if snapshot.q_weights is not None:
            scores = snapshot.feature_basis.evaluate(np.atleast_2d(state)) @ snapshot.q_weights
            return int(np.argmax(scores))
        # Interpolation is linear; policy_at also works for scores after adding
        # a common positive offset, without changing the maximizing action.
        return int(np.argmax(policy_at(model, positive_scores, state)))

    positive_scores = snapshot.q - min(0, float(snapshot.q.min())) + 1
    if model.kind == "slip":
        return slip.rollout(float(start[0]), lambda state: model.actions[choose([state])],
                            model.params, steps=steps)
    if model.kind == "walker":
        trace = walker.rollout_policy(model.context["grid"], snapshot.q, float(start[0]),
                                      model.context["region"], max_decisions=min(steps, 30),
                                      action_selector=lambda speed: choose([speed]))
        return {"walker": trace, "states": walker.section_speeds(trace)[:, None],
                "actions": np.array([d["alpha"] for d in trace.decisions]), "outcome": trace.outcome}
    if model.kind == "mdp":
        indices, action_indices = algorithms.rollout(model, snapshot.policy, int(start[0]), steps=steps, seed=seed)
        return {"states": model.states[indices], "actions": model.actions[action_indices],
                "outcome": "Step limit reached"}
    state = np.array(start, dtype=float)
    states, actions = [state.copy()], []
    outcome = "Step limit reached"
    for _ in range(steps):
        action = float(model.actions[choose(state)])
        state = pendulum.step(model.context["plant"], state, action)
        actions.append(action)
        states.append(state.copy())
        if len(state) > 1 and abs(state[1]) > model.params["omega_limit"]:
            outcome = "Left the velocity grid"
            break
    return {"states": np.array(states), "actions": np.array(actions), "outcome": outcome}


def rollout_figure(model, trace):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3), layout="constrained")
    for i, label in enumerate(model.state_labels):
        axes[0].plot(trace["states"][:, i], label=label)
    axes[0].set(xlabel="Decision", title="States under the selected policy")
    axes[0].legend()
    axes[1].step(np.arange(len(trace["actions"])), trace["actions"], where="post")
    axes[1].set(xlabel="Decision", ylabel=model.action_label, title="Actions taken")
    for ax in axes:
        ax.grid(alpha=0.2)
    plt.close(fig)
    return fig


def animation_html(model, trace, label):
    if model.kind == "walker":
        return walker.animation_html(trace["walker"], model.context["grid"].params, policy_label=label)
    if model.kind == "slip":
        payload = json.dumps({"frames": trace["trace"].tolist(), "label": label,
                              "outcome": trace["outcome"], "height": model.params["total_energy"] / (model.params["mass"] * model.params["gravity"])},
                             allow_nan=False).replace("<", "\\u003c")
        return _SLIP_PLAYER.replace("__DATA__", payload)
    payload = json.dumps({"kind": model.kind, "states": trace["states"].tolist(),
                          "actions": trace["actions"].tolist(), "dt": model.params.get("dt", 0.35),
                          "nodes": len(model.states), "label": label, "outcome": trace["outcome"]},
                         allow_nan=False).replace("<", "\\u003c")
    return _PLAYER.replace("__DATA__", payload)


def slip_map_figure(model):
    """Physical return map, with failures distinct from infeasible touchdown."""
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dedede")
    image = _q_map(ax, model, model.context["next_states"], "SLIP apex return map · next s",
                   cmap=cmap, vmin=0, vmax=1, rasterized=True)
    infeasible = np.ma.masked_where(model.context["status"] != slip.INFEASIBLE,
                                    np.ones(model.shape))
    ax.pcolormesh(model.actions, model.axes[0], infeasible, cmap=ListedColormap(["#30343b"]), shading="nearest", rasterized=True)
    fig.colorbar(image, ax=ax, label="Normalized next apex height")
    ax.set_title("SLIP apex return map · next s\nDark: foot starts below ground · gray: no next apex")
    plt.close(fig)
    return fig


_SLIP_PLAYER = '''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;font:14px system-ui;color:#263445}main{border:1px solid #dce3e9;border-radius:12px;overflow:hidden}
header{background:#edf3f8;padding:12px 16px;font-weight:600}canvas{width:100%;height:260px;display:block;background:#f8fbfd}
footer{display:flex;gap:16px;align-items:center;padding:14px}button{padding:6px 16px;border:1px solid #c6d2dc;background:white;border-radius:6px;cursor:pointer}
input{flex:1}#info{padding:0 16px 12px}</style></head><body><main><header id="label"></header>
<canvas id="scene" width="900" height="260" aria-label="SLIP physical rollout"></canvas>
<footer><button id="play">Pause</button><button id="replay">Replay</button><input id="seek" aria-label="Time" type="range" min="0" step="1"><span id="time"></span></footer><div id="info"></div></main>
<script>const data=__DATA__,canvas=document.getElementById('scene'),ctx=canvas.getContext('2d'),seek=document.getElementById('seek');
let k=0,playing=true,last=null,clock=0;seek.max=data.frames.length-1;document.getElementById('label').textContent=data.label;
const play=document.getElementById('play'),scale=190/Math.max(1.2,data.height),ground=230;
function draw(){const f=data.frames[k],x=v=>400+(v-f[1])*scale,y=v=>ground-v*scale;
ctx.clearRect(0,0,900,260);ctx.strokeStyle='#a9b9c5';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(0,ground);ctx.lineTo(900,ground);ctx.stroke();
ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillStyle='#768998';for(let v=Math.floor(f[1]-400/scale);v<f[1]+500/scale;v++){ctx.fillRect(x(v),ground,1,8);ctx.fillText(v+' m',x(v),ground+23)}
ctx.strokeStyle='#c3d2df';ctx.lineWidth=1;ctx.beginPath();for(let i=Math.max(0,k-150);i<=k;i++){const p=data.frames[i];if(i===Math.max(0,k-150))ctx.moveTo(x(p[1]),y(p[2]));else ctx.lineTo(x(p[1]),y(p[2]))}ctx.stroke();
const bx=x(f[1]),by=y(f[2]),fx=x(f[5]),fy=y(f[6]),dx=bx-fx,dy=by-fy,len=Math.max(1,Math.hypot(dx,dy));
ctx.strokeStyle=f[7]?'#246c9f':'#89a9bf';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(fx,fy);
for(let j=1;j<16;j++){const t=j/16,w=(j<3||j>13)?0:(j%2?5:-5);ctx.lineTo(fx+t*dx-w*dy/len,fy+t*dy+w*dx/len)}ctx.lineTo(bx,by);ctx.stroke();
ctx.fillStyle=f[7]?'#246c9f':'#89a9bf';ctx.beginPath();ctx.arc(fx,fy,4,0,2*Math.PI);ctx.fill();ctx.fillStyle='#d77919';ctx.beginPath();ctx.arc(bx,by,12,0,2*Math.PI);ctx.fill();
seek.value=k;document.getElementById('time').textContent=f[0].toFixed(2)+' s';document.getElementById('info').textContent=(f[7]?'Stance':'Flight')+' · forward speed '+f[3].toFixed(2)+' m/s · α '+(f[8]*180/Math.PI).toFixed(1)+'°'+(k===data.frames.length-1?' · '+data.outcome:'');}
play.onclick=()=>{playing=!playing;last=null;play.textContent=playing?'Pause':'Play'};
document.getElementById('replay').onclick=()=>{k=0;clock=0;last=null;playing=true;play.textContent='Pause';draw()};
seek.oninput=()=>{k=Number(seek.value);clock=data.frames[k][0];playing=false;play.textContent='Play';draw()};
function tick(t){if(playing){if(last!==null)clock+=(t-last)/1000;while(k<data.frames.length-1&&data.frames[k+1][0]<=clock)k++;if(k===data.frames.length-1){playing=false;play.textContent='Play'}draw()}last=t;requestAnimationFrame(tick)}draw();requestAnimationFrame(tick);
</script></body></html>'''


_PLAYER = '''<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;font:14px system-ui;color:#263445}main{border:1px solid #dce3e9;border-radius:12px;overflow:hidden}
header{background:#edf3f8;padding:12px 16px;font-weight:600}canvas{width:100%;height:260px;display:block;background:#f8fbfd}
footer{display:flex;gap:16px;align-items:center;padding:14px}button{padding:6px 16px;border:1px solid #c6d2dc;background:white;border-radius:6px;cursor:pointer}
input{flex:1}#info{padding:0 16px 12px}</style></head><body><main><header id="label"></header>
<canvas id="scene" width="900" height="260" aria-label="Selected policy rollout"></canvas>
<footer><button id="play">Pause</button><button id="replay">Replay</button><input id="seek" aria-label="Decision" type="range" min="0" step="1"><span id="step"></span></footer><div id="info"></div></main>
<script>const data=__DATA__,canvas=document.getElementById('scene'),ctx=canvas.getContext('2d'),seek=document.getElementById('seek');
let k=0,playing=true,last=0;seek.max=data.states.length-1;document.getElementById('label').textContent=data.label;
function draw(){ctx.clearRect(0,0,900,260);const s=data.states[k];ctx.lineWidth=5;
if(data.kind==='mdp'){for(let i=0;i<data.nodes;i++){let a=2*Math.PI*i/data.nodes,x=450+310*Math.cos(a),y=130+94*Math.sin(a);ctx.beginPath();ctx.arc(x,y,17,0,2*Math.PI);ctx.fillStyle=i===s[0]?'#d77919':'#dce3e9';ctx.fill();ctx.fillStyle='#263445';ctx.font='14px system-ui';ctx.textAlign='center';ctx.fillText(i,x,y+5);}}
else{let x=450+96*Math.sin(s[0]),y=135-96*Math.cos(s[0]);ctx.strokeStyle='#2a689a';ctx.beginPath();ctx.moveTo(450,135);ctx.lineTo(x,y);ctx.stroke();ctx.fillStyle='#d77919';ctx.beginPath();ctx.arc(x,y,13,0,2*Math.PI);ctx.fill();ctx.fillStyle='#263445';ctx.beginPath();ctx.arc(450,135,5,0,2*Math.PI);ctx.fill();}
seek.value=k;document.getElementById('step').textContent='Decision '+k;document.getElementById('info').textContent='State: '+s.map(v=>v.toFixed(3)).join(', ')+' · Action: '+(data.actions[k]===undefined?'—':data.actions[k].toFixed(3))+(k===data.states.length-1?' · '+data.outcome:'');}
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'Pause':'Play';last=0};
document.getElementById('replay').onclick=()=>{k=0;playing=true;last=0;document.getElementById('play').textContent='Pause';draw()};
seek.oninput=()=>{k=Number(seek.value);playing=false;document.getElementById('play').textContent='Play';draw()};
function tick(t){if(playing&&t-last>Math.max(40,data.dt*1000)){k=(k+1)%data.states.length;last=t;draw()}requestAnimationFrame(tick)}draw();requestAnimationFrame(tick);
</script></body></html>'''
