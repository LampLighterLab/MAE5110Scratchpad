"""Rollout settings and previews shared by the reward-design notebooks."""

import json

import marimo as mo
import matplotlib.pyplot as plt
import numpy as np


def initial_distribution_fields(kind):
    """UI fields for the collection form; changing these requires resubmitting it."""
    fields = {
        "initial_distribution": mo.ui.dropdown(
            options={"Uniform": "uniform", "Normal": "normal", "Single state": "fixed"},
            value="Uniform", label="Initial distribution", allow_select_none=False),
        "theta_bounds": mo.ui.range_slider(
            -np.pi, np.pi, step=0.01, value=[-np.pi, np.pi], show_value=True,
            full_width=True, label="Uniform: initial angle bounds (rad)"),
        "theta_mean": mo.ui.number(
            value=0.0, step=0.1, label="Normal mean / single-state angle (rad)"),
        "theta_std": mo.ui.number(
            0, 6, value=0.5, step=0.05, label="Normal: angle standard deviation (rad)"),
    }
    if kind == "inverted":
        fields.update({
            "omega_bounds": mo.ui.range_slider(
                -10, 10, step=0.05, value=[-1, 1], show_value=True,
                full_width=True, label="Uniform: initial velocity bounds"),
            "omega_mean": mo.ui.number(
                value=0.0, step=0.1, label="Normal mean / single-state velocity"),
            "omega_std": mo.ui.number(
                0, 10, value=0.5, step=0.05, label="Normal: velocity standard deviation"),
        })
    return fields


def rollout_preview(data):
    """All saved trajectories in state space; opacity reveals visitation density."""
    states = data["states"]
    inertial = states.shape[-1] == 2
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    for state in states:
        theta = (state[:, 0] + np.pi) % (2 * np.pi) - np.pi
        if inertial:
            broken = theta.copy()
            broken[np.flatnonzero(np.abs(np.diff(theta)) > np.pi) + 1] = np.nan
            axis.plot(broken, state[:, 1], color="grey", alpha=0.12, lw=0.8)
        else:
            axis.plot(np.sin(state[:, 0]), np.cos(state[:, 0]),
                      color="grey", alpha=0.12, lw=0.8)
    starts = states[:, 0]
    theta0 = (starts[:, 0] + np.pi) % (2 * np.pi) - np.pi
    if inertial:
        axis.scatter(theta0, starts[:, 1], color="#dc2626", marker="o", s=20,
                     alpha=0.75, zorder=4, label="Initial conditions (all episodes)")
        axis.set(xlabel="Angle θ (rad; 0 = upright)", ylabel="Angular velocity ω",
                 title=f"State space · all {len(states)} trajectories", xlim=(-np.pi, np.pi))
        goal = (0, 0)
    else:
        axis.scatter(np.sin(theta0), np.cos(theta0), color="#dc2626", marker="o",
                     s=20, alpha=0.75, zorder=4, label="Initial conditions (all episodes)")
        axis.set(xlabel="sin θ", ylabel="cos θ",
                 title=f"Circular state space · all {len(states)} trajectories")
        axis.set_aspect("equal")
        goal = (0, 1)
    axis.scatter(*goal, marker="*", color="black", s=100, zorder=5, label="Upright goal")
    axis.legend(fontsize=8)
    axis.axhline(0, color="#111827", lw=1, ls="--", alpha=0.5)
    axis.axvline(0, color="#111827", lw=1, ls="--", alpha=0.5)
    axis.grid(alpha=0.15)
    return figure


def pendulum_animation_html(data, count=12):
    """Canvas player for the first N physical pendulums, using saved angles exactly."""
    payload = json.dumps({"theta": data["states"][:count, :, 0].T.tolist(),
                          "dt": data["dt"]}, allow_nan=False).replace("<", "\\u003c")
    return _PENDULUM_PLAYER.replace("__DATA__", payload)


_PENDULUM_PLAYER = r'''<!doctype html>
<html><head><meta charset="utf-8"><style>
body{margin:0;font:14px system-ui;color:#263445}
main{border:1px solid #dce3e9;border-radius:10px;padding:12px}
h3{margin:0 0 8px;font-size:15px}
canvas{width:100%;max-width:700px;height:auto;aspect-ratio:7 / 3;display:block;margin:0 auto}
.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
button,select{padding:5px 10px}input{flex:1;min-width:120px}
p{margin:8px 0 0;font-size:12px;color:#536171}
</style></head><body><main><h3 id="title"></h3>
<canvas id="scene" width="700" height="300" aria-label="Overlaid physical pendulum simulations"></canvas>
<div class="controls"><button id="play">Play</button><button id="reset">Reset</button>
<input id="seek" type="range" min="0" step="1" value="0" aria-label="Simulation time">
<label>Speed <select id="speed"><option value="0.25">¼×</option><option value="0.5">½×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label>
<span id="time"></span></div>
<p>All pendulums share a pivot. Grey rods and bobs show the current states; red circles mark the initial positions. Upright is at the top. Starts paused at t = 0.</p>
</main><script>
const data=__DATA__, canvas=document.getElementById('scene'), ctx=canvas.getContext('2d');
const seek=document.getElementById('seek'), play=document.getElementById('play'), speed=document.getElementById('speed');
let k=0, playing=false, last=null, elapsed=0;
seek.max=data.theta.length-1;
document.getElementById('title').textContent='Physical pendulums · first '+data.theta[0].length+' episodes';
function bob(theta){return [350+115*Math.sin(theta),150-115*Math.cos(theta)]}
function draw(){
 ctx.clearRect(0,0,700,300);
 ctx.setLineDash([4,4]);ctx.strokeStyle='#d1d5db';ctx.lineWidth=1;
 ctx.beginPath();ctx.arc(350,150,115,0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);
 ctx.fillStyle='#263445';ctx.textAlign='center';ctx.font='12px system-ui';ctx.fillText('upright',350,18);
 for(const theta of data.theta[k]){
  const [x,y]=bob(theta);ctx.strokeStyle='rgba(100,100,100,0.2)';ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(350,150);ctx.lineTo(x,y);ctx.stroke();
  ctx.fillStyle='rgba(100,100,100,0.25)';ctx.beginPath();ctx.arc(x,y,6,0,2*Math.PI);ctx.fill();
 }
 for(const theta of data.theta[0]){
  const [x,y]=bob(theta);ctx.strokeStyle='rgba(220,38,38,0.7)';ctx.lineWidth=1.5;
  ctx.beginPath();ctx.arc(x,y,7,0,2*Math.PI);ctx.stroke();
 }
 ctx.fillStyle='#111827';ctx.beginPath();ctx.arc(350,150,4,0,2*Math.PI);ctx.fill();
 seek.value=k;document.getElementById('time').textContent='t = '+(k*data.dt).toFixed(2);
}
play.onclick=()=>{
 if(!playing && k===data.theta.length-1){k=0;elapsed=0;draw()}
 playing=!playing;last=null;play.textContent=playing?'Pause':'Play';
};
document.getElementById('reset').onclick=()=>{k=0;elapsed=0;playing=false;last=null;play.textContent='Play';draw()};
seek.oninput=()=>{k=Number(seek.value);elapsed=k*data.dt;playing=false;last=null;play.textContent='Play';draw()};
function tick(now){
 if(playing){
  if(last!==null)elapsed+=(now-last)/1000*Number(speed.value);
  k=Math.min(data.theta.length-1,Math.floor(elapsed/data.dt));
  if(k===data.theta.length-1){playing=false;play.textContent='Play'}
  draw();
 }
 last=now;requestAnimationFrame(tick);
}
draw();requestAnimationFrame(tick);
</script></body></html>'''
