"""Continuous walker rollouts and a small, self-contained animation player."""

from dataclasses import dataclass
import json

import numpy as np

from . import walker_stopping as stopping


@dataclass
class Rollout:
    # Columns: time, theta, omega, ankle torque, alpha, walking flag.
    trace: np.ndarray
    decisions: list
    outcome: str
    footstrikes: int
    capture_time: float | None


def choose_action(grid, q, speed):
    """Greedy lookup using the grid's interpolation, without rounding physics."""
    if q.shape != grid.shape or not np.all(np.isfinite(q)):
        raise ValueError("Q must be finite and match the state-action grid.")
    if not np.isfinite(speed) or not grid.states[0] <= speed <= grid.states[-1]:
        raise ValueError("Crossing speed is outside the policy grid.")
    nodes = np.flatnonzero(~grid.source_goal)
    if not nodes.size:
        raise ValueError("No walking states are available for a foot-placement decision.")
    upper = int(np.clip(np.searchsorted(grid.states[nodes], speed), 0, len(nodes) - 1))
    lower = max(0, upper - 1)
    lo, hi = nodes[lower], nodes[upper]
    if grid.interpolation == "nearest":
        node = lo if abs(speed - grid.states[lo]) <= abs(grid.states[hi] - speed) else hi
        values = q[node]
    else:
        weight = 0 if hi == lo else np.clip((speed - grid.states[lo]) / (grid.states[hi] - grid.states[lo]), 0, 1)
        values = (1 - weight) * q[lo] + weight * q[hi]
    return int(np.argmax(values))  # Same tie rule as Q iteration.


def rollout_policy(grid, q, initial_speed, region, *, max_decisions=15, max_time=20.0):
    """Apply the current greedy policy to the continuous hybrid model.

    Alpha is selected only at theta=0 and held through impact. Returned speeds
    are not snapped to grid nodes. After capture, the actual bounded balancer
    runs to rest; the animation does not replace this motion with a zero state.
    """
    if q.shape != grid.shape or not np.all(np.isfinite(q)):
        raise ValueError("Q must be finite and match the state-action grid.")
    if not np.isfinite(initial_speed) or not 0 <= initial_speed <= grid.states[-1]:
        raise ValueError("Initial speed must lie in the policy grid.")
    if not isinstance(max_decisions, (int, np.integer)) or max_decisions < 1:
        raise ValueError("max_decisions must be a positive integer.")
    if not np.isfinite(max_time) or max_time <= 0:
        raise ValueError("max_time must be finite and positive.")
    params = grid.params
    state = np.array([0.0, float(initial_speed)])
    captured = bool(stopping.capture_margin(region, state) >= 0)
    elapsed, footstrikes = 0.0, 0
    capture_time = None
    decisions, segments = [], []
    alpha = float(params["angle_of_attack"])
    outcome = "Decision limit reached"

    def append_trace(trace):
        segment = np.asarray(trace, dtype=float).copy()
        segment[:, 0] += elapsed
        segments.append(segment)

    for _ in range(max_decisions + 1):
        if captured:
            capture_time = elapsed
            balance_params = params | {"angle_of_attack": alpha,
                                       "balance_horizon": min(params["balance_horizon"], max_time - elapsed)}
            status, _, trace = stopping.simulate_balance(state, balance_params, record=True)
            append_trace(trace)
            outcome = {stopping.CAPTURE: "Standing (settled)", stopping.FAILURE: "Ankle balancing failed",
                       stopping.TIMEOUT: "Balancing time limit reached"}[int(status[0])]
            break
        if elapsed >= max_time:
            outcome = "Rollout time limit reached"
            break
        if len(decisions) == max_decisions:
            break
        if not grid.states[0] <= state[1] <= grid.states[-1]:
            outcome = "Left the policy speed grid"
            break
        action_index = choose_action(grid, q, state[1])
        alpha = float(grid.actions[action_index])
        decisions.append({"time": elapsed, "speed": float(state[1]), "alpha": alpha,
                          "action_index": action_index})
        transition_params = params | {"transition_horizon": min(params["transition_horizon"], max_time - elapsed)}
        transition = stopping.simulate_transitions(state[1], alpha, region, transition_params, record=True)
        append_trace(transition["trace"])
        elapsed += float(transition["elapsed"])
        footstrikes += int(transition["impacts"])
        state = transition["states"].copy()
        status = int(transition["status"])
        decisions[-1].update({"status": status, "next_speed":
            float(state[1]) if status == stopping.RETURN else
            0.0 if status == stopping.CAPTURE else np.nan})
        if status == stopping.CAPTURE:
            # Handle balancing at the top, including capture on the last step.
            captured = True
            continue
        if status == stopping.FAILURE:
            outcome = "Step failed (reversal or ground contact)"
            break
        if status == stopping.TIMEOUT:
            outcome = "Walking time limit reached"
            break
        if not grid.states[0] <= state[1] <= grid.states[-1]:
            outcome = "Left the policy speed grid"
            break
    return Rollout(np.concatenate(segments), decisions, outcome, footstrikes, capture_time)


def section_speeds(rollout):
    """Physical section speeds, folding successful ankle settling into state 0.

    Keep a returned endpoint even when the decision limit prevents another
    action. Failure, timeout, or unsuccessful balancing must not invent a stop.
    """
    if not rollout.decisions:
        initial = float(rollout.trace[0, 2])
        return np.array([initial, 0.0] if initial != 0 and rollout.outcome == "Standing (settled)" else [initial])
    speeds = [d["speed"] for d in rollout.decisions]
    last = rollout.decisions[-1]
    if last["status"] == stopping.RETURN or rollout.outcome == "Standing (settled)":
        speeds.append(last["next_speed"])
    return np.asarray(speeds)


def world_foot_positions(trace, length):
    """Move the stance foot at impact while keeping the world hip continuous."""
    trace = np.asarray(trace)
    feet = np.zeros((len(trace), 2))
    counts = np.zeros(len(trace), dtype=int)
    for i in range(1, len(trace)):
        feet[i] = feet[i - 1]
        counts[i] = counts[i - 1]
        # A reset is recorded twice at its event time; no interpolation across it.
        if abs(trace[i, 0] - trace[i - 1, 0]) < 1e-10 and trace[i, 1] - trace[i - 1, 1] < -0.1:
            before, after = trace[i - 1, 1], trace[i, 1]
            feet[i] += length * np.array([np.sin(before) - np.sin(after), np.cos(before) - np.cos(after)])
            counts[i] += 1
    return feet, counts


def animation_data(rollout, params, *, fps=30):
    """Sample physical time at a fixed frame rate, keeping event resets sharp."""
    trace = rollout.trace
    feet, counts = world_foot_positions(trace, params["length"])
    times = np.unique(np.r_[np.arange(0, trace[-1, 0], 1 / fps), trace[-1, 0]])
    indices = np.maximum(0, np.searchsorted(trace[:, 0], times, side="right") - 1)
    frames = np.column_stack([times, trace[indices, 1:], feet[indices], counts[indices]])
    return {
        "frames": frames.tolist(), "length": params["length"], "slope": params["incline"],
        "torque_scale": params["mass"] * params["gravity"] * params["length"],
        "outcome": rollout.outcome,
    }


def animation_html(rollout, params, *, policy_label="Current greedy policy", autoplay=True):
    """Mount a fresh player that visibly replays the newly selected policy.

    Include the learning stage in the payload, even if two Q tables select the
    same actions. This also refreshes the iframe when its physical path agrees.
    Automatic playback repeats after a short hold on the rollout's outcome.
    """
    data = animation_data(rollout, params) | {"policy_label": policy_label, "autoplay": bool(autoplay)}
    payload = json.dumps(data, allow_nan=False).replace("<", "\\u003c")
    return _PLAYER.replace("__ROLLOUT_DATA__", payload)


_PLAYER = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box}body{margin:0;font:14px system-ui,sans-serif;color:#263445;background:#fff}
.player{max-width:1000px;border:1px solid #dce3e9;border-radius:12px;overflow:hidden}
.policy{padding:12px 16px;background:#edf3f8;font-weight:600}
canvas{display:block;width:100%;height:340px;background:#f8fbfd}
.controls{display:flex;align-items:center;gap:12px;padding:14px;flex-wrap:wrap}
button,select{font:inherit;background:#fff;border:1px solid #c6d2dc;border-radius:6px;padding:6px 12px;color:#263445;cursor:pointer}
button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid #3979b7}
button:disabled{opacity:.5;cursor:default}.seek{flex:1;min-width:140px;accent-color:#3979b7}
.readout{padding:0 14px 14px;display:flex;gap:20px;flex-wrap:wrap;font-variant-numeric:tabular-nums}
.badge{font-weight:650;color:#237962}.time{min-width:105px;font-variant-numeric:tabular-nums}
</style></head><body>
<div class="player">
<div id="policy" class="policy"></div>
<canvas id="scene" role="img" aria-label="Walker following the current greedy policy on an inclined ground"></canvas>
<div class="controls">
<button id="play" aria-label="Play rollout">Play</button>
<button id="restart" aria-label="Replay rollout from the beginning">Replay</button>
<input id="seek" class="seek" type="range" min="0" step="0.001" value="0" aria-label="Rollout time">
<span id="clock" class="time"></span>
<label>Speed <select id="speed" aria-label="Playback speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label>
</div>
<div class="readout"><span id="mode" class="badge"></span><span id="state"></span><span id="action"></span><span id="torque"></span><span id="steps"></span></div>
</div>
<script>
const data = __ROLLOUT_DATA__;
const frames=data.frames, duration=frames[frames.length-1][0], L=data.length;
const canvas=document.getElementById('scene'), ctx=canvas.getContext('2d');
const play=document.getElementById('play'), seek=document.getElementById('seek');
const speed=document.getElementById('speed');
let t=0, running=false, previous=null, endHold=0;
seek.max=duration;seek.disabled=duration===0;play.disabled=duration===0;
document.getElementById('policy').textContent=data.policy_label;
function setPlaying(value){running=value;previous=null;endHold=0;play.textContent=value?'Pause':'Play';play.setAttribute('aria-label',value?'Pause rollout':'Play rollout');}
function frameAt(time){let lo=0,hi=frames.length;while(lo<hi){let mid=(lo+hi)>>1;if(frames[mid][0]<=time)lo=mid+1;else hi=mid;}return frames[Math.max(0,lo-1)];}
function draw(){
 const f=frameAt(t), [time,theta,omega,tau,alpha,walking,fx,fy,steps]=f;
 const W=canvas.clientWidth||900,H=340,dpr=window.devicePixelRatio||1;
 if(canvas.width!==Math.round(W*dpr)||canvas.height!==Math.round(H*dpr)){canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
 const hx=fx+L*Math.sin(theta),hy=fy+L*Math.cos(theta),scale=Math.min(W/(4.5*L),H/(2.3*L));
 const cx=hx,cy=-Math.tan(data.slope)*cx+0.55*L;
 const X=x=>W/2+(x-cx)*scale,Y=y=>H/2-(y-cy)*scale;
 const xmin=cx-W/(2*scale),xmax=cx+W/(2*scale),ground=x=>-Math.tan(data.slope)*x;
 ctx.fillStyle='#eee7dc';ctx.beginPath();ctx.moveTo(0,Y(ground(xmin)));ctx.lineTo(W,Y(ground(xmax)));ctx.lineTo(W,H);ctx.lineTo(0,H);ctx.fill();
 ctx.strokeStyle='#9b8874';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(0,Y(ground(xmin)));ctx.lineTo(W,Y(ground(xmax)));ctx.stroke();
 ctx.font='12px system-ui';ctx.textAlign='center';ctx.fillStyle='#7a7068';
 for(let x=Math.ceil(xmin);x<xmax;x++){ctx.beginPath();ctx.moveTo(X(x),Y(ground(x)));ctx.lineTo(X(x),Y(ground(x))+7);ctx.stroke();ctx.fillText(x+' m',X(x),Y(ground(x))+24);}
 function line(x1,y1,x2,y2,color,width,dash=[]){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(X(x1),Y(y1));ctx.lineTo(X(x2),Y(y2));ctx.stroke();ctx.setLineDash([]);}
 function dot(x,y,r,color){ctx.fillStyle=color;ctx.beginPath();ctx.arc(X(x),Y(y),r,0,Math.PI*2);ctx.fill();}
 line(fx,fy,fx,fy+1.25*L,'#c3ccd3',1,[4,5]);
 if(walking&&theta>=0){const sx=hx-L*Math.sin(theta-2*alpha),sy=hy-L*Math.cos(theta-2*alpha);line(hx,hy,sx,sy,'#df8a25',4,[7,5]);dot(sx,sy,5,'#df8a25');}
 const color=walking?'#3979b7':'#24937d';line(fx,fy,hx,hy,color,7);dot(fx,fy,6,'#263445');dot(hx,hy,12,color);
 ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.beginPath();ctx.arc(X(hx),Y(hy),12,0,Math.PI*2);ctx.stroke();
 if(!walking&&Math.abs(tau)>1e-5){ctx.strokeStyle='#24937d';ctx.lineWidth=3;const a0=tau>0?Math.PI*.1:Math.PI*.9,a1=tau>0?Math.PI*.9:Math.PI*.1;ctx.beginPath();ctx.arc(X(fx),Y(fy),22,a0,a1,tau<0);ctx.stroke();}
 ctx.textAlign='left';ctx.fillStyle='#526575';ctx.font='13px system-ui';ctx.fillText('Stance leg',16,23);ctx.fillStyle='#df8a25';ctx.fillText('Swing leg (held clear before the section and during balancing)',16,44);
 document.getElementById('clock').textContent=t.toFixed(2)+' / '+duration.toFixed(2)+' s';
 document.getElementById('mode').textContent=t>=duration?data.outcome:(walking?'Walking · ankle off':'Ankle balancing');
 document.getElementById('state').textContent='θ = '+theta.toFixed(3)+' rad, ω = '+omega.toFixed(3)+' rad/s';
 document.getElementById('action').textContent=walking?'α = '+alpha.toFixed(4)+' rad':'Swing leg clear';
 document.getElementById('torque').textContent='τ/(mgℓ) = '+(tau/data.torque_scale).toFixed(3);
 document.getElementById('steps').textContent=steps+' footstrikes';seek.value=t;
}
play.onclick=()=>{if(t>=duration)t=0;setPlaying(!running);draw();};
document.getElementById('restart').onclick=()=>{t=0;setPlaying(duration>0);draw();};
seek.oninput=()=>{setPlaying(false);t=Number(seek.value);draw();};
function tick(now){
 if(running){
  const dt=previous===null?0:(now-previous)/1000;previous=now;
  if(t>=duration){
   if(data.autoplay){endHold+=dt;if(endHold>=1.2){t=0;endHold=0;}}
   else setPlaying(false);
  }else t=Math.min(duration,t+dt*Number(speed.value));
  draw();
 }
 requestAnimationFrame(tick);
}
window.addEventListener('resize',draw);setPlaying(data.autoplay&&duration>0);draw();requestAnimationFrame(tick);
</script></body></html>'''
