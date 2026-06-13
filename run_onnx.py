"""
Run ONNX policy in real-time MuJoCo viewer with live command sliders.
Usage:
    python run_onnx.py [policy.onnx] [env_name] [--x X] [--y Y] [--yaw YAW]

Defaults:
    policy   : eros_policy.onnx (same folder)
    env_name : ErosJoystickFlatTerrain
    command  : x=0.0  y=-1.0  yaw=0.0
"""
import sys
import os
import time
import argparse
import threading
import tkinter as tk

import numpy as np
import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import onnxruntime as rt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from mujoco_playground._src import registry

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("policy", nargs="?", default=os.path.join(ROOT, "eros_policy.onnx"))
parser.add_argument("env",    nargs="?", default="ErosJoystickFlatTerrain")
parser.add_argument("--x",   type=float, default=0.0)
parser.add_argument("--y",   type=float, default=-1.0)
parser.add_argument("--yaw", type=float, default=0.0)
parser.add_argument("--rng", type=int,   default=1)
args = parser.parse_args()

print(f"Policy : {args.policy}")
print(f"Env    : {args.env}")

# ── Shared command (written by slider thread, read by sim thread) ─────────────
cmd = {"x": args.x, "y": args.y, "yaw": args.yaw}

# ── Slider window (runs in its own thread) ────────────────────────────────────
def run_slider_window():
    root = tk.Tk()
    root.title("Command Control")
    root.geometry("340x200")
    root.resizable(False, False)

    def make_slider(parent, label, from_, to, init, key, row):
        tk.Label(parent, text=label, width=6, anchor="e").grid(row=row, column=0, padx=6, pady=8)
        var = tk.DoubleVar(value=init)
        def on_change(*_):
            cmd[key] = round(var.get(), 2)
        slider = tk.Scale(
            parent, variable=var, from_=from_, to=to,
            resolution=0.05, orient=tk.HORIZONTAL, length=220,
            command=on_change,
        )
        slider.grid(row=row, column=1, padx=6)
        val_label = tk.Label(parent, textvariable=var, width=5)
        val_label.grid(row=row, column=2, padx=4)

    make_slider(root, "X vel",   -2.0,  2.0,  args.x,   "x",   0)
    make_slider(root, "Y vel",   -2.0,  2.0,  args.y,   "y",   1)
    make_slider(root, "Yaw vel", -1.5,  1.5,  args.yaw, "yaw", 2)

    def reset_cmd():
        cmd["x"] = 0.0; cmd["y"] = 0.0; cmd["yaw"] = 0.0

    tk.Button(root, text="Stop (0,0,0)", command=reset_cmd, width=16).grid(
        row=3, column=0, columnspan=3, pady=10)

    root.mainloop()

slider_thread = threading.Thread(target=run_slider_window, daemon=True)
slider_thread.start()

# ── ONNX ──────────────────────────────────────────────────────────────────────
onnx_session = rt.InferenceSession(args.policy, providers=["CPUExecutionProvider"])

def onnx_policy(obs_dict, _):
    state_np = np.asarray(jax.device_get(obs_dict["state"]), dtype=np.float32)[np.newaxis, :]
    out = onnx_session.run(["continuous_actions"], {"obs": state_np})[0]
    return jp.asarray(out[0]), None

# ── Env ───────────────────────────────────────────────────────────────────────
eval_env  = registry.load(args.env)
jit_reset = jax.jit(eval_env.reset)
jit_step  = jax.jit(eval_env.step)

phase_dt = 2 * jp.pi * eval_env.dt * 1.5
phase    = jp.array([0.0, jp.pi])

_ARROW_BODY_CANDIDATES = ["neck_link", "head_lnk", "torso_link", "base_link"]
arrow_body_id = None
for _name in _ARROW_BODY_CANDIDATES:
    try:
        arrow_body_id = eval_env.mj_model.body(_name).id
        print(f"Arrow body: '{_name}' (id={arrow_body_id})")
        break
    except Exception:
        pass
if arrow_body_id is None:
    arrow_body_id = eval_env._torso_body_id

# ── Init ──────────────────────────────────────────────────────────────────────
rng   = jax.random.PRNGKey(args.rng)
state = jit_reset(rng)
state.info["phase_dt"] = phase_dt
state.info["phase"]    = phase
state.info["command"]  = jp.array([cmd["x"], cmd["y"], cmd["yaw"]])

mj_model = eval_env.mj_model
mj_data  = mujoco.MjData(mj_model)

def sync_viewer(state):
    mj_data.qpos[:] = np.array(state.data.qpos)
    mj_data.qvel[:] = np.array(state.data.qvel)
    mujoco.mj_forward(mj_model, mj_data)

sync_viewer(state)

# ── Sim loop ──────────────────────────────────────────────────────────────────
step_count = 0
episode    = 0
last_print = time.time()

print("\nViewer running. Adjust sliders in the Control window.\n")

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    viewer.cam.azimuth   = 100
    viewer.cam.elevation = -12
    viewer.cam.distance  = 2.0
    viewer.cam.lookat[:] = np.array(mj_data.xpos[arrow_body_id])

    while viewer.is_running():
        t0 = time.time()

        command = jp.array([cmd["x"], cmd["y"], cmd["yaw"]])

        act_rng, rng = jax.random.split(rng)
        ctrl, _  = onnx_policy(state.obs, act_rng)
        state    = jit_step(state, ctrl)
        state.info["command"] = command
        step_count += 1

        if state.done:
            episode += 1
            print(f"Episode {episode} ended at step {step_count}, resetting...")
            step_count = 0
            rng, reset_rng = jax.random.split(rng)
            state = jit_reset(reset_rng)
            state.info["phase_dt"] = phase_dt
            state.info["phase"]    = phase
            state.info["command"]  = command

        sync_viewer(state)
        viewer.cam.lookat[:] = np.array(mj_data.xpos[arrow_body_id])
        viewer.sync()

        now = time.time()
        if now - last_print >= 1.0:
            d = state.data
            # IMU site index (same as env uses)
            imu_id = eval_env.mj_model.site("imu").id

            gyro      = np.array(d.sensordata[0:3])    # rad/s  — angular velocity
            linvel    = np.array(d.sensordata[3:6])    # m/s    — local linear velocity
            accel     = np.array(d.sensordata[6:9])    # m/s²   — accelerometer
            gravity   = np.array(d.site_xmat[imu_id]).reshape(3,3).T @ np.array([0,0,-1])  # gravity in IMU frame

            bz = float(d.qpos[2])
            print(f"\nep={episode} step={step_count:4d}  base_z={bz:+.4f}")
            print(f"  CMD    : x={cmd['x']:+.2f}  y={cmd['y']:+.2f}  yaw={cmd['yaw']:+.2f}")
            print(f"  GYRO   : x={gyro[0]:+.3f}  y={gyro[1]:+.3f}  z={gyro[2]:+.3f}  rad/s")
            print(f"  ACCEL  : x={accel[0]:+.3f}  y={accel[1]:+.3f}  z={accel[2]:+.3f}  m/s²")
            print(f"  LINVEL : x={linvel[0]:+.3f}  y={linvel[1]:+.3f}  z={linvel[2]:+.3f}  m/s")
            print(f"  GRAVITY: x={gravity[0]:+.3f}  y={gravity[1]:+.3f}  z={gravity[2]:+.3f}  (IMU frame)")
            last_print = now

        wait = eval_env.dt - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)
