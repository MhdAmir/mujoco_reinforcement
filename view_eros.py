"""
Interactive EROS robot viewer.
- Opens MuJoCo viewer with full ctrl sliders
- Press R to reset to home keyframe pose
- Press Space to pause/unpause simulation
- Ctrl sliders: r_hip_yaw, r_hip_roll, r_hip_pitch, r_knee_pitch, r_ankle_pitch, r_ankle_roll,
                l_hip_yaw, l_hip_roll, l_hip_pitch, l_knee_pitch, l_ankle_pitch, l_ankle_roll,
                r_sh_pitch, r_sh_roll, r_elb_pitch, l_sh_pitch, l_sh_roll, l_elb_pitch, neck, head
- Terminal prints base z and foot z every second so you can monitor standing height
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))

import mujoco
import mujoco.viewer
import numpy as np
import time

XML_PATH = os.path.join(os.path.dirname(__file__),
    'mujoco_playground/_src/locomotion/eros/xmls/scene_mjx_feetonly_flat_terrain.xml')
ASSET_DIRS = [
    os.path.join(os.path.dirname(__file__), 'mujoco_playground/_src/locomotion/eros/xmls'),
    os.path.join(os.path.dirname(__file__), 'mujoco_playground/_src/locomotion/eros/xmls/assets'),
]

# Load assets
assets = {}
for d in ASSET_DIRS:
    if os.path.exists(d):
        for f in glob.glob(os.path.join(d, '*')):
            if os.path.isfile(f):
                with open(f, 'rb') as fh:
                    assets[os.path.basename(f)] = fh.read()

with open(XML_PATH) as f:
    xml_str = f.read()

model = mujoco.MjModel.from_xml_string(xml_str, assets=assets)
data = mujoco.MjData(model)

# Print actuator names and indices
print("Actuators (ctrl index -> name):")
for i in range(model.nu):
    print(f"  ctrl[{i:2d}] = {model.actuator(i).name}")

print("\nJoint qpos addresses:")
for i in range(model.njnt):
    jnt = model.joint(i)
    if jnt.name:
        print(f"  qpos[{jnt.qposadr}] = {jnt.name}")

# Reset to home keyframe
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

last_print = time.time()
r_foot_geom = model.geom('r_ankle_roll_geom').id
l_foot_geom = model.geom('l_ankle_roll_geom').id

def get_foot_z(geom_id):
    return data.geom_xpos[geom_id][2] - model.geom_size[geom_id][2]

print("\nViewer opened. Use ctrl sliders on the right panel to tune joint positions.")
print("Press R in viewer to reset to home keyframe.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # set camera
    viewer.cam.azimuth = 100
    viewer.cam.elevation = -10
    viewer.cam.distance = 1.8
    viewer.cam.lookat[:] = [0.1, 0.077, 0.2]

    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()

        now = time.time()
        if now - last_print >= 1.0:
            r_fz = get_foot_z(r_foot_geom)
            l_fz = get_foot_z(l_foot_geom)
            bz = data.qpos[2]
            by = data.qpos[1]
            print(f"base_z={bz:+.4f}  r_foot_z={r_fz:+.4f}  l_foot_z={l_fz:+.4f}  base_y={by:+.4f}")
            print(f"  ctrl: r_hp={data.ctrl[2]:.3f} r_kn={data.ctrl[3]:.3f} r_ak={data.ctrl[4]:.3f} "
                  f"l_hp={data.ctrl[8]:.3f} l_kn={data.ctrl[9]:.3f} l_ak={data.ctrl[10]:.3f}")
            last_print = now

        elapsed = time.time() - step_start
        sleep_t = model.opt.timestep - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
