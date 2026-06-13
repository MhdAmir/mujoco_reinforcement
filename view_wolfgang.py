"""
Interactive Wolfgang robot viewer.
- Opens MuJoCo viewer with full ctrl sliders
- Press R to reset to home keyframe pose
- Press Space to pause/unpause simulation
- Ctrl sliders: LR_HR, LR_HAA, LR_HFE, LR_KFE, LR_FFE, LR_FAA,
                LL_HR, LL_HAA, LL_HFE, LL_KFE, LL_FFE, LL_FAA
- Terminal prints base z and foot z every second so you can monitor standing height
"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(__file__))

import mujoco
import mujoco.viewer
import numpy as np
import time

XML_PATH = os.path.join(os.path.dirname(__file__),
    'mujoco_playground/_src/locomotion/wolfgang/xmls/scene_mjx_feetonly_flat_terrain.xml')
ASSET_DIRS = [
    os.path.join(os.path.dirname(__file__), 'mujoco_playground/_src/locomotion/wolfgang/xmls'),
    os.path.join(os.path.dirname(__file__), 'mujoco_playground/_src/locomotion/wolfgang/xmls/assets'),
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
# Wolfgang keyframe ctrl is all-zero; initialize ctrl to match joint qpos so
# position actuators hold the standing pose instead of driving joints to 0.
for i in range(model.nu):
    jnt_id = model.actuator_trnid[i, 0]
    data.ctrl[i] = data.qpos[model.jnt_qposadr[jnt_id]]
mujoco.mj_forward(model, data)

last_print = time.time()
r_foot_geom = model.geom('r_foot1').id
l_foot_geom = model.geom('l_foot1').id

def get_foot_z(geom_id):
    return data.geom_xpos[geom_id][2] - model.geom_size[geom_id][2]

print("\nViewer opened. Use ctrl sliders on the right panel to tune joint positions.")
print("Press R in viewer to reset to home keyframe.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # set camera
    viewer.cam.azimuth = 100
    viewer.cam.elevation = -10
    viewer.cam.distance = 1.8
    viewer.cam.lookat[:] = [0.0, 0.0, 0.28]

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
            print(f"  ctrl: r_hfe={data.ctrl[2]:.3f} r_kfe={data.ctrl[3]:.3f} r_ffe={data.ctrl[4]:.3f} "
                  f"l_hfe={data.ctrl[8]:.3f} l_kfe={data.ctrl[9]:.3f} l_ffe={data.ctrl[10]:.3f}")
            last_print = now

        elapsed = time.time() - step_start
        sleep_t = model.opt.timestep - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
