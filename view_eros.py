"""
Interactive EROS robot viewer.
- Opens MuJoCo viewer with full ctrl sliders (right panel -> Control tab)
- Press R in viewer to reset to home keyframe pose
- Press Space to pause/unpause simulation
- IMU site is highlighted RED with an XYZ axis frame so you can see exactly
  where it is and how it's oriented. Foot sites are highlighted GREEN.
- Terminal shows a live-updating line with gyro / accel / gravity / linvel
  read straight from the IMU sensors (same data the RL policy/real robot sees).
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

# ── Highlight IMU + foot sites so they're visible in the viewer ───────────────
def highlight_site(name, size, rgba):
    sid = model.site(name).id
    model.site_size[sid] = size
    model.site_rgba[sid] = rgba

highlight_site('imu',    [0.018, 0.018, 0.018], [1.0, 0.0, 0.0, 1.0])  # red ball
highlight_site('r_foot', [0.012, 0.012, 0.012], [0.0, 1.0, 0.0, 1.0])  # green ball
highlight_site('l_foot', [0.012, 0.012, 0.012], [0.0, 1.0, 0.0, 1.0])  # green ball

imu_site_id = model.site('imu').id
print(f"\nIMU site id={imu_site_id}  local pos (relative to base_link) = {model.site_pos[imu_site_id]}")

# ── Sensor helper: read by name regardless of ordering in XML ─────────────────
def sensor(name):
    sid = model.sensor(name).id
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return data.sensordata[adr:adr + dim]

# Reset to home keyframe
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

r_foot_geom = model.geom('r_ankle_roll_geom').id
l_foot_geom = model.geom('l_ankle_roll_geom').id

def get_foot_z(geom_id):
    return data.geom_xpos[geom_id][2] - model.geom_size[geom_id][2]

print("\nViewer opened. Use ctrl sliders (right panel -> Control) to tune joint positions.")
print("RED ball = IMU location. GREEN balls = foot sites.")
print("Press R in viewer to reset to home keyframe.\n")

last_print = 0.0
PRINT_HZ = 10.0  # live IMU readout refresh rate

with mujoco.viewer.launch_passive(model, data) as viewer:
    # set camera
    viewer.cam.azimuth = 100
    viewer.cam.elevation = -10
    viewer.cam.distance = 1.8
    viewer.cam.lookat[:] = [0.1, 0.077, 0.2]

    # Sites are visible by default (group 0). Show XYZ axis frames at each site
    # too, so IMU orientation is visible.
    viewer.opt.sitegroup[:] = 1
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()

        now = time.time()
        if now - last_print >= 1.0 / PRINT_HZ:
            gyro    = sensor('gyro')           # rad/s
            accel   = sensor('accelerometer')  # m/s^2
            linvel  = sensor('local_linvel')   # m/s (IMU frame)
            upvec   = sensor('upvector')       # gravity-aligned "up" in IMU frame
            r_fz, l_fz = get_foot_z(r_foot_geom), get_foot_z(l_foot_geom)
            bz = data.qpos[2]

            line = (
                f"\rt={data.time:6.2f}s  base_z={bz:+.3f}  foot(r,l)=({r_fz:+.3f},{l_fz:+.3f})  "
                f"GYRO=({gyro[0]:+.2f},{gyro[1]:+.2f},{gyro[2]:+.2f})  "
                f"ACCEL=({accel[0]:+.2f},{accel[1]:+.2f},{accel[2]:+.2f})  "
                f"LINVEL=({linvel[0]:+.2f},{linvel[1]:+.2f},{linvel[2]:+.2f})  "
                f"UP=({upvec[0]:+.2f},{upvec[1]:+.2f},{upvec[2]:+.2f})   "
            )
            sys.stdout.write(line)
            sys.stdout.flush()
            last_print = now

        elapsed = time.time() - step_start
        sleep_t = model.opt.timestep - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)
