#@title Rollout, Render, and Save MP4
from mujoco_playground._src.gait import draw_joystick_command
import mediapy as media

env = registry.load(env_name)
eval_env = registry.load(env_name)
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

rng = jax.random.PRNGKey(1)

rollout = []
modify_scene_fns = []

x_vel = 0.0   #@param {type: "number"}
y_vel = -1.0  #@param {type: "number"}
yaw_vel = 0.0 #@param {type: "number"}
command = jp.array([x_vel, y_vel, yaw_vel])

phase_dt = 2 * jp.pi * eval_env.dt * 1.5
phase = jp.array([0, jp.pi])

# Cari body ID yang representatif untuk posisi arrow command
# neck_link ada di atas torso (~0.44m dari ground), jauh lebih baik dari base_link
_arrow_body_names = ["neck_link", "head_lnk", "torso_link", "base_link"]
_arrow_body_id = None
for _name in _arrow_body_names:
    try:
        _arrow_body_id = eval_env.mj_model.body(_name).id
        print(f"Using body '{_name}' (id={_arrow_body_id}) for command arrow position")
        break
    except Exception:
        pass
if _arrow_body_id is None:
    _arrow_body_id = eval_env._torso_body_id
    print(f"Fallback to _torso_body_id={_arrow_body_id}")

# z offset di atas body yang dipilih agar arrow muncul di atas kepala
_ARROW_Z_OFFSET = 0.35 if "neck" in _arrow_body_names[0] else 0.55

for j in range(1):
    print(f"episode {j}")
    state = jit_reset(rng)
    state.info["phase_dt"] = phase_dt
    state.info["phase"] = phase
    for i in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        if state.done:
            break
        state.info["command"] = command
        rollout.append(state)

        xyz = np.array(state.data.xpos[_arrow_body_id])
        xyz += np.array([0, 0.0, _ARROW_Z_OFFSET])

        x_axis = state.data.xmat[_arrow_body_id, 0]
        yaw = -np.arctan2(x_axis[1], x_axis[0])

        modify_scene_fns.append(
            functools.partial(
                draw_joystick_command,
                cmd=state.info["command"],
                xyz=xyz,
                theta=yaw,
                scl=np.linalg.norm(state.info["command"]),
            )
        )

render_every = 1
fps = 1.0 / eval_env.dt / render_every
print(f"fps: {fps}")
traj = rollout[::render_every]
mod_fns = modify_scene_fns[::render_every]

scene_option = mujoco.MjvOption()
scene_option.geomgroup[2] = True
scene_option.geomgroup[3] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

camera_names = [eval_env.mj_model.camera(i).name for i in range(eval_env.mj_model.ncam)]
camera_to_use = "track" if "track" in camera_names else -1

frames = eval_env.render(
    traj,
    camera=camera_to_use,
    scene_option=scene_option,
    width=640,
    height=480,
    modify_scene_fns=mod_fns,
)

media.show_video(frames, fps=fps, loop=False)

video_filename = "rollout_video.mp4"
media.write_video(video_filename, frames, fps=fps)
print(f"✅ Video saved as {video_filename}")
