"""Shared tooling for scripted single-arm oracles on TwoArmOpposedEnv tasks: env construction, plan rollout, oracle test
(plots + videos), pi-about-z symmetry check and demo generation in the generic per-arm dataset format.

A task module (plate_lift_oracle, block_sort_oracle) defines
  ENV_NAME, PROJECT, MAX_STEPS (policy rollouts)
  METRIC, METRIC_LABEL, METRIC_SCALE   per-step quantity plotted vs time
  BOX_KEY, BOX_LABEL, BOX_SCALE        per-episode quantity of the box plot
  SCORE_LABEL, SCORE_SUCCESS           rollout score (max reward) axis label and success threshold
  plan(raw, rng)          -> iterable of (world eef pose 4x4, gripper); a generator is closed-loop (reads live state)
  metrics(raw)            -> dict of per-step scalars, incl. METRIC and 'reward' (= env reward)
  summarize(log)          -> dict of episode scalars incl. 'score' (max reward) and 'success'
  oracle_ok(summary)      -> bool, the oracle-test / demo-keeping criterion
  mirror(r0, r1)          -> copy r0's non-robot state into r1 rotated by pi about z
"""
import importlib
import json
import multiprocessing as mp
import pathlib

import numpy as np
import robosuite.utils.transform_utils as T

ARM_CAMS = ['shouldercamera{}', 'robot{}_eye_in_hand']
LOWDIM = ['robot{}_eef_base_pos', 'robot{}_eef_base_quat', 'robot{}_gripper_qpos']
GENERIC_KEYS = ['shouldercamera_image', 'robot_eye_in_hand_image', 'robot_eef_base_pos', 'robot_eef_base_quat',
                'robot_gripper_qpos']
VIDEO_CAM = 'agentview'
IMG_HW = 84
CONTROL_FREQ = 20
TEST_SEED0, GEN_SEED0, EVAL_SEED0 = 50000, 0, 100000


def env_meta(task, active_arm):
    import robosuite
    ctrl = robosuite.load_controller_config(default_controller='OSC_POSE')
    ctrl['control_delta'] = False
    cams = [VIDEO_CAM] + [c.format(i) for c in ARM_CAMS for i in range(2)]
    return {'env_name': task.ENV_NAME, 'type': 1, 'env_kwargs': dict(
        robots=['Panda', 'Panda'], active_arm=active_arm, controller_configs=ctrl, control_freq=CONTROL_FREQ,
        camera_names=cams, camera_heights=IMG_HW, camera_widths=IMG_HW, horizon=10000)}


def make_env(task, active_arm, images=True, postprocess=True, camera_names=None):
    """postprocess=False keeps images uint8 HWC (dataset format); True gives CHW float like the runner."""
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import diffusion_policy.env.opposed  # noqa: F401  registers the envs
    ObsUtils.initialize_obs_modality_mapping_from_dict({
        'rgb': [c.format(i) + '_image' for c in ARM_CAMS for i in range(2)] + [VIDEO_CAM + '_image'],
        'low_dim': [k.format(i) for k in LOWDIM for i in range(2)]})
    meta = env_meta(task, active_arm)
    if camera_names is not None:
        meta['env_kwargs']['camera_names'] = camera_names
    env = EnvUtils.create_env_from_metadata(env_meta=meta, render=False, render_offscreen=images,
                                            use_image_obs=images)
    env.postprocess_visual_obs = postprocess   # create_env hardcodes True; EnvRobosuite reads this attr per obs
    env.env.hard_reset = False
    return env


# ------------------------------------------------------------------ plan helpers
def rot_about(u, theta):
    return T.quat2mat(T.axisangle2quat(np.asarray(u, dtype=float) * theta))


def finger_axis_world(raw):
    g = raw.robots[raw.active_arm].gripper
    pads = [raw.sim.data.geom_xpos[raw.sim.model.geom_name2id(g.important_geoms[k][0])]
            for k in ('left_fingerpad', 'right_fingerpad')]
    d = pads[0] - pads[1]
    return d / np.linalg.norm(d)


def yaw_to_align(f, d):
    """Smallest world-z rotation taking horizontal finger axis f onto +-d (the finger axis is sign-free)."""
    phi = np.arctan2(f[0] * d[1] - f[1] * d[0], f[:2] @ d[:2])
    return (phi + np.pi / 2) % np.pi - np.pi / 2


def interp(p0, p1, n):
    q0, q1 = T.mat2quat(p0[:3, :3]), T.mat2quat(p1[:3, :3])
    return [T.make_pose(p0[:3, 3] + (p1[:3, 3] - p0[:3, 3]) * s, T.quat2mat(T.quat_slerp(q0, q1, s)))
            for s in np.linspace(0, 1, n + 1)[1:]]


def n_steps(p0, p1, speed):
    return max(int(np.ceil(np.linalg.norm(p1[:3, 3] - p0[:3, 3]) / (speed / CONTROL_FREQ))), 1)


def move(p0, p1, speed, grip):
    return [(p, grip) for p in interp(p0, p1, n_steps(p0, p1, speed))]


def idle_plan(raw, length):
    return [(raw.eef_pose_world(raw.active_arm), -1.)] * length


# ------------------------------------------------------------------ rollout
def rollout(env, task, plan_fn, seed=None, state=None, record=False, video=False):
    """Run plan_fn(raw) from reset(seed) or reset_to(state). Logs task.metrics per step; optionally records the
    robomimic obs/states before each step and the base-frame 7-D actions (dataset), and agentview frames."""
    assert (seed is None) != (state is None)
    if seed is not None:
        np.random.seed(seed)
        env.reset()
    else:
        env.reset_to({'states': state})
    raw = env.env
    i = raw.active_arm
    log, obs_list, actions, states, frames = {}, [], [], [], []
    obs = env.get_observation()
    for pose, grip in plan_fn(raw):
        if record:
            obs_list.append(obs)
            states.append(env.get_state()['states'])
        if video:
            frames.append(env.render(mode='rgb_array', height=256, width=256, camera_name=VIDEO_CAM))
        a = np.zeros(14)
        a[7 * i:7 * i + 7] = raw.world_pose_to_base_action(i, pose, grip)
        actions.append(a[7 * i:7 * i + 7])
        obs, _, _, _ = env.step(a)
        for k, v in task.metrics(raw).items():
            log.setdefault(k, []).append(v)
    log = {k: np.array(v) for k, v in log.items()}
    return log, dict(obs=obs_list, actions=np.array(actions), states=np.array(states), frames=frames)


def oracle_rollout(env, task, seed, **kw):
    return rollout(env, task, lambda raw: task.plan(raw, np.random.RandomState(seed)), seed=seed, **kw)


# ------------------------------------------------------------------ symmetry
def symmetry_check(task, seed=TEST_SEED0 - 1000, atol=5e-3):
    """An arm1 oracle episode from the pi-about-z mirror of an arm0 initial state must reproduce arm0's base-frame
    observations and actions (so the shared policy's inputs carry no arm identity)."""
    e0, e1 = make_env(task, 0, images=False), make_env(task, 1, images=False)
    np.random.seed(seed)
    e0.reset()
    e1.reset()
    r0, r1 = e0.env, e1.env
    for dst, src in [(r1.robots[0], r0.robots[1]), (r1.robots[1], r0.robots[0])]:
        for a, b in zip(dst.robot_joints + dst.gripper.joints, src.robot_joints + src.gripper.joints):
            r1.sim.data.set_joint_qpos(a, r0.sim.data.get_joint_qpos(b))
    task.mirror(r0, r1)
    r1.sim.forward()
    for arm, r in [(0, r0), (1, r1)]:
        assert r.active_arm == arm
    p0, p1 = r0.eef_pose_base(0), r1.eef_pose_base(1)
    assert np.allclose(p0, p1, atol=1e-6), (p0, p1)
    s0, s1 = e0.get_state()['states'], e1.get_state()['states']
    rng = lambda: np.random.RandomState(seed)
    l0, d0 = rollout(e0, task, lambda raw: task.plan(raw, rng()), state=s0)
    l1, d1 = rollout(e1, task, lambda raw: task.plan(raw, rng()), state=s1)
    n = min(len(d0['actions']), len(d1['actions']))
    err = np.abs(d0['actions'][:n, :3] - d1['actions'][:n, :3]).max()
    m_err = np.abs(l0[task.METRIC][:n] - l1[task.METRIC][:n]).max()
    print(f'symmetry check: lengths {len(d0["actions"])}/{len(d1["actions"])}, max base-frame action pos diff '
          f'{err * 1000:.3f} mm, max {task.METRIC} diff {m_err:.4g}')
    assert len(d0['actions']) == len(d1['actions']) and err < atol and m_err <= task.SYM_METRIC_TOL, (err, m_err)


# ------------------------------------------------------------------ oracle test
def _test_worker(args):
    task_name, seed, arm, video = args
    task = importlib.import_module(task_name)
    env = make_env(task, arm, images=video)
    o_log, o = oracle_rollout(env, task, seed, video=video)
    i_log, _ = rollout(env, task, lambda raw: idle_plan(raw, len(o['actions'])), seed=seed)
    return seed, arm, o_log, i_log, o['frames']


def test_oracle(task, out, n_per_arm, n_proc):
    """Oracle vs idle on held-out seeds: METRIC vs time plot, videos, per-episode summaries; asserts oracle_ok."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import imageio
    import wandb
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(task.__name__, TEST_SEED0 + 2 * k + a, a, k < 2) for a in (0, 1) for k in range(n_per_arm)]
    with mp.Pool(n_proc) as pool:
        res = pool.map(_test_worker, jobs)
    run = wandb.init(project=task.PROJECT, job_type='oracle_test', config=dict(n_per_arm=n_per_arm))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    rows = []
    for seed, arm, o, idle, frames in res:
        for lg, c in [(o, 'C0'), (idle, 'C7')]:
            axes[arm].plot(np.arange(len(lg[task.METRIC])) / CONTROL_FREQ, lg[task.METRIC] * task.METRIC_SCALE,
                           color=c, alpha=0.5, lw=1)
        rows.append(dict(seed=seed, arm=arm, n_steps=len(o[task.METRIC]), **task.summarize(o),
                         **{f'idle_{k}': v for k, v in task.summarize(idle).items()}))
        if frames:
            path = out / f'oracle_arm{arm}_seed{seed}.mp4'
            imageio.mimwrite(path, frames, fps=CONTROL_FREQ)
            wandb.log({f'video/arm{arm}_seed{seed}': wandb.Video(str(path), fps=CONTROL_FREQ, format='mp4')})
    for arm in (0, 1):
        axes[arm].set_title(f'arm {arm} active (blue: oracle, grey: idle)')
        axes[arm].set_xlabel('time [s]')
    axes[0].set_ylabel(task.METRIC_LABEL)
    fig.tight_layout()
    fig.savefig(out / 'oracle_vs_time.png', dpi=150)
    json.dump(rows, open(out / 'oracle_test.json', 'w'), indent=1)
    wandb.log({'oracle_vs_time': wandb.Image(str(out / 'oracle_vs_time.png')),
               'oracle_test': wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])})
    run.finish()
    for r in rows:
        print(r)
    bad = [r for r in rows if not task.oracle_ok(r)]
    assert not bad, f'oracle failed on {len(bad)}/{len(rows)} episodes: {bad}'
    print(f'oracle OK on all {len(rows)} episodes; episode length {min(r["n_steps"] for r in rows)}-'
          f'{max(r["n_steps"] for r in rows)} steps')


# ------------------------------------------------------------------ generate
def _gen_worker(args):
    task_name, seed = args
    task = importlib.import_module(task_name)
    arm = seed % 2
    env = make_env(task, arm, images=True, postprocess=False)
    log, d = oracle_rollout(env, task, seed, record=True)
    from diffusion_policy.env.opposed import arm_key
    obs = {g: np.stack([o[arm_key(g, arm)] for o in d['obs']]) for g in GENERIC_KEYS}
    return dict(seed=seed, arm=arm, ok=task.oracle_ok(task.summarize(log)), obs=obs, actions=d['actions'],
                states=d['states'], rewards=log['reward'])


def generate(task, out, n, n_proc, min_rate=0.95):
    """n oracle demos (active arm = seed % 2) in robomimic hdf5 with generic per-arm keys and base-frame actions."""
    import h5py
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with h5py.File(out, 'w') as f, mp.Pool(n_proc) as pool:
        data = f.create_group('data')
        for ep in pool.imap(_gen_worker, [(task.__name__, s) for s in range(GEN_SEED0, GEN_SEED0 + n)]):
            if not ep['ok']:
                print('dropping failed demo, seed', ep['seed'])
                continue
            g = data.create_group(f'demo_{n_ok}')
            for k, v in ep['obs'].items():
                g.create_dataset(f'obs/{k}', data=v, compression='gzip' if v.dtype == np.uint8 else None)
            T_ = len(ep['actions'])
            g.create_dataset('actions', data=ep['actions'])
            g.create_dataset('states', data=ep['states'])
            g.create_dataset('rewards', data=ep['rewards'])
            g.create_dataset('dones', data=np.eye(T_)[-1])
            g.attrs.update(num_samples=T_, active_arm=ep['arm'], seed=ep['seed'])
            n_ok += 1
        data.attrs['total'] = sum(data[k].attrs['num_samples'] for k in data)
        data.attrs['env_args'] = json.dumps(env_meta(task, 0))
    rate = n_ok / n
    print(f'oracle success {n_ok}/{n} = {rate:.3f}; wrote {out}')
    assert rate >= min_rate, rate


def main(task, extra_modes=None):
    """CLI: <script> {test,generate,symmetry,*extra_modes} [--out] [--n] [--n_per_arm] [--n_proc]."""
    import argparse
    import os
    extra_modes = extra_modes or {}
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['test', 'generate', 'symmetry', *extra_modes])
    ap.add_argument('--out')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--n_per_arm', type=int, default=20)
    ap.add_argument('--n_proc', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    args = ap.parse_args()
    mp.set_start_method('spawn')
    if args.mode in extra_modes:
        extra_modes[args.mode](args)
    elif args.mode == 'symmetry':
        symmetry_check(task)
    elif args.mode == 'test':
        test_oracle(task, args.out, args.n_per_arm, args.n_proc)
    else:
        generate(task, args.out, args.n, args.n_proc)
