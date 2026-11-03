"""Scripted oracle for TwoArmPlateLift (single active arm): top-down grasp of the near rim, then a pivot lift about the
far bottom edge to THETA_MAX, then a hold.

  python plate_lift_oracle.py calibrate                 # one arm can't lift plate off; THETA_MAX, DCOM_ORACLE; symmetry
  python plate_lift_oracle.py symmetry
  python plate_lift_oracle.py test --out results/plate_lift/oracle_test
  python plate_lift_oracle.py generate --n 200 --out data/plate_lift/oracle/image_abs.hdf5
"""
import multiprocessing as mp
import pathlib

import numpy as np
import robosuite.utils.transform_utils as T

import diffusion_policy.env.opposed.plate_lift as PL
from diffusion_policy.scripts import opposed_oracle as O

ENV_NAME = 'TwoArmPlateLift'
PROJECT = 'plate_lift'
MAX_STEPS = 300
METRIC, METRIC_LABEL, METRIC_SCALE = 'dcom', 'plate COM rise [cm]', 100
BOX_KEY, BOX_LABEL, BOX_SCALE = 'max_dcom', 'max plate COM rise [cm]', 100
SCORE_LABEL = 'rollout score: max COM rise [m]'
SCORE_SUCCESS = 0.9 * PL.DCOM_ORACLE
SYM_METRIC_TOL = 2e-3   # m

GRASP_DEPTH = 0.025     # grip site below rim top
PREGRASP_HEIGHT = 0.10
SPEED = 0.15            # m/s translational cap
ANG_SPEED = np.deg2rad(20)  # rad/s pivot rate
HOLD_CLOSE = 15
HOLD_END = 20


def grasp_frame(raw):
    """World grasp pose on the active rim, the pivot point, and the pivot axis (rotating +theta lifts the rim)."""
    i = raw.active_arm
    R_p = raw.sim.data.body_xmat[raw.plate_body_id].reshape(3, 3)
    g_pos = raw.sim.data.site_xpos[raw.rim_site_ids[i]] - np.array([0, 0, GRASP_DEPTH])
    # yaw the current (top-down) eef so its finger axis is parallel to the plate y axis (across the rim)
    phi = O.yaw_to_align(O.finger_axis_world(raw), R_p[:, 1])
    g_rot = O.rot_about([0, 0, 1], phi) @ raw.eef_pose_world(i)[:3, :3]
    far_y = raw.plate.half_size[1] * (1 if i == 0 else -1)
    pivot = raw.sim.data.body_xpos[raw.plate_body_id] + R_p @ np.array([0, far_y, -raw.plate.total_height / 2])
    h = (g_pos - pivot) * np.array([1, 1, 0])
    u = np.cross(h, [0, 0, 1.])
    return T.make_pose(g_pos, g_rot), pivot, u / np.linalg.norm(u)


def plan(raw, rng=None, theta_max=None, lift='pivot'):
    """(world eef pose, gripper) targets. lift='pivot' arcs about the far edge; 'vertical' lifts 0.25 m.
    rng jitters the speeds by up to +-20%."""
    theta_max = PL.THETA_MAX if theta_max is None else theta_max
    s = 1.0 if rng is None else rng.uniform(0.8, 1.2)
    start = raw.eef_pose_world(raw.active_arm)
    grasp, pivot, u = grasp_frame(raw)
    pre = grasp.copy()
    pre[2, 3] += PREGRASP_HEIGHT
    out = O.move(start, pre, SPEED * s, -1.) + O.move(pre, grasp, 0.5 * SPEED * s, -1.) + [(grasp, 1.)] * HOLD_CLOSE
    if lift == 'pivot':
        n = max(int(np.ceil(theta_max / (ANG_SPEED * s / O.CONTROL_FREQ))), 1)
        for th in np.linspace(0, theta_max, n + 1)[1:]:
            R = O.rot_about(u, th)
            out.append((T.make_pose(pivot + R @ (grasp[:3, 3] - pivot), R @ grasp[:3, :3]), 1.))
    else:
        top = grasp.copy()
        top[2, 3] += 0.25
        out += O.move(grasp, top, 0.5 * SPEED * s, 1.)
    return out + [out[-1]] * HOLD_END


def metrics(raw):
    return dict(dcom=raw.dcom, tilt=raw.plate_tilt, far_rim=raw.far_rim_bottom_height(),
                holding=raw.active_holding(), reward=raw.reward())


def summarize(log):
    score = float(log['reward'].max())
    return dict(max_dcom=float(log['dcom'].max()), final_dcom=float(log['dcom'][-1]),
                final_tilt_deg=float(np.rad2deg(log['tilt'][-1])), far_rim_max=float(log['far_rim'].max()),
                hold_contact_frac=float(log['holding'][-HOLD_END:].mean()), score=score,
                success=bool(score >= SCORE_SUCCESS))


def oracle_ok(s):
    # contact may flicker during the static hold; a dropped plate would fall flat, so require the COM to stay up
    return s['final_dcom'] >= 0.9 * PL.DCOM_ORACLE and s['hold_contact_frac'] >= 0.8 and s['far_rim_max'] < 0.005


def mirror(r0, r1):
    # the plate is itself pi-about-z symmetric, so the mirrored scene keeps its yaw (rim labels swap) and negates xy
    q = r0.sim.data.get_joint_qpos(r0.plate.joints[0]).copy()
    r1.sim.data.set_joint_qpos(r1.plate.joints[0], np.concatenate([q[:3] * [-1, -1, 1], q[3:]]))


# ------------------------------------------------------------------ calibrate
THETA_GRID = np.deg2rad(np.arange(20, 86, 5))


def _calib_worker(args):
    """Vertical single-arm lift (must not get the whole plate off the table) and pivot holds over THETA_GRID.
    An angle is feasible if, at the end of the hold, the plate tilt matches it within 7 deg and the gripper still
    holds the rim (OSC sag of a few cm under the 4 kg plate is irrelevant to the goal, so not checked)."""
    import diffusion_policy.scripts.plate_lift_oracle as task
    seed, arm = args
    env = O.make_env(task, arm, images=False)
    vert, _ = O.rollout(env, task, lambda raw: plan(raw, theta_max=0., lift='vertical'), seed=seed)
    ok, dcoms = [], []
    for th in THETA_GRID:
        log, _ = O.rollout(env, task, lambda raw: plan(raw, theta_max=th), seed=seed)
        ok.append(bool(abs(log['tilt'][-1] - th) < np.deg2rad(7) and log['holding'][-HOLD_END:].all()))
        dcoms.append(log['dcom'][-HOLD_END:].min())
    return dict(far_rim_max=vert['far_rim'].max(), far_rim_end=vert['far_rim'][-HOLD_END:].max(),
                ok=np.array(ok), dcoms=np.array(dcoms))


def calibrate(args, n_inits=20):
    import diffusion_policy.scripts.plate_lift_oracle as task
    jobs = [(s, s % 2) for s in range(O.TEST_SEED0 - n_inits, O.TEST_SEED0)]   # disjoint from test/gen/eval seeds
    with mp.Pool(args.n_proc) as pool:
        res = pool.map(_calib_worker, jobs)
    far_max, far_end = max(r['far_rim_max'] for r in res), max(r['far_rim_end'] for r in res)
    print(f'vertical single-arm lift over inits: transient max far-rim height {far_max * 100:.2f} cm, '
          f'max over final hold {far_end * 100:.2f} cm')
    print('feasible fraction per angle:', dict(zip(np.rad2deg(THETA_GRID).round().astype(int),
                                                    np.mean([r['ok'] for r in res], 0).round(2))))
    assert far_end < 0.005, 'one arm lifted the whole plate off the table: make the plate heavier'
    all_ok = np.flatnonzero(np.all([r['ok'] for r in res], 0))
    assert len(all_ok), 'no pivot angle is feasible for every init'
    k = all_ok.max()
    theta_max, dcom_oracle = float(THETA_GRID[k]), float(min(r['dcoms'][k] for r in res))
    print(f'THETA_MAX = {theta_max:.4f} rad ({np.rad2deg(theta_max):.0f} deg); DCOM_ORACLE = {dcom_oracle:.4f} m')
    O.symmetry_check(task)
    p = pathlib.Path(PL.__file__)
    lines = [f'THETA_MAX = {theta_max:.6f}' + l[l.index('  #'):] if l.startswith('THETA_MAX = ') else
             f'DCOM_ORACLE = {dcom_oracle:.6f}' + l[l.index('  #'):] if l.startswith('DCOM_ORACLE = ') else l
             for l in p.read_text().split('\n')]
    p.write_text('\n'.join(lines))
    print('wrote constants to', p)


if __name__ == '__main__':
    import diffusion_policy.scripts.plate_lift_oracle as task   # the importable module, not __main__ (spawn workers)
    O.main(task, {'calibrate': calibrate})
