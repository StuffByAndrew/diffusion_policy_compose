"""Scripted closed-loop oracle for TwoArmBlockSort (single active arm): repeatedly pick the unsorted cube nearest the
gripper (top-down, finger axis along the cube face normal with the most clearance from neighbours), carry it above the
other cubes and place it in the next free slot of its color's zone (slots ordered nearest-to-base first). All choices
are functions of the scene in the active robot's base frame, so arm0/arm1 episodes are exact pi-about-z mirrors.

  python block_sort_oracle.py symmetry
  python block_sort_oracle.py test --out results/block_sort/oracle_test
  python block_sort_oracle.py generate --n 200 --out data/block_sort/oracle/image_abs.hdf5
"""
import numpy as np
import robosuite.utils.transform_utils as T

from diffusion_policy.scripts import opposed_oracle as O

ENV_NAME = 'TwoArmBlockSort'
PROJECT = 'block_sort'
MAX_STEPS = 900
METRIC, METRIC_LABEL, METRIC_SCALE = 'n_sorted', 'cubes in their zone', 1
BOX_KEY, BOX_LABEL, BOX_SCALE = 'final_sorted', 'cubes in their zone at episode end', 1
SCORE_LABEL = 'rollout score: max fraction sorted'
SCORE_SUCCESS = 1.0
SYM_METRIC_TOL = 0

SPEED = 0.3             # m/s free / carry motion
SLOW = 0.12             # m/s last few cm of descents
CARRY_Z = 0.13          # grip site height above the table while moving (carried cube clears standing cubes)
APPROACH = 0.04         # slow-descent distance
PLACE_LIFT = 0.008      # release height of the cube bottom above the table
SLOT_DY = 0.08          # slot spacing along y within a zone
HOLD_CLOSE, HOLD_OPEN, SETTLE, HOLD_END = 10, 6, 5, 10
MAX_PICKS = 10


def _with_z(pose, z):
    p = pose.copy()
    p[2, 3] = z
    return p


def grasp_axis(raw, k):
    """Horizontal cube face normal for the finger axis, maximizing clearance to neighbours near the finger line."""
    p = raw.block_pos(k)[:2]
    R = T.quat2mat(raw.block_quat(k))
    axes = [a for a in R.T if abs(a[2]) < 0.5]   # the two horizontal body axes, in body order (mirror-stable)
    assert len(axes) == 2, R
    best, best_clear = None, -1
    for a in axes:
        d = a[:2] / np.linalg.norm(a[:2])
        clear = np.inf
        for j in range(raw.n_blocks):
            v = raw.block_pos(j)[:2] - p
            if j != k and abs(v[0] * d[1] - v[1] * d[0]) < 0.06:
                clear = min(clear, abs(v @ d))
        if clear > best_clear + 1e-9:
            best, best_clear = d, clear
    return best


def free_slot(raw, color):
    zc = raw.zone_center(color)
    base_y = raw.base_pose(raw.active_arm)[1, 3]
    slots = sorted([np.array([zc[0], dy]) for dy in (-SLOT_DY, 0, SLOT_DY)], key=lambda s: abs(s[1] - base_y))
    for s in slots:
        if all(np.linalg.norm(raw.block_pos(j)[:2] - s) > 0.045 for j in range(raw.n_blocks)):
            return s
    raise AssertionError(f'no free slot in zone {color}')


def plan(raw, rng=None):
    """Generator of (world eef pose, gripper); re-reads the scene before every pick. rng jitters speeds +-20%."""
    s = 1.0 if rng is None else rng.uniform(0.8, 1.2)
    cmd = raw.eef_pose_world(raw.active_arm)
    R0, f0 = cmd[:3, :3].copy(), O.finger_axis_world(raw)
    z_table = raw.table_top_z
    z_carry, z_grasp = z_table + CARRY_Z, z_table + raw.block_half
    z_place = z_grasp + PLACE_LIFT

    def go(target, speed, grip, settle=0):
        nonlocal cmd
        steps = O.move(cmd, target, speed * s, grip) + [(target, grip)] * settle
        cmd = target
        return steps

    for _ in range(MAX_PICKS):
        todo = [k for k in range(raw.n_blocks) if not raw.block_sorted(k)]
        if not todo:
            break
        k = min(todo, key=lambda j: np.linalg.norm(raw.block_pos(j)[:2] - cmd[:2, 3]))
        rot = O.rot_about([0, 0, 1], O.yaw_to_align(f0, grasp_axis(raw, k))) @ R0
        above = T.make_pose(np.array([*raw.block_pos(k)[:2], z_carry]), rot)
        slot = free_slot(raw, raw.block_color(k))
        drop = T.make_pose(np.array([*slot, z_carry]), rot)
        yield from go(above, SPEED, -1., SETTLE)
        yield from go(_with_z(above, z_grasp + APPROACH), SPEED, -1.)
        yield from go(_with_z(above, z_grasp), SLOW, -1.)
        yield from go(cmd, SPEED, 1., HOLD_CLOSE)
        yield from go(above, SLOW * 1.5, 1.)
        yield from go(drop, SPEED, 1., SETTLE)
        yield from go(_with_z(drop, z_place + APPROACH), SPEED, 1.)
        yield from go(_with_z(drop, z_place), SLOW, 1.)
        yield from go(cmd, SPEED, -1., HOLD_OPEN)
        yield from go(drop, SPEED, -1.)
    yield from go(cmd, SPEED, -1., HOLD_END)


def metrics(raw):
    return dict(n_sorted=raw.n_sorted, reward=raw.reward())


def summarize(log):
    return dict(max_sorted=int(log['n_sorted'].max()), final_sorted=int(log['n_sorted'][-1]),
                score=float(log['reward'].max()), success=bool(log['n_sorted'][-1] == 6))


def oracle_ok(s):
    return s['success']


def mirror(r0, r1):
    from diffusion_policy.env.opposed.base import rotz_pi_quat_wxyz
    for b0, b1 in zip(r0.blocks, r1.blocks):
        q = r0.sim.data.get_joint_qpos(b0.joints[0]).copy()
        r1.sim.data.set_joint_qpos(b1.joints[0], np.concatenate([q[:3] * [-1, -1, 1], rotz_pi_quat_wxyz(q[3:])]))
    r1.sim.data.set_joint_qpos('zones_hinge', r0.sim.data.get_joint_qpos('zones_hinge') + np.pi)


if __name__ == '__main__':
    import diffusion_policy.scripts.block_sort_oracle as task   # the importable module, not __main__ (spawn workers)
    O.main(task)
