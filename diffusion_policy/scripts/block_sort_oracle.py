"""Scripted closed-loop oracle for TwoArmBlockSort (single active arm): repeatedly pick the unsorted cube nearest the
gripper (top-down, fingers across the cube face pair needing the smaller wrist yaw unless a neighbour blocks it), carry it above the
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
MAX_STEPS = 1300    # oracle episodes take 784-1158 steps
METRIC, METRIC_LABEL, METRIC_SCALE = 'n_sorted', 'cubes in their zone', 1
BOX_KEY, BOX_LABEL, BOX_SCALE = 'final_sorted', 'cubes in their zone at episode end', 1
SCORE_LABEL = 'rollout score: max fraction sorted'
SCORE_SUCCESS = 1.0
SYM_METRIC_TOL = 1   # a cube can cross the in-zone threshold one step apart in the mirrored episode

SPEED = 0.3             # m/s free / carry motion
SLOW = 0.12             # m/s last few cm of descents
CARRY_Z = 0.13          # grip site height above the table while moving (carried cube clears standing cubes)
APPROACH = 0.04         # slow-descent distance
PLACE_LIFT = 0.008      # release height of the cube bottom above the table
SLOT_DY = 0.08          # slot spacing along y within a zone
HOLD_CLOSE, HOLD_OPEN, HOLD_END = 10, 6, 10
SETTLE_TOL, SETTLE_MAX = 0.005, 30   # wait until the eef is within 5 mm of the target (OSC lags the command)
MAX_PICKS = 10
CLEAR_MIN = 0.085       # m, neighbour distance along the finger line that the open fingers clear


def _with_z(pose, z):
    p = pose.copy()
    p[2, 3] = z
    return p


def grasp_yaw(raw, k, f0):
    """World-z rotation (from the reset gripper yaw, finger axis f0) putting the fingers across a pair of cube faces.
    Prefer the smaller rotation (<= 45 deg; large wrist yaw at long reach drives the arm into joint limits) unless a
    neighbour sits within CLEAR_MIN along that finger line, then take the face pair with more clearance."""
    p = raw.block_pos(k)[:2]
    R = T.quat2mat(raw.block_quat(k))
    # the two most horizontal body axes (a knocked cube may lean); rounding + index tie-break keeps mirrors identical
    axes = [R[:, i] for i in sorted(range(3), key=lambda i: (round(abs(R[2, i]), 3), i))[:2]]
    cands = []
    for a in axes:
        d = a[:2] / np.linalg.norm(a[:2])
        clear = min([abs(v @ d) for v in (raw.block_pos(j)[:2] - p for j in range(raw.n_blocks) if j != k)
                     if abs(v[0] * d[1] - v[1] * d[0]) < 0.06], default=np.inf)
        cands.append((O.yaw_to_align(f0, d), clear))
    cands.sort(key=lambda c: abs(c[0]))
    return cands[0][0] if cands[0][1] >= CLEAR_MIN else max(cands, key=lambda c: c[1])[0]


def free_slot(raw, k):
    """Nearest-to-base slot of cube k's zone not occupied by another cube."""
    color = raw.block_color(k)
    zc = raw.zone_center(color)
    base_y = raw.base_pose(raw.active_arm)[1, 3]
    slots = sorted([np.array([zc[0], dy]) for dy in (-SLOT_DY, 0, SLOT_DY)], key=lambda s: abs(s[1] - base_y))
    for s in slots:
        if all(np.linalg.norm(raw.block_pos(j)[:2] - s) > 0.045 for j in range(raw.n_blocks) if j != k):
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

    def go(target, speed, grip, hold=0, settle=False):
        nonlocal cmd
        yield from O.move(cmd, target, speed * s, grip) + [(target, grip)] * hold
        cmd = target
        for _ in range(SETTLE_MAX if settle else 0):
            if np.linalg.norm(raw.eef_pose_world(raw.active_arm)[:3, 3] - target[:3, 3]) < SETTLE_TOL:
                break
            yield target, grip

    for _ in range(MAX_PICKS):
        todo = [k for k in range(raw.n_blocks) if not raw.block_sorted(k)]
        if not todo:
            break
        k = min(todo, key=lambda j: np.linalg.norm(raw.block_pos(j)[:2] - cmd[:2, 3]))
        rot = O.rot_about([0, 0, 1], grasp_yaw(raw, k, f0)) @ R0
        above = T.make_pose(np.array([*raw.block_pos(k)[:2], z_carry]), rot)
        slot = free_slot(raw, k)
        drop = T.make_pose(np.array([*slot, z_carry]), rot)
        yield from go(above, SPEED, -1., settle=True)
        yield from go(_with_z(above, z_grasp + APPROACH), SPEED, -1.)
        yield from go(_with_z(above, z_grasp), SLOW, -1., settle=True)
        yield from go(cmd, SPEED, 1., hold=HOLD_CLOSE)
        yield from go(above, SLOW * 1.5, 1.)
        yield from go(drop, SPEED, 1., settle=True)
        yield from go(_with_z(drop, z_place + APPROACH), SPEED, 1.)
        yield from go(_with_z(drop, z_place), SLOW, 1., settle=True)
        yield from go(cmd, SPEED, -1., hold=HOLD_OPEN)
        yield from go(drop, SPEED, -1.)
    yield from go(cmd, SPEED, -1., hold=HOLD_END)


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
