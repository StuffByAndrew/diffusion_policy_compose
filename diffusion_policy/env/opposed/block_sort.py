"""TwoArmBlockSort: 3 red and 3 blue cubes start at random poses in the middle of the table; the active arm must put
every cube into the zone of its color.

The two zones lie at the table's +x and -x sides, between the bases, so both arms reach both. Which side is red is
random per episode. It is encoded as the angle (0 or pi) of a hinge carrying both zone patches, so it is part of
the sim state (reset_to restores it); policies must read it from the images.
"""
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.models.objects import BoxObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_inertial, new_joint

from diffusion_policy.env.opposed.base import TwoArmOpposedEnv, look_at_quat_wxyz

N_PER_COLOR = 3
BLOCK_RGBA = [(0.8, 0.1, 0.1, 1)] * N_PER_COLOR + [(0.1, 0.25, 0.85, 1)] * N_PER_COLOR   # color 0 red, 1 blue
ZONE_RGBA = [(1, 0.6, 0.6, 1), (0.6, 0.72, 1, 1)]


class TwoArmBlockSort(TwoArmOpposedEnv):
    def __init__(self, robots, active_arm, block_half=0.025, zone_x=0.28, zone_half=(0.08, 0.14),
                 spawn_half=(0.14, 0.10), min_spacing=0.09, idle_offset=(-0.2, 0, 0.2), table_full_size=(1.0, 0.6, 0.05),
                 **kwargs):
        self.idle_offset = np.array(idle_offset)
        self.block_half = block_half
        self.zone_x = zone_x
        self.zone_half = np.array(zone_half)
        self.spawn_half = np.array(spawn_half)
        self.min_spacing = min_spacing
        self.n_blocks = 2 * N_PER_COLOR
        super().__init__(robots=robots, active_arm=active_arm, table_full_size=table_full_size, **kwargs)

    def _camera_poses(self):
        """Policy camera high above the table's near-right corner (robot0's view; both zones and the cubes stay
        visible beside the arm); video camera oblique from the +x side."""
        target = self.table_offset + np.array([0, 0.05, 0])
        sh, av = np.array([0.35, -0.2, 2.3]), np.array([0.9, 0, 1.8])
        return {'agentview': (av, look_at_quat_wxyz(av, self.table_offset)),
                'shouldercamera0': (sh, look_at_quat_wxyz(sh, target))}

    def _load_objects(self, arena):
        zones = new_body(name='zones', pos=(0, 0, self.table_offset[2] + 0.001))
        zones.append(new_joint(name='zones_hinge', type='hinge', axis=(0, 0, 1), damping=10.))
        zones.append(new_inertial(mass=0.1, diaginertia=(1e-3, 1e-3, 1e-3)))   # visual-only geoms carry no mass
        for color, sx in [(0, 1), (1, -1)]:   # hinge angle 0: red zone at +x
            zones.append(new_geom(name=f'zone{color}', type='box', size=(*self.zone_half, 0.0005),
                                  pos=(sx * self.zone_x, 0, 0), rgba=ZONE_RGBA[color], group=1,
                                  contype=0, conaffinity=0))
        arena.worldbody.append(zones)
        self.blocks = [BoxObject(name=f'block{k}', size=[self.block_half] * 3, rgba=BLOCK_RGBA[k])
                       for k in range(self.n_blocks)]
        return self.blocks

    def _idle_pose(self, arm):
        """Tuck the idle hand back and up by idle_offset (its base frame: x forward, z up): with the bases at +-0.46
        its start pose hovers over the table center, in the active arm's way."""
        pose = self.eef_pose_world(arm)
        pose[:3, 3] += self.base_pose(arm)[:3, :3] @ self.idle_offset
        return pose

    def _sample_placements(self):
        """Uniform positions in the spawn rectangle with pairwise spacing >= min_spacing (rejection), uniform yaw."""
        xy = []
        for _ in range(10000):
            p = np.random.uniform(-1, 1, 2) * self.spawn_half
            if all(np.linalg.norm(p - q) >= self.min_spacing for q in xy):
                xy.append(p)
                if len(xy) == self.n_blocks:
                    break
        assert len(xy) == self.n_blocks, 'spawn area too small for min_spacing'
        z = self.table_offset[2] + self.block_half
        return [(b, (*p, z), T.convert_quat(T.axisangle2quat(np.array([0, 0, np.random.uniform(0, np.pi / 2)])),
                                            to='wxyz')) for b, p in zip(self.blocks, xy)]

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            self.sim.data.set_joint_qpos('zones_hinge', np.pi * np.random.randint(2))
            self.sim.forward()

    def _setup_references(self):
        super()._setup_references()
        self.block_body_ids = [self.sim.model.body_name2id(b.root_body) for b in self.blocks]

    # ---------------------------------------------------------------- task state
    @staticmethod
    def block_color(k):
        return int(k >= N_PER_COLOR)

    @property
    def red_side(self):
        """+1 if the red zone is at +x, else -1."""
        c = np.cos(self.sim.data.get_joint_qpos('zones_hinge'))
        assert abs(abs(c) - 1) < 1e-6, c
        return float(np.sign(c))

    def zone_center(self, color):
        return np.array([self.red_side * (1 - 2 * color) * self.zone_x, 0.])

    def block_pos(self, k):
        return self.sim.data.body_xpos[self.block_body_ids[k]].copy()

    def block_quat(self, k):
        return T.convert_quat(self.sim.data.body_xquat[self.block_body_ids[k]], to='xyzw')

    def block_sorted(self, k):
        """Resting (not held up) with its center inside the zone of its color."""
        p = self.block_pos(k)
        return bool(np.all(np.abs(p[:2] - self.zone_center(self.block_color(k))) < self.zone_half)
                    and p[2] < self.table_top_z + self.block_half + 0.01)

    @property
    def n_sorted(self):
        return sum(self.block_sorted(k) for k in range(self.n_blocks))

    def reward(self, action=None):
        """Fraction of cubes resting in the zone of their color."""
        return self.n_sorted / self.n_blocks

    def _check_success(self):
        return self.n_sorted == self.n_blocks

    # ---------------------------------------------------------------- observables
    def _object_sensors(self):
        out = [('red_side', lambda c: np.array([self.red_side]))]
        for k in range(self.n_blocks):
            out += [(f'block{k}_pos', lambda c, k=k: self.block_pos(k)),
                    (f'block{k}_quat', lambda c, k=k: self.block_quat(k))]
        return out
