"""TwoArmPlateLift: a heavy rimmed plate that one arm can only pivot up about its far bottom edge."""
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import add_to_dict, array_to_string
from robosuite.utils.placement_samplers import UniformRandomSampler

from diffusion_policy.env.opposed.base import TwoArmOpposedEnv

# Filled in by `plate_lift_oracle.py calibrate`.
THETA_MAX = 1.047198      # rad, pivot angle the oracle lifts the near rim to
DCOM_ORACLE = 0.143375    # m, COM rise achieved at THETA_MAX
CLEARANCE_SUCCESS = 0.01  # m, active_arm='both': every bottom corner this far above the table counts as lifted


class PlateWithRimObject(CompositeObject):
    """Flat slab with four rim walls. Rims 0/1 run along x at -y/+y (facing robot0/robot1)."""

    def __init__(self, name, half_size=(0.15, 0.20), base_thickness=0.01, rim_height=0.04,
                 rim_thickness=0.012, density=2000., rim_friction=3.0):
        self._name = name
        self.half_size = np.array(half_size)
        self.base_thickness = base_thickness
        self.rim_height = rim_height
        self.rim_thickness = rim_thickness
        self.density = density
        self.rim_friction = rim_friction
        self._important_sites = {}
        self.rim_geoms = [[f'rim{i}'] for i in range(2)]
        super().__init__(**self._get_geom_attrs())
        self.rim_geoms = [self.correct_naming(g) for g in self.rim_geoms]

    @property
    def important_sites(self):
        return {**super().important_sites, **self._important_sites}

    @property
    def total_height(self):
        return self.base_thickness + self.rim_height

    def _get_geom_attrs(self):
        hx, hy = self.half_size
        hz = self.total_height / 2
        t = self.rim_thickness
        obj_args = {}
        add_to_dict(dic=obj_args, geom_types='box', geom_locations=(0, 0, -hz + self.base_thickness / 2),
                    geom_quats=(1, 0, 0, 0), geom_sizes=(hx, hy, self.base_thickness / 2), geom_names='base',
                    geom_rgbas=(0.85, 0.85, 0.8, 1), geom_materials=None, geom_frictions=None, density=self.density)
        friction = (self.rim_friction, 0.005, 0.0001)
        rims = [  # name, location, half sizes
            ('rim0', (0, -hy + t / 2, 0), (hx, t / 2, hz)),
            ('rim1', (0, hy - t / 2, 0), (hx, t / 2, hz)),
            ('rim2', (-hx + t / 2, 0, 0), (t / 2, hy - t, hz)),
            ('rim3', (hx - t / 2, 0, 0), (t / 2, hy - t, hz)),
        ]
        for name, loc, size in rims:
            add_to_dict(dic=obj_args, geom_types='box', geom_locations=loc, geom_quats=(1, 0, 0, 0),
                        geom_sizes=size, geom_names=name, geom_rgbas=(0.2, 0.4, 0.8, 1) if name in ('rim0', 'rim1')
                        else (0.85, 0.85, 0.8, 1), geom_materials=None, geom_frictions=friction,
                        density=self.density)
        sites = []
        for name, pos in [('rim0', (0, -hy + t / 2, hz)), ('rim1', (0, hy - t / 2, hz)), ('center', (0, 0, 0))]:
            site = self.get_site_attrib_template()
            site.update({'name': name, 'pos': array_to_string(np.array(pos)), 'size': '0.005'})
            sites.append(site)
            self._important_sites[name] = self.naming_prefix + name
        obj_args.update({'total_size': (hx, hy, hz), 'name': self.name, 'locations_relative_to_center': True,
                         'obj_types': 'all', 'sites': sites})
        return obj_args


class TwoArmPlateLift(TwoArmOpposedEnv):
    def __init__(self, robots, active_arm, plate_kwargs=None, placement_xy_range=0.03,
                 placement_yaw_range=np.pi / 12, **kwargs):
        self.plate_kwargs = plate_kwargs or {}
        self.placement_xy_range = placement_xy_range
        self.placement_yaw_range = placement_yaw_range
        super().__init__(robots=robots, active_arm=active_arm, **kwargs)

    def _load_objects(self, arena):
        self.plate = PlateWithRimObject(name='plate', **self.plate_kwargs)
        self.placement_initializer = UniformRandomSampler(
            name='ObjectSampler', mujoco_objects=self.plate,
            x_range=[-self.placement_xy_range, self.placement_xy_range],
            y_range=[-self.placement_xy_range, self.placement_xy_range],
            rotation=(-self.placement_yaw_range, self.placement_yaw_range), rotation_axis='z',
            ensure_object_boundary_in_range=False, ensure_valid_placement=True,
            reference_pos=self.table_offset, z_offset=0.)
        return [self.plate]

    def _sample_placements(self):
        return [(obj, pos, quat) for pos, quat, obj in self.placement_initializer.sample().values()]

    def _setup_references(self):
        super()._setup_references()
        self.plate_body_id = self.sim.model.body_name2id(self.plate.root_body)
        self.rim_site_ids = [self.sim.model.site_name2id(self.plate.important_sites[f'rim{i}']) for i in range(2)]

    # ---------------------------------------------------------------- plate state
    @property
    def plate_com(self):
        return self.sim.data.xipos[self.plate_body_id].copy()

    @property
    def rest_com_z(self):
        """COM height of the plate resting flat on the table."""
        return float(self.table_top_z + self.plate.total_height / 2
                     + self.sim.model.body_ipos[self.plate_body_id][2])

    @property
    def plate_tilt(self):
        R = self.sim.data.body_xmat[self.plate_body_id].reshape(3, 3)
        return float(np.arccos(np.clip(R[2, 2], -1, 1)))

    @property
    def dcom(self):
        return float(self.plate_com[2] - self.rest_com_z)

    def bottom_corner_height(self, y_signs):
        """Height above the table of the lowest bottom corner on the given rim sides (+1: robot1's rim, -1: robot0's)."""
        R = self.sim.data.body_xmat[self.plate_body_id].reshape(3, 3)
        p = self.sim.data.body_xpos[self.plate_body_id]
        hx, hy = self.plate.half_size
        hz = self.plate.total_height / 2
        corners = [p + R @ np.array([sx * hx, sy * hy, -hz]) for sy in y_signs for sx in (-1, 1)]
        return min(c[2] for c in corners) - self.table_top_z

    def far_rim_bottom_height(self):
        """Height above the table of the lowest point of the rim far from the active arm."""
        return self.bottom_corner_height([1 if self.active_arm == 0 else -1])

    def plate_clearance(self):
        """Height above the table of the plate's lowest bottom corner (> 0: fully off the table)."""
        return self.bottom_corner_height([-1, 1])

    def holding(self, arm):
        """Gripper `arm` touches its rim. (A two-pad pinch check is too strict: once the plate tilts, gravity along
        the finger axis pushes the weak Panda fingers sideways and the rim rests on one finger.)"""
        return self.touching(arm, self.plate.rim_geoms[arm])

    def active_holding(self):
        return self.holding(self.active_arm)

    def reward(self, action=None):
        """Plate COM rise [m], only while every active gripper holds its rim (so flinging scores 0)."""
        return max(self.dcom, 0.) if all(self.holding(a) for a in self.active_arms) else 0.

    def _check_success(self):
        if self.active_arm == 'both':
            return self.reward() > 0 and self.plate_clearance() > CLEARANCE_SUCCESS
        assert DCOM_ORACLE is not None, 'run plate_lift_oracle.py calibrate first'
        return self.reward() >= 0.9 * DCOM_ORACLE

    # ---------------------------------------------------------------- observables
    def _object_sensors(self):
        return [('plate_pos', lambda c: self.sim.data.body_xpos[self.plate_body_id].copy()),
                ('plate_quat', lambda c: T.convert_quat(self.sim.data.body_xquat[self.plate_body_id], to='xyzw')),
                ('rim0_pos', lambda c: self.sim.data.site_xpos[self.rim_site_ids[0]].copy()),
                ('rim1_pos', lambda c: self.sim.data.site_xpos[self.rim_site_ids[1]].copy())]
