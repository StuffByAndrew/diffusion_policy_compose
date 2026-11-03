"""Two opposed Pandas whose bases are related by a pi rotation about world z; one arm is active per episode, or both
with active_arm='both' (for composing two copies of the per-arm policy).

The active arm's absolute OSC_POSE action (pos, axis-angle, gripper) is given in that robot's base frame and its
proprio is exposed in the base frame, so one arm-agnostic policy can drive either arm. The idle arm holds its reset
pose with the gripper open. Shoulder camera 1 is the exact pi-about-z copy of shoulder camera 0.

Subclasses implement
  _load_objects(arena) -> list of MujocoObjects (free-jointed)
  _sample_placements() -> [(obj, pos, quat_wxyz)]
  _object_sensors()    -> [(name, sensor fn)]     (low-dim object state, for oracles only)
"""
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.environments.manipulation.two_arm_env import TwoArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

# shoulder camera of two_arm_transport, for robot0's base at y = -0.56 (table depth 0.8); shifted with the base
SHOULDER_CAM0_POS = np.array([0.4430096057365183, -1.0697399743660143, 1.3639950119362048])
SHOULDER_CAM0_QUAT = np.array([0.804057240486145, 0.5531665086746216, 0.11286306381225586, 0.18644218146800995])  # wxyz


def arm_key(generic_key, arm):
    """'robot_eef_base_pos' -> 'robot0_eef_base_pos', 'shouldercamera_image' -> 'shouldercamera0_image'."""
    return generic_key.replace('_', f'{arm}_', 1)


def rotz_pi_quat_wxyz(q):
    """Rotate a wxyz quaternion by pi about world z."""
    return T.convert_quat(T.quat_multiply(np.array([0, 0, 1., 0]), T.convert_quat(q, to='xyzw')), to='wxyz')


class TwoArmOpposedEnv(TwoArmEnv):
    def __init__(self, robots, active_arm, table_full_size=(0.8, 0.8, 0.05), table_friction=(1., 5e-3, 1e-4),
                 env_configuration='single-arm-opposed', use_object_obs=True, **kwargs):
        assert active_arm in (0, 1, 'both'), active_arm
        assert env_configuration == 'single-arm-opposed', env_configuration
        self.active_arm = active_arm
        self.active_arms = (0, 1) if active_arm == 'both' else (active_arm,)
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.use_object_obs = use_object_obs
        self.idle_pose = None
        super().__init__(robots=robots, env_configuration=env_configuration, **kwargs)

    # ---------------------------------------------------------------- model
    def _load_model(self):
        super()._load_model()
        depth = self.table_full_size[1]   # the arms face each other along world y
        for robot, rotation in zip(self.robots, (np.pi / 2, -np.pi / 2)):
            xpos = robot.robot_model.base_xpos_offset['table'](depth)
            rot = np.array((0, 0, rotation))
            robot.robot_model.set_base_xpos(T.euler2mat(rot) @ np.array(xpos))
            robot.robot_model.set_base_ori(rot)

        arena = TableArena(table_full_size=self.table_full_size, table_friction=self.table_friction,
                           table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])
        arena.set_camera(camera_name='agentview', pos=[0.8894354364730311, 0, 1.7383813133506494],
                         quat=[0.6530981063842773, 0.2710406184196472, 0.27104079723358154, 0.6530979871749878])
        cam0 = SHOULDER_CAM0_POS + np.array([0, 0.4 - depth / 2, 0])
        arena.set_camera(camera_name='shouldercamera0', pos=cam0, quat=SHOULDER_CAM0_QUAT)
        arena.set_camera(camera_name='shouldercamera1', pos=cam0 * np.array([-1, -1, 1]),
                         quat=rotz_pi_quat_wxyz(SHOULDER_CAM0_QUAT))
        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[r.robot_model for r in self.robots],
                                      mujoco_objects=self._load_objects(arena))

    def _setup_references(self):
        super()._setup_references()
        self.table_top_id = self.sim.model.site_name2id('table_top')

    def _reset_internal(self):
        super()._reset_internal()
        if not self.deterministic_reset:
            for obj, pos, quat in self._sample_placements():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(pos), np.array(quat)]))
        self.sim.forward()

    @property
    def table_top_z(self):
        return float(self.sim.data.site_xpos[self.table_top_id][2])

    # ---------------------------------------------------------------- frames
    def base_pose(self, arm):
        body = self.robots[arm].robot_model.root_body
        return T.make_pose(self.sim.data.get_body_xpos(body).copy(),
                           self.sim.data.get_body_xmat(body).reshape(3, 3).copy())

    def eef_pose_world(self, arm):
        robot = self.robots[arm]
        return T.make_pose(self.sim.data.site_xpos[robot.eef_site_id].copy(),
                           self.sim.data.site_xmat[robot.eef_site_id].reshape(3, 3).copy())

    def eef_pose_base(self, arm):
        return T.pose_inv(self.base_pose(arm)) @ self.eef_pose_world(arm)

    def base_action_to_world(self, arm, a):
        """7-D (pos, axis-angle, gripper) absolute action: base frame -> world frame."""
        pose_w = self.base_pose(arm) @ T.make_pose(a[:3], T.quat2mat(T.axisangle2quat(a[3:6])))
        return np.concatenate([pose_w[:3, 3], T.quat2axisangle(T.mat2quat(pose_w[:3, :3])), a[6:7]])

    def world_pose_to_base_action(self, arm, pose_w, gripper):
        pose_b = T.pose_inv(self.base_pose(arm)) @ pose_w
        return np.concatenate([pose_b[:3, 3], T.quat2axisangle(T.mat2quat(pose_b[:3, :3])), [gripper]])

    def _pre_action(self, action, policy_step=False):
        assert len(action) == 14, len(action)
        assert not self.robots[0].controller.use_delta, f'{type(self).__name__} expects absolute OSC_POSE actions'
        if self.active_arm == 'both':
            world = np.concatenate([self.base_action_to_world(i, np.asarray(action[7 * i:7 * i + 7])) for i in (0, 1)])
            return super()._pre_action(world, policy_step)
        if policy_step and self.sim.data.time == 0.:   # first control step of an episode (also after reset_to)
            self.idle_pose = self.eef_pose_world(1 - self.active_arm)
        assert self.idle_pose is not None
        world = np.zeros(14)
        i, j = self.active_arm, 1 - self.active_arm
        world[7 * i:7 * i + 7] = self.base_action_to_world(i, np.asarray(action[7 * i:7 * i + 7]))
        world[7 * j:7 * j + 7] = np.concatenate([
            self.idle_pose[:3, 3], T.quat2axisangle(T.mat2quat(self.idle_pose[:3, :3])), [-1.]])
        super()._pre_action(world, policy_step)

    def touching(self, arm, geoms):
        return self.check_contact(self.robots[arm].gripper, geoms)

    def active_touching(self, geoms):
        return self.touching(self.active_arm, geoms)

    # ---------------------------------------------------------------- observables
    def _setup_observables(self):
        observables = super()._setup_observables()
        for arm in range(2):
            pf = self.robots[arm].robot_model.naming_prefix

            @sensor(modality=f'{pf}proprio')
            def eef_base_pos(obs_cache, arm=arm):
                return self.eef_pose_base(arm)[:3, 3]

            @sensor(modality=f'{pf}proprio')
            def eef_base_quat(obs_cache, arm=arm):
                return T.mat2quat(self.eef_pose_base(arm)[:3, :3])

            for name, s in [(f'{pf}eef_base_pos', eef_base_pos), (f'{pf}eef_base_quat', eef_base_quat)]:
                observables[name] = Observable(name=name, sensor=s, sampling_rate=self.control_freq)
        if self.use_object_obs:
            for name, s in self._object_sensors():
                observables[name] = Observable(name=name, sensor=sensor(modality='object')(s),
                                               sampling_rate=self.control_freq)
        return observables
