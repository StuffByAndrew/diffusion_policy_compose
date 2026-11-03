"""Single-arm rollouts of one arm-agnostic policy on a TwoArmOpposedEnv task: one RobomimicImageRunner per active arm.

The policy only ever receives the active arm's observations, renamed to generic (arm-free) keys.
"""
import torch
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env_runner.robomimic_image_runner import RobomimicImageRunner
from diffusion_policy.env.opposed import arm_key


class ArmPolicy:
    """Routes arm `arm`'s obs (generic names) to the shared policy; places its 10-D action in the 20-D layout.
    The idle slice is a copy (valid rotation) and is overwritten by the env with the idle hold pose."""

    def __init__(self, policy, arm, generic_keys):
        self.policy, self.arm, self.generic_keys = policy, arm, list(generic_keys)

    @property
    def device(self):
        return self.policy.device

    @property
    def dtype(self):
        return self.policy.dtype

    def reset(self):
        self.policy.reset()

    def predict_action(self, obs_dict):
        out = self.policy.predict_action({g: obs_dict[arm_key(g, self.arm)] for g in self.generic_keys})
        a = out['action']
        assert a.shape[-1] == 10, a.shape
        return {'action': torch.cat([a, a], dim=-1)}


class OpposedArmRunner(BaseImageRunner):
    def __init__(self, output_dir, shape_meta, render_obs_key, **kwargs):
        super().__init__(output_dir)
        self.generic_keys = list(shape_meta['obs'].keys())
        self.runners = []
        for arm in (0, 1):
            arm_meta = {'obs': {arm_key(k, arm): v for k, v in shape_meta['obs'].items()},
                        'action': {'shape': [20]}}
            cams = [arm_key(k, arm)[:-len('_image')] for k, v in shape_meta['obs'].items() if v.get('type') == 'rgb']
            self.runners.append(RobomimicImageRunner(
                output_dir=output_dir, shape_meta=arm_meta, render_obs_key=arm_key(render_obs_key, arm),
                env_kwargs={'active_arm': arm, 'camera_names': cams}, **kwargs))

    def run(self, policy):
        log = {}
        for arm, runner in enumerate(self.runners):
            for k, v in runner.run(ArmPolicy(policy, arm, self.generic_keys)).items():
                log[f'arm{arm}/{k}'] = v
        for split in ('train', 'test'):
            log[f'{split}/mean_score'] = sum(log[f'arm{a}/{split}/mean_score'] for a in (0, 1)) / 2
        return log
