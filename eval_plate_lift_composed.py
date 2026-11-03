"""Bimanual plate lift by score composition: two copies of the unchanged per-arm policy, one per arm, sampled in one joint
diffusion chain (ComposedDiffusionPolicy), vs the single-arm policy (arm 0 active, arm 1 idle) on the same seeds.

python eval_plate_lift_composed.py -c <ckpt> -o results/plate_lift/<run>/<ckpt>_composed
"""
import json
import pathlib

import click
import imageio
import numpy as np
import torch
import wandb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import diffusion_policy.env.opposed.plate_lift as PL
from diffusion_policy.env.opposed import arm_key
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.policy.composed_diffusion_policy import ComposedDiffusionPolicy
from diffusion_policy.scripts import opposed_oracle as O
from diffusion_policy.scripts import plate_lift_oracle as task
from eval_opposed import load_policy, policy_rollout

METHODS = ['composed', 'single']   # both arms active | arm 0 active, arm 1 holds its reset pose
HOLD_WINDOW = 20                    # final steps over which the both-holding fraction is measured
PLOTS = [('dcom', 'plate COM rise [cm]', 100), ('clearance', 'lowest plate corner above table [cm]', 100),
         ('tilt', 'plate tilt [deg]', 180 / np.pi)]


def metrics(raw):
    return dict(dcom=raw.dcom, tilt=raw.plate_tilt, clearance=raw.plate_clearance(), hold0=raw.holding(0),
                hold1=raw.holding(1), reward=raw.reward())


def summarize(log):
    both = log['hold0'] & log['hold1']
    lifted = both & (log['clearance'] > PL.CLEARANCE_SUCCESS)
    return dict(max_dcom=float(log['dcom'].max()), final_dcom=float(log['dcom'][-1]),
                max_clearance=float(log['clearance'].max()), final_clearance=float(log['clearance'][-1]),
                final_tilt_deg=float(np.rad2deg(log['tilt'][-1])), both_hold_frac=float(both[-HOLD_WINDOW:].mean()),
                lifted_frac=float(lifted.mean()), success=bool(lifted.any()))


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('--n_seeds', default=50)
@click.option('--n_video', default=3)
@click.option('--max_steps', default=task.MAX_STEPS)
def main(checkpoint, output_dir, n_seeds, n_video, max_steps):
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy, cfg = load_policy(checkpoint, out)
    keys = list(cfg.task.shape_meta['obs'].keys())
    rgb = [k for k, v in cfg.task.shape_meta['obs'].items() if v.get('type') == 'rgb']
    rot_tf = RotationTransformer('axis_angle', 'rotation_6d')
    composer = ComposedDiffusionPolicy([(policy, slice(0, 10)), (policy, slice(10, 20))])
    predict = {'composed': lambda bs: composer.predict_action(bs)['action'][0].cpu().numpy(),
               'single': lambda bs: policy.predict_action(bs[0])['action'][0].cpu().numpy()}
    envs = {m: O.make_env(task, active_arm, camera_names=[arm_key(k, a)[:-len('_image')]
                                                          for a in arms for k in rgb])
            for m, active_arm, arms in [('composed', 'both', (0, 1)), ('single', 0, (0,))]}
    run = wandb.init(project=task.PROJECT, job_type='eval_composed',
                     config=dict(checkpoint=checkpoint, n_seeds=n_seeds, max_steps=max_steps))

    logs = {m: [] for m in METHODS}
    rows = []
    for k in range(n_seeds):
        seed = O.EVAL_SEED0 + k
        for m in METHODS:
            torch.manual_seed(seed)
            lg, frames = policy_rollout(metrics, envs[m], seed, predict[m], policy.device, keys, cfg.n_obs_steps,
                                        max_steps, rot_tf, m == 'composed' and k < n_video)
            logs[m].append(lg)
            rows.append(dict(method=m, seed=seed, **summarize(lg)))
            print(rows[-1])
            if frames:
                path = out / f'{m}_seed{seed}.mp4'
                imageio.mimwrite(path, frames, fps=O.CONTROL_FREQ)
                wandb.log({f'video/{m}_seed{seed}': wandb.Video(str(path), format='mp4')})

    colors = dict(composed='C0', single='C7')
    fig, axes = plt.subplots(1, len(PLOTS), figsize=(5 * len(PLOTS), 3.8))
    t = np.arange(max_steps) / O.CONTROL_FREQ
    for ax, (key, label, scale) in zip(axes, PLOTS):
        for m in METHODS:
            X = np.stack([lg[key] for lg in logs[m]]) * scale
            ax.plot(t, X.mean(0), color=colors[m], label=m)
            ax.fill_between(t, X.mean(0) - X.std(0), X.mean(0) + X.std(0), color=colors[m], alpha=0.2)
        ax.set_xlabel('time [s]')
        ax.set_ylabel(label)
    axes[1].axhline(PL.CLEARANCE_SUCCESS * 100, color='gray', ls=':', label='lifted threshold')
    axes[0].legend()
    fig.suptitle(f'composed (both arms) vs single arm, {n_seeds} held-out seeds, mean ± std')
    fig.tight_layout()
    fig.savefig(out / 'metrics_vs_time.png', dpi=150)

    summary = {f'{m}/{k}': float(np.mean([r[k] for r in rows if r['method'] == m]))
               for m in METHODS for k in rows[0] if k not in ('method', 'seed')}
    summary.update(checkpoint=checkpoint, n_seeds=n_seeds, max_steps=max_steps)
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
    json.dump(rows, open(out / 'episodes.json', 'w'), indent=1)
    wandb.log({**{k: v for k, v in summary.items() if isinstance(v, float)},
               'metrics_vs_time': wandb.Image(str(out / 'metrics_vs_time.png'))})
    run.finish()
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
