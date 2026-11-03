"""Single-arm evaluation of a shared per-arm policy on a TwoArmOpposedEnv task: per-step task metric for policy vs
oracle vs idle on the same held-out seeds, for each active arm.

python eval_opposed.py --task plate_lift_oracle -c <ckpt> -o results/plate_lift/... --train_log <run_dir>/logs.json.txt
"""
import collections
import importlib
import json
import pathlib

import click
import dill
import hydra
import imageio
import matplotlib
import numpy as np
import torch
import wandb

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diffusion_policy.env.opposed import arm_key
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.scripts import opposed_oracle as O

METHODS = ['policy', 'oracle', 'idle']


def load_policy(checkpoint, out):
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=str(out))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    return (workspace.ema_model if cfg.training.use_ema else workspace.model).to('cuda').eval(), cfg


def policy_rollout(metrics, env, seed, predict, device, keys, n_obs, max_steps, rot_tf, video):
    """Each active arm gets its own generic-key obs history. predict(list of per-arm obs batches, in
    raw.active_arms order) -> (n_act, 10 * n_active) base-frame actions, arm j in columns 10j:10j+10."""
    raw = env.env
    arms = raw.active_arms
    np.random.seed(seed)
    env.reset()

    def generic(o, arm):
        return {g: o[arm_key(g, arm)].astype(np.float32) for g in keys}

    obs = env.get_observation()
    hist = {arm: collections.deque([generic(obs, arm)] * n_obs, maxlen=n_obs) for arm in arms}
    log = collections.defaultdict(list)
    frames = []
    while len(log['reward']) < max_steps:
        batches = [{k: torch.from_numpy(np.stack([h[k] for h in hist[arm]]))[None].to(device) for k in keys}
                   for arm in arms]
        with torch.no_grad():
            act = predict(batches)
        assert act.shape[1] == 10 * len(arms), act.shape
        for a_t in act:
            a = np.zeros(14)
            for j, arm in enumerate(arms):
                a10 = a_t[10 * j:10 * j + 10]
                a[7 * arm:7 * arm + 7] = np.concatenate([a10[:3], rot_tf.inverse(a10[3:9]), a10[9:]])
            obs, _, _, _ = env.step(a)
            for arm in arms:
                hist[arm].append(generic(obs, arm))
            for k, v in metrics(raw).items():
                log[k].append(v)
            if video:
                frames.append(env.render(mode='rgb_array', height=256, width=256, camera_name=O.VIDEO_CAM))
    return {k: np.array(v[:max_steps]) for k, v in log.items()}, frames


@click.command()
@click.option('--task', 'task_name', required=True, help='oracle module in diffusion_policy.scripts')
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('--train_log', required=True)
@click.option('--n_seeds', default=50)
@click.option('--n_video', default=3)
@click.option('--max_steps', default=None, type=int)
def main(task_name, checkpoint, output_dir, train_log, n_seeds, n_video, max_steps):
    task = importlib.import_module(f'diffusion_policy.scripts.{task_name}')
    max_steps = max_steps or task.MAX_STEPS
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    policy, cfg = load_policy(checkpoint, out)
    shape_meta = cfg.task.shape_meta
    keys = list(shape_meta['obs'].keys())
    rot_tf = RotationTransformer('axis_angle', 'rotation_6d')
    run = wandb.init(project=task.PROJECT, job_type='eval', config=dict(checkpoint=checkpoint, n_seeds=n_seeds))

    logs = {(m, a): [] for m in METHODS for a in (0, 1)}
    rows = []
    for arm in (0, 1):
        cams = [arm_key(k, arm)[:-len('_image')] for k, v in shape_meta['obs'].items() if v.get('type') == 'rgb']
        env = O.make_env(task, arm, camera_names=cams)
        oenv = O.make_env(task, arm, images=False)
        for k in range(n_seeds):
            seed = O.EVAL_SEED0 + k
            p_log, frames = policy_rollout(task.metrics, env, seed,
                                           lambda bs: policy.predict_action(bs[0])['action'][0].cpu().numpy(),
                                           policy.device, keys, cfg.n_obs_steps, max_steps, rot_tf, k < n_video)
            o_log, _ = O.oracle_rollout(oenv, task, seed)
            i_log, _ = O.rollout(oenv, task, lambda raw: O.idle_plan(raw, max_steps), seed=seed)
            for m, lg in zip(METHODS, (p_log, o_log, i_log)):
                logs[(m, arm)].append(lg[task.METRIC])
                rows.append(dict(method=m, arm=arm, seed=seed, **task.summarize(lg)))
            if frames:
                path = out / f'policy_arm{arm}_seed{seed}.mp4'
                imageio.mimwrite(path, frames, fps=O.CONTROL_FREQ)
                wandb.log({f'video/policy_arm{arm}_seed{seed}': wandb.Video(str(path), format='mp4')})
            print(rows[-3])

    # metric vs time
    colors = dict(policy='C0', oracle='C1', idle='C7')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for arm in (0, 1):
        for m in METHODS:
            L = max(len(x) for x in logs[(m, arm)])
            X = np.stack([np.pad(x, (0, L - len(x)), mode='edge') for x in logs[(m, arm)]]) * task.METRIC_SCALE
            t = np.arange(L) / O.CONTROL_FREQ
            axes[arm].plot(t, X.mean(0), color=colors[m], label=m)
            axes[arm].fill_between(t, X.mean(0) - X.std(0), X.mean(0) + X.std(0), color=colors[m], alpha=0.2)
        axes[arm].set_title(f'arm {arm} active ({n_seeds} held-out seeds, mean ± std)')
        axes[arm].set_xlabel('time [s]')
    axes[0].set_ylabel(task.METRIC_LABEL)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out / 'metric_vs_time.png', dpi=150)

    # per-episode summary
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for j, (arm, m) in enumerate([(a, m) for a in (0, 1) for m in METHODS]):
        v = [r[task.BOX_KEY] * task.BOX_SCALE for r in rows if r['arm'] == arm and r['method'] == m]
        ax.boxplot(v, positions=[j], widths=0.6, showfliers=False)
        ax.scatter(np.full(len(v), j) + np.random.uniform(-0.15, 0.15, len(v)), v, s=6, color=colors[m])
    ax.set_xticks(range(6), [f'arm{a}\n{m}' for a in (0, 1) for m in METHODS])
    ax.set_ylabel(task.BOX_LABEL)
    fig.tight_layout()
    fig.savefig(out / 'episode_summary.png', dpi=150)

    # training curve
    entries = [json.loads(l) for l in open(train_log)]
    ev = list({e['epoch']: e for e in entries if 'arm0/test/mean_score' in e}.values())  # last per epoch (resumes repeat epochs)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for arm in (0, 1):
        ax.plot([e['epoch'] for e in ev], [e[f'arm{arm}/test/mean_score'] for e in ev], 'o-', label=f'arm {arm}')
    ax.set_xlabel('epoch')
    ax.set_ylabel(task.SCORE_LABEL)
    ax.axhline(np.mean([r['score'] for r in rows if r['method'] == 'oracle']), color='k', ls='--',
               label='oracle (eval mean)')
    ax.axhline(task.SCORE_SUCCESS, color='gray', ls=':', label='success threshold')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'train_curve.png', dpi=150)

    summary = {f'{m}/arm{a}/{k}': float(np.mean([r[k] for r in rows if r['method'] == m and r['arm'] == a]))
               for m in METHODS for a in (0, 1) for k in rows[0] if k not in ('method', 'arm', 'seed')}
    summary.update(task=task_name, checkpoint=checkpoint, n_seeds=n_seeds, max_steps=max_steps)
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
    json.dump(rows, open(out / 'episodes.json', 'w'), indent=1)
    wandb.log({**{k: v for k, v in summary.items() if isinstance(v, float)},
               **{n: wandb.Image(str(out / f'{n}.png')) for n in ['metric_vs_time', 'episode_summary', 'train_curve']}})
    run.finish()
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
