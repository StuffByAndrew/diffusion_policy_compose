"""Product-of-experts composition of diffusion policies that own disjoint slices of a joint action.

    p(A | o_1..o_n)  ∝  prod_i p_i(A[s_i] | o_i)

Expert i sees only its own observation o_i and scores only its slice s_i. The log of the product is the sum of the
experts' log-densities, and expert i is constant along every other slice, so the joint score (equivalently the joint
noise prediction) at diffusion step k is the experts' predictions placed side by side:

    eps(x, k) [..., s_i] = eps_i(x[..., s_i], k, o_i)

One shared DDPM chain (same timesteps, same update) then samples the joint action. For disjoint slices this is exact
(the noised product is the product of the noised factors) and equivalent in distribution to independent sampling;
coupling terms between experts would be added to `eps` inside the loop.
"""
import torch

from diffusion_policy.common.pytorch_util import dict_apply


class ComposedDiffusionPolicy:
    def __init__(self, experts):
        """experts: [(DiffusionUnetHybridImagePolicy, slice of joint action dims), ...]; the same policy object may
        appear several times (copies share weights)."""
        ref = experts[0][0]
        for p, s in experts:
            assert p.obs_as_global_cond
            assert dict(p.noise_scheduler.config) == dict(ref.noise_scheduler.config)
            for a in ('horizon', 'n_obs_steps', 'n_action_steps', 'num_inference_steps', 'kwargs'):
                assert getattr(p, a) == getattr(ref, a), a
            assert s.stop - s.start == p.action_dim and s.step is None, (s, p.action_dim)
        dims = sorted(d for _, s in experts for d in range(s.start, s.stop))
        assert dims == list(range(len(dims))), 'expert slices must be disjoint and cover the joint action'
        self.experts, self.ref, self.action_dim = experts, ref, len(dims)

    @property
    def device(self):
        return self.ref.device

    def reset(self):
        for p, _ in self.experts:
            p.reset()

    @torch.no_grad()
    def predict_action(self, obs_dicts):
        """obs_dicts[i]: expert i's observation dict (its own generic keys), each (B, To, ...)."""
        assert len(obs_dicts) == len(self.experts)
        ref = self.ref
        conds = []
        for (p, _), obs in zip(self.experts, obs_dicts):
            nobs = p.normalizer.normalize(obs)
            B = next(iter(nobs.values())).shape[0]
            feats = p.obs_encoder(dict_apply(nobs, lambda x: x[:, :p.n_obs_steps].reshape(-1, *x.shape[2:])))
            conds.append(feats.reshape(B, -1))

        x = torch.randn((B, ref.horizon, self.action_dim), device=ref.device, dtype=ref.dtype)
        scheduler = ref.noise_scheduler
        scheduler.set_timesteps(ref.num_inference_steps)
        for t in scheduler.timesteps:
            eps = torch.empty_like(x)
            for (p, s), c in zip(self.experts, conds):
                eps[..., s] = p.model(x[..., s], t, local_cond=None, global_cond=c)
            x = scheduler.step(eps, t, x, **ref.kwargs).prev_sample

        action = torch.empty_like(x)
        for p, s in self.experts:
            action[..., s] = p.normalizer['action'].unnormalize(x[..., s])
        start = ref.n_obs_steps - 1
        return {'action': action[:, start:start + ref.n_action_steps], 'action_pred': action}
