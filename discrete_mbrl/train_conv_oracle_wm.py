"""
Caveat 3 (deeper): Spatial ConvNet transition model — same offline-to-convergence
recipe as Phase E2, different architecture class entirely.

The MLP transition model in the paper flattens the (D, H, W) latent into a
single vector and applies dense layers. A ConvNet transition model preserves
spatial structure and applies 3x3 convs over the latent grid — a natural
inductive bias for grid-world dynamics.

If the gap closure of Phase E2 also holds with this architecturally-distinct
model, the dissociation is not specific to MLPs at all; it is a property of
the joint training procedure.

VAE-only first round: encoder output is (B, D, H, W) continuous; ConvNet
operates directly. VQ extension would need straight-through to codebook indices
on the output.

Usage:
    python train_conv_oracle_wm.py --model_path src.pt --output_path conv.pt
"""
import argparse
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(__file__))

from env_helpers import make_env  # noqa: E402

from analyze_wm_multistep import load_checkpoint  # noqa: E402

from analyze_wm_phaseB_e8 import (  # noqa: E402
    build_mlp, encode_flat,
)


# ── Spatial ConvNet transition model (continuous, drop-in for ContinuousTransitionModel) ──

class SpatialConvTransitionModel(nn.Module):
    """Spatial ConvNet replacement for the flat MLP transition model.

    Input: flat (B, D*H*W) continuous latent (compatible with the analyzer's
    free-running rollout convention).
    Action conditioning: action embedding tiled to (B, hidden, H, W) and
    concatenated as extra channels.
    Output: same shape as input — flat (B, D*H*W).
    Reward + gamma heads operate on a globally-pooled feature.
    """

    def __init__(self, encoder_out_shape, n_actions, hidden=64, depth=3):
        super().__init__()
        D, H, W = encoder_out_shape
        self.D, self.H, self.W = D, H, W
        self.input_dim = D * H * W
        self.act_dim = n_actions
        self.act_dtype = torch.long

        self.action_embed = nn.Embedding(n_actions, hidden)

        layers = []
        in_c = D + hidden
        for i in range(depth - 1):
            layers += [nn.Conv2d(in_c, hidden, kernel_size=3, padding=1), nn.ReLU()]
            in_c = hidden
        # Final layer projects back to D channels
        layers.append(nn.Conv2d(in_c, D, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

        # Reward + discount heads on globally-pooled output
        self.reward_head = nn.Sequential(
            nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 1))
        self.gamma_head = nn.Sequential(
            nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 1))

    def prepare_acts(self, acts):
        if self.act_dtype == torch.long:
            return F.one_hot(acts, self.act_dim).float()
        return acts

    def forward(self, x, acts, return_logits=False):
        """x: (B, D*H*W) flat. Returns (states_flat, reward, gamma)."""
        B = x.shape[0]
        x_sp = x.view(B, self.D, self.H, self.W)
        a_emb = self.action_embed(acts.long())  # (B, hidden)
        a_emb_sp = a_emb[:, :, None, None].expand(-1, -1, self.H, self.W)
        h = torch.cat([x_sp, a_emb_sp], dim=1)  # (B, D+hidden, H, W)
        out_sp = self.net(h)
        out_flat = out_sp.view(B, -1)
        pooled = out_sp.mean(dim=[2, 3])  # (B, D)
        reward = self.reward_head(pooled)
        gamma = torch.sigmoid(self.gamma_head(pooled))
        return out_flat, reward, gamma


# ── Buffer (continuous) ─────────────────────────────────────────────────

def collect_buffer(env, ae_model, policy, n_random_eps, n_policy_eps, max_steps, device):
    z_in, z_out, actions, rewards, gammas = [], [], [], [], []
    n_total = n_random_eps + n_policy_eps
    for ep in range(n_total):
        use_policy = (ep >= n_random_eps) and (policy is not None)
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs_t = torch.from_numpy(obs).float()
        z_t = encode_flat(ae_model, obs_t, device, 'continuous').squeeze(0).cpu().numpy()
        for step in range(max_steps):
            if use_policy:
                with torch.no_grad():
                    a = int(policy(torch.from_numpy(z_t).unsqueeze(0).to(device)).argmax(dim=-1).item())
            else:
                a = env.action_space.sample()
            sr = env.step(a)
            obs_next = sr[0] if isinstance(sr, tuple) else sr
            r = sr[1] if len(sr) >= 2 else 0.0
            done = sr[2] if len(sr) >= 3 else False
            obs_t = torch.from_numpy(obs_next).float()
            z_tp1 = encode_flat(ae_model, obs_t, device, 'continuous').squeeze(0).cpu().numpy()
            z_in.append(z_t)
            z_out.append(z_tp1)
            actions.append(a)
            rewards.append(float(r))
            gammas.append(0.0 if done else 0.99)
            z_t = z_tp1
            if done:
                break
        if (ep + 1) % 50 == 0:
            print(f'  buffer {ep+1}/{n_total} eps  N={len(actions)}')
    return {
        'z_in':    np.stack(z_in).astype(np.float32),
        'z_out':   np.stack(z_out).astype(np.float32),
        'actions': np.array(actions, dtype=np.int64),
        'rewards': np.array(rewards, dtype=np.float32),
        'gammas':  np.array(gammas, dtype=np.float32),
    }


class TransDataset(Dataset):
    def __init__(self, buf):
        self.z_in    = buf['z_in']
        self.actions = buf['actions']
        self.z_out   = buf['z_out']
        self.rewards = buf['rewards']
        self.gammas  = buf['gammas']

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        return self.z_in[i], self.actions[i], self.z_out[i], self.rewards[i], self.gammas[i]


def train(buf, model, device, epochs, batch_size, lr):
    model.train()
    ds = TransDataset(buf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        ep_state, ep_r, ep_g, n = 0.0, 0.0, 0.0, 0
        for z_in, acts, z_out, r_t, g_t in loader:
            z_in = z_in.float().to(device)
            acts = acts.long().to(device)
            z_out = z_out.float().to(device)
            r_t = r_t.float().to(device).unsqueeze(-1)
            g_t = g_t.float().to(device).unsqueeze(-1)
            preds, r_pred, g_pred = model(z_in, acts)
            loss_state = F.mse_loss(preds, z_out)
            loss_r = F.mse_loss(r_pred, r_t)
            loss_g = F.mse_loss(g_pred, g_t)
            loss = loss_state + loss_r + loss_g
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_state += float(loss_state.item()) * z_in.shape[0]
            ep_r     += float(loss_r.item()) * z_in.shape[0]
            ep_g     += float(loss_g.item()) * z_in.shape[0]
            n += z_in.shape[0]
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f'  ep {ep+1:3d}/{epochs}  state_mse={ep_state/n:.5f}  r_mse={ep_r/n:.5f}  g_mse={ep_g/n:.5f}')
    model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--n_random_eps', type=int, default=300)
    parser.add_argument('--n_policy_eps', type=int, default=300)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--conv_hidden', type=int, default=64)
    parser.add_argument('--conv_depth', type=int, default=3)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f'Loading {args.model_path}')
    ae_model, _trans_orig, trans_kind, env, margs = load_checkpoint(args.model_path, args.device)
    if trans_kind != 'continuous':
        raise NotImplementedError('Spatial ConvNet first-pass implementation handles only continuous trans models (VAE).')
    ae_model.eval()
    encoder_out_shape = getattr(ae_model, 'encoder_out_shape')
    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None) or ae_model.latent_dim // L
    n_actions = env.action_space.n
    src = torch.load(args.model_path, map_location='cpu', weights_only=False)

    policy = None
    if 'policy_state_dict' in src:
        try:
            policy = build_mlp(L * embedding_dim,
                               src['args'].get('policy_hidden', [256, 256]),
                               n_actions, src['args'].get('rl_activation', 'relu'))
            policy.load_state_dict(src['policy_state_dict'])
            policy = policy.to(args.device).eval()
            print('  policy loaded for buffer-mix')
        except Exception as e:
            print(f'  policy load failed: {e}')

    print(f'\n[1/3] Buffer collection ({args.n_random_eps} random + {args.n_policy_eps} policy)')
    t0 = time.time()
    buf = collect_buffer(env, ae_model, policy, args.n_random_eps, args.n_policy_eps,
                         args.max_steps, args.device)
    print(f'  collected {len(buf["actions"])} transitions in {time.time()-t0:.1f}s')

    print(f'\n[2/3] ConvNet training (hidden={args.conv_hidden}, depth={args.conv_depth}, {args.epochs} epochs)')
    t0 = time.time()
    model = SpatialConvTransitionModel(
        encoder_out_shape=encoder_out_shape, n_actions=n_actions,
        hidden=args.conv_hidden, depth=args.conv_depth,
    ).to(args.device)
    train(buf, model, args.device, args.epochs, args.batch_size, args.lr)
    print(f'  trained in {time.time()-t0:.1f}s')

    print(f'\n[3/3] Saving to {args.output_path}')
    out = dict(src)
    out['trans_model_state_dict'] = {k: v.cpu() for k, v in model.state_dict().items()}
    out['args']['oracle_arch'] = 'spatial_conv'
    out['args']['conv_hidden'] = args.conv_hidden
    out['args']['conv_depth'] = args.conv_depth
    out['args']['oracle_offline_epochs'] = args.epochs
    out['args']['oracle_offline_n_transitions'] = int(buf['actions'].shape[0])
    out['args']['oracle_source_ckpt'] = args.model_path
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(out, args.output_path)
    print('  saved.  Note: analyzer must rebuild trans_model with this architecture.')


if __name__ == '__main__':
    main()
