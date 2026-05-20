"""
Phase E2: Train an "oracle" world model offline-to-convergence on a frozen encoder.

The paper's WM is trained jointly with the encoder via online PPO, so the WM
sees a non-stationary distribution (encoder drifts, policy drifts). A reviewer
might argue: "your WM fails not because the latent is unusable but because the
joint-training procedure is broken; train a fresh transition model offline on
a frozen encoder to convergence and the dissociation will close."

This script tests that hypothesis directly.

Pipeline:
  1. Load a paper checkpoint; freeze its encoder + policy.
  2. Collect a 50k-transition buffer mixing random and policy-driven rollouts.
  3. Train a fresh DiscreteTransitionModel (or ContinuousTransitionModel) from
     scratch on that buffer for many epochs with the standard online loss.
  4. Save the new transition model alongside the original encoder as a
     checkpoint compatible with analyze_wm_multistep.py.

Resulting ckpt has the SAME encoder as the source ckpt and a transition model
trained offline-to-convergence on a fixed distribution. Running the analyzer on
it isolates the encoder representation from the joint-training procedure.

Usage:
    python train_oracle_wm.py \
        --model_path path/to/source_best_model.pt \
        --n_random_eps 250 --n_policy_eps 250 \
        --epochs 100 \
        --output_path path/to/oracle_best_model.pt
"""
import argparse
import copy
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

from analyze_wm_multistep import load_checkpoint, encode_codes  # noqa: E402

from analyze_wm_phaseB_e8 import (  # noqa: E402
    build_mlp, encode_flat,
)


# ── Buffer collection ────────────────────────────────────────────────────

def collect_buffer(env, ae_model, policy, n_random_eps, n_policy_eps, max_steps,
                   device, trans_kind):
    """Collect a buffer of (z_t, code_t, action_t, code_{t+1}, z_{t+1}) tuples
    under a 50/50 mix of random actions and policy actions."""
    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None)
    if embedding_dim is None:
        embedding_dim = ae_model.latent_dim // L
    n_total = n_random_eps + n_policy_eps

    z_in, z_out, codes_in, codes_out, actions = [], [], [], [], []

    for ep_idx in range(n_total):
        use_policy = (ep_idx >= n_random_eps) and (policy is not None)
        obs = env.reset()
        obs = obs[0] if isinstance(obs, tuple) else obs
        obs_t = torch.from_numpy(obs).float()
        z_t = encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy()
        if trans_kind == 'discrete':
            c_t = encode_codes(ae_model, obs_t, device).squeeze(0).cpu().numpy()
        else:
            c_t = None

        for step in range(max_steps):
            if use_policy:
                with torch.no_grad():
                    a = int(policy(torch.from_numpy(z_t).unsqueeze(0).to(device))
                            .argmax(dim=-1).item())
            else:
                a = env.action_space.sample()

            sr = env.step(a)
            obs_next = sr[0] if isinstance(sr, tuple) else sr
            done = sr[2] if len(sr) >= 3 else False
            obs_t = torch.from_numpy(obs_next).float()
            z_tp1 = encode_flat(ae_model, obs_t, device, trans_kind).squeeze(0).cpu().numpy()
            if trans_kind == 'discrete':
                c_tp1 = encode_codes(ae_model, obs_t, device).squeeze(0).cpu().numpy()
            else:
                c_tp1 = None

            z_in.append(z_t)
            z_out.append(z_tp1)
            actions.append(a)
            if trans_kind == 'discrete':
                codes_in.append(c_t)
                codes_out.append(c_tp1)

            z_t = z_tp1
            c_t = c_tp1
            if done:
                break

        if (ep_idx + 1) % 50 == 0:
            print(f'  buffer: {ep_idx + 1}/{n_total} eps, {len(actions)} transitions')

    buf = {
        'z_in':      np.stack(z_in).astype(np.float32),
        'z_out':     np.stack(z_out).astype(np.float32),
        'actions':   np.array(actions, dtype=np.int64),
    }
    if trans_kind == 'discrete':
        buf['codes_in']  = np.stack(codes_in).astype(np.int64)
        buf['codes_out'] = np.stack(codes_out).astype(np.int64)
    print(f'  buffer total: {len(actions)} transitions  (random {n_random_eps} eps, policy {n_policy_eps} eps)')
    return buf


# ── Offline trainer ────────────────────────────────────────────────────

class TransDataset(Dataset):
    def __init__(self, buf, trans_kind):
        self.kind = trans_kind
        self.actions = buf['actions']
        self.z_in    = buf['z_in']
        self.z_out   = buf['z_out']
        if trans_kind == 'discrete':
            self.codes_in  = buf['codes_in']
            self.codes_out = buf['codes_out']

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, i):
        if self.kind == 'discrete':
            return (self.codes_in[i], self.actions[i], self.codes_out[i])
        return (self.z_in[i], self.actions[i], self.z_out[i])


def train_oracle_discrete(buf, env, ae_model, n_codes, embedding_dim,
                          hidden, depth, device, epochs, batch_size, lr):
    from shared.models.transition_models import DiscreteTransitionModel
    L = ae_model.n_latent_embeds

    trans_model = DiscreteTransitionModel(
        L, n_codes, embedding_dim,
        env.action_space,
        hidden_sizes=[hidden] * depth,
        stochastic=False,
        stoch_hidden_sizes=[256, 256],
        discretizer_hidden_sizes=[256],
        use_soft_embeds=False,
        return_logits=False,
    ).to(device)
    trans_model.train()

    ds = TransDataset(buf, 'discrete')
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=0)
    opt = torch.optim.Adam(trans_model.parameters(), lr=lr)

    n_iters = 0
    for ep in range(epochs):
        ep_loss, ep_acc, ep_n = 0.0, 0.0, 0
        for codes_in, acts, codes_out in loader:
            codes_in  = codes_in.long().to(device)
            acts      = acts.long().to(device)
            codes_out = codes_out.long().to(device)
            logits, _, _ = trans_model(codes_in, acts, return_logits=True)
            # logits: (B, n_codes, L); target: (B, L) long
            loss = F.cross_entropy(logits, codes_out)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * codes_in.shape[0]
            ep_acc  += float((logits.argmax(dim=1) == codes_out).float().mean().item()) * codes_in.shape[0]
            ep_n += codes_in.shape[0]
            n_iters += 1
        print(f'  ep {ep+1:3d}/{epochs}  loss={ep_loss/ep_n:.4f}  acc={ep_acc/ep_n:.4f}')

    trans_model.eval()
    return trans_model


def train_oracle_continuous(buf, env, ae_model, embedding_dim,
                            hidden, depth, device, epochs, batch_size, lr):
    from shared.models.transition_models import ContinuousTransitionModel
    L = ae_model.n_latent_embeds
    input_dim = L * embedding_dim

    trans_model = ContinuousTransitionModel(
        input_dim, env.action_space,
        hidden_sizes=[hidden] * depth,
    ).to(device)
    trans_model.train()

    ds = TransDataset(buf, 'continuous')
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                        num_workers=0)
    opt = torch.optim.Adam(trans_model.parameters(), lr=lr)

    for ep in range(epochs):
        ep_loss, ep_n = 0.0, 0
        for z_in, acts, z_out in loader:
            z_in  = z_in.float().to(device)
            acts  = acts.long().to(device)
            z_out = z_out.float().to(device)
            preds, _, _ = trans_model(z_in, acts)
            loss = F.mse_loss(preds, z_out)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * z_in.shape[0]
            ep_n += z_in.shape[0]
        print(f'  ep {ep+1:3d}/{epochs}  mse={ep_loss/ep_n:.5f}')

    trans_model.eval()
    return trans_model


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True,
                        help='Source checkpoint; encoder+policy taken from here.')
    parser.add_argument('--output_path', required=True,
                        help='Where to save the oracle ckpt.')
    parser.add_argument('--n_random_eps', type=int, default=300)
    parser.add_argument('--n_policy_eps', type=int, default=300)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--trans_hidden', type=int, default=None,
                        help='Override transition-model hidden width (default: take from src args).')
    parser.add_argument('--trans_depth', type=int, default=None,
                        help='Override transition-model depth (default: take from src args).')
    args = parser.parse_args()

    print(f'Loading source ckpt: {args.model_path}')
    ae_model, _trans_orig, trans_kind, env, margs = load_checkpoint(args.model_path, args.device)
    ae_model.eval()
    L = ae_model.n_latent_embeds
    embedding_dim = getattr(ae_model, 'embedding_dim', None)
    if embedding_dim is None:
        embedding_dim = ae_model.latent_dim // L
    n_codes = getattr(margs, 'codebook_size', 64)
    hidden = args.trans_hidden if args.trans_hidden is not None else getattr(margs, 'trans_hidden', 256)
    depth  = args.trans_depth  if args.trans_depth  is not None else getattr(margs, 'trans_depth', 3)
    print(f'  trans_model architecture: hidden={hidden}, depth={depth}')

    # Load policy if available (for buffer-mix)
    policy = None
    src = torch.load(args.model_path, map_location='cpu', weights_only=False)
    if 'policy_state_dict' in src:
        try:
            policy_hidden = src['args'].get('policy_hidden', [256, 256])
            n_actions = env.action_space.n
            policy = build_mlp(L * embedding_dim, policy_hidden, n_actions, 'relu')
            policy.load_state_dict(src['policy_state_dict'])
            policy = policy.to(args.device).eval()
            print(f'  policy loaded for buffer-mix (input={L*embedding_dim}, n_actions={n_actions})')
        except Exception as e:
            print(f'  policy load failed: {e}; using random-only buffer')
            policy = None

    print(f'\n[1/3] Collecting buffer ({args.n_random_eps} random + {args.n_policy_eps} policy eps)')
    t0 = time.time()
    buf = collect_buffer(env, ae_model, policy, args.n_random_eps, args.n_policy_eps,
                         args.max_steps, args.device, trans_kind)
    print(f'  buffer collected in {time.time()-t0:.1f}s')

    print(f'\n[2/3] Training oracle WM offline ({args.epochs} epochs, batch={args.batch_size}, lr={args.lr})')
    t0 = time.time()
    if trans_kind == 'discrete':
        trans_oracle = train_oracle_discrete(
            buf, env, ae_model, n_codes, embedding_dim,
            hidden, depth, args.device, args.epochs, args.batch_size, args.lr)
    else:
        trans_oracle = train_oracle_continuous(
            buf, env, ae_model, embedding_dim,
            hidden, depth, args.device, args.epochs, args.batch_size, args.lr)
    print(f'  trained in {time.time()-t0:.1f}s')

    print(f'\n[3/3] Saving oracle ckpt to {args.output_path}')
    out_ckpt = dict(src)
    out_ckpt['trans_model_state_dict'] = {k: v.cpu() for k, v in trans_oracle.state_dict().items()}
    out_ckpt['args']['oracle_offline_epochs'] = args.epochs
    out_ckpt['args']['oracle_offline_n_transitions'] = int(buf['actions'].shape[0])
    out_ckpt['args']['oracle_source_ckpt'] = args.model_path
    # Persist architecture override so the analyzer rebuilds with matching shape.
    out_ckpt['args']['trans_hidden'] = hidden
    out_ckpt['args']['trans_depth'] = depth
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(out_ckpt, args.output_path)
    print(f'  saved.  Now run analyze_wm_multistep.py on it to compare.')


if __name__ == '__main__':
    main()
