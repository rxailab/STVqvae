"""Full-rigor item (C) — RL transfer curves for DK-8 variants.

Extracts running reward from existing training logs and produces
`logs/rl_transfer_curves.png` comparing v2, v5+dc, v6, VAE on DoorKey-8x8.
"""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, 'logs')
OUT = os.path.join(LOG_DIR, 'rl_transfer_curves.png')

# (label, log-file glob, colour)
RUNS = [
    ('v2 (baseline VQ)',       'exp*v2*.log',            '#888888'),
    ('v5 + dead-code restart', 'exp70_doorkey_v5_deadcode.log', '#1f77b4'),
    ('v6 (RGB+coord+trunk)',   'exp38_v6enc.log',        '#2ca02c'),
    ('VAE baseline (no VQ)',   'vae_baseline_doorkey_v6enc.log', '#d62728'),
]

BEST_RE  = re.compile(r'New best average reward:\s+([\d.]+)')
ITER_RE  = re.compile(r'(\d+)/\d+\s+\[')

def extract(log_path):
    """Return (iters, rewards) from log — tracks running best reward per iter."""
    if not os.path.exists(log_path):
        return None, None
    with open(log_path, 'rb') as f:
        data = f.read().decode('utf-8', 'ignore')
    # Walk character-by-character is wasteful; instead find all iter markers and
    # all "New best" events and align them by file-offset.
    iter_hits   = [(m.start(), int(m.group(1))) for m in ITER_RE.finditer(data)]
    best_hits   = [(m.start(), float(m.group(1))) for m in BEST_RE.finditer(data)]
    if not iter_hits or not best_hits:
        return None, None
    # For each best event use the nearest preceding iter as x-coord.
    iters, rewards, run_best = [], [], 0.0
    bi = 0
    for pos, rew in best_hits:
        while bi + 1 < len(iter_hits) and iter_hits[bi + 1][0] <= pos:
            bi += 1
        it = iter_hits[bi][1]
        run_best = max(run_best, rew)
        iters.append(it)
        rewards.append(run_best)
    return np.array(iters), np.array(rewards)

plt.figure(figsize=(7, 4.5))
for label, pattern, colour in RUNS:
    matches = sorted(glob.glob(os.path.join(LOG_DIR, pattern)))
    if not matches:
        print(f'  {label:30s}  no log found for {pattern}')
        continue
    log_path = matches[0]
    iters, rew = extract(log_path)
    if iters is None:
        print(f'  {label:30s}  could not parse {os.path.basename(log_path)}')
        continue
    print(f'  {label:30s}  {os.path.basename(log_path):45s}  '
          f'n_events={len(iters):4d}  best={rew.max():.3f}')
    plt.plot(iters, rew, label=label, color=colour, lw=1.8)

plt.xlabel('PPO iteration')
plt.ylabel('Running best avg reward')
plt.title('DoorKey-8x8 — RL transfer across encoder / VQ variants')
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f'\nSaved {OUT}')
