"""
Phase I correlation analysis: tie action-dependence metrics to the §4 paper
metrics (probe accuracy, WM_k accuracy, planning rank-corr).

Joins:
  logs/phaseI_actcond_v2/actcond_v2_summary.csv   (one row per ckpt)
  logs/phaseA/phaseA_summary.csv                  (4 rows per ckpt, k∈{1,3,5,10})

For each (ckpt, k) pair, reports Pearson/Spearman correlation between:
  - overall action-dep metrics vs overall WM_k metrics
  - per-class action_dep_ratio[c] vs per-class WM_k accuracy[c]
    (pooled scatter: one point per (ckpt, class))
  - action metrics vs probe accuracy (expected: NO correlation,
    contrast with the WM correlations).

Output: prints a tidy table + saves logs/phaseI_actcond_v2/correlations.csv.
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np

REPO = '/mmfs1/storage/users/xiar3/exp/STVqvae'
ACT_CSV = f'{REPO}/logs/phaseI_actcond_v2/actcond_v2_summary.csv'
PHASEA_CSV = f'{REPO}/logs/phaseA/phaseA_summary.csv'
OUT_CSV = f'{REPO}/logs/phaseI_actcond_v2/correlations.csv'

CLASSES = ['wall', 'door', 'key', 'goal', 'agent']


def pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float('nan'), 0
    x, y = x[m], y[m]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float('nan'), int(m.sum())
    return float(np.corrcoef(x, y)[0, 1]), int(m.sum())


def spearman(x, y):
    """Pearson on ranks."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float('nan'), 0
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float('nan'), int(m.sum())
    return float(np.corrcoef(rx, ry)[0, 1]), int(m.sum())


def main():
    # Load action-dep summary (one row per ckpt) ────────────────────────────
    act = {}
    with open(ACT_CSV) as f:
        for r in csv.DictReader(f):
            ckpt = r['checkpoint']
            act[ckpt] = r
    print(f'Loaded {len(act)} ckpts from {os.path.basename(ACT_CSV)}')

    # Load phaseA (4 rows per ckpt) ──────────────────────────────────────────
    phasea = defaultdict(dict)   # phasea[ckpt][k] = row
    with open(PHASEA_CSV) as f:
        for r in csv.DictReader(f):
            phasea[r['checkpoint']][int(r['k'])] = r
    print(f'Loaded {len(phasea)} ckpts × k from {os.path.basename(PHASEA_CSV)}')

    # Build joined rows ──────────────────────────────────────────────────────
    rows = []
    for ckpt, krows in phasea.items():
        if ckpt not in act:
            continue
        a = act[ckpt]
        for k, pr in krows.items():
            row = {'ckpt': ckpt, 'k': k, 'env': pr.get('env')}
            for fld in ('action_dep_ratio', 'action_residual_rank',
                        'epsilon_action_collapse', 'state_change_norm',
                        'action_diff_norm', 'reward_action_var',
                        'effective_action_rank'):
                row[fld] = float(a[fld]) if a.get(fld) else float('nan')
            for c in CLASSES:
                row[f'pc_ratio_{c}'] = (float(a.get(f'pc_ratio_{c}', 'nan'))
                                       if a.get(f'pc_ratio_{c}') else float('nan'))
                row[f'probe_{c}'] = (float(pr[f'probe_{c}'])
                                    if pr.get(f'probe_{c}') else float('nan'))
                row[f'wm_exact_{c}'] = (float(pr[f'wm_exact_{c}'])
                                       if pr.get(f'wm_exact_{c}') else float('nan'))
            rows.append(row)
    print(f'Joined {len(rows)} (ckpt × k) rows.')

    # ── Correlation analyses ───────────────────────────────────────────────
    results = []

    # (1) Overall action-dep metrics vs overall WM_k accuracy (mean over classes).
    #     "WM_k overall" = mean of wm_exact_{wall,door,key,goal,agent}.
    print('\n=== (1) overall action-dep vs mean WM_k accuracy (per-k) ===')
    print(f"{'k':>2s} {'metric':>22s} {'r_pearson':>10s} {'r_spearman':>10s} {'n':>4s}")
    for k in (1, 3, 5, 10):
        sub = [r for r in rows if r['k'] == k]
        wm_mean = [np.nanmean([r[f'wm_exact_{c}'] for c in CLASSES]) for r in sub]
        for mname in ('action_dep_ratio', 'action_residual_rank',
                      'epsilon_action_collapse', 'effective_action_rank'):
            x = [r[mname] for r in sub]
            rp, n = pearson(x, wm_mean); rs, _ = spearman(x, wm_mean)
            print(f'{k:>2d} {mname:>22s} {rp:>+10.3f} {rs:>+10.3f} {n:>4d}')
            results.append({'analysis': '1_overall', 'k': k, 'metric': mname,
                            'r_pearson': rp, 'r_spearman': rs, 'n': n})

    # (2) Per-class action_dep_ratio[c] vs per-class WM_k[c] — pooled.
    print('\n=== (2) per-class action_dep_ratio[c] vs WM_k[c]  (pooled over ckpts+classes) ===')
    print(f"{'k':>2s} {'r_pearson':>10s} {'r_spearman':>10s} {'n':>5s}")
    for k in (1, 3, 5, 10):
        sub = [r for r in rows if r['k'] == k]
        xs, ys = [], []
        for r in sub:
            for c in CLASSES:
                if not np.isnan(r[f'pc_ratio_{c}']) and not np.isnan(r[f'wm_exact_{c}']):
                    xs.append(r[f'pc_ratio_{c}']); ys.append(r[f'wm_exact_{c}'])
        rp, n = pearson(xs, ys); rs, _ = spearman(xs, ys)
        print(f'{k:>2d} {rp:>+10.3f} {rs:>+10.3f} {n:>5d}')
        results.append({'analysis': '2_per_class_pooled', 'k': k, 'metric': 'pc_ratio_vs_wm',
                        'r_pearson': rp, 'r_spearman': rs, 'n': n})

    # (3) Contrast: action-dep vs probe accuracy (should be uncorrelated).
    print('\n=== (3) CONTRAST: action-dep vs probe accuracy  (k=1 row, expect near zero) ===')
    sub = [r for r in rows if r['k'] == 1]
    probe_mean = [np.nanmean([r[f'probe_{c}'] for c in CLASSES]) for r in sub]
    for mname in ('action_dep_ratio', 'action_residual_rank',
                  'epsilon_action_collapse'):
        x = [r[mname] for r in sub]
        rp, n = pearson(x, probe_mean); rs, _ = spearman(x, probe_mean)
        print(f'{mname:>22s}  vs probe  r_pearson={rp:+.3f}  r_spearman={rs:+.3f}  n={n}')
        results.append({'analysis': '3_vs_probe_contrast', 'k': 1, 'metric': mname,
                        'r_pearson': rp, 'r_spearman': rs, 'n': n})

    # (4) Per-class on each (env, encoder) cell, k=1, to show per-cohort structure.
    print('\n=== (4) Per-(env,encoder) per-class action_dep_ratio vs WM_1[c]  (k=1) ===')
    def encoder_of(ckpt):
        if 'cb1024' in ckpt: return 'v6_cb1024'
        if 'cb256_thr1' in ckpt: return 'v6_cb256'
        if 'cb512_thr2' in ckpt: return 'v6_cb512'
        if 'v5dc' in ckpt or 'v5enc_deadcode' in ckpt: return 'v5dc'
        if '_v5enc_' in ckpt or '_v5_s' in ckpt: return 'v5'
        if '_vae_' in ckpt: return 'vae'
        if '_v2_' in ckpt: return 'v2'
        if '_v6_' in ckpt or 'v6enc' in ckpt or 'v6aenc' in ckpt or 'v6benc' in ckpt or 'v6cenc' in ckpt: return 'v6'
        if '_v9_' in ckpt or 'v9enc' in ckpt: return 'v9'
        return '?'
    cells = defaultdict(list)
    for r in [x for x in rows if x['k'] == 1]:
        cells[(r['env'], encoder_of(r['ckpt']))].append(r)
    print(f"{'env':16s} {'enc':12s} {'n_ckpt':>6s} {'r_pearson':>10s} {'r_spearman':>10s} {'n_pts':>5s}")
    for (env, enc), grp in sorted(cells.items()):
        xs, ys = [], []
        for r in grp:
            for c in CLASSES:
                if not np.isnan(r[f'pc_ratio_{c}']) and not np.isnan(r[f'wm_exact_{c}']):
                    xs.append(r[f'pc_ratio_{c}']); ys.append(r[f'wm_exact_{c}'])
        rp, n = pearson(xs, ys); rs, _ = spearman(xs, ys)
        print(f'{env:16s} {enc:12s} {len(grp):>6d} {rp:>+10.3f} {rs:>+10.3f} {n:>5d}')
        results.append({'analysis': '4_per_cell', 'k': 1, 'metric': f'{env}__{enc}',
                        'r_pearson': rp, 'r_spearman': rs, 'n': n})

    # Save results CSV ──────────────────────────────────────────────────────
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['analysis', 'k', 'metric', 'r_pearson', 'r_spearman', 'n'])
        w.writeheader()
        for r in results: w.writerow(r)
    print(f'\nSaved {OUT_CSV}')


if __name__ == '__main__':
    main()
