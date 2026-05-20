"""Within-ckpt Spearman: does action_dep_ratio rank the classes in the same
order as WM_1 accuracy, ckpt by ckpt? This is the right test for "action-dep
predicts which class the WM gets right" — the pooled correlation in
phaseI_correlate.py mixes encoder-quality variation with class variation.
"""
import csv
import numpy as np
from collections import defaultdict

REPO = '/mmfs1/storage/users/xiar3/exp/STVqvae'
ACT = f'{REPO}/logs/phaseI_actcond_v2/actcond_v2_summary.csv'
PHA = f'{REPO}/logs/phaseA/phaseA_summary.csv'

CLASSES = ['wall', 'door', 'key', 'goal', 'agent']

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


def spearman(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return float('nan')
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m]))
    if rx.std() < 1e-12 or ry.std() < 1e-12: return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    act = {}
    with open(ACT) as f:
        for r in csv.DictReader(f):
            act[r['checkpoint']] = r

    phasea = defaultdict(dict)
    with open(PHA) as f:
        for r in csv.DictReader(f):
            phasea[r['checkpoint']][int(r['k'])] = r

    # Within-ckpt rank correlation, k=1 ───────────────────────────────────
    per_ckpt = []
    for ckpt in act:
        if ckpt not in phasea or 1 not in phasea[ckpt]: continue
        a = act[ckpt]; pr = phasea[ckpt][1]
        xs = [float(a.get(f'pc_ratio_{c}', 'nan') or 'nan') for c in CLASSES]
        ys = [float(pr.get(f'wm_exact_{c}', 'nan') or 'nan') for c in CLASSES]
        per_ckpt.append({'ckpt': ckpt, 'env': pr.get('env'),
                         'enc': encoder_of(ckpt),
                         'spearman': spearman(xs, ys),
                         'overall_dep': float(a['action_dep_ratio']),
                         'eps_coll': float(a['epsilon_action_collapse']),
                         'resid_rank': float(a['action_residual_rank'])})

    valid = [r for r in per_ckpt if np.isfinite(r['spearman'])]
    arr = np.array([r['spearman'] for r in valid])
    print(f'Within-ckpt Spearman (per-class action_dep_ratio vs WM_1 across 5 classes)')
    print(f'  Pooled over {len(valid)} ckpts: mean = {arr.mean():+.3f}, std = {arr.std():.3f}')
    print(f'  P(r > 0) = {(arr > 0).mean():.2f}, P(r > 0.5) = {(arr > 0.5).mean():.2f}')

    # By cohort
    print(f"\n{'env':25s} {'enc':12s} {'n_ckpt':>6s} {'mean_r':>8s} {'std_r':>8s} {'P(r>0)':>8s}")
    groups = defaultdict(list)
    for r in valid: groups[(r['env'], r['enc'])].append(r['spearman'])
    for (env, enc), rs in sorted(groups.items()):
        rs = np.array(rs)
        print(f'{env:25s} {enc:12s} {len(rs):>6d} {rs.mean():+8.3f} {rs.std():>8.3f} {(rs > 0).mean():>+8.2f}')

    # ── Alternative: WM_1 accuracy split by collapsed vs healthy ────────
    print('\n— Compare WM_1 mean by action-dep regime —')
    health = {'collapsed (eps_coll≥0.7)': [], 'healthy (eps_coll<0.3)': [], 'middle': []}
    for r in valid:
        wm_mean = np.nanmean([float(phasea[r['ckpt']][1][f'wm_exact_{c}'] or 'nan') for c in CLASSES])
        if r['eps_coll'] >= 0.7: health['collapsed (eps_coll≥0.7)'].append(wm_mean)
        elif r['eps_coll'] < 0.3: health['healthy (eps_coll<0.3)'].append(wm_mean)
        else: health['middle'].append(wm_mean)
    for k, vs in health.items():
        if vs:
            vs = np.array([v for v in vs if np.isfinite(v)])
            print(f'  {k:35s} n={len(vs):>3d}  WM_1 mean = {vs.mean():.3f} ± {vs.std():.3f}')


if __name__ == '__main__':
    main()
