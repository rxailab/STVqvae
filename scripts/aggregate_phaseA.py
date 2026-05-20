"""Flatten Phase A per-ckpt JSONs into one wide CSV for plotting.

Output: <out_dir>/phaseA_summary.csv -- one row per (checkpoint, horizon).

Columns include:
  checkpoint, encoder_version, ae_model_type, trans_model_type, env_name, k,
  probe_<class>             # per-class probe recall (z_0)
  wm_exact_<class>          # original metric (paper §3.3)
  wm_probe_<class>          # E1 -- shared probe applied to z_hat_k
  wm_class_<class>          # E2 -- VQ code->class lookup
  wm_centroid_<class>       # E3 -- VAE class-centroid arg-max-cosine
  wm_swap_<class>           # E6 -- VAE same-class spatial swap
  pearson_exact / pearson_probe / pearson_class / pearson_centroid / pearson_swap

Confusion matrices stay in the JSONs (not in the CSV).
"""
import csv
import json
import os
import sys
from glob import glob


TASK_CRITICAL = ['wall', 'door', 'key', 'goal', 'agent']
HORIZONS = [1, 3, 5, 10]


def _acc_by_name(d, class_name):
    """Pull accuracy for class_name out of a {str(class_id): {acc, name, count}} dict."""
    if not d:
        return None
    for v in d.values():
        if isinstance(v, dict) and v.get('name') == class_name:
            return v.get('acc')
    return None


def main():
    if len(sys.argv) < 2:
        print('usage: aggregate_phaseA.py <phaseA_dir>', file=sys.stderr)
        sys.exit(2)
    out_dir = sys.argv[1]
    files = sorted(glob(os.path.join(out_dir, '*.json')))
    if not files:
        print(f'no JSONs under {out_dir}', file=sys.stderr)
        sys.exit(2)

    rows = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        ckpt = os.path.basename(fp).replace('.json', '')
        probe = d.get('probe_acc_per_class', {})
        wm_exact_per_h = d.get('wm_acc_per_class_per_horizon', {})
        ext_per_h = d.get('extended_metrics_per_horizon', {})
        pearson_orig = d.get('pearson_r_per_horizon', {})

        for k in HORIZONS:
            ks = str(k)
            wm_exact = wm_exact_per_h.get(ks)
            ext = ext_per_h.get(ks, {})
            if wm_exact is None and not ext:
                continue
            row = {
                'checkpoint':       ckpt,
                'encoder_version':  d.get('encoder_version', '?'),
                'ae_model_type':    d.get('ae_model_type', '?'),
                'trans_model_type': d.get('trans_model_type', '?'),
                'env':              d.get('env_name', '?'),
                'k':                k,
            }
            for cls in TASK_CRITICAL:
                row[f'probe_{cls}']       = _acc_by_name(probe, cls)
                row[f'wm_exact_{cls}']    = _acc_by_name(wm_exact, cls)
                row[f'wm_probe_{cls}']    = _acc_by_name(ext.get('wm_probe_per_class'), cls)
                row[f'wm_class_{cls}']    = _acc_by_name(ext.get('wm_class_per_class'), cls)
                row[f'wm_centroid_{cls}'] = _acc_by_name(ext.get('wm_centroid_per_class'), cls)
                row[f'wm_swap_{cls}']     = _acc_by_name(ext.get('wm_swap_per_class'), cls)
            row['pearson_exact'] = pearson_orig.get(ks)
            ext_pearson = ext.get('pearson_r', {}) or {}
            for variant in ['probe', 'class', 'centroid', 'swap']:
                row[f'pearson_{variant}'] = ext_pearson.get(variant)
            rows.append(row)

    if not rows:
        print('no rows extracted', file=sys.stderr)
        sys.exit(2)

    csv_path = os.path.join(out_dir, 'phaseA_summary.csv')
    fieldnames = list(rows[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f'wrote {csv_path}  ({len(rows)} rows from {len(files)} checkpoints)')


if __name__ == '__main__':
    main()
