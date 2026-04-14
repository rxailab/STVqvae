#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data_helpers import prepare_dataloaders
from model_construction import construct_ae_model
from training_helpers import make_argparser, process_args


def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def aggregate_metrics(metric_list):
    out = {}
    if not metric_list:
        return out
    keys = metric_list[0].keys()
    for key in keys:
        out[key] = float(np.mean([to_float(m[key]) for m in metric_list]))
    return out


def main():
    parser = make_argparser()
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=0,
                        help="Optional cap on number of test batches (0 = full test set).")
    args = parser.parse_args()
    args = process_args(args)

    train_loader, test_loader, _ = prepare_dataloaders(
        args.env_name,
        n=args.max_transitions,
        batch_size=args.batch_size,
        preprocess=args.preprocess,
        randomize=True,
        n_preload=0,
        preload_all=True,
        extra_buffer_keys=args.extra_buffer_keys,
    )

    sample_item = train_loader.dataset[0]
    sample_obs = sample_item[0] if isinstance(sample_item, (tuple, list)) else sample_item
    if isinstance(sample_obs, torch.Tensor):
        if sample_obs.ndim == 3:
            sample_obs = sample_obs.unsqueeze(0)
        elif sample_obs.ndim >= 4:
            sample_obs = sample_obs[:1]

    model, trainer = construct_ae_model(sample_obs.shape[1:], args, load=True)
    model = model.to(args.device).eval()
    if trainer is None:
        raise RuntimeError("Selected encoder type has no trainer/loss function for validation.")

    loss_rows = []
    stat_rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if args.max_eval_batches > 0 and batch_idx >= args.max_eval_batches:
                break
            losses, stats = trainer.calculate_losses(batch, return_stats=True)
            loss_rows.append(losses)
            stat_rows.append(stats)

    results = {
        "env_name": args.env_name,
        "model_dir": args.model_dir,
        "ae_model_type": args.ae_model_type,
        "ae_model_version": args.ae_model_version,
        "ae_model_hash": args.ae_model_hash,
        "test_metrics": aggregate_metrics(loss_rows),
        "test_stats": aggregate_metrics(stat_rows),
        "n_test_batches": len(loss_rows),
    }

    print(json.dumps(results, indent=2))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
