"""Evaluate BC on validation, historical holdout, or a pre-reserved independent test."""

import argparse
from pathlib import Path
import json

import torch

from training.bc_pick_place_common import evaluate_policy, load_bundle
from training.bc_pick_place_protocol import evaluation_seeds
from training.pick_place_common import sha256, write_json


def evaluate(bundle, split, output, episodes=None, device="cpu", human=False):
    bundle, output = Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    model, normalizer, metadata = load_bundle(bundle, device)
    try:
        config = metadata["dataset_protocol"]["config"]
        seeds = evaluation_seeds(metadata, split)
        if episodes is not None:
            if type(episodes) is not int or not 0 < episodes <= len(seeds):
                raise ValueError("episodes must be within the chosen seed block")
            seeds = seeds[:episodes]

        def progress(done, total, successes):
            if done % 20 == 0 or done == total:
                print(f"{split.upper()} {done}/{total} successes={successes}", flush=True)

        report = evaluate_policy(model, normalizer, config, seeds, human=human, progress=progress)
        report.update(bundle=str(bundle.resolve()), split=split,
                      epoch=metadata["epoch"], deterministic=True,
                      model_sha256=metadata["sha256"]["model.zip"],
                      bundle_manifest_sha256=sha256(bundle / "manifest.json"),
                      selection_seeds=metadata.get("selection_seeds"),
                      role=metadata.get("role"))
        if split == "test":
            report.update(evaluation_protocol=metadata["evaluation_protocol"],
                          seed_disjoint_from_train_and_validation=True,
                          purpose="final_evaluation_only_not_model_selection")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, report)
        print(json.dumps(report["summary"], indent=2))
        return report
    finally:
        normalizer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path)
    source.add_argument("--run", type=Path, help="Load this run's validation-selected best bundle")
    parser.add_argument("--split", choices=("validation", "holdout", "test"), default="validation")
    parser.add_argument("--episodes", type=int, help="Default: all seeds in the chosen split")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--human", action="store_true", help="macOS: run with mjpython")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.run is not None:
        result = json.loads((args.run / "result.json").read_text())
        if result.get("status") != "completed":
            raise ValueError("--run requires completed training with a fixed validation-best model")
        bundle = Path(result["best_model"])
    else:
        bundle = args.bundle
    evaluate(bundle, args.split, args.output, args.episodes, args.device, args.human)


if __name__ == "__main__":
    main()
