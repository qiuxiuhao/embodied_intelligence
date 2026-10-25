"""Train a Day15 SB3 actor with BC; no PPO updates or holdout-based selection."""

import argparse
from copy import deepcopy
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch

from training.bc_pick_place_common import (
    DEFAULT_BC_CONFIG, action_losses, actor_mean, build_model, evaluate_policy,
    load_bc_config, load_demonstrations, make_normalizer, offline_metrics,
    sampling_probabilities, save_bundle, validate_bc_config)
from training.bc_pick_place_protocol import (
    DEFAULT_EVALUATION_CONFIG, bind_evaluation_protocol, load_evaluation_config,
    parse_eval_episodes, selection_seeds)
from training.pick_place_common import ROOT, sha256, write_json


def train(dataset, output, config, evaluation_config=None):
    config = deepcopy(config)
    validate_bc_config(config)
    output, dataset = Path(output), Path(dataset)
    if output.exists():
        raise FileExistsError(f"Choose a new run directory: {output}")
    torch.set_num_threads(config["torch_threads"])
    protocol, splits, audit = load_demonstrations(dataset)
    selected_seeds = selection_seeds(protocol["config"], config["eval_episodes"])
    evaluation_protocol = bind_evaluation_protocol(protocol["config"],
        load_evaluation_config() if evaluation_config is None else evaluation_config)
    # The test block is reserved now, never passed to training or selection.
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [ROOT / name for name in (
        "training/bc_pick_place_common.py", "training/train_bc_pick_place.py",
        "training/evaluate_bc_pick_place.py", "training/selective_normalize.py",
        "training/bc_pick_place_protocol.py")]
    run_metadata = {
        "bc_config": config, "dataset_protocol": protocol,
        "dataset_path": str(dataset.resolve()),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "selection_seeds": selected_seeds, "selection_episodes": len(selected_seeds),
        "evaluation_protocol": evaluation_protocol, "test_used_during_training": False,
        "reserved_holdout_used": False,
        "selection_rule": "closed_loop_success_then_validation_phase_balanced_loss",
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths},
        "versions": {name: version(name) for name in ("torch", "numpy", "stable-baselines3", "mujoco")},
    }
    write_json(output / "config.json", run_metadata)
    write_json(output / "dataset_audit.json", audit)
    print(f"SELECTION validation episodes={len(selected_seeds)}; "
          f"test episodes={evaluation_protocol['test']['episodes']} reserved, unused", flush=True)
    normalizer = make_normalizer(protocol["config"], splits["train"]["observations"])
    started, epoch, updates, seen_samples = time.perf_counter(), 0, 0, 0
    model = None
    try:
        model = build_model(normalizer, config)
        normalized = {name: normalizer.normalize_obs(data["observations"])
                      for name, data in splits.items()}
        x = torch.as_tensor(normalized["train"], device=model.device)
        y = torch.as_tensor(splits["train"]["actions"], device=model.device)
        probabilities = sampling_probabilities(splits["train"]["phase_ids"], config["phase_balance"])
        rng = np.random.default_rng(config["seed"])
        parameters = [p for p in model.policy.parameters() if p.requires_grad]
        stats_count = float(normalizer.obs_rms.count)
        write_json(output / "normalization.json", {
            "fit_split": "train", "pre_action_observations": len(x),
            "running_count_including_prior": stats_count,
            "normalized_fields": normalizer.normalized_fields, "frozen": True})
        best_key, best_path = None, None

        def assess(current_epoch):
            nonlocal best_key, best_path
            offline = {split: offline_metrics(model, normalized[split], data["actions"],
                                               data["phase_ids"], config)
                       for split, data in splits.items()}
            print(f"EVAL epoch={current_epoch} starting {len(selected_seeds)} validation episodes", flush=True)

            def progress(done, total, successes):
                if done % 20 == 0 or done == total:
                    print(f"EVAL epoch={current_epoch} {done}/{total} "
                          f"successes={successes}", flush=True)

            closed = evaluate_policy(model, normalizer, protocol["config"], selected_seeds,
                                     progress=progress)
            if float(normalizer.obs_rms.count) != stats_count:
                raise RuntimeError("BC/evaluation unexpectedly updated normalization")
            key = (closed["summary"]["success_rate"], -offline["validation"]["phase_balanced_loss"])
            report = {"epoch": current_epoch, "offline": offline, "closed_loop": closed,
                      "optimizer_steps": updates, "sampled_transitions": seen_samples}
            write_json(output / f"evaluation_{current_epoch:04d}.json", report)
            if best_key is None or key > best_key:
                best_key = key
                path = output / "best_model" / f"epoch_{current_epoch:04d}"
                save_bundle(path, model, normalizer, {
                    **run_metadata, **report, "role": "validation_best"})
                best_path = str(path.resolve())
            print(f"EVAL epoch={current_epoch} success={key[0]:.1%} "
                  f"val_balanced_loss={-key[1]:.6f} "
                  f"grasp={closed['summary']['grasp_rate']:.1%} "
                  f"lift={closed['summary']['lift_rate']:.1%}", flush=True)
            return report

        last_report = assess(0)
        with (output / "epochs.jsonl").open("x") as log:
            for epoch in range(1, config["epochs"]+1):
                model.policy.set_training_mode(True)
                indices = rng.choice(len(x), size=len(x), replace=True, p=probabilities)
                total_loss = 0.
                for start in range(0, len(indices), config["batch_size"]):
                    selected = torch.as_tensor(indices[start:start+config["batch_size"]],
                                               dtype=torch.long, device=model.device)
                    prediction = actor_mean(model.policy, x[selected])
                    per_sample, _, _ = action_losses(prediction, y[selected], config)
                    loss = per_sample.mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite BC loss")
                    model.policy.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
                    model.policy.optimizer.step()
                    total_loss += float(loss.detach())*len(selected)
                    updates += 1
                    seen_samples += len(selected)
                metrics = {
                    "epoch": epoch, "sampled_train_loss": total_loss/len(x),
                    "validation": offline_metrics(model, normalized["validation"],
                        splits["validation"]["actions"], splits["validation"]["phase_ids"], config),
                    "optimizer_steps": updates, "sampled_transitions": seen_samples,
                    "wall_time_s": time.perf_counter()-started}
                log.write(json.dumps(metrics, allow_nan=False)+"\n")
                log.flush()
                print(f"BC epoch={epoch}/{config['epochs']} loss={metrics['sampled_train_loss']:.6f} "
                      f"val_xyz={metrics['validation']['xyz_mse']:.6f} "
                      f"val_grip={metrics['validation']['gripper_mse']:.6f}", flush=True)
                if epoch % config["eval_interval"] == 0 or epoch == config["epochs"]:
                    last_report = assess(epoch)
        final_path = output / "final_model"
        save_bundle(final_path, model, normalizer, {
            **run_metadata, **last_report, "role": "final"})
        result = {
            "status": "completed", "epochs": epoch, "optimizer_steps": updates,
            "selection_episodes": len(selected_seeds), "test_used_during_training": False,
            "sampled_transitions": seen_samples, "ppo_timesteps": model.num_timesteps,
            "best_model": best_path, "final_model": str(final_path.resolve()),
            "best_selection_success_rate": best_key[0],
            "final_selection": last_report["closed_loop"]["summary"],
            "wall_time_s": time.perf_counter()-started}
        write_json(output / "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return result
    except BaseException:
        if model is not None:
            save_bundle(output / "interrupted_model", model, normalizer, {
                **run_metadata, "epoch": epoch, "optimizer_steps": updates,
                "sampled_transitions": seen_samples, "role": "interrupted",
                "epoch_may_be_partial": True})
        raise
    finally:
        normalizer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs/day15_demos_v1")
    parser.add_argument("--config", type=Path, default=DEFAULT_BC_CONFIG)
    parser.add_argument("--evaluation-config", type=Path, default=DEFAULT_EVALUATION_CONFIG,
                        help="Reserve test seeds before training; never evaluated during training")
    parser.add_argument("--output", type=Path, required=True, help="New run directory, never overwritten")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--eval-episodes", type=parse_eval_episodes,
                        help="all (default in config), or a positive validation episode count")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()
    config = load_bc_config(args.config)
    for name in ("epochs", "batch_size", "eval_interval", "eval_episodes", "seed", "device"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    train(args.dataset, args.output, config, load_evaluation_config(args.evaluation_config))


if __name__ == "__main__":
    main()
