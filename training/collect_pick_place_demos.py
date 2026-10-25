"""Collect and independently verify raw Day15 expert episodes.

Every scheduled seed is attempted exactly once. Failures remain in the index and
on disk; only successful episodes are listed as eligible for BC. Validation
episodes must never be used to fit normalization or supervised training.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from training.pick_place_common import (DEFAULT_CONFIG, ROOT, load_config, make_env, provenance,
                                       rollout_expert, seed_range, sha256, source_hashes, summarize,
                                       validate_episode, write_json)


def verify_dataset(directory, *, pending_manifest=None):
    """Reload every NPZ without pickle and audit hashes, splits and trajectories."""
    directory = Path(directory)
    manifest = (pending_manifest if pending_manifest is not None else
                json.loads((directory / "manifest.json").read_text()))
    if manifest["status"] != "complete":
        raise ValueError("Dataset has no completion marker")
    index = directory / "episodes.jsonl"
    if sha256(index) != manifest["index_sha256"]:
        raise ValueError("Episode index checksum mismatch")
    protocol_file = directory / "protocol.json"
    if sha256(protocol_file) != manifest["protocol_sha256"]:
        raise ValueError("Protocol checksum mismatch")
    protocol = json.loads(protocol_file.read_text())
    if sha256(directory / "acceptance.json") != protocol["acceptance_sha256"]:
        raise ValueError("Acceptance evidence checksum mismatch")
    for relative, digest in protocol["source_sha256"].items():
        if sha256(directory / "source" / relative) != digest:
            raise ValueError(f"Source snapshot checksum mismatch: {relative}")
    records = [json.loads(line) for line in index.read_text().splitlines()]
    seen, files = set(), set()
    config = protocol["config"]
    counts, transitions, eligible = Counter(), Counter(), Counter()
    phase_counts = {split: Counter() for split in config["demonstrations"]}
    for record in records:
        split, seed, relative = record["split"], record["seed"], record["file"]
        if split not in config["demonstrations"] or seed not in seed_range(config["demonstrations"][split]):
            raise ValueError("Unexpected split/seed")
        if seed in seen or relative in files:
            raise ValueError("Repeated episode or cross-split seed leakage")
        seen.add(seed)
        files.add(relative)
        path = directory / relative
        if path.resolve().parent != (directory / split).resolve():
            raise ValueError("Unexpected episode path")
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Episode checksum mismatch: {relative}")
        with np.load(path, allow_pickle=False) as file:
            arrays = {key: file[key] for key in file.files}
        validate_episode(arrays, record)
        counts[split] += 1
        transitions[split] += record["steps"]
        eligible[split] += int(record["eligible_for_bc"])
        phase_counts[split].update(int(p) for p in arrays["phase_ids"])
    for split, spec in config["demonstrations"].items():
        if counts[split] != spec["episodes"]:
            raise ValueError("Missing scheduled attempts")
        if {p.name for p in (directory / split).glob("*.npz")} != {
                Path(r["file"]).name for r in records if r["split"] == split}:
            raise ValueError("Unindexed NPZ or missing episode")
        expected = [r["file"] for r in records if r["split"] == split and r["eligible_for_bc"]]
        if manifest["bc_eligible_files"][split] != expected:
            raise ValueError("Eligible file list mismatch")
        if manifest["summaries"][split] != summarize([r for r in records if r["split"] == split]):
            raise ValueError("Manifest summary mismatch")
    return {"passed": True, "episodes": dict(counts), "transitions": dict(transitions),
            "bc_eligible_episodes": dict(eligible),
            "phase_step_counts": {k: dict(v) for k, v in phase_counts.items()},
            "disjoint_episode_seeds": True, "all_file_hashes_verified": True,
            "raw_observations": True, "normalization_fitted": False}


def collect(config, output, acceptance_report):
    acceptance_report = Path(acceptance_report)
    acceptance = json.loads(acceptance_report.read_text())
    if not acceptance["passed"]:
        raise ValueError("Expert acceptance did not pass")
    if acceptance["protocol"]["config"] != config:
        raise ValueError("Acceptance and collection configurations differ")
    if acceptance["protocol"]["source_sha256"] != source_hashes():
        raise ValueError("Source changed since acceptance; rerun physical acceptance")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = provenance(config)
    protocol["acceptance_report"] = str(acceptance_report.resolve())
    protocol["acceptance_sha256"] = sha256(acceptance_report)
    protocol["split_policy"] = "Whole episodes by disjoint seeds; no frame-level random split"
    protocol["failure_policy"] = "Retain every attempt; BC eligibility only for physical SUCCESS"
    protocol["reserved_holdout_used"] = False
    write_json(output / "protocol.json", protocol)
    (output / "acceptance.json").write_bytes(acceptance_report.read_bytes())
    # Small source/XML snapshots make the dataset self-describing even with a
    # dirty worktree. Meshes remain the repository's unchanged Panda assets.
    for relative in protocol["source_sha256"]:
        path = output / "source" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    records, started = [], time.perf_counter()
    env = make_env(config)
    try:
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
            for split, spec in config["demonstrations"].items():
                (output / split).mkdir()
                for seed in seed_range(spec):
                    arrays, record = rollout_expert(env, config, seed, {"start_pose": "home"})
                    relative = f"{split}/episode_{seed:09d}.npz"
                    with (output / relative).open("xb") as file:
                        np.savez_compressed(file, **arrays)
                    record.update(split=split, file=relative, sha256=sha256(output / relative))
                    records.append(record)
                    stream.write(json.dumps(record, allow_nan=False)+"\n")
                    stream.flush()
                    if len(records) % 25 == 0 or not record["success"]:
                        print(f"COLLECT attempts={len(records)} split={split} seed={seed} "
                              f"success={record['success']} reason={record['reason']}", flush=True)
    finally:
        env.close()
    summaries = {split: summarize([r for r in records if r["split"] == split])
                 for split in config["demonstrations"]}
    manifest = {
        "status": "complete", "dataset_version": config["dataset_version"],
        "protocol_sha256": sha256(output / "protocol.json"),
        "index_sha256": sha256(output / "episodes.jsonl"),
        "summaries": summaries,
        "bc_eligible_files": {split: [r["file"] for r in records
                                     if r["split"] == split and r["eligible_for_bc"]]
                              for split in config["demonstrations"]},
        "wall_time_s": time.perf_counter()-started,
        "normalization_note": "Fit statistics on train only; retain raw observations and pair stats with BC/PPO",
    }
    # Verify the on-disk dataset before publishing its completion marker.
    verification = verify_dataset(output, pending_manifest=manifest)
    write_json(output / "verification.json", verification)
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"summaries": summaries, "verification": verification}, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="New dataset directory")
    parser.add_argument("--acceptance-report", type=Path)
    parser.add_argument("--verify", type=Path, help="Read-only re-audit of an existing dataset")
    args = parser.parse_args()
    if args.verify:
        if args.output or args.acceptance_report:
            parser.error("--verify cannot be combined with collection")
        print(json.dumps(verify_dataset(args.verify), indent=2))
    else:
        if args.output is None or args.acceptance_report is None:
            parser.error("Collection requires --output and --acceptance-report")
        collect(load_config(args.config), args.output, args.acceptance_report)


if __name__ == "__main__":
    main()
