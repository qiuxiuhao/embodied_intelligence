"""Headless physical acceptance before collecting Day15 demonstrations."""

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np

from training.pick_place_common import (DEFAULT_CONFIG, load_config, make_env, provenance,
                                       rollout_expert, summarize, write_json)


def acceptance_cases(config, env):
    z = env.table_top + env.half_size[2]
    fixed = {"cube_position": [.5, 0., z], "cube_yaw": 0.,
             "target_position": [.5, .12, z]}
    cases = [
        ("fixed_home", 300000, {**fixed, "start_pose": "home"}),
        ("fixed_pre_grasp", 300001, {**fixed, "start_pose": "pre_grasp"}),
        ("alternate_target", 300002, {**fixed, "target_position": [.56, -.08, z]}),
    ]
    cube_range = np.asarray(config["env"]["cube_random_xy_half_range_m"])
    target_range = np.asarray(config["env"]["target_random_xy_half_range_m"])
    yaw = config["env"]["cube_yaw_half_range_rad"]
    signs = list(itertools.product((-1, 1), repeat=2))
    for index, (cube_sign, target_sign, yaw_sign) in enumerate(itertools.product(signs, signs, (-1, 1))):
        cube_xy = np.array([.5, 0.]) + cube_range*np.array(cube_sign)
        target_xy = np.array([.5, .12]) + target_range*np.array(target_sign)
        cases.append(("boundary", 300010+index, {
            "cube_position": [*cube_xy, z], "cube_yaw": yaw_sign*yaw,
            "target_position": [*target_xy, z], "start_pose": "home"}))
    a = config["acceptance"]
    cases.extend(("random_home", a["random_seed_start"]+i, {"start_pose": "home"})
                 for i in range(a["random_episodes"]))
    return cases


def accept(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = provenance(config)
    write_json(output / "protocol.json", protocol)
    records, started = [], time.perf_counter()
    env = make_env(config)
    try:
        with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
            for case, seed, options in acceptance_cases(config, env):
                arrays, record = rollout_expert(env, config, seed, options)
                record["case"] = case
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                stream.flush()
                if not record["success"]:
                    with (output / f"failure_{seed}.npz").open("xb") as file:
                        np.savez_compressed(file, **arrays)
                if len(records) % 10 == 0 or not record["success"]:
                    print(f"ACCEPT {len(records)} seed={seed} case={case} "
                          f"success={record['success']} reason={record['reason']}", flush=True)
    finally:
        env.close()
    groups = {case: summarize([r for r in records if r["case"] == case])
              for case in dict.fromkeys(r["case"] for r in records)}
    required = ("fixed_home", "fixed_pre_grasp", "alternate_target", "boundary")
    passed = (all(groups[k]["success_rate"] == 1. for k in required)
              and groups["random_home"]["success_rate"] >= config["acceptance"]["minimum_success_rate"])
    report = {"passed": passed, "protocol": protocol, "groups": groups,
              "summary": summarize(records), "wall_time_s": time.perf_counter()-started,
              "episodes": records,
              "scope": "State-based, fixed home/pre_grasp; one cube, current XY/yaw ranges; no learned policy"}
    write_json(output / "report.json", report)
    print(json.dumps({"passed": passed, "groups": groups}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True, help="New acceptance directory")
    args = parser.parse_args()
    report = accept(load_config(args.config), args.output)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
