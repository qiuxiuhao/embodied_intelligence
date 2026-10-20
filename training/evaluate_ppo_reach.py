"""加载成套 PPO/VecNormalize，按相同 seeds 做原始物理评估与公平基线。"""

import argparse
from pathlib import Path
import json

import torch

from training.reach_common import HOLDOUT_SEEDS, evaluate, load_bundle, write_json


def evaluation_seeds(config, episodes=None, seed_start=None):
    """Use the bundle's held-out protocol unless the caller explicitly overrides it."""
    saved = list(config.get("holdout_seeds", HOLDOUT_SEEDS))
    count = len(saved) if episodes is None else episodes
    if count <= 0:
        raise ValueError("episodes 必须为正数")
    if seed_start is not None:
        return list(range(seed_start, seed_start+count))
    if count > len(saved):
        raise ValueError("episodes 超过存档 holdout_seeds 数量；扩展评估请显式设置 --seed-start")
    return saved[:count]


def main():
    """默认使用存档的全部 hold-out seeds；Viewer 也加载同一个训练模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", choices=("ppo", "random", "zero"), default=["ppo"])
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--human", action="store_true")
    args = parser.parse_args()
    if args.episodes is not None and args.episodes <= 0:
        parser.error("episodes 必须为正数")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    model, vector, config = load_bundle(args.bundle)
    try:
        seeds = evaluation_seeds(config, args.episodes, args.seed_start)
        reports = {p: evaluate(config, seeds, model if p == "ppo" else None,
                              vector, policy=p, human=args.human) for p in args.policies}
        write_json(args.output, {"bundle": str(args.bundle.resolve()), "config": config, "reports": reports})
        for policy, report in reports.items():
            print(policy, json.dumps(report["summary"], indent=2))
    finally:
        vector.close()


if __name__ == "__main__":
    main()
