"""Validation selection and a separately reserved, seed-disjoint BC test block."""

from copy import deepcopy
import json
from pathlib import Path

from training.pick_place_common import ROOT, seed_range


DEFAULT_EVALUATION_CONFIG = ROOT / "configs/day15_bc_evaluation.json"
EVALUATION_VERSION = "day15_bc_evaluation_v1"


def parse_eval_episodes(value):
    """CLI accepts 'all' or a positive integer; old integer configurations work."""
    if value == "all":
        return value
    try:
        count = int(value)
    except (ValueError, TypeError) as error:
        raise ValueError("eval_episodes must be 'all' or a positive integer") from error
    if count <= 0:
        raise ValueError("eval_episodes must be 'all' or a positive integer")
    return count


def selection_seeds(task, count):
    seeds = seed_range(task["demonstrations"]["validation"])
    if count == "all":
        return seeds
    if type(count) is not int or not 0 < count <= len(seeds):
        raise ValueError("eval_episodes must fit the validation seed count or be 'all'")
    return seeds[:count]


def load_evaluation_config(path=DEFAULT_EVALUATION_CONFIG):
    return json.loads(Path(path).read_text())


def bind_evaluation_protocol(task, config):
    """Validate before training and again before test; no test rollout occurs here.

    Historical exclusions may overlap the current train/validation blocks. Only
    test must be disjoint from their union. Existing dataset protocols stay intact.
    """
    if (set(config) != {"version", "test", "historical_exclusions"}
            or config["version"] != EVALUATION_VERSION):
        raise ValueError("Unknown BC evaluation protocol")
    test = set(seed_range(config["test"]))
    acceptance = task["acceptance"]
    blocks = {
        "train": task["demonstrations"]["train"],
        "validation": task["demonstrations"]["validation"],
        "reserved_holdout": task["reserved_holdout"],
        "acceptance_fixed": {"seed_start": 300000, "episodes": 100},
        "acceptance_random": {"seed_start": acceptance["random_seed_start"],
                              "episodes": acceptance["random_episodes"]},
    }
    blocks.update({f"historical/{name}": spec
                   for name, spec in config["historical_exclusions"].items()})
    for name, spec in blocks.items():
        if test.intersection(seed_range(spec)):
            raise ValueError(f"Test seeds overlap {name}")
    if set(seed_range(blocks["train"])) & set(seed_range(blocks["validation"])):
        raise ValueError("Training and validation seeds overlap")
    return deepcopy(config)


def evaluation_seeds(metadata, split):
    task = metadata["dataset_protocol"]["config"]
    if split == "test":
        if "evaluation_protocol" not in metadata:
            raise ValueError("This legacy bundle has no test protocol bound before training")
        protocol = bind_evaluation_protocol(task, metadata["evaluation_protocol"])
        seeds = seed_range(protocol["test"])
        if set(seeds).intersection(metadata.get("selection_seeds", [])):
            raise ValueError("Test seeds overlap recorded model-selection seeds")
        return seeds
    if split == "validation":
        return seed_range(task["demonstrations"]["validation"])
    if split == "holdout":
        return seed_range(task["reserved_holdout"])
    raise ValueError(f"Unknown evaluation split: {split}")
