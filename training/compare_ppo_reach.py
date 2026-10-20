"""核对同训练步数、同物理条件后评估Reach模型，防止把阶段诊断混作等预算对照。"""

import argparse
import json
from pathlib import Path

import torch

from training.reach_common import evaluate, load_bundle, write_json, environment_version


def compare(models, expected_steps, seeds, output, allow_protocol_differences=False):
    """models为名称到成套存档目录的映射；主结果只使用指定步数模型。

    奖励允许作为实验变量变化，物理控制/成功门槛必须一致。D同时保存原奖励口径；
    C在独立评估时使用原随机目标，不能继承训练课程的固定reset选项。
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    physical, roles, protocols = None, {}, {}
    for label, directory in models.items():
        manifest = json.loads((Path(directory)/"manifest.json").read_text())
        roles[label] = manifest["role"]
        if manifest["timesteps"] != expected_steps:
            raise ValueError(f"{label}只有{manifest['timesteps']}步，要求{expected_steps}步")
        parameters = {k: v for k, v in manifest["config"]["env"].items() if k != "rewards"}
        if physical is not None and parameters != physical:
            raise ValueError(f"{label}物理/控制/成功配置与参考模型不一致")
        physical = parameters
        config = manifest["config"]
        protocols[label] = {
            "environment_version": environment_version(config),
            "gravity_compensation": config.get("gravity_compensation", False),
            "timeout_mode": config.get("timeout_mode", "external_truncation"),
        }
    protocols_equal = all(p == next(iter(protocols.values())) for p in protocols.values())
    if not protocols_equal and not allow_protocol_differences:
        raise ValueError("控制器/环境/超时协议不同；须显式使用 --allow-protocol-differences 比较")
    output.mkdir(parents=True, exist_ok=False)
    reports, reference_initials = {}, None
    for label, directory in models.items():
        model, normalizer, config = load_bundle(directory)
        try:
            if model.num_timesteps != expected_steps:
                raise ValueError(f"{label}模型内部步数与manifest不一致")
            policies = ("ppo", "random", "zero") if label == next(iter(models)) else ("ppo",)
            for policy in policies:
                key = label if policy == "ppo" else policy
                if key in reports:
                    raise ValueError(f"重复实验名称{key}")
                report = evaluate(config, seeds, model if policy == "ppo" else None,
                                  normalizer, policy=policy)
                initials = [(episode["seed"], episode["initial"]) for episode in report["episodes"]]
                if reference_initials is not None and initials != reference_initials:
                    raise AssertionError(f"{key}初始条件与参考不一致")
                reference_initials = initials
                write_json(output/(key+".json"), {"bundle": str(Path(directory).resolve()),
                                                 "config": config, "report": report})
                reports[key] = report
                print(key, json.dumps(report["summary"], ensure_ascii=False), flush=True)
        finally:
            normalizer.close()
    result = {"primary_training_steps": expected_steps, "seeds": seeds,
              "identical_initial_conditions": True, "all_physical_configs_equal": True,
              "control_protocols": protocols, "all_control_protocols_equal": protocols_equal,
              "protocol_differences_explicitly_allowed": allow_protocol_differences,
              "primary_uses_final_models": all(role == "final" for role in roles.values()),
              "primary_uses_exact_budget_snapshots": True, "model_roles": roles, "reports": reports}
    write_json(output/"comparison.json", result)
    plot_comparison(result, output/"comparison.png")
    return result


def plot_comparison(result, path):
    """同预算末位置误差、真实成功率与原Reward口径；数字来自真实评估。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(result["reports"])
    summaries = [result["reports"][label]["summary"] for label in labels]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    reference_available = all(s["mean_reference_original_return"] is not None for s in summaries)
    columns = (("success_rate", 100., "True task success (%)"),
               ("mean_final_tcp_error_m", 1000., "Final TCP error (mm)"),
               ("mean_reference_original_return", 1., "Return in ORIGINAL reward units")
               if reference_available else
               ("mean_max_reach_stable_time_s", 1., "Longest stable hold per episode (s)"))
    for axis, (name, factor, title) in zip(axes, columns):
        values = [factor*summary[name] for summary in summaries]
        bars = axis.bar(labels, values)
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:.2f}", (bar.get_x()+bar.get_width()/2, value),
                          ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylim(0, 105)
    fig.suptitle(f"Matched budget: {result['primary_training_steps']:,} training steps; {len(result['seeds'])} hold-out episodes")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    """命令行显式指定标签=配对目录，先验步数检查不通过就不开始评估。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, help="标签=成套存档目录")
    parser.add_argument("--expected-steps", type=int, default=98304)
    parser.add_argument("--seed-start", type=int, default=50000)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-protocol-differences", action="store_true",
                        help="显式比较不同控制器/动作/超时版本，报告中标记差异")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes必须为正")
    models = {}
    for entry in args.models:
        label, directory = entry.split("=", 1)
        if not label or label in models or label in ("random", "zero") or not label.replace("_", "").isalnum():
            parser.error("模型标签必须唯一且只包含字母、数字或下划线")
        models[label] = Path(directory)
    torch.set_num_threads(1)
    compare(models, args.expected_steps, list(range(args.seed_start,args.seed_start+args.episodes)), args.output,
            allow_protocol_differences=args.allow_protocol_differences)


if __name__ == "__main__":
    main()
