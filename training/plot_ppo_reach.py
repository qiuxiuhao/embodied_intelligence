"""仅从已有真实日志画训练曲线，不插值出未执行的成功实验。"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(runs, output):
    """可串联同一实验的 Pilot/恢复阶段；按真实累计步数绘图，不伪造缺失数据。"""
    runs, output = [Path(run) for run in runs], Path(output)
    if output.exists():
        raise FileExistsError(output)
    metrics, episodes, by_step = [], [], {}
    for run in runs:
        metrics.extend(json.loads((run / "metrics.json").read_text()))
        episodes.extend(json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines())
        for path in sorted((run / "evaluations").glob("step_*.json")):
            evaluation = json.loads(path.read_text())
            by_step[evaluation["training_timesteps"]] = evaluation
    metrics.sort(key=lambda m: m["timesteps"])
    episodes.sort(key=lambda e: e["timesteps"])
    evaluations = [by_step[step] for step in sorted(by_step)]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)
    axes = axes.ravel()
    axes[0].scatter([e["timesteps"] for e in episodes], [e["r"] for e in episodes], s=8, alpha=.5)
    axes[0].set_title("Raw episode return")
    for axis, key, title, scale in zip(axes[1:3], ("success_rate", "mean_final_tcp_error_m"),
            ("Selection evaluation success rate", "Selection final TCP error (mm)"), (1, 1000)):
        axis.plot([e["training_timesteps"] for e in evaluations],
                  [scale*e["summary"][key] for e in evaluations], "o-")
        axis.set_title(title)
    axes[3].scatter([e["timesteps"] for e in episodes], [e["l"] for e in episodes], s=8)
    axes[3].set_title("Episode length (steps)")
    for axis, key in zip(axes[4:], ("entropy_loss", "value_loss", "explained_variance", "approx_kl", "clip_fraction")):
        axis.plot([m["timesteps"] for m in metrics], [m["train/"+key] for m in metrics])
        axis.set_title(key)
    for axis in axes:
        axis.set_xlabel("Environment timesteps")
        axis.grid(alpha=.2)
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--combined-output", type=Path)
    args = parser.parse_args()
    if args.combined_output:
        print(plot(args.runs, args.combined_output))
    else:
        for run in args.runs:
            print(plot([run], run / "curves.png"))
