from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. 配置
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs" / "day07"

SEEDS = [42, 43, 44]


# ============================================================
# 2. 读取 CSV
# ============================================================

def load_metrics(seed):

    csv_path = (
        OUTPUT_DIR
        / f"seed_{seed}"
        / "metrics.csv"
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Missing metrics file: {csv_path}"
        )

    with open(
        csv_path,
        "r",
        encoding="utf-8",
    ) as file:

        rows = list(
            csv.DictReader(file)
        )

    return rows


# ============================================================
# 3. 从 CSV 提取数值列
# ============================================================

def get_column(rows, column_name):

    return np.asarray(
        [
            float(row[column_name])
            for row in rows
        ],
        dtype=np.float64,
    )


# ============================================================
# 4. 绘制单个指标
# ============================================================

def plot_metric(
    metric_name,
    title,
    ylabel,
    filename,
):

    plt.figure(figsize=(10, 6))

    for seed in SEEDS:

        rows = load_metrics(seed)

        steps = get_column(
            rows,
            "total_steps",
        )

        values = get_column(
            rows,
            metric_name,
        )

        # 去除未测量的 NaN
        mask = (
            np.isfinite(steps)
            &
            np.isfinite(values)
        )

        plt.plot(
            steps[mask],
            values[mask],
            marker="o",
            markersize=3,
            label=f"Seed {seed}",
        )

    plt.xlabel("Environment Steps")

    plt.ylabel(ylabel)

    plt.title(title)

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    save_path = OUTPUT_DIR / filename

    plt.savefig(
        save_path,
        dpi=200,
    )

    plt.close()

    print("Saved:", save_path)


# ============================================================
# 5. 主程序
# ============================================================

def main():

    plot_metric(
        metric_name="val_success",
        title="PPO Validation Success Rate",
        ylabel="Success Rate",
        filename="validation_success.png",
    )

    plot_metric(
        metric_name="val_return",
        title="PPO Validation Mean Return",
        ylabel="Mean Return",
        filename="validation_return.png",
    )

    plot_metric(
        metric_name="value_loss",
        title="PPO Value Loss",
        ylabel="Value Loss",
        filename="value_loss.png",
    )

    plot_metric(
        metric_name="approx_kl",
        title="PPO Approximate KL",
        ylabel="Approximate KL",
        filename="approx_kl.png",
    )

    plot_metric(
        metric_name="clip_fraction",
        title="PPO Clip Fraction",
        ylabel="Clip Fraction",
        filename="clip_fraction.png",
    )

    print("\nAll plots generated.")


if __name__ == "__main__":

    main()