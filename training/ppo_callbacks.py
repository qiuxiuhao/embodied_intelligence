"""Day 14：更新后评估、不可覆盖的成套 Checkpoint 与真实 PPO 指标。"""

import csv
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from training.reach_common import evaluate, reference_original_reward, save_bundle, write_json, environment_version


class ReachTrainingCallback(BaseCallback):
    """在下一次 rollout 开始时读取刚完成的参数更新，而不是保存更新前的策略。"""

    def __init__(self, run_dir, config):
        super().__init__()
        self.run_dir, self.config = Path(run_dir), config
        self.records, self.episode_records, self.evaluations = [], [], []
        self.last_metrics_step = -1
        self.last_eval_step = -1
        self.last_checkpoint_step = 0
        self.best_success_key = (-1., -float("inf"))
        self.best_return = -float("inf")
        self.best_paths = {}
        self.reference_episode_return = None
        self.diagnostics_enabled = (environment_version(config) == "reach_v2"
                                    and not config["ppo"].get("use_sde", False))
        self.pending_distribution = None
        self.rollout_samples = []
        self.selection_protocol = json.loads(json.dumps({key: config.get(key) for key in
            ("selection_seeds", "env", "environment_version", "reward_version", "reach_rewards",
             "gravity_compensation", "timeout_mode")}))

    def _on_training_start(self):
        """恢复实验按当前步数重新安排定期评估；首次评估作为该实验的初始点。"""
        self.started = time.perf_counter()
        self.start_step = self.num_timesteps
        self.reference_episode_return = np.zeros(self.training_env.num_envs, dtype=float)
        # 恢复模型含历史 _n_updates，但新 logger 尚无本轮指标，不能当作 NaN。
        self.last_metrics_step = self.start_step
        self.last_checkpoint_step = self.start_step
        previous = getattr(self.model, "reach_selection_state", None)
        if (previous and previous["protocol"] == self.selection_protocol
                and all(Path(p).exists() for p in previous["paths"].values())):
            self.best_success_key = tuple(previous["success_key"])
            self.best_return = previous["best_return"]
            self.best_paths = dict(previous["paths"])
        self._evaluate()

    def _on_step(self):
        """Monitor 回报位于 VecNormalize 内层，始终为原始 Episode Return。"""
        for rank, info in enumerate(self.locals["infos"]):
            terms = info.get("reward_terms", {})
            reference = reference_original_reward(sum(terms.values()), terms, self.config)
            if reference is not None:
                self.reference_episode_return[rank] += reference
            if self.diagnostics_enabled:
                self.rollout_samples.append({
                    "action": self.locals["actions"][rank].copy(),
                    "success": bool(info.get("is_success")),
                    "episode_end": "episode" in info,
                    "error": info.get("tcp_position_error"),
                    "speed": info.get("arm_speed_rad_s"),
                    "gap": info.get("command_tracking_gap_m"),
                    "hold": info.get("stable_times_s", [0.])[0],
                    "ik_failure": not info.get("ik_success", False),
                    "workspace_clipped": bool(info.get("workspace_clipped")),
                    "terms": terms,
                })
            if "episode" in info:
                item = {"timesteps": self.num_timesteps, **info["episode"],
                        "reference_original_return": float(self.reference_episode_return[rank]) if reference is not None else None,
                        "reward_version": self.config.get("reward_version", "legacy"),
                        "environment_rank": rank}
                # 不同环境可在同一向量步结束；只清零该环境，避免回报串入另一回合。
                self.reference_episode_return[rank] = 0.
                self.episode_records.append(item)
                with (self.run_dir / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(item)+"\n")
        return True

    def _optimizer_step(self):
        parameter = next(self.model.policy.parameters())
        step = self.model.policy.optimizer.state.get(parameter, {}).get("step", 0)
        return int(step.item() if isinstance(step, torch.Tensor) else step)

    def _on_rollout_end(self):
        """Capture the exact pre-update Gaussian on the same stored observations."""
        if not self.diagnostics_enabled or self.model.use_sde:
            return
        buffer = self.model.rollout_buffer
        observations = torch.as_tensor(
            buffer.observations.reshape((-1, *self.model.observation_space.shape)),
            device=self.model.device).clone()
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(observations).distribution
            self.pending_distribution = (observations, distribution.mean.detach().clone(),
                                         distribution.stddev.detach().clone(), self._optimizer_step(),
                                         float(buffer.advantages.std()))

    def _diagnostics_after_update(self):
        if self.pending_distribution is None:
            return
        observations, mean, std, optimizer_step, advantage_std = self.pending_distribution
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(observations).distribution
            kl = (torch.log(distribution.stddev/std)
                  + (std.square()+(mean-distribution.mean).square())/(2*distribution.stddev.square())-.5)
            kl_total = kl.sum(-1)
        samples = self.rollout_samples
        actions = np.stack([s["action"] for s in samples])
        near = [s for s in samples if s["error"] is not None
                and s["error"] < self.config["env"]["tcp_position_tolerance_m"]]
        terms = Counter()
        for sample in samples:
            terms.update(sample["terms"])

        def average(rows, key):
            values = [r[key] for r in rows if r[key] is not None]
            return float(np.mean(values)) if values else None

        report = {
            "timesteps": self.num_timesteps, "samples": len(samples),
            "exact_gaussian_kl": float(kl_total.mean()),
            "exact_gaussian_kl_p95": float(torch.quantile(kl_total, .95)),
            "exact_kl_per_dimension": kl.mean(0).cpu().tolist(),
            "optimizer_steps": self._optimizer_step()-optimizer_step,
            "gaussian_std_per_dimension": distribution.stddev.mean(0).cpu().tolist(),
            "raw_action_mean": actions.mean(0).tolist(),
            "raw_action_std": actions.std(0).tolist(),
            "raw_action_clip_fraction_per_dimension": (np.abs(actions) > 1).mean(0).tolist(),
            "raw_xyz_any_clip_fraction": float(np.any(np.abs(actions[:, :3]) > 1, axis=1).mean()),
            "training_successes": sum(s["success"] for s in samples),
            "completed_episodes": sum(s["episode_end"] for s in samples),
            "near_goal_control_steps": len(near),
            "near_goal_mean_speed_rad_s": average(near, "speed"),
            "mean_command_tracking_gap_m": average(samples, "gap"),
            "max_hold_at_control_step_s": max(s["hold"] for s in samples),
            "ik_failure_fraction": float(np.mean([s["ik_failure"] for s in samples])),
            "workspace_clip_fraction": float(np.mean([s["workspace_clipped"] for s in samples])),
            "reward_terms_mean": {key: value/len(samples) for key, value in terms.items()},
            # SB3 resets the rollout buffer before the next rollout-start callback.
            "advantage_std": advantage_std,
        }
        with (self.run_dir/"diagnostics.jsonl").open("a") as stream:
            stream.write(json.dumps(report, allow_nan=False)+"\n")
        for key in ("exact_gaussian_kl", "exact_gaussian_kl_p95", "optimizer_steps"):
            self.logger.record("train/"+key, report[key])
        self.pending_distribution = None
        self.rollout_samples = []

    def _metrics(self):
        """SB3 在 train() 后留下本次更新指标；逐更新存 CSV/JSON/TensorBoard。"""
        if self.num_timesteps == self.last_metrics_step or self.model._n_updates == 0:
            return
        values = {k: float(v) for k, v in self.logger.name_to_value.items()
                  if k.startswith("train/") and np.isscalar(v)}
        if not values or not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError(f"无效 PPO 指标：{values}")
        values.update(timesteps=self.num_timesteps, wall_time_s=time.perf_counter()-self.started)
        values["end_to_end_fps"] = (self.num_timesteps-self.start_step)/max(values["wall_time_s"], 1e-9)
        self.records.append(values)
        path = self.run_dir / "metrics.csv"
        with path.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values))
            if len(self.records) == 1:
                writer.writeheader()
            writer.writerow(values)
        # dump_logs 在 train() 前运行；这里主动 dump 才不会漏掉最后一次更新。
        for key, value in values.items():
            self.logger.record(key, value)
        self.logger.dump(self.num_timesteps)
        self.last_metrics_step = self.num_timesteps

    def _evaluate(self):
        """仅选模 seeds；两种选模规则分别保存，绝不使用最终 hold-out 选择模型。"""
        report = evaluate(self.config, self.config["selection_seeds"], self.model, self.training_env)
        write_json(self.run_dir / "evaluations" / f"step_{self.num_timesteps:09d}.json", report)
        self.evaluations.append(report)
        summary = report["summary"]
        error = summary["mean_final_tcp_error_m"]
        key = (summary["success_rate"], -error if error is not None else -float("inf"))
        improvements = []
        if key > self.best_success_key:
            self.best_success_key = key
            improvements.append("success_then_error")
        if summary["mean_return"] > self.best_return:
            self.best_return = summary["mean_return"]
            improvements.append("mean_return")
        if improvements:
            path = self.run_dir / "best_model" / f"step_{self.num_timesteps:09d}"
            for criterion in improvements:
                self.best_paths[criterion] = str(path.resolve())
            self.model.reach_selection_state = {
                "protocol": self.selection_protocol, "success_key": self.best_success_key,
                "best_return": self.best_return, "paths": dict(self.best_paths)}
            save_bundle(self.model, self.training_env, path, self.config, "selection_best")
        with (self.run_dir / "selection_history.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": self.num_timesteps, "selected": self.best_paths})+"\n")
        for name, value in summary.items():
            if isinstance(value, (int, float)):
                self.logger.record("eval/"+name, value)
        print(f"EVAL step={self.num_timesteps} success={summary['success_rate']:.1%} "
              f"error={error*1000 if error is not None else float('nan'):.2f}mm "
              f"return={summary['mean_return']:.4f}", flush=True)
        if self.config.get("training_reset_options"):
            # 课程反馈单独记录；best仍只按完整原随机Reach选模，不能混算成功率。
            curriculum = evaluate(self.config, list(range(11000, 11005)), self.model, self.training_env,
                                  reset_options=self.config["training_reset_options"])
            write_json(self.run_dir / "curriculum_evaluations" / f"step_{self.num_timesteps:09d}.json", curriculum)
            for name, value in curriculum["summary"].items():
                if isinstance(value, (int, float)):
                    self.logger.record("curriculum/"+name, value)
            print(f"CURRICULUM step={self.num_timesteps} success={curriculum['summary']['success_rate']:.1%} "
                  f"error={curriculum['summary']['mean_final_tcp_error_m']*1000:.2f}mm", flush=True)
        self.last_eval_step = self.num_timesteps

    def _on_rollout_start(self):
        """滚动结束本身还未 PPO 更新；在下个滚动开始时才能得到已更新的模型。"""
        self._diagnostics_after_update()
        self._metrics()
        if self.num_timesteps-self.last_eval_step >= self.config["eval_interval"]:
            self._evaluate()
        if self.num_timesteps-self.last_checkpoint_step >= self.config["checkpoint_interval"]:
            save_bundle(self.model, self.training_env,
                self.run_dir / "checkpoints" / f"step_{self.num_timesteps:09d}", self.config, "periodic")
            self.last_checkpoint_step = self.num_timesteps

    def _on_training_end(self):
        """最后一次参数更新也记录并评估，不把训练前的策略误标为最终策略。"""
        self._diagnostics_after_update()
        self._metrics()
        if self.last_eval_step != self.num_timesteps:
            self._evaluate()
        write_json(self.run_dir / "metrics.json", self.records)
        write_json(self.run_dir / "best_selection.json", self.best_paths)
