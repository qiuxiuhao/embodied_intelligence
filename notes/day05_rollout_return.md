# day5 rollout_return
## 知识图
                    Policy
                      │
               obs → │ → action
                      │
                      ▼
                 Environment
                      │
              ┌───────┼────────┐
              ▼       ▼        ▼
           next_obs  reward   done
              │
              ▼
           Transition
              │
              ▼
            Rollout
              │
              ▼
          Trajectory
              │
        ┌─────┴─────┐
        ▼           ▼
      Return     Dataset
        │           │
        │           └──→ BC / ACT
        │
        └──────────────→ PPO / RL