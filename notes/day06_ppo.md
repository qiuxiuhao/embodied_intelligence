# day06: PPO
## 一些文字记录
1. actor ： 学习 π(a|s)
2. critic ： 学习 V(s)
3. Stochastic Policy: 训练时从action distribution sample
4. log_prob ：PPO 用来比较 old/new policy
5. Value： 当前状态未来预计 return
6. Advantage： 一个 action 相对当前平均水平好多少
7. GAE ： 利用 TD error 估计 advantage
8. ratio： new probability / old probability
9. PPO clip: 限制 policy 一次更新过大
10. On-policy: 新策略需要重新采 rollout

