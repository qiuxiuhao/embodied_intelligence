import torch
import torch.nn as nn

from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_low, action_high, hidden_dim=64,):
        super().__init__()
        action_low = torch.as_tensor(action_low,dtype=torch.float32,)
        action_high = torch.as_tensor(action_high,dtype=torch.float32,)
        action_dim = action_low.numel()

        # ==========================================
        # Actor
        # ==========================================
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim,),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim,),
            nn.Tanh(),
            nn.Linear(hidden_dim,action_dim,),
        )

        # Gaussian 的 log sigma
        self.log_std = nn.Parameter(
            torch.full((action_dim,),-0.5)
        )

        # ==========================================
        # Critic
        # ==========================================
        self.critic = nn.Sequential(
            nn.Linear(obs_dim,hidden_dim,),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim,),
            nn.Tanh(),
            nn.Linear(hidden_dim,1),
        )

        # ==========================================
        # Action transform
        # ==========================================
        self.register_buffer("action_low",action_low)
        self.register_buffer("action_high",action_high)

        action_scale = ( action_high - action_low ) / 2.0
        action_bias = ( action_high + action_low ) / 2.0

        self.register_buffer("action_scale", action_scale,)
        self.register_buffer("action_bias", action_bias,)

    def _distribution(self, obs,):
        mean = self.actor(obs)
        log_std = torch.clamp(self.log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std,)

    def _raw_to_action(self, raw_action,):
        squashed = torch.tanh(raw_action)
        action = (self.action_bias + self.action_scale*squashed)
        return action, squashed

    def _log_prob_from_raw(self, distribution, raw_action, squashed,):
        log_prob = distribution.log_prob(raw_action)
        correction = torch.log(self.action_scale * ( 1.0 - squashed.pow(2)) + 1e-6 )
        log_prob = (log_prob - correction)
        return log_prob.sum(dim=-1)

    @torch.no_grad()
    def act(self,obs, deterministic=False):
        distribution = self._distribution(obs)
        if deterministic:
            raw_action =( distribution.mean )
        else:
            raw_action =( distribution.sample() )

        action, squashed = (self._raw_to_action(raw_action))

        log_prob = (self._log_prob_from_raw(distribution,raw_action,squashed))
        value = self.critic(obs).squeeze(-1)

        return (action,log_prob,value)

    def evaluate_actions(self,obs,action,):
        distribution = self._distribution(obs)

        # ==========================================
        # action -> [-1, 1]
        # ==========================================
        normalized_action = (action - self.action_bias) / self.action_scale

        normalized_action = (normalized_action.clamp(-0.999999,0.999999,))

        # inverse tanh
        raw_action = torch.atanh(normalized_action)
        log_prob = ( self._log_prob_from_raw(distribution, raw_action, normalized_action,))
        value = self.critic(obs).squeeze(-1)

        # 这里作为探索程度的近似诊断
        entropy = ( distribution.entropy().sum(dim=-1) )

        return (log_prob, entropy, value)

    def value(self,obs,):
        return self.critic(obs).squeeze(-1)



