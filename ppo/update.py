import torch

def ppo_update(model, optimizer, batch, advantages, returns, clip_coef=0.2, value_coef=0.5,
               entropy_coef=0.0,update_epochs=10,minibatch_size=256,max_grad_norm=0.5,):

    observations = ( batch["observations"])
    actions = ( batch["actions"] )
    old_log_probs = ( batch["old_log_probs"])

    # ==========================================
    # Advantage normalization
    # ==========================================

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    num_samples = (observations.shape[0])

    metrics = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
    }

    for _ in range(update_epochs):

        indices = torch.randperm(num_samples, device = observations.device)

        for start in range(0, num_samples, minibatch_size):

            md_idx = indices[start: start+minibatch_size]

            # ======================================
            # New policy evaluation
            # ======================================
            (new_log_prob, entropy, new_value) = model.evaluate_actions(observations[md_idx],actions[md_idx])

            log_ratio = ( new_log_prob - old_log_probs[md_idx])
            ratio = torch.exp(log_ratio)
            mb_advantage = (advantages[md_idx])

            # ======================================
            # PPO policy loss
            # ======================================
            unclipped = ( ratio * mb_advantage)
            clipped =  (torch.clamp(ratio, 1.0-clip_coef, 1.0+clip_coef) * mb_advantage)

            policy_loss = (-torch.min(unclipped,clipped,).mean())

            # ======================================
            # Value loss
            # ======================================
            value_loss = (0.5 * (new_value-returns[md_idx]).pow(2).mean())

            # ======================================
            # Entropy
            # ======================================
            entropy_mean = (entropy.mean())

            # ======================================
            # Total
            # ======================================
            loss = (policy_loss + value_coef*value_loss - entropy_coef +entropy_mean)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm,)
            optimizer.step()

            # ======================================
            # Diagnostics
            # ======================================
            with torch.no_grad():

                approx_kl = ( (ratio -1.0) - log_ratio ).mean()
                clip_fraction = ( (torch.abs(ratio - 1.0) > clip_coef).float().mean())

            metrics["policy_loss"].append( policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(entropy_mean.item())
            metrics["approx_kl"].append(approx_kl.item())
            metrics["clip_fraction"].append(clip_fraction.item())

    return {
        key: sum(values)/len(values) for key,values in metrics.items()
    }



