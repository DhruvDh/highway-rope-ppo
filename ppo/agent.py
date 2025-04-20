# ppo/agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import logging
from typing import Any, Optional


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor head
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Log std parameter
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Move to device
        self.to(self.device)

    def forward(self, x):
        # Accept numpy arrays or torch tensors
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        shared_features = self.shared(x)
        action_mean = self.actor_mean(shared_features)
        action_std = self.log_std.exp()
        state_value = self.critic(shared_features)
        return action_mean, action_std, state_value

    def get_action(self, state, deterministic: bool = False):
        action_mean, action_std, value = self.forward(state)
        dist = Normal(action_mean, action_std)
        if deterministic:
            z = action_mean
            action = torch.tanh(z)
            log_prob = None
        else:
            z = dist.sample()
            action = torch.tanh(z)
            # Change of variables for tanh
            log_prob = (dist.log_prob(z) - torch.log1p(-action.pow(2) + 1e-6)).sum(dim=-1)
        action_np = action.detach().cpu().numpy()
        pre_tanh_np = z.detach().cpu().numpy()
        log_prob_val = log_prob.item() if log_prob is not None else None
        value_val = value.detach().cpu().numpy()[0]
        return action_np, pre_tanh_np, log_prob_val, value_val

    def evaluate(self, states, actions, pre_tanh_actions):
        action_means, action_stds, state_values = self.forward(states)
        dist = Normal(action_means, action_stds)
        log_probs = dist.log_prob(pre_tanh_actions)
        tanh_actions = torch.tanh(pre_tanh_actions)
        log_probs = log_probs - torch.log1p(-tanh_actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, state_values, entropy


class PPOMemory:
    def __init__(
        self, batch_size: int = 64, device: torch.device = torch.device("cpu")
    ):
        # Buffers for experience; annotate types for mypy
        self.states: list[Any] = []
        self.actions: list[Any] = []
        self.pre_tanh_actions: list[Any] = []
        self.rewards: list[float] = []
        self.next_states: list[Any] = []
        self.log_probs: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []
        self.batch_size = batch_size
        self.device = device

    def store(
        self, state, action, pre_tanh_action, reward, next_state, log_prob, done, value
    ):
        self.states.append(state)
        self.actions.append(action)
        self.pre_tanh_actions.append(pre_tanh_action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        # Reset all buffers
        self.states = []
        self.actions = []
        self.pre_tanh_actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.dones = []
        self.values = []

    def compute_advantages(self, gamma: float, lam: float, last_value: float):
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0
        # GAE computation
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        returns = advantages + np.array(self.values)
        return advantages, returns

    def get_batches(self):
        n = len(self.states)
        indices = np.arange(n, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [
            indices[i : i + self.batch_size] for i in range(0, n, self.batch_size)
        ]
        return batches

    def get_tensors(self):
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        pre_tanh = torch.FloatTensor(np.array(self.pre_tanh_actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        return states, actions, pre_tanh, old_log_probs


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        eps_clip: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.005,
        max_grad_norm: float = 0.5,
        epochs: int = 6,
        batch_size: int = 64,
        hidden_dim: int = 128,
        logger: Optional[logging.Logger] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.device = device
        self.actor_critic = ActorCritic(
            state_dim, action_dim, hidden_dim, device=device
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.logger = logger or logging.getLogger(__name__)
        self.memory = PPOMemory(batch_size=batch_size, device=device)

    def select_action(self, state, deterministic: bool = False):
        action, pre_tanh, log_prob, value = self.actor_critic.get_action(
            state, deterministic
        )
        return action, pre_tanh, log_prob, value

    def update(self, last_value: float = 0.0):
        states, actions, pre_tanh, old_log_probs = self.memory.get_tensors()
        advantages, returns = self.memory.compute_advantages(
            self.gamma, self.lam, last_value
        )
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batches = self.memory.get_batches()
        # Initialize accumulators as floats for mypy compatibility
        total_policy_loss = total_value_loss = total_entropy = total_loss = 0.0
        clip_frac = approx_kl = explained_var = 0.0
        for epoch in range(self.epochs):
            # Initialize epoch-level accumulators as floats
            epoch_policy_loss = epoch_value_loss = epoch_entropy = epoch_total_loss = 0.0
            epoch_clip_count = 0.0
            epoch_kl_sum = 0.0
            for batch_idxs in batches:
                b_states = states[batch_idxs]
                b_actions = actions[batch_idxs]
                b_pre_tanh = pre_tanh[batch_idxs]
                b_old_log_probs = old_log_probs[batch_idxs]
                b_adv = advantages[batch_idxs]
                b_ret = returns[batch_idxs]
                new_log_probs, state_values, entropy = self.actor_critic.evaluate(
                    b_states, b_actions, b_pre_tanh
                )
                ratios = torch.exp(new_log_probs - b_old_log_probs)
                # KL divergence
                with torch.no_grad():
                    log_ratio = new_log_probs - b_old_log_probs
                    batch_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                    epoch_kl_sum += batch_kl
                # Surrogate losses
                surr1 = ratios * b_adv
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * b_adv
                )
                actor_loss = -torch.min(surr1, surr2).mean()
                state_values = state_values.squeeze(-1)
                critic_loss = F.mse_loss(state_values, b_ret)
                entropy_bonus = entropy.mean()
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy_bonus
                )
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                # Clip fraction count
                with torch.no_grad():
                    epoch_clip_count += (
                        torch.abs(ratios - 1) > self.eps_clip
                    ).float().sum().item() / len(b_states)
                # Accumulate epoch losses
                epoch_policy_loss += actor_loss.item()
                epoch_value_loss += critic_loss.item()
                epoch_entropy += entropy_bonus.item()
                epoch_total_loss += loss.item()
            # Aggregate epoch metrics
            num_batches = len(batches)
            total_policy_loss += epoch_policy_loss / num_batches
            total_value_loss += epoch_value_loss / num_batches
            total_entropy += epoch_entropy / num_batches
            total_loss += epoch_total_loss / num_batches
            clip_frac += epoch_clip_count / num_batches
            approx_kl += epoch_kl_sum / num_batches
        # Explained variance
        with torch.no_grad():
            y_pred = torch.FloatTensor(self.memory.values).to(self.device)
            y_true = returns[:-1] if len(returns) > len(y_pred) else returns
            var_y = torch.var(y_true)
            # Compute explained variance safely as float
            if var_y.item() > 0:
                explained_var = (1 - torch.var(y_true - y_pred) / var_y).item()
            else:
                explained_var = 0.0
        # Normalize metrics
        avg_policy_loss = total_policy_loss / self.epochs
        avg_value_loss = total_value_loss / self.epochs
        avg_entropy = total_entropy / self.epochs
        avg_loss = total_loss / self.epochs
        avg_clip_frac = clip_frac / self.epochs
        avg_kl = approx_kl / self.epochs
        # Log results
        self.logger.info(
            "update_complete loss=%.4f policy_loss=%.4f value_loss=%.4f entropy=%.4f clip_frac=%.3f kl=%.5f explained_var=%.3f",
            avg_loss,
            avg_policy_loss,
            avg_value_loss,
            avg_entropy,
            avg_clip_frac,
            avg_kl,
            explained_var,
        )
        self.memory.clear()
        return {
            "loss": avg_loss,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "clip_fraction": avg_clip_frac,
            "approx_kl": avg_kl,
            "explained_variance": explained_var,
        }

    def save(self, path: str):
        # Save model and optimizer state
        torch.save(
            {
                "model": self.actor_critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        self.logger.info(f"model_saved path={path}")

    def load(self, path: str, load_optimizer: bool = True):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["model"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.logger.info(f"model_loaded path={path}")
        return checkpoint.get("config", {})
