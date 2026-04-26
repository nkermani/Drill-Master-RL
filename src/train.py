"""Training loop for Multi-Agent RL"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import os


class ReplayBuffer:
    def __init__(self, capacity: int = 100000, num_agents: int = 10):
        self.capacity = capacity
        self.num_agents = num_agents
        self.position = 0
        self.size = 0
        
        self.obs = None
        self.actions = None
        self.rewards = None
        self.next_obs = None
        self.dones = None
    
    def _allocate(self, batch_size: int):
        if self.obs is None:
            max_capacity = self.capacity
            
            self.obs = np.zeros((max_capacity, self.num_agents, 6), dtype=np.float32)
            self.actions = np.zeros((max_capacity, self.num_agents), dtype=np.int64)
            self.rewards = np.zeros((max_capacity, self.num_agents), dtype=np.float32)
            self.next_obs = np.zeros((max_capacity, self.num_agents, 6), dtype=np.float32)
            self.dones = np.zeros((max_capacity, self.num_agents), dtype=np.bool_)
    
    def add(self, obs, actions, rewards, next_obs, dones):
        self._allocate(1)
        
        self.obs[self.position] = obs
        self.actions[self.position] = actions
        self.rewards[self.position] = rewards
        self.next_obs[self.position] = next_obs
        self.dones[self.position] = dones
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple:
        if self.size < batch_size:
            return None
        
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.tensor(self.obs[indices], dtype=torch.float32),
            torch.tensor(self.actions[indices], dtype=torch.long),
            torch.tensor(self.rewards[indices], dtype=torch.float32),
            torch.tensor(self.next_obs[indices], dtype=torch.float32),
            torch.tensor(self.dones[indices], dtype=torch.bool)
        )


class PPOAgent:
    def __init__(
        self,
        policy,
        value_network,
        optimizer,
        num_agents: int = 10,
        action_dim: int = 5,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5
    ):
        self.policy = policy
        self.value_network = value_network
        self.optimizer = optimizer
        
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def compute_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                next_value = 0
            else:
                next_value = next_values[t]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t].float()) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t].float()) * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        return returns, advantages
    
    def update(
        self,
        batch: Tuple,
        epoch: int
    ) -> Dict:
        obs_batch, actions_batch, rewards_batch, next_obs_batch, dones_batch = batch
        
        action_probs, values = self.policy(obs_batch, torch.zeros(obs_batch.shape[1], 2, dtype=torch.long))
        
        _, next_values = self.policy(next_obs_batch, torch.zeros(obs_batch.shape[1], 2, dtype=torch.long))
        
        returns, advantages = self.compute_returns(
            rewards_batch, dones_batch, values, next_values.detach()
        )
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions_batch)
        
        entropy = dist.entropy().mean()
        
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        value_loss = F.mse_loss(values, returns.detach())
        
        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item()
        }


class Trainer:
    def __init__(
        self,
        env,
        policy,
        value_network,
        num_agents: int = 10,
        learning_rate: float = 3e-4,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        num_epochs: int = 10,
        max_steps: int = 1000
    ):
        self.env = env
        self.policy = policy
        self.value_network = value_network
        
        params = list(policy.parameters()) + list(value_network.parameters())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        
        self.agent = PPOAgent(
            policy=policy,
            value_network=value_network,
            optimizer=self.optimizer,
            num_agents=num_agents
        )
        
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, num_agents=num_agents)
        
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_steps = max_steps
    
    def collect_experience(self, num_steps: int = 100) -> Dict:
        obs, info = self.env.reset()
        
        total_reward = 0
        total_steps = 0
        
        for _ in range(num_steps):
            robot_features = torch.tensor(obs['robot_features'], dtype=torch.float32).unsqueeze(0)
            edge_index = torch.tensor([[0, 0]], dtype=torch.long)
            
            with torch.no_grad():
                action_probs, value = self.policy(robot_features, edge_index)
            
            actions = action_probs[0].argmax(dim=-1).numpy()
            
            next_obs, rewards, terminations, truncations, info = self.env.step(actions)
            
            self.replay_buffer.add(
                obs['robot_features'],
                actions,
                rewards.numpy(),
                next_obs['robot_features'],
                terminations
            )
            
            total_reward += rewards.sum()
            total_steps += 1
            
            obs = next_obs
            
            if truncations.any():
                break
        
        return {
            'total_reward': total_reward,
            'total_steps': total_steps,
            'info': info
        }
    
    def train(self, num_updates: int = 1000, log_interval: int = 10) -> Dict:
        history = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_reward': []
        }
        
        for update in tqdm(range(num_updates), desc="Training"):
            exp_info = self.collect_experience(num_steps=self.max_steps)
            
            for _ in range(self.num_epochs):
                batch = self.replay_buffer.sample(self.batch_size)
                
                if batch is None:
                    break
                
                update_stats = self.agent.update(batch, epoch=update)
                
                history['policy_loss'].append(update_stats['policy_loss'])
                history['value_loss'].append(update_stats['value_loss'])
                history['entropy'].append(update_stats['entropy'])
            
            history['total_reward'].append(exp_info['total_reward'])
            
            if update % log_interval == 0:
                print(f"Update {update}: Reward={exp_info['total_reward']:.2f}")
        
        return history
    
    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def visualize_training(history: Dict, save_path: str = 'training_curves.png'):
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(history['policy_loss'])
        axes[0, 0].set_title('Policy Loss')
        axes[0, 0].set_xlabel('Update')
        
        axes[0, 1].plot(history['value_loss'])
        axes[0, 1].set_title('Value Loss')
        axes[0, 1].set_xlabel('Update')
        
        axes[1, 0].plot(history['entropy'])
        axes[1, 0].set_title('Entropy')
        axes[1, 0].set_xlabel('Update')
        
        axes[1, 1].plot(history['total_reward'])
        axes[1, 1].set_title('Total Reward')
        axes[1, 1].set_xlabel('Update')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except ImportError:
        print("Matplotlib not available for visualization")