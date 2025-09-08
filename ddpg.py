import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict
from copy import deepcopy
import torch
import os
from torch.nn import LazyLinear
import torch.nn.functional as F
import torch.optim as optim
import pickle

from utils import trunc_normal
import time

from IPython.display import clear_output
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pvt_graph import PVTGraph

class ReplayBuffer:
    """
    Algorithm: Experience Replay Buffer with PVT Corner Support
    Features:
    - Separate buffers for each PVT corner
    - Efficient storage and sampling mechanisms
    - Support for attention weights and rewards
    """
    def __init__(self, CktGraph, PVT_Graph, size: int, batch_size: int = 32):
        self.size = size
        self.batch_size = batch_size
        self.num_corners = PVT_Graph.num_corners
        self.obs_shape = CktGraph.obs_shape
        self.action_shape = CktGraph.action_shape
        self.device = CktGraph.device
        
        # Initialize corner buffers
        self.corner_buffers = {}
        for corner_idx in range(self.num_corners):
            self.corner_buffers[corner_idx] = {
                'obs': np.zeros((size,) + self.obs_shape, dtype=np.float32),
                'next_obs': np.zeros((size,) + self.obs_shape, dtype=np.float32),
                'action': np.zeros((size,) + self.action_shape, dtype=np.float32),
                'reward': np.zeros(size, dtype=np.float32),
                'total_reward': np.zeros(size, dtype=np.float32),
                'done': np.zeros(size, dtype=bool),
                'attention_weights': np.zeros(size, dtype=np.float32),
                'ptr': 0,
                'size': 0
            }

    def store(
        self,
        pvt_state: np.ndarray,
        action: np.ndarray,
        results_dict: dict,
        next_pvt_state: np.ndarray,
        corner_indices: list,
        attention_weights: np.ndarray,
        total_reward: float,
        done: bool,
    ):
        """Store experience for each selected corner"""
        for i, corner_idx in enumerate(corner_indices):
            if corner_idx in self.corner_buffers:
                buffer = self.corner_buffers[corner_idx]
                ptr = buffer['ptr']
                
                # Store experience
                buffer['obs'][ptr] = pvt_state
                buffer['next_obs'][ptr] = next_pvt_state
                buffer['action'][ptr] = action
                buffer['reward'][ptr] = results_dict.get(corner_idx, {}).get('reward', 0.0)
                buffer['total_reward'][ptr] = total_reward
                buffer['done'][ptr] = done
                buffer['attention_weights'][ptr] = attention_weights[i] if i < len(attention_weights) else 1.0
                
                # Update pointers
                buffer['ptr'] = (ptr + 1) % self.size
                buffer['size'] = min(buffer['size'] + 1, self.size)

    def sample_corner_batch(self, corner_idx: int) -> Dict[str, np.ndarray]:
        """Sample batch from specific corner buffer"""
        if corner_idx not in self.corner_buffers:
            return None
            
        buffer = self.corner_buffers[corner_idx]
        if buffer['size'] < self.batch_size:
            return None
            
        # Random sampling
        indices = np.random.choice(buffer['size'], self.batch_size, replace=False)
        
        return {
            'obs': torch.FloatTensor(buffer['obs'][indices]).to(self.device),
            'next_obs': torch.FloatTensor(buffer['next_obs'][indices]).to(self.device),
            'action': torch.FloatTensor(buffer['action'][indices]).to(self.device),
            'reward': torch.FloatTensor(buffer['reward'][indices]).to(self.device),
            'total_reward': torch.FloatTensor(buffer['total_reward'][indices]).to(self.device),
            'done': torch.BoolTensor(buffer['done'][indices]).to(self.device),
            'attention_weights': torch.FloatTensor(buffer['attention_weights'][indices]).to(self.device)
        }
    
    def can_sample(self, corner_idx: int) -> bool:
        """Check if corner buffer has enough samples"""
        if corner_idx not in self.corner_buffers:
            return False
        return self.corner_buffers[corner_idx]['size'] >= self.batch_size


class DDPGAgent:
    def __init__(
        self,
        env,
        CktGraph,
        PVT_Graph,
        Actor,
        Critic,
        memory_size: int,
        batch_size: int,
        noise_sigma: float,
        noise_sigma_min: float,
        noise_sigma_decay: float,
        noise_type: str,
        gamma: float = 0.99,
        tau: float = 5e-3,
        initial_random_steps: int = 1e4,
        sample_num: int = 3,
        agent_folder: str = None,
        old = False
    ):
        """
        Algorithm: DDPG Agent Initialization
        """
        self.env = env
        self.CktGraph = CktGraph
        self.PVT_Graph = PVT_Graph
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.initial_random_steps = int(initial_random_steps)
        self.sample_num = sample_num
        
        # Noise parameters
        self.noise_sigma = noise_sigma
        self.noise_sigma_min = noise_sigma_min
        self.noise_sigma_decay = noise_sigma_decay
        self.noise_type = noise_type
        
        # Initialize networks
        self.device = CktGraph.device
        self.actor = Actor.to(self.device)
        self.actor_target = deepcopy(self.actor)
        self.critic = Critic.to(self.device)
        self.critic_target = deepcopy(self.critic)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(CktGraph, PVT_Graph, memory_size, batch_size)
        
        # Training state
        self.total_step = 0
        self.is_test = False
        
        # Load agent if folder provided
        if agent_folder and old:
            self.load_agent(agent_folder)

    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select action using actor network with exploration noise"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        # Add exploration noise
        if not self.is_test and self.total_step < self.initial_random_steps:
            # Random exploration
            action = np.random.uniform(-1, 1, size=action.shape)
        elif not self.is_test:
            # Add noise to policy action
            if self.noise_type == 'uniform':
                noise = np.random.uniform(-self.noise_sigma, self.noise_sigma, size=action.shape)
            else:  # gaussian
                noise = np.random.normal(0, self.noise_sigma, size=action.shape)
            action = np.clip(action + noise, -1, 1)
        
        # Sample corners using attention mechanism
        if hasattr(self.actor, 'sample_corners'):
            attention_weights, corner_indices = self.actor.sample_corners(self.sample_num)
            attention_weights = attention_weights.cpu().numpy()
        else:
            corner_indices = np.random.choice(self.PVT_Graph.num_corners, self.sample_num, replace=False)
            attention_weights = np.ones(self.sample_num) / self.sample_num
        
        return action, corner_indices, attention_weights

    def step(self, action: Tuple[np.ndarray, np.ndarray, bool]) -> Tuple:
        """Take step in environment"""
        return self.env.step(action)

    def update_model(self) -> Tuple[float, float]:
        """Update actor and critic networks"""
        actor_loss_total = 0.0
        critic_loss_total = 0.0
        updates = 0
        
        # Update for each corner that has enough samples
        for corner_idx in range(self.PVT_Graph.num_corners):
            if not self.memory.can_sample(corner_idx):
                continue
                
            batch = self.memory.sample_corner_batch(corner_idx)
            if batch is None:
                continue
            
            # Update critic
            critic_loss = self._update_critic(batch)
            
            # Update actor
            actor_loss = self._update_actor(batch)
            
            actor_loss_total += actor_loss
            critic_loss_total += critic_loss
            updates += 1
        
        if updates > 0:
            actor_loss_total /= updates
            critic_loss_total /= updates
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target, self.tau)
            self._soft_update(self.critic, self.critic_target, self.tau)
        
        return actor_loss_total, critic_loss_total

    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update critic network"""
        obs = batch['obs']
        next_obs = batch['next_obs']
        action = batch['action']
        reward = batch['reward'].unsqueeze(1)
        done = batch['done'].unsqueeze(1)
        
        # Compute target Q value
        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_action)
            target_q = reward + (1 - done.float()) * self.gamma * target_q
        
        # Compute current Q value
        current_q = self.critic(obs, action)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()

    def _update_actor(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update actor network"""
        obs = batch['obs']
        
        # Actor loss
        action = self.actor(obs)
        actor_loss = -self.critic(obs, action).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item()

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self, num_steps: int, plot_interval: int, check_interval: int, continue_training: bool = False):
        """Train the agent"""
        print(f"Starting training for {num_steps} steps...")
        
        for step in range(num_steps):
            # Reset environment
            pvt_state, _ = self.env.reset()
            
            # Select action
            action, corner_indices, attention_weights = self.select_action(pvt_state)
            
            # Take step
            results_dict, flag, terminated, truncated, info = self.step(
                (action, corner_indices, False)
            )
            
            # Calculate total reward
            total_reward = sum(result.get('reward', 0) for result in results_dict.values())
            
            # Store experience
            self.memory.store(
                pvt_state=pvt_state,
                action=action,
                results_dict=results_dict,
                next_pvt_state=pvt_state,  # Same state for episodic env
                corner_indices=corner_indices,
                attention_weights=attention_weights,
                total_reward=total_reward,
                done=terminated or truncated
            )
            
            # Update networks
            if self.total_step > self.batch_size:
                actor_loss, critic_loss = self.update_model()
                
                if step % check_interval == 0:
                    print(f"Step {step}: Actor Loss = {actor_loss:.4f}, Critic Loss = {critic_loss:.4f}")
            
            # Decay noise
            if self.noise_sigma > self.noise_sigma_min:
                self.noise_sigma *= self.noise_sigma_decay
            
            self.total_step += 1
            
            if step % plot_interval == 0:
                print(f"Step {step}, Total Reward: {total_reward:.4f}, Noise: {self.noise_sigma:.4f}")

    def load_agent(self, agent_folder: str):
        """Load agent from folder"""
        try:
            # Load actor weights
            actor_path = os.path.join(agent_folder, "actor.pth")
            if os.path.exists(actor_path):
                self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                self.actor_target.load_state_dict(self.actor.state_dict())
                print(f"Loaded actor weights from {actor_path}")
            
            # Load critic weights
            critic_path = os.path.join(agent_folder, "critic.pth")
            if os.path.exists(critic_path):
                self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                self.critic_target.load_state_dict(self.critic.state_dict())
                print(f"Loaded critic weights from {critic_path}")
                
        except Exception as e:
            print(f"Error loading agent: {e}")

    def load_replay_buffer(self, buffer_path: str):
        """Load replay buffer from file"""
        try:
            with open(buffer_path, 'rb') as f:
                self.memory = pickle.load(f)
            print(f"Loaded replay buffer from {buffer_path}")
        except Exception as e:
            print(f"Error loading replay buffer: {e}")

    def _normalize_pvt_graph_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize PVT graph state"""
        state = state.clone()
        # Simple normalization - can be enhanced based on domain knowledge
        state = (state - state.mean()) / (state.std() + 1e-6)
        return state