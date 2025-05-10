# policy.py - Policy network implementation
import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
    """
    Combined actor-critic network for the Gold Rush environment.
    
    The network takes a flattened observation tensor and outputs:
    1. Action logits for each agent
    2. Value estimate for the current state
    
    Architecture is intentionally simple for ease of use:
    - Two hidden layers with ReLU activations
    - Separate output heads for actor and critic
    """
    def __init__(self, input_dim, hidden_dim, action_dim):
        """
        Initialize the network.
        
        Args:
            input_dim: Dimension of flattened observation
            hidden_dim: Size of hidden layers
            action_dim: Total number of action logits (5*num_agents)
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor or numpy array
            
        Returns:
            logits: Action logits (batch, action_dim)
            value: Value estimate (batch,)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        feat = self.fc(x)
        logits = self.actor(feat)        # tensor (batch, action_dim)
        value = self.critic(feat).squeeze(-1)
        return logits, value

    def act(self, obs):
        """
        Sample actions stochastically for training.
        
        Args:
            obs: Observation tensor or numpy array
            
        Returns:
            Tuple of (actions, log_probs, values) as NumPy arrays
        """
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        with torch.no_grad():                       # Prevents gradient tracking
            logits, value = self.forward(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)
        return (action.detach().cpu().numpy(),
                logp.detach().cpu().numpy(),
                value.detach().cpu().numpy())