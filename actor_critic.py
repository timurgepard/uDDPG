import math
import torch
import torch.nn as nn


# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=32, max_action=1.0):
        super(Actor, self).__init__()
        self.device = device

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
         )
        
        self.max_action = torch.mean(max_action).item()
        self.std = 3.0
        self.x_coor = 0.0

    def accuracy(self):
        if self.std>0.01:
            self.std = 3.0 * self.max_action * math.exp(-self.x_coor)
            self.x_coor += 3e-5
            return True
        return False

    def forward(self, state, mean=False):
        x = self.max_action*self.net(state)
        if mean: return x
        if self.accuracy(): x += torch.normal(torch.zeros_like(x), self.std)
        return x.clamp(-1.0, 1.0)

        
        
# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim+action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        return self.net(x)
