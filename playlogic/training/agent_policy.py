import torch
import torch.nn as nn
import torch.nn.functional as F

class SoccerAgent(nn.Module):
    """Small MLP for one player. Outputs move dx,dy and pass/shoot logits."""
    def __init__(self, input_dim=40, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.move_head = nn.Linear(hidden_dim, 2)     # dx, dy in [-8,8]
        self.pass_head = nn.Linear(hidden_dim, 1)     # logit for pass
        self.shoot_head = nn.Linear(hidden_dim, 1)    # logit for shoot

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        move = torch.tanh(self.move_head(x)) * 8.0
        pass_logit = self.pass_head(x)
        shoot_logit = self.shoot_head(x)
        return move, pass_logit, shoot_logit