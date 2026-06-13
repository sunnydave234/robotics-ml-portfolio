"""
model.py — W4D3
Trivial 3-layer MLP for the behavior-cloning smoke test.

Input:  concat(state, prev_action)  -> (state_dim + action_dim,)
Output: action                       -> (action_dim,)

For pusht: input_dim=4, output_dim=2
"""
import torch
import torch.nn as nn

class BCMLP(nn.Module):
    """ 3-layer MLP: input_dim -> hidden -> hidden -> output_dim """
    def __init__(self, input_dim: int, hidden_dim: int=256, output_dim: int=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
