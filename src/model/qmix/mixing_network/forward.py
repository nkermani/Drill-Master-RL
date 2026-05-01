# src/model/qmix/mixing_network/forward.py

"""Forward method for MixingNetwork"""

import torch.nn.functional as F


def forward(
    self,
    agent_qs: torch.Tensor,
    state: torch.Tensor
) -> torch.Tensor:
    w1 = torch.abs(self.hyper_w1(state))
    w1 = w1.reshape(-1, self.num_agents, self.hidden_dim)

    b1 = self.hyper_b1(state).reshape(-1, 1, self.hidden_dim)

    hidden = agent_qs.unsqueeze(-1)

    hidden = torch.bmm(hidden, w1) + b1
    hidden = F.relu(hidden)

    w2 = torch.abs(self.hyper_w2(state))
    w2 = w2.reshape(-1, self.hidden_dim, 1)

    b2 = self.hyper_b2(state).reshape(-1, 1, 1)

    q_tot = torch.bmm(hidden, w2) + b2
    q_tot = q_tot.squeeze(-1)

    return q_tot
