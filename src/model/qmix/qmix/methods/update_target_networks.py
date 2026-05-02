# src/model/qmix/qmix/methods/update_target_networks.py

"""Update target networks method for QMIX"""

def _update_target_networks(self):
    for target_agent, agent in zip(self.target_agents, self.agents):
        target_agent.load_state_dict(agent.state_dict())

    self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
