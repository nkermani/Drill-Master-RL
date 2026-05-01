# src/train/replay_buffer/methods/add.py

"""Add method for ReplayBuffer"""


def add(self, obs, actions, rewards, next_obs, dones):
    self._allocate(1)

    self.obs[self.position] = obs
    self.actions[self.position] = actions
    self.rewards[self.position] = rewards
    self.next_obs[self.position] = next_obs
    self.dones[self.position] = dones

    self.position = (self.position + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)
