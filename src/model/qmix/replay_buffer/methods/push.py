# src/model/qmix/replay_buffer/methods/push.py

"""Push method for ReplayBuffer"""


def push(self, *experience):
    if len(self.buffer) < self.capacity:
        self.buffer.append(experience)
    else:
        self.buffer[self.position] = experience

    self.position = (self.position + 1) % self.capacity
