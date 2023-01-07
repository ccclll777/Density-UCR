"""
带有优先级的buffer
"""
import numpy as np

class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1  # for beta calculation
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    def reset(self):
        self.__init__(capacity =self.capacity, alpha=0.6, beta_start=0.4, beta_frames=100000)
        # self.buffer.clear()
        # self.pos = 0
        # self.frame = 1  # for beta calculation
    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.

        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        max_prio = self.priorities.max() if self.buffer else 1.0  # gives max priority if buffer is not empty else 1
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity  # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
    def push_with_priority(self, state, action, reward, next_state, done,priority):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity  # lets the pos circle in the ranges of capacity if pos+1 > cap --> new posi = 0
    def sample(self, batch_size):
        N = len(self.buffer)
        if N == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        # calc P = p^a/sum(p^a)
        probs = prios ** self.alpha
        P = probs / probs.sum()
        # gets the indices depending on the probability p
        indices = np.random.choice(N, batch_size, p=P)
        samples = [self.buffer[idx] for idx in indices]
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        # Compute importance-sampling weight
        weights = (N * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        states, actions, rewards, next_states, dones = zip(*samples)
        return {"state_list": np.concatenate(states), "next_state_list":  np.concatenate(next_states),
                "action_list": actions, "reward_list": rewards,"indices_list": indices , "done_list": dones,"weights":weights}

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio)

    def __len__(self):
        return len(self.buffer)