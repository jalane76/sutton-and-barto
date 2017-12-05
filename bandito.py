import random

class MultiArmedBandit:

    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.action_values = [random.gauss(0, 1) for i in range(0, k)]
        self.estimated_action_values = [0] * k
        self.action_counts = [0] * k

    def step(self):
        self.action = self.next_action()
        self.optimal_action = self.estimated_action_values.index(max(self.estimated_action_values))
        self.reward = self._reward()
        self.action_counts[self.action] += 1
        self.reward_update()

    # Action selection
    def next_action(self):
        chance = random.random()
        if chance < self.epsilon:
            return random.randint(0, self.k - 1)
        else:
            # do this to give equal chance to all max values
            max_reward = max(self.estimated_action_values)
            max_indices = [self.estimated_action_values.index(r) for r in self.estimated_action_values if
                           r == max_reward]
            if len(max_indices) > 1:
                return random.randint(0, len(max_indices) - 1)
            else:
                return max_indices[0]

    # Reward calculations
    def _reward(self):
        return random.gauss(self.action_values[self.action], 1)

    def reward_update(self):
        self.estimated_action_values[self.action] = self._incremental_average(self.estimated_action_values[self.action], self.reward, self.action_counts[self.action])

    # Helpers
    def _incremental_average(self, old_value, new_value, step_size):
        if step_size < 1:
            step_size = 1
        return old_value + (new_value - old_value) / step_size