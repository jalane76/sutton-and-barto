import random

class MultiArmedBandit:

    def __init__(self, k=10, epsilon=0.1, alpha=0, initial_values=None):
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        if initial_values is None:
            self.action_values = [random.gauss(0, 1) for i in range(0, k)]
        else:
            self.action_values = initial_values
        self.estimated_action_values = [0] * k
        self.action_counts = [0] * k

    def step(self):
        self.action = self.next_action()
        self.optimal_action = self.estimated_action_values.index(max(self.estimated_action_values))
        self.reward = self._reward()
        self.action_counts[self.action] += 1
        self.reward_update()

    def random_walk_values(self, mean, stddev):
        self.action_values = [v + random.gauss(mean, stddev) for v in self.action_values]

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
        if self.alpha > 0:
            step_size = self.alpha
        else:
            step_size = self.action_counts[self.action]
        self.estimated_action_values[self.action] = self._incremental_average(self.estimated_action_values[self.action], self.reward, step_size)

    # Helpers
    def _incremental_average(self, old_value, new_value, step_size):
        if step_size < 1:
            step_size = 1
        return old_value + (new_value - old_value) / step_size