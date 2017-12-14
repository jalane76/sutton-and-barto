import random
import copy
import math

class Bandit:
    def __init__(self, k=10, initial_mu=0, initial_sigma=1, initial_values=None, step_mu=0, step_sigma=0, reward_sigma=1):
        self._k = k
        self._step_mu = step_mu
        self._step_sigma = step_sigma
        self._reward_sigma = reward_sigma

        if self._reward_sigma > 0:
            self._reward = self._stochastic_reward
        else:
            self._reward = self._deterministic_reward

        if initial_values is not None and len(initial_values) == self._k:
            self._action_values = copy.deepcopy(initial_values)
        else:
            self._action_values = [random.gauss(initial_mu, initial_sigma) for i in range(self._k)]

    def reward_action(self, action):
        if action not in range(self._k):
            reward = 0
        else:
            reward = self._reward(action)

        if self._step_sigma > 0:
            self._action_values = [v + random.gauss(self._step_mu, self._step_sigma) for v in self._action_values]

        return reward

    @property
    def k(self):
        return self._k

    @property
    def optimal_action(self):
        return self._action_values.index(max(self._action_values))

    @property
    def action_values(self):
        return copy.copy(self._action_values)

    def _stochastic_reward(self, action):
        return random.gauss(self._action_values[action], self._reward_sigma)

    def _deterministic_reward(self, action):
        return self._action_values[action]


class Agent:
    def __init__(self, k=10, epsilon=0.1, alpha=0, c=0, grad_step=0, initial_estimates=None):
        self._k = k
        self._epsilon = epsilon
        self._alpha = alpha
        self._c = c
        self._grad_step = grad_step

        if initial_estimates is None:
            self._estimates = [0] * k
        else:
            self._estimates = initial_estimates

        self._action_counts = [0] * k
        self._reward_average = 0

        if c > 0:
            self._choose_action = self._upper_conf_bound_selection
        elif grad_step > 0:
            self._choose_action = self._gradient_selection
        else:
            self._choose_action = self._epsilon_greedy_selection

        if alpha > 0 and grad_step == 0:
            self.process_reward = self._fixed_step_reward
        elif grad_step > 0:
            self.process_reward = self._gradient_reward
        else:
            self.process_reward = self._sample_avg_reward

    @property
    def k(self):
        return self._k

    # Action selection
    @property
    def action(self):
        action =  self._choose_action()
        self._action_counts[action] += 1
        return action

    def _epsilon_greedy_selection(self):
        chance = random.random()
        if chance < self._epsilon:
            return random.randint(0, self.k - 1)
        else:
            # do this to give equal chance to all max values
            max_estimate = max(self._estimates)
            max_indices = []
            for i, estimate in enumerate(self._estimates):
                if estimate == max_estimate:
                    max_indices.append(i)
            return random.choice(max_indices)

    def _upper_conf_bound_selection(self):
        bounds = [self._calc_bound(e, c) for e, c in zip(self._estimates, self._action_counts)]
        return bounds.index(max(bounds))

    def _gradient_selection(self):
        if self._estimates == [0] * self._k:
            return random.randint(0, len(self._estimates) - 1)
        else:
            return self._estimates.index(random.choices(self._estimates, self._calc_pis(), k=1)[0])

    # Reward processing
    def _sample_avg_reward(self, action, reward):
        self._estimates[action] = self._incremental_average(self._estimates[action], reward, self._action_counts[action])

    def _fixed_step_reward(self, action, reward):
        self._estimates[action] = self._incremental_average(self._estimates[action], reward, 1 / self._alpha)

    def _gradient_reward(self, action, reward):
        if self._alpha > 0:
            self._reward_average = self._incremental_average(self._reward_average, reward, 1 / self._alpha)
        else:
            self._reward_average = self._incremental_average(self._reward_average, reward, sum(self._action_counts))
        pis = self._calc_pis()
        for index in range(len(self._estimates)):
            if index == action:
                self._estimates[index] = self._estimates[index] + self._grad_step * (reward - self._reward_average) * (1 - pis[action])
            else:
                self._estimates[index] = self._estimates[index] - self._grad_step * (reward - self._reward_average) * pis[action]
        # clamp estimates
        self._estimates = [self._clamp(e, -100, 100) for e in self._estimates]


    # Helpers
    def _incremental_average(self, old_value, new_value, step_size):
        return old_value + (new_value - old_value) / step_size

    def _calc_bound(self, estimate, count):
        if count == 0:
            return math.inf
        return estimate + self._c * math.sqrt(math.log(sum(self._action_counts)) / count)

    def _calc_pis(self):
        normalizer = sum([math.exp(estimate) for estimate in self._estimates])
        return [math.exp(estimate) / normalizer for estimate in self._estimates]

    def _clamp(self, value, min_limit, max_limit):
        return max(min(value, max_limit), min_limit)