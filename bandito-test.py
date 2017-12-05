import time
import matplotlib.pyplot as plt

import bandito

def main():
    k = 10
    epsilon = 0.01
    num_steps = 1000

    bandit = bandito.MultiArmedBandit(k, epsilon)
    actions = []
    optimal_actions = []
    rewards = []
    for step in range(0, num_steps):
        bandit.step()
        actions.append(bandit.action)
        optimal_actions.append(bandit.optimal_action)
        rewards.append(bandit.reward)

    plt.subplot(311)
    plt.scatter(range(0, k), bandit.action_values)
    plt.subplot(312)
    plt.scatter(range(0, num_steps), actions, c='g')
    plt.scatter(range(0, num_steps), optimal_actions, c='r')
    plt.subplot(313)
    plt.plot(range(0, num_steps), rewards)
    plt.show()


if __name__ == "__main__":
    start = time.time()
    main()
    print('RUNTIME: ', time.time() - start, 's')