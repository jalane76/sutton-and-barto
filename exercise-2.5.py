import sys
import argparse
import time
import matplotlib.pyplot as plt

import bandito

def main():
    k = 10
    epsilon = 0.1
    alpha = 0.1
    mean = 0
    stddev = 0.01
    max_steps = 10000
    max_runs = 2000

    avg_reward_average = [0] * max_steps
    optimal_action_average = [0] * max_steps
    avg_reward_fixed = [0] * max_steps
    optimal_action_fixed = [0] * max_steps

    for run in range(0, max_runs):
        sys.stdout.write('\r %d/%d' % (run + 1, max_runs))
        sys.stdout.flush()

        average_bandit = bandito.MultiArmedBandit(k=k, epsilon=epsilon, initial_values=[1]*k)
        fixed_bandit = bandito.MultiArmedBandit(k=k, epsilon=epsilon, alpha=alpha, initial_values=[1]*k)

        for step in range(0, max_steps):
            average_bandit.step()
            avg_reward_average[step] += average_bandit.reward
            if average_bandit.action == average_bandit.action_values.index(max(average_bandit.action_values)):
                optimal_action_average[step] += 1
            average_bandit.random_walk_values(mean, stddev)

            fixed_bandit.step()
            avg_reward_fixed[step] += fixed_bandit.reward
            if fixed_bandit.action == fixed_bandit.action_values.index(max(fixed_bandit.action_values)):
                optimal_action_fixed[step] += 1
            fixed_bandit.random_walk_values(mean, stddev)
    print() # basically a newline

    #normalize
    avg_reward_average = [r / max_runs for r in avg_reward_average]
    optimal_action_average = [a / max_runs * 100 for a in optimal_action_average]
    avg_reward_fixed = [r / max_runs for r in avg_reward_fixed]
    optimal_action_fixed = [a / max_runs * 100 for a in optimal_action_fixed]

    plt.subplot(311)
    plt.plot(range(0, max_steps), avg_reward_average, label='Average')
    plt.plot(range(0, max_steps), avg_reward_fixed, label='Fixed')
    plt.subplot(312)
    plt.plot(range(0, max_steps), optimal_action_average, label='Average')
    plt.plot(range(0, max_steps), optimal_action_fixed, label='Fixed')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    start = time.time()
    main()
    print('RUNTIME: ', time.time() - start, 's')