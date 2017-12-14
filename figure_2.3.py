import sys
import time
from datetime import timedelta
import matplotlib.pyplot as plt

import bandido

def main():
    k = 10
    epsilon = 0.1
    alpha = 0.1
    max_steps = 1000
    max_runs = 2000


    average_reward_optimistic = [0] * max_steps
    optimal_action_optimistic = [0] * max_steps

    average_reward_realistic = [0] * max_steps
    optimal_action_realistic = [0] * max_steps

    for run in range(0, max_runs):
        sys.stdout.write('\r %d/%d' % (run + 1, max_runs))
        sys.stdout.flush()

        # Optimistic run
        optimistic_bandit = bandido.Bandit(k=k)
        optimistic_agent = bandido.Agent(k=k, epsilon=0, alpha=alpha, initial_estimates=[5]*k)
        for step in range(0, max_steps):
            action = optimistic_agent.action
            reward = optimistic_bandit.reward_action(action)
            optimistic_agent.process_reward(action, reward)

            average_reward_optimistic[step] += reward
            if action == optimistic_bandit.optimal_action:
                optimal_action_optimistic[step] += 1

        # Realistic run
        realistic_bandit = bandido.Bandit(k=k)
        realistic_agent = bandido.Agent(k=k, epsilon=epsilon, alpha=alpha)
        for step in range(0, max_steps):
            action = realistic_agent.action
            reward = realistic_bandit.reward_action(action)
            realistic_agent.process_reward(action, reward)

            average_reward_realistic[step] += reward
            if action == realistic_bandit.optimal_action:
                optimal_action_realistic[step] += 1

    print() # basically a newline

    #normalize
    average_reward_optimistic = [r / max_runs for r in average_reward_optimistic]
    optimal_action_optimistic = [a / max_runs * 100 for a in optimal_action_optimistic]

    average_reward_realistic = [r / max_runs for r in average_reward_realistic]
    optimal_action_realistic = [a / max_runs * 100 for a in optimal_action_realistic]

    ax1 = plt.subplot(311)
    ax1.set_title('Average reward', loc='left')
    plt.plot(range(0, max_steps), average_reward_optimistic, label='Optimistic')
    plt.plot(range(0, max_steps), average_reward_realistic, label='Realistic')
    ax2 = plt.subplot(312)
    ax2.set_title('% Optimal action', loc='left')
    plt.plot(range(0, max_steps), optimal_action_optimistic, label='Optimistic')
    plt.plot(range(0, max_steps), optimal_action_realistic, label='Realistic')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    start = time.time()
    main()
    print('RUNTIME: ', str(timedelta(seconds=(time.time() - start))))