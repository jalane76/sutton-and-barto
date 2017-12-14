import sys
import time
from datetime import timedelta
import matplotlib.pyplot as plt

import bandido

def main():
    k = 10
    epsilon = 0.1
    c = 2
    max_steps = 1000
    max_runs = 2000


    average_reward_epsilon_greedy = [0] * max_steps
    optimal_action_epsilon_greedy = [0] * max_steps

    average_reward_ucb = [0] * max_steps
    optimal_action_ucb = [0] * max_steps

    for run in range(0, max_runs):
        sys.stdout.write('\r %d/%d' % (run + 1, max_runs))
        sys.stdout.flush()

        # Epsilon greedy
        epsilon_greedy_bandit = bandido.Bandit(k=k)
        epsilon_greedy_agent = bandido.Agent(k=k, epsilon=epsilon)
        for step in range(0, max_steps):
            action = epsilon_greedy_agent.action
            reward = epsilon_greedy_bandit.reward_action(action)
            epsilon_greedy_agent.process_reward(action, reward)

            average_reward_epsilon_greedy[step] += reward
            if action == epsilon_greedy_bandit.optimal_action:
                optimal_action_epsilon_greedy[step] += 1

        # UCB
        ucb_bandit = bandido.Bandit(k=k)
        ucb_agent = bandido.Agent(k=k, epsilon=epsilon, c=c)
        for step in range(0, max_steps):
            action = ucb_agent.action
            reward = ucb_bandit.reward_action(action)
            ucb_agent.process_reward(action, reward)

            average_reward_ucb[step] += reward
            if action == ucb_bandit.optimal_action:
                optimal_action_ucb[step] += 1

    print() # basically a newline

    #normalize
    average_reward_epsilon_greedy = [r / max_runs for r in average_reward_epsilon_greedy]
    optimal_action_epsilon_greedy = [a / max_runs * 100 for a in optimal_action_epsilon_greedy]

    average_reward_ucb = [r / max_runs for r in average_reward_ucb]
    optimal_action_ucb = [a / max_runs * 100 for a in optimal_action_ucb]

    ax1 = plt.subplot(311)
    ax1.set_title('Average reward', loc='left')
    plt.plot(range(0, max_steps), average_reward_epsilon_greedy, label='Epsilon greedy')
    plt.plot(range(0, max_steps), average_reward_ucb, label='UCB')
    ax2 = plt.subplot(312)
    ax2.set_title('% Optimal action', loc='left')
    plt.plot(range(0, max_steps), optimal_action_epsilon_greedy, label='Epsilon greedy')
    plt.plot(range(0, max_steps), optimal_action_ucb, label='UCB')
    plt.legend(loc='lower right')
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    start = time.time()
    main()
    print('RUNTIME: ', time.time() - start, 's')