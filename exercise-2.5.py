import sys
import time
from datetime import timedelta
import matplotlib.pyplot as plt

import bandido

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

        average_bandit = bandido.Bandit(k=k, step_mu=mean, step_sigma=stddev)
        average_agent = bandido.Agent(k=k, epsilon=epsilon)
        fixed_bandit = bandido.Bandit(k=k, step_mu=mean, step_sigma=stddev)
        fixed_agent = bandido.Agent(k=k, epsilon=epsilon, alpha=alpha)

        for step in range(0, max_steps):
            # sample average
            action = average_agent.action
            reward = average_bandit.reward_action(action)
            average_agent.process_reward(action, reward)

            avg_reward_average[step] += reward
            if action == average_bandit.optimal_action:
                optimal_action_average[step] += 1

            # fixed step
            action = fixed_agent.action
            reward = fixed_bandit.reward_action(action)
            fixed_agent.process_reward(action, reward)

            avg_reward_fixed[step] += reward
            if action == fixed_bandit.optimal_action:
                optimal_action_fixed[step] += 1

    print() # basically a newline

    #normalize
    avg_reward_average = [r / max_runs for r in avg_reward_average]
    optimal_action_average = [a / max_runs * 100 for a in optimal_action_average]
    avg_reward_fixed = [r / max_runs for r in avg_reward_fixed]
    optimal_action_fixed = [a / max_runs * 100 for a in optimal_action_fixed]

    ax1 = plt.subplot(311)
    ax1.set_title('Average reward', loc='left')
    plt.plot(range(0, max_steps), avg_reward_average, label='Average')
    plt.plot(range(0, max_steps), avg_reward_fixed, label='Fixed')
    ax2 = plt.subplot(312)
    ax2.set_title('% Optimal action', loc='left')
    plt.plot(range(0, max_steps), optimal_action_average, label='Average')
    plt.plot(range(0, max_steps), optimal_action_fixed, label='Fixed')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start = time.time()
    main()
    print('RUNTIME: ', str(timedelta(seconds=(time.time() - start))))