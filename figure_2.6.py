import sys
import time
from datetime import timedelta
import math
import matplotlib.pyplot as plt

import bandido

def main():
    k = 10

    epsilons = [1.0 / 128.0, 1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0]
    grad_steps = [1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0, 2.0]
    cs = [1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0, 2.0, 4.0]
    init_estimates = [1.0 / 4.0, 1.0 / 2.0, 1.0, 2.0, 4.0]

    max_steps = 1000
    max_runs = 2000

    # switches for dev/debug
    doEpsilon = True
    doGradient = True
    doUCB = True
    doOptimistic = True

    x_vals = [1.0 / 128.0, 1.0 / 64.0, 1.0 / 32.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 1.0, 2.0, 4.0]
    ax = plt.subplot(111)
    ax.set_title('Parameter study, stationary bandit', loc='left')
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Average reward over first 1000 steps')
    plt.xscale('log')
    plt.xticks(x_vals, [str(v) for v in x_vals])

    if doEpsilon:
        print('Epsilon-greedy agent...')
        average_rewards = [math.nan for x in x_vals if x < epsilons[0]]
        for epsilon in epsilons:
            average_run_reward = 0
            print('epsilon = ', epsilon)
            for run in range(max_runs):
                sys.stdout.write('\r %d/%d' % (run + 1, max_runs))
                sys.stdout.flush()

                bandit = bandido.Bandit(k=k)
                agent = bandido.Agent(k=k, epsilon=epsilon)
                average_steps_reward = 0

                for step in range(max_steps):
                    action = agent.action
                    reward = bandit.reward_action(action)
                    agent.process_reward(action, reward)

                    average_steps_reward = average_steps_reward + (reward - average_steps_reward) / (step + 1)

                average_run_reward = average_run_reward + (average_steps_reward - average_run_reward) / (run + 1)

            average_rewards.append(average_run_reward)
            print()  # basically a newline
        average_rewards = average_rewards + [math.nan for x in x_vals if x > epsilons[-1]]
        plt.plot(x_vals, average_rewards, label='Epsilon')

    if doGradient:
        print('Gradient agent...')
        average_rewards = [math.nan for x in x_vals if x < grad_steps[0]]
        for grad_step in grad_steps:
            average_run_reward = 0
            print('alpha = ', grad_step)
            for run in range(max_runs):
                sys.stdout.write('\r %d/%d' % (run + 1, max_runs))
                sys.stdout.flush()

                bandit = bandido.Bandit(k=k)
                agent = bandido.Agent(k=k, grad_step=grad_step)
                average_steps_reward = 0

                for step in range(max_steps):
                    action = agent.action
                    reward = bandit.reward_action(action)
                    agent.process_reward(action, reward)

                    average_steps_reward = average_steps_reward + (reward - average_steps_reward) / (step + 1)

                average_run_reward = average_run_reward + (average_steps_reward - average_run_reward) / (run + 1)

            average_rewards.append(average_run_reward)
            print()  # basically a newline

        average_rewards = average_rewards + [math.nan for x in x_vals if x > grad_steps[-1]]
        plt.plot(x_vals, average_rewards, label='Gradient')

    if doUCB:
        print('UCB agent...')
        average_rewards = [math.nan for x in x_vals if x < cs[0]]
        for c in cs:
            average_run_reward = 0
            print('c = ', c)
            for run in range(max_runs):
                sys.stdout.write('\r %d/%d' % (run + 1, max_runs))
                sys.stdout.flush()

                bandit = bandido.Bandit(k=k)
                agent = bandido.Agent(k=k, c=c)
                average_steps_reward = 0

                for step in range(max_steps):
                    action = agent.action
                    reward = bandit.reward_action(action)
                    agent.process_reward(action, reward)

                    average_steps_reward = average_steps_reward + (reward - average_steps_reward) / (step + 1)

                average_run_reward = average_run_reward + (average_steps_reward - average_run_reward) / (run + 1)

            average_rewards.append(average_run_reward)
            print()  # basically a newline

        average_rewards = average_rewards + [math.nan for x in x_vals if x > cs[-1]]
        plt.plot(x_vals, average_rewards, label='UCB')

    if doOptimistic:
        print('Optimistic epsilon-greedy agent')
        average_rewards = [math.nan for x in x_vals if x < init_estimates[0]]
        for estimate in init_estimates:
            average_run_reward = 0
            print('estimate = ', estimate)
            for run in range(max_runs):
                sys.stdout.write('\r %d/%d' % (run + 1, max_runs))
                sys.stdout.flush()

                bandit = bandido.Bandit(k=k)
                agent = bandido.Agent(k=k, epsilon=0, alpha=0.1, initial_estimates=[estimate]*k)
                average_steps_reward = 0

                for step in range(max_steps):
                    action = agent.action
                    reward = bandit.reward_action(action)
                    agent.process_reward(action, reward)

                    average_steps_reward = average_steps_reward + (reward - average_steps_reward) / (step + 1)

                average_run_reward = average_run_reward + (average_steps_reward - average_run_reward) / (run + 1)

            average_rewards.append(average_run_reward)
            print()  # basically a newline

        average_rewards = average_rewards + [math.nan for x in x_vals if x > init_estimates[-1]]
        plt.plot(x_vals, average_rewards, label='Optimistic')

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start = time.time()
    main()
    print('RUNTIME: ', str(timedelta(seconds=(time.time() - start))))