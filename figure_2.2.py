import sys
import time
from datetime import timedelta
import matplotlib.pyplot as plt

import bandido

def main():
    k = 10
    epsilons = [0, 0.01, 0.1]
    max_steps = 1000
    max_runs = 2000


    for epsilon in epsilons:
        average_reward = [0] * max_steps
        optimal_action = [0] * max_steps

        for run in range(0, max_runs):
            sys.stdout.write('\r %d/%d' % (run + 1, max_runs))
            sys.stdout.flush()

            bandit = bandido.Bandit(k=k)
            agent = bandido.Agent(k=k, epsilon=epsilon)

            for step in range(0, max_steps):
                action = agent.action
                reward = bandit.reward_action(action)
                agent.process_reward(action, reward)

                average_reward[step] += reward
                if action == bandit.optimal_action:
                    optimal_action[step] += 1

        print() # basically a newline

        #normalize
        average_reward = [r / max_runs for r in average_reward]
        optimal_action = [a / max_runs * 100 for a in optimal_action]

        ax1 = plt.subplot(311)
        ax1.set_title('Average reward', loc='left')
        plt.plot(range(0, max_steps), average_reward, label=r'$\epsilon='+str(epsilon)+'$')
        ax2 = plt.subplot(312)
        ax2.set_title('% Optimal action', loc='left')
        plt.plot(range(0, max_steps), optimal_action, label=r'$\epsilon='+str(epsilon)+'$')
        plt.legend(loc='lower right')
        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    start = time.time()
    main()
    print('RUNTIME: ', str(timedelta(seconds=(time.time() - start))))