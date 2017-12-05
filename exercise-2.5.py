import sys
import argparse
import time
import matplotlib.pyplot as plt

import bandito

def main():
    k = 10
    epsilons = [0, 0.01, 0.1]
    max_steps = 1000
    max_runs = 2000

    for epsilon in epsilons:
        avg_reward = [0] * max_steps
        percent_optimal_action = [0] * max_steps

        for run in range(0, max_runs):
            sys.stdout.write('\r %d/%d' % (run + 1, max_runs))
            sys.stdout.flush()

            bandit = bandito.MultiArmedBandit(k, epsilon)

            for step in range(0, max_steps):
                bandit.step()
                avg_reward[step] += bandit.reward
                if bandit.action == bandit.action_values.index(max(bandit.action_values)):
                    percent_optimal_action[step] += 1
        print() # basically a newline

        #normalize
        avg_reward = [r / max_runs for r in avg_reward]
        percent_optimal_action = [a / max_runs * 100 for a in percent_optimal_action]

        plt.subplot(311)
        plt.plot(range(0, max_steps), avg_reward)
        plt.subplot(312)
        plt.plot(range(0, max_steps), percent_optimal_action)
    plt.show()

if __name__ == "__main__":
    start = time.time()
    main()
    print('RUNTIME: ', time.time() - start, 's')