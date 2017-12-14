import sys
import time
from datetime import timedelta
import matplotlib.pyplot as plt

import bandido

def main():
    k = 10
    grad_steps = [0.1, 0.4]
    baseline_means = [0, 4]
    max_steps = 1000
    max_runs = 2000

    ax = plt.subplot(111)
    ax.set_title('% Optimal action', loc='left')

    for grad_step in grad_steps:
        for mean in baseline_means:
            optimal_action = [0] * max_steps

            for run in range(0, max_runs):
                sys.stdout.write('\r %d/%d' % (run + 1, max_runs))
                sys.stdout.flush()

                bandit = bandido.Bandit(k=k, initial_mu=mean)
                agent = bandido.Agent(k=k, grad_step=grad_step)
                for step in range(max_steps):
                    action = agent.action
                    reward = bandit.reward_action(action)
                    agent.process_reward(action, reward)

                    if action == bandit.optimal_action:
                        optimal_action[step] += 1

            print() # basically a newline

            #normalize
            optimal_action = [a / max_runs * 100 for a in optimal_action]

            label = r'$\alpha\, =\, {}\, with\, {}\, baseline$'.format(str(grad_step), ' no ' if mean == baseline_means[0] else ' ')
            plt.plot(range(0, max_steps), optimal_action, label=label)

    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start = time.time()
    main()
    print('RUNTIME: ', str(timedelta(seconds=(time.time() - start))))