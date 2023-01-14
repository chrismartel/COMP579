import numpy as np
import matplotlib.pyplot as plt
from bandit_algorithms import BernoulliSimulator

def main():
    np.random.seed(0)
    K = 3
    delta = 0.1
    p = [0.5, 0.5 - delta, 0.5 - 2*delta]
    n_trials = 50
    bernoulliSimulator = BernoulliSimulator(K, p)
    rewards = bernoulliSimulator.simulate(n_trials)

    fig, axes = plt.subplots(1,K)
    fig.set_size_inches(20, 5)

    for arm in range(K):
        means = np.zeros(n_trials)
        for trial in range(n_trials):
            means[trial] = np.mean(rewards[arm,:trial+1])
        true = p[arm]
        axes[arm].scatter(np.arange(n_trials), rewards[arm], label="samples", s=1)
        axes[arm].plot(np.arange(n_trials), means, label='estimated mean')
        axes[arm].plot(np.arange(n_trials), np.full(n_trials, true), label='true value')
        axes[arm].set_xlabel('Timesteps')
        axes[arm].set_ylabel('Reward')
        axes[arm].set_ylim(0,1)
        axes[arm].legend()
    plt.suptitle("Bernoulli Simulation for a {arm}-arm Bandit".format(arm=K))
    plt.savefig('plots/q1.png')

if __name__ == "__main__":
    main()