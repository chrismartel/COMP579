import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import BernoulliSimulator
import argparse

def main():

    parser = argparse.ArgumentParser(description='Assignment 1 - Q1')
    parser.add_argument("-t", "--n_trials", type=int, help="number of trials per experiment", default=1000)
    args = parser.parse_args()

    n_trials = args.n_trials

    # k-arm bandit parameters
    K = 3
    DELTA = 0.1
    p = [0.5, 0.5 - DELTA, 0.5 - 2*DELTA]

    # Bernoulli simulation
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
    plt.savefig('plots/q1/q1.png')

if __name__ == "__main__":
    main()