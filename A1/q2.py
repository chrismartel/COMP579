import numpy as np
import matplotlib.pyplot as plt
from bandit_algorithms import BernoulliSimulator
from bandit_algorithms import Bandit
import argparse

def main():
    
    parser = argparse.ArgumentParser(description='Assignment 1 - Q2')
    parser.add_argument("-t", "--n_trials", type=int, help="number of trials per experiment", default=1000)
    parser.add_argument("-a", "--alpha", type=float, help="fixed learning rate - alpha", default=0.01)
    args = parser.parse_args()

    n_trials = args.n_trials
    alpha = args.alpha

    # k-arm bandit parameters
    K = 3
    DELTA = 0.1
    p = [0.5, 0.5 - DELTA, 0.5 - 2*DELTA]

    # Bernoulli simulation
    bernoulliSimulator = BernoulliSimulator(K, p)
    rewards = bernoulliSimulator.simulate(n_trials)

    fig, axes = plt.subplots(1,K)
    fig.set_size_inches(20, 6)

    bandit = Bandit(k=K, alpha=alpha)

    q_averaging = np.zeros((K,n_trials))
    q_fixed_learning = np.zeros((K,n_trials))

    for arm in range(K):
        for trial in range(n_trials):
            bandit.N += 1
            q_averaging[arm,trial] = bandit.updateAvg(arm, rewards[arm,trial])
            q_fixed_learning[arm,trial] = bandit.update(arm, rewards[arm,trial])
    
        true = p[arm]
        axes[arm].plot(np.arange(n_trials), q_averaging[arm], label='averaging method')
        axes[arm].plot(np.arange(n_trials), q_averaging[arm], label="fixed learning method - alpha = {alpha}".format(alpha=alpha))
        axes[arm].plot(np.arange(n_trials), np.full(n_trials, true), label='true value')
        axes[arm].set_xlabel('Timesteps')
        axes[arm].set_ylabel('Reward')
        axes[arm].set_title("arm {arm} - alpha={alpha}".format(arm=arm, alpha=alpha))
        axes[arm].set_ylim(0,1)
        axes[arm].legend()
    plt.suptitle("Fixed Learning Rate Q-Value Estimation | alpha = {alpha}".format(arm=K, alpha=alpha))
    plt.savefig("plots/q2/q2_alpha={alpha}.png".format(alpha=alpha))


if __name__ == "__main__":
    main()