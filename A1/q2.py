import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import BernoulliSimulator
from bernoulli_bandit import BernoulliBandit
import argparse

def main():
    
    parser = argparse.ArgumentParser(description='Assignment 1 - Q2')
    parser.add_argument("-t", "--n_trials", type=int, help="number of trials per experiment", default=1000)
    parser.add_argument("-a", "--alphas", type=str, help="alpha values to compare. Must a string be of the form [a1,a2,a3] where ais are the alpha values", default="[0.1]")
    args = parser.parse_args()

    n_trials = args.n_trials
    alphas = args.alphas

    alphas = [float(alpha) for alpha in args.alphas.lstrip('[').rstrip(']').split(',')]

    # k-arm bandit parameters
    K = 3
    DELTA = 0.1
    p = [0.5, 0.5 - DELTA, 0.5 - 2*DELTA]

    # Bernoulli simulation
    bernoulliSimulator = BernoulliSimulator(K, p)
    rewards = bernoulliSimulator.simulate(n_trials)

    fig, axes = plt.subplots(1,K)
    fig.set_size_inches(20, 6)

    for update_method in range(len(alphas) + 1):
        bandit = BernoulliBandit(k=K)

        q = np.zeros((K,n_trials))

        for arm in range(K):
            for trial in range(n_trials):
                bandit.N += 1
                if (update_method == 0):
                    q[arm,trial] = bandit.updateAvg(arm, rewards[arm,trial])
                else:
                    q[arm,trial] = bandit.update(arm, rewards[arm,trial], alphas[update_method-1])
        
            label = "averaging method" if update_method == 0 else "fixed learning method - alpha = {alpha}".format(alpha=alphas[update_method-1])
            axes[arm].plot(np.arange(n_trials), q[arm], label=label)
        
    for arm in range(K):
        true = p[arm]
        axes[arm].plot(np.arange(n_trials), np.full(n_trials, true), label='true value')
        axes[arm].set_xlabel('Timesteps')
        axes[arm].set_ylabel('Reward')
        axes[arm].set_title("arm {arm}".format(arm=arm))
        axes[arm].set_ylim(0,1)
        axes[arm].legend()
    plt.suptitle("Performance of Q-Value Estimation Methods")
    plt.savefig("plots/q2/q2.png")


if __name__ == "__main__":
    main()