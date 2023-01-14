import numpy as np
import matplotlib.pyplot as plt
from bandit_algorithms import BernoulliSimulator
from bandit_algorithms import Bandit

def main():
    np.random.seed(0)

    K = 3
    delta = 0.1
    p = [0.5, 0.5 - delta, 0.5 - 2*delta]
    bernoulliSimulator = BernoulliSimulator(K, p)
    n_trials = 50
    rewards = bernoulliSimulator.simulate(n_trials)

    alphas = [0.01, 0.1]

    fig, axes = plt.subplots(len(alphas),K)
    fig.set_size_inches(20, 12)

    for i, alpha in enumerate(alphas):

        bandit = Bandit(k=K, alpha=alpha)

        q_averaging = np.zeros((K,n_trials))
        q_fixed_learning = np.zeros((K,n_trials))

        for arm in range(K):
            for trial in range(n_trials):
                bandit.N += 1
                q_averaging[arm,trial] = bandit.updateAvg(arm, rewards[arm,trial])
                q_fixed_learning[arm,trial] = bandit.update(arm, rewards[arm,trial])
        
            true = p[arm]
            axes[i,arm].plot(np.arange(n_trials), q_averaging[arm], label='averaging method')
            axes[i,arm].plot(np.arange(n_trials), q_averaging[arm], label="fixed learning method - alpha = {alpha}".format(alpha=alpha))
            axes[i,arm].plot(np.arange(n_trials), np.full(n_trials, true), label='true value')
            axes[i,arm].set_xlabel('Timesteps')
            axes[i,arm].set_ylabel('Reward')
            axes[i,arm].set_title("arm {arm} - alpha={alpha}".format(arm=arm, alpha=alpha))
            axes[i,arm].set_ylim(0,1)
            axes[i,arm].legend()
    plt.suptitle("Q Value Estimation Method for a {arm}-arm Bandit".format(arm=K))
    plt.savefig("plots/q2.png")


if __name__ == "__main__":
    main()