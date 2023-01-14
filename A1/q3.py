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
    n_trials = 100
    n_experiments = 100

    alphas = [0.01, 0.1]

    for _, alpha in enumerate(alphas):
        fig, axes = plt.subplots(1,K)
        fig.set_size_inches(20, 5)
        q_averaging = np.zeros((n_experiments, K, n_trials))
        q_fixed_learning = np.zeros((n_experiments, K, n_trials))

        for exp in range(n_experiments):
            rewards = bernoulliSimulator.simulate(n_trials)
            bandit_averaging = Bandit(k=K)
            bandit_fixed_learning = Bandit(k=K, alpha=alpha)
            for arm in range(K):
                for trial in range(n_trials):
                    bandit_averaging.N += 1
                    bandit_fixed_learning.N += 1
                    q_averaging[exp,arm,trial] = bandit_averaging.updateAvg(arm, rewards[arm,trial])
                    q_fixed_learning[exp,arm,trial] = bandit_fixed_learning.update(arm, rewards[arm,trial])
        
        for arm in range(K):
            true = p[arm]
            axes[arm].plot(np.arange(n_trials), np.mean(q_averaging, axis=0)[arm], label='averaging method')
            axes[arm].plot(np.arange(n_trials), np.mean(q_fixed_learning, axis=0)[arm], label="fixed learning method - alpha = {alpha}".format(K=K, alpha=alpha))
            axes[arm].plot(np.arange(n_trials), np.full(n_trials, true), label='true value')
            axes[arm].set_xlabel('Timesteps')
            axes[arm].set_ylabel('Reward')
            axes[arm].set_title("arm {arm}".format(arm=arm))
            axes[arm].set_ylim(0,1)
            axes[arm].legend()
        plt.suptitle("Q Value Estimation Methods Comparison for a {K}-arm Bandit".format(K=K))
        plt.savefig("plots/q3_alpha={alpha}.png".format(alpha=alpha))


if __name__ == "__main__":
    main()