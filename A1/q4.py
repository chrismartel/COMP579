import numpy as np
import matplotlib.pyplot as plt
from bandit_algorithms import BernoulliSimulator
from bandit_algorithms import Bandit

def main():
    np.random.seed(0)

    K = 3
    delta = 0.1
    p = np.array([0.5, 0.5 - delta, 0.5 - 2*delta])
    bernoulliSimulator = BernoulliSimulator(K, p)

    n_trials = 1000
    n_experiments = 100

    epsilons = [0, 0.125, 0.25, 0.5, 1]

    for _, epsilon in enumerate(epsilons):
        fig, axes = plt.subplots(1,4)
        fig.set_size_inches(24, 5)
        # the reward received at each trial of each experiment
        reward_received = np.zeros((n_experiments, n_trials))
        average_reward_received = np.zeros((n_experiments, n_trials))

        # the fraction of the trials in which the best action is 
        # truly selected
        fraction_first_action = np.zeros((n_experiments, n_trials))

        # the instantaneous regret for each trial of each experiment
        instantaneous_regret = np.zeros((n_experiments, n_trials))
        average_instantaneous_regret = np.zeros((n_experiments, n_trials))

        # the total regret up to timestep t for each experiment
        total_regret = np.zeros((n_experiments, n_trials))

        for exp in range(n_experiments):
            rewards = bernoulliSimulator.simulate(n_trials)
            bandit = Bandit(k=K)

            for trial in range(n_trials):
                arm = bandit.greedySelection(epsilon=epsilon)
                reward_received[exp, trial] = rewards[arm,trial]
                average_reward_received[exp, trial] = np.mean(reward_received[exp][0:trial+1])

                bandit.N[arm] += 1
                bandit.updateAvg(arm, reward_received[exp, trial])
    
                best_action_indices = np.argwhere(p == np.amax(p))
                fraction_first_action[exp, trial] = np.sum(bandit.N[best_action_indices]) / (trial + 1)

                instantaneous_regret[exp, trial] = np.max(rewards[:,trial]) - reward_received[exp, trial]
                average_instantaneous_regret[exp, trial] = np.mean(instantaneous_regret[exp][0:trial+1])

                total_regret[exp, trial] = instantaneous_regret[exp, trial] if trial == 0 else total_regret[exp, trial-1] + instantaneous_regret[exp, trial]

        axes[0].plot(np.arange(n_trials), np.mean(average_reward_received, axis=0), color='blue')
        axes[0].set_xlabel('Timesteps')
        axes[0].set_ylabel('Reward Received')
        axes[0].set_title('Reward Received Over Time')

        axes[1].plot(np.arange(n_trials), np.mean(fraction_first_action, axis=0), color='red')
        axes[1].set_xlabel('Timesteps')
        axes[1].set_ylabel('Fraction of Truly Best Action Selected')
        axes[1].set_title('Fraction of Truly Best Action Selected')

        axes[2].plot(np.arange(n_trials), np.mean(average_instantaneous_regret, axis=0), color='orange')
        axes[2].set_xlabel('Timesteps')
        axes[2].set_ylabel('Instantaneous Regret')
        axes[2].set_title('Instantaneous Regret Over Time')

        axes[3].plot(np.arange(n_trials), np.mean(total_regret, axis=0), color='green')
        axes[3].set_xlabel('Timesteps')
        axes[3].set_ylabel('Total Regret')
        axes[3].set_title('Total Regret Up to Timestep t')
        fig.suptitle("epsilon = {epsilon}".format(epsilon=epsilon))
        plt.savefig("plots/q4_epsilon={epsilon}.png".format(epsilon=epsilon))

if __name__ == "__main__":
    main()