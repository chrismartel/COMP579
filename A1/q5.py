import numpy as np
import matplotlib.pyplot as plt
from bandit_algorithms import BernoulliSimulator
from bandit_algorithms import Bandit
import argparse

def main():

    parser = argparse.ArgumentParser(description='Assignment 1 - Q4')
    parser.add_argument("-t", "--n_trials", type=int, help="number of trials per experiment", default=1000)
    parser.add_argument("-e", "--n_experiments", type=int, help="number of experiments", default=100)
    parser.add_argument("-g", "--epsilons", type=str, help="epsilon values to compare for e-greedy algorithm. Must a string be of the form [e1,e2,e3] where eis are the epsilon values", default="[0]")
    parser.add_argument("-a", "--alpha", type=float, help="fixed learning rate - alpha", default=0.01)
    args = parser.parse_args()

    n_trials = args.n_trials
    n_experiments = args.n_experiments
    alpha = args.alpha
    epsilons = [float(epsilon) for epsilon in args.epsilons.lstrip('[').rstrip(']').split(',')]
    epsilon_colors = ['red', 'blue', 'green', 'orange', 'black', 'yellow']

    # k-arm bandit parameters
    K = 3
    DELTA = 0.1
    p = np.array([0.5, 0.5 - DELTA, 0.5 - 2*DELTA])

    # Bernoulli simulation
    bernoulliSimulator = BernoulliSimulator(K, p)

    fig, axes = plt.subplots(1,4)
    fig.set_size_inches(24, 5)

    # the reward received at each trial of each experiment
    instant_reward_received = np.zeros((len(epsilons), n_experiments, n_trials))
    average_reward_received = np.zeros((len(epsilons), n_experiments, n_trials))

    # the fraction of the trials in which the best action is 
    # truly selected
    fraction_first_action = np.zeros((len(epsilons), n_experiments, n_trials))

    # the instantaneous regret for each trial of each experiment
    instantaneous_regret = np.zeros((len(epsilons), n_experiments, n_trials))
    average_instantaneous_regret = np.zeros((len(epsilons), n_experiments, n_trials))

    # the total regret up to timestep t for each experiment
    total_regret = np.zeros((len(epsilons), n_experiments, n_trials))

    bandit = Bandit(k=K, alpha=alpha)

    for epsilon in range(len(epsilons)):
        for exp in range(n_experiments):
            bandit.reset()
            rewards = bernoulliSimulator.simulate(n_trials)

            for trial in range(n_trials):
                arm = bandit.greedySelection(epsilon=epsilons[epsilon])
                instant_reward_received[epsilon, exp, trial] = rewards[arm,trial]
                average_reward_received[epsilon, exp, trial] = np.mean(instant_reward_received[epsilon, exp][0:trial+1])

                bandit.N[arm] += 1
                bandit.update(arm, instant_reward_received[epsilon, exp, trial])

                best_action_indices = np.argwhere(p == np.max(p))
                fraction_first_action[epsilon, exp, trial] = np.sum(bandit.N[best_action_indices]) / (trial + 1)

                instantaneous_regret[epsilon, exp, trial] = np.max(rewards[:,trial]) - instant_reward_received[epsilon, exp, trial]
                average_instantaneous_regret[epsilon, exp, trial] = np.mean(instantaneous_regret[epsilon, exp][0:trial+1])

                total_regret[epsilon, exp, trial] = instantaneous_regret[epsilon, exp, trial] if trial == 0 else total_regret[epsilon, exp, trial-1] + instantaneous_regret[epsilon, exp, trial]
        
        axes[0].plot(np.arange(n_trials), np.mean(average_reward_received[epsilon], axis=0), color=epsilon_colors[epsilon], label="epsilon={epsilon}".format(epsilon=epsilons[epsilon]))
        axes[1].plot(np.arange(n_trials), np.mean(fraction_first_action[epsilon], axis=0), color=epsilon_colors[epsilon], label="epsilon={epsilon}".format(epsilon=epsilons[epsilon]))
        axes[2].plot(np.arange(n_trials), np.mean(average_instantaneous_regret[epsilon], axis=0), color=epsilon_colors[epsilon], label="epsilon={epsilon}".format(epsilon=epsilons[epsilon]))
        axes[3].plot(np.arange(n_trials), np.mean(total_regret[epsilon], axis=0), color=epsilon_colors[epsilon], label="epsilon={epsilon}".format(epsilon=epsilons[epsilon]))

    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Reward Received')
    axes[0].set_title('Reward Received Over Time')
    axes[0].legend()

    axes[1].set_xlabel('Timesteps')
    axes[1].set_ylabel('Fraction of Truly Best Action Selected')
    axes[1].set_title('Fraction of Truly Best Action Selected')
    axes[0].legend()

    axes[2].set_xlabel('Timesteps')
    axes[2].set_ylabel('Instantaneous Regret')
    axes[2].set_title('Instantaneous Regret Over Time')
    axes[0].legend()

    axes[3].set_xlabel('Timesteps')
    axes[3].set_ylabel('Total Regret')
    axes[3].set_title('Total Regret Up to Timestep t')
    axes[0].legend()
    fig.suptitle("Performance of E-Greedy Algorithm | Fixed Learning Rate Method - alpha = {alpha}".format(alpha=alpha))
    
    plt.savefig("plots/q5/q5_alpha={alpha}.png".format(alpha=alpha))

if __name__ == "__main__":
    main()