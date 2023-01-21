import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import BernoulliSimulator
from bernoulli_bandit import BernoulliBandit
import argparse

def main():

    parser = argparse.ArgumentParser(description='Assignment 1 - Q7')
    parser.add_argument("-t", "--n_trials", type=int, help="number of trials per experiment", default=1000)
    parser.add_argument("-e", "--n_experiments", type=int, help="number of experiments", default=100)
    args = parser.parse_args()

    n_trials = args.n_trials
    n_experiments = args.n_experiments
    
    # k-arm bandit parameters
    K = 3
    DELTA = 0.1
    p = np.array([0.5, 0.5 - DELTA, 0.5 - 2*DELTA])

    # Bernoulli simulation
    bernoulliSimulator = BernoulliSimulator(K, p)

    fig, axes = plt.subplots(1,4)
    fig.set_size_inches(24, 5)

    # the reward received at each trial of each experiment
    instant_reward_received = np.zeros((n_experiments, n_trials))
    average_reward_received = np.zeros((n_experiments, n_trials))

    # the fraction of the trials in which the best action is 
    # truly selected
    fraction_first_action = np.zeros((n_experiments, n_trials))

    # the instantaneous regret for each trial of each experiment
    instantaneous_regret = np.zeros((n_experiments, n_trials))
    average_instantaneous_regret = np.zeros((n_experiments, n_trials))

    # the total regret up to timestep t for each experiment
    total_regret = np.zeros((n_experiments, n_trials))

        
    bandit = BernoulliBandit(k=K)

    for exp in range(n_experiments):
        bandit.reset()
        rewards = bernoulliSimulator.simulate(n_trials)

        for trial in range(n_trials):
            arm = bandit.thompsonSamplingSelection()
            bandit.thompsonSamplingUpdate(arm, rewards[arm,trial])

            instant_reward_received[exp, trial] = rewards[arm,trial]
            average_reward_received[exp, trial] = np.mean(instant_reward_received[exp][0:trial+1])

            best_action_indices = np.argwhere(p == np.max(p))
            fraction_first_action[exp, trial] = np.sum(bandit.N[best_action_indices]) / (trial + 1)

            instantaneous_regret[exp, trial] = np.max(rewards[:,trial]) - instant_reward_received[exp, trial]
            average_instantaneous_regret[exp, trial] = np.mean(instantaneous_regret[exp][0:trial+1])

            total_regret[exp, trial] = instantaneous_regret[exp, trial] if trial == 0 else total_regret[exp, trial-1] + instantaneous_regret[exp, trial]
    

    axes[0].plot(np.arange(n_trials), np.mean(average_reward_received, axis=0))
    axes[1].plot(np.arange(n_trials), np.mean(fraction_first_action, axis=0))
    axes[2].plot(np.arange(n_trials), np.mean(average_instantaneous_regret, axis=0))
    axes[3].plot(np.arange(n_trials), np.mean(total_regret, axis=0))

    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Reward Received')
    axes[0].set_title('Reward Received Over Time')

    axes[1].set_xlabel('Timesteps')
    axes[1].set_ylabel('Fraction of Truly Best Action Selected')
    axes[1].set_title('Fraction of Truly Best Action Selected')

    axes[2].set_xlabel('Timesteps')
    axes[2].set_ylabel('Instantaneous Regret')
    axes[2].set_title('Instantaneous Regret Over Time')

    axes[3].set_xlabel('Timesteps')
    axes[3].set_ylabel('Total Regret')
    axes[3].set_title('Total Regret Up to Timestep t')
    fig.suptitle("Performance of Thompson Sampling Algorithm")
    
    plt.savefig("plots/q7/q7.png")

if __name__ == "__main__":
    main()