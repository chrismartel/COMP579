import numpy as np
import matplotlib.pyplot as plt
from bernoulli_bandit import BernoulliSimulator
from bernoulli_bandit import BernoulliBandit
import argparse


def main():

    parser = argparse.ArgumentParser(description='Assignment 1 - Q9')
    parser.add_argument("-t", "--n_trials", type=int,
                        help="number of trials per experiment", default=1000)
    parser.add_argument("-e", "--n_experiments", type=int,
                        help="number of experiments", default=100)
    args = parser.parse_args()

    n_trials = args.n_trials
    n_experiments = args.n_experiments

    # k-arm bandit parameters
    K = 3

    fixed_learning_label_format = "fixed learning - alpha = {alpha}"
    e_greedy_label_format = "E-Greedy | epsilon = {epsilon} | {method}"
    ucb_label_format = "UCB | c = {c} | {method}"
    thompson_sampling_label = "Thompson Sampling"

    configs = [
        {"type": "e_greedy", "epsilon": 0.125, "alpha": 0.1, "averaging": False, "label": e_greedy_label_format.format(
            epsilon=0.125, method=fixed_learning_label_format.format(alpha=0.1))},
        {"type": "e_greedy", "epsilon": 0.125, "alpha": 0, "averaging": True,
            "label": e_greedy_label_format.format(epsilon=0.125, method="averaging")},
        {"type": "e_greedy", "epsilon": 0.25, "alpha": 0.1, "averaging": False, "label": e_greedy_label_format.format(
            epsilon=0.25, method=fixed_learning_label_format.format(alpha=0.1))},
        {"type": "e_greedy", "epsilon": 0.25, "alpha": 0, "averaging": True,
            "label": e_greedy_label_format.format(epsilon=0.25, method="averaging")},
        {"type": "ucb", "c": 2, "alpha": 0.1, "averaging": False, "label": ucb_label_format.format(
            c=2, method=fixed_learning_label_format.format(alpha=0.1))},
        {"type": "ucb", "c": 2, "alpha": 0, "averaging": True,
            "label": ucb_label_format.format(c=2, method="averaging")},
        {"type": "ts", "label": thompson_sampling_label},
    ]

    # Bernoulli simulation
    DELTA = 0.1
    p_0_500 = np.array([0.5, 0.5 - DELTA, 0.5 - 2*DELTA])
    p_501_1000 = np.array([0.5, 0.5 + DELTA, 0.5 + 2*DELTA])
    bernoulliSimulator = BernoulliSimulator(K, p_0_500)

    fig, axes = plt.subplots(1, 4)
    fig.set_size_inches(35, 8)

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

    for config in configs:
        for exp in range(n_experiments):
            bandit.reset()

            # non-stationary bandit
            bernoulliSimulator.p = p_0_500
            rewards_0_500 = bernoulliSimulator.simulate(int(n_trials/2))
            bernoulliSimulator.p = p_501_1000
            rewards_501_1000 = bernoulliSimulator.simulate(int(n_trials/2))

            rewards = np.concatenate((rewards_0_500, rewards_501_1000), axis=1)

            for trial in range(n_trials):

                # E-Greedy Algorithm
                if config["type"] == "e_greedy":
                    arm = bandit.greedySelection(epsilon=config["epsilon"])
                    if config["averaging"]:
                        bandit.updateAvg(arm, rewards[arm, trial])
                    else:
                        bandit.update(
                            arm, rewards[arm, trial], alpha=config["alpha"])

                # UCB Algorithm
                elif config["type"] == "ucb":
                    arm = bandit.ucbSelection(c=config["c"])
                    if config["averaging"]:
                        bandit.updateAvg(arm, rewards[arm, trial])
                    else:
                        bandit.update(
                            arm, rewards[arm, trial], alpha=config["alpha"])

                # Thompson Sampling Algorithm
                elif config["type"] == "ts":
                    arm = bandit.thompsonSamplingSelection()
                    bandit.thompsonSamplingUpdate(arm, rewards[arm, trial])

                instant_reward_received[exp, trial] = rewards[arm, trial]
                average_reward_received[exp, trial] = np.mean(
                    instant_reward_received[exp][0:trial+1])

                best_action_indices = np.argwhere(bernoulliSimulator.p == np.max(bernoulliSimulator.p))
                fraction_first_action[exp, trial] = np.sum(
                    bandit.N[best_action_indices]) / (trial + 1)

                instantaneous_regret[exp, trial] = np.max(
                    rewards[:, trial]) - instant_reward_received[exp, trial]
                average_instantaneous_regret[exp, trial] = np.mean(
                    instantaneous_regret[exp][0:trial+1])

                total_regret[exp, trial] = instantaneous_regret[exp,
                                                                trial] if trial == 0 else total_regret[exp, trial-1] + instantaneous_regret[exp, trial]

        axes[0].plot(np.arange(n_trials), np.mean(
            average_reward_received, axis=0), label=config["label"])
        axes[1].plot(np.arange(n_trials), np.mean(
            fraction_first_action, axis=0), label=config["label"])
        axes[2].plot(np.arange(n_trials), np.mean(
            average_instantaneous_regret, axis=0), label=config["label"])
        axes[3].plot(np.arange(n_trials), np.mean(
            total_regret, axis=0), label=config["label"])

    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Reward Received')
    axes[0].set_title('Reward Received Over Time')
    axes[0].legend()

    axes[1].set_xlabel('Timesteps')
    axes[1].set_ylabel('Fraction of Truly Best Action Selected')
    axes[1].set_title('Fraction of Truly Best Action Selected')
    axes[1].legend()

    axes[2].set_xlabel('Timesteps')
    axes[2].set_ylabel('Instantaneous Regret')
    axes[2].set_title('Instantaneous Regret Over Time')
    axes[2].legend()

    axes[3].set_xlabel('Timesteps')
    axes[3].set_ylabel('Total Regret')
    axes[3].set_title('Total Regret Up to Timestep t')
    axes[3].legend()
    fig.suptitle("Performance Comparison of Selection Algorithms")

    plt.savefig("plots/q9/q9.png")


if __name__ == "__main__":
    main()
