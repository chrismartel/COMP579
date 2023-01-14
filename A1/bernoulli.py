import numpy as np
import matplotlib.pyplot as plt

def main():
    np.random.seed(0)
    bernoulliSimulator = BernoulliSimulator(3, [0.5, 0.5, 0.4])
    bernoulliSimulator.simulate(50)

class BernoulliSimulator:

    def __init__(self, k, p):
        '''
            k : number of arms
            p : numpy array containing the probability of success for each arm.
        '''
        self.k = k
        self.p = p


    def sample(self, i):
        '''
            Run a bernoulli trial for an arm. Returns 1 if the trial is
            successfull, 0 otherwise.
        '''
        return np.random.binomial(1,self.p[i],1)

    def simulate(self, n):
        '''
            Simulate n trials for each arm, plot and return the data. The x axis represents the number of 
            trials and the y axis represents the reward for each trial.
        '''
        fig, axes = plt.subplots(1,self.k)
        fig.set_size_inches(20, 5)
        rewards = np.zeros((self.k,n))

        for i in range(self.k):
            means = np.zeros(n)
            for j in range(n):
                rewards[i,j] = self.sample(i)
                means[j] = np.mean(rewards[i,:j+1])

            true = self.p[i]

            axes[i].scatter(np.arange(n), rewards[i], label="sample values", s=1)
            axes[i].plot(np.arange(n), means, label='estimated mean')
            axes[i].plot(np.arange(n), np.full(n, true), label='true mean')
            axes[i].set_xlabel('samples')
            axes[i].set_ylabel('reward')
            axes[i].legend()

        plt.savefig('bernoulli.png')
        return rewards


if __name__ == "__main__":
    main()

