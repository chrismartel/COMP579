import numpy as np
import random
import matplotlib.pyplot as plt

class BernoulliSimulator:

    def __init__(self, k, p):
        '''
            k : number of arms
            p : numpy array containing the probability of success for each arm.
        '''
        self.K = k
        self.p = p


    def sample(self, i):
        '''
            Run a bernoulli trial for an arm. Returns 1 if the trial is
            successfull, 0 otherwise.
        '''
        return np.random.binomial(1,self.p[i],1)

    def simulate(self, n):
        '''
            Simulate n trials for each arm and return the data.
        '''
        rewards = np.zeros((self.K,n))

        for i in range(self.K):
            for j in range(n):
                rewards[i,j] = self.sample(i)

        return rewards

class Bandit:
    def __init__(self, k, alpha=0):
        '''
            k: the number of arms
            alpha: the learning rate
        '''
        self.K = k
        self.alpha = alpha
        self.Q = np.zeros(k)
        self.N = np.zeros(k)

    def greedySelection(self, epsilon=0):
        '''
            Select a random action with probability epsilon or select the
            greedy action with probability 1 - epsilon. Return the chosen
            action index.
            epsilon: value between 0 and 1
        '''
        if (random.uniform(0,1) <= epsilon):
            return random.randint(0, self.K-1)
        else:
            # if multiple max values, return one at random
            q_max_indices = np.argwhere(self.Q == np.amax(self.Q))
            return random.choice(q_max_indices)

    def update(self, i, R):
        self.Q[i] = self.Q[i] + self.alpha * (R-self.Q[i])
        return self.Q[i]

    def updateAvg(self, i, R):
        self.Q[i] = self.Q[i] + (1/self.N[i]) * (R-self.Q[i])
        return self.Q[i]