import numpy as np
import random
import matplotlib.pyplot as plt
import math

class BernoulliSimulator:

    def __init__(self, k, p):
        '''
            k : number of arms
            p : numpy array containing the probability of success for each arm.
        '''
        np.random.seed(1234)
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

class BernoulliBandit:
    def __init__(self, k):
        '''
            k: the number of arms
        '''
        self.K = k # number of arms
        self.reset()

    def greedySelection(self, epsilon=0):
        '''
            Select a random action with probability epsilon or select the
            greedy action with probability 1 - epsilon. Return the chosen
            action index.
            epsilon: value between 0 and 1
        '''
        if (random.uniform(0,1) <= epsilon):
            arm = random.randint(0, self.K-1)
        else:
            # if multiple max values, return one at random
            q_max_indices = np.argwhere(self.Q == np.amax(self.Q))
            arm = random.choice(q_max_indices)
        self.N[arm] += 1
        return arm
        
    def ucbSelection(self, c=2):
        self.t += 1
        ucbs = np.array([self.Q[i] + c * math.sqrt(math.log(self.t) / max(self.N[i],1)) for i in range(self.K)])
        q_max_indices = np.argwhere(ucbs == np.max(ucbs))
        arm = random.choice(q_max_indices)
        self.N[arm] += 1
        return arm
    
    def thompsonSamplingSelection(self):
        self.t += 1
        samples = np.array([np.random.beta(max(self.S[i],1),max(self.F[i],1)) for i in range(self.K)])
        q_max_indices = np.argwhere(samples == np.max(samples))
        arm = random.choice(q_max_indices)
        self.N[arm] += 1
        return arm

    def thompsonSamplingUpdate(self, arm, reward):
        if reward == 1:
            self.S[arm] += 1
        else:
            self.F[arm] += 1

    def update(self, arm, reward, alpha=0.1):
        self.Q[arm] = self.Q[arm] + alpha * (reward-self.Q[arm])
        return self.Q[arm]

    def updateAvg(self, arm, reward):
        self.Q[arm] = self.Q[arm] + (1/self.N[arm]) * (reward-self.Q[arm])
        return self.Q[arm]
        
    def reset(self):
        self.Q = np.zeros(self.K) # action-values
        self.N = np.full(self.K,0) # number of time each action is chosen
        self.t = 0 # time counter
        self.S = np.full(self.K, 0) # successes for each arm
        self.F = np.full(self.K, 0) # failures for each arm   
