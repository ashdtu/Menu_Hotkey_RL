import numpy as np
import random

class Agent():
    def __init__(self,environment,learner):
        self.epsilon=0.3
        self.temp=50
        self.env=environment
        self.learn=learner


    def _actionProbs(self, state):

        q_mean = []
        q_cov = []
        i = 0
        cum_reward = np.reshape(self.learn.ret_reward(), (len(self.learn.ret_reward()), 1))
        for act in self.env.actions:
            K = []
            for i in range(self.learn.ret_dict().shape[0]):
                K = np.append(K, self.learn.kernel(self.learn.ret_dict()[i],np.append(state, act)))  # k(s,a) with previous sequence
            K = np.reshape(K, (1, len(K)))
            q_mean = np.append(q_mean, np.dot(K, np.dot(self.learn.inv,cum_reward)))  # q mean list for every action
            #self.q_cov = np.append(self.q_cov,self.learn.kernel(np.append(state, act), np.append(state, act)) - np.dot(np.dot(self.K, self.learn.inv),np.dot(self.learn.ret_h(), self.K.T)))  # q_covariance for every action

        return q_mean

    def getAction(self, state):

        if (self.learn.ret_dict() is not None):
            q_meanlist= self._actionProbs(state)
            rand = random.random()
            if (rand > self.epsilon):
                max_index = np.argwhere(q_meanlist == np.amax(q_meanlist))
                max_index = max_index.flatten().tolist()
                action = self.env.actions[random.choice(max_index)]
            else:
                action = random.choice(self.env.actions)
                # cov_index = np.argwhere(q_covlist == np.amax(q_covlist))
                # cov_index = cov_index.flatten().tolist()
                # action = self.env.actions[random.choice(cov_index)]

        else:
            action = random.choice(self.env.actions)

        return action

