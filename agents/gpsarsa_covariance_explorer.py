import numpy as np
import random
from collections import defaultdict


class Agent():
    def __init__(self,environment,learner):
        self.env=environment
        self.learn=learner
        self.epsilon=0.1                                                        # Epsilon for policy
        self.mean_dict = defaultdict(lambda: [0.0, 0.0, 0.0])                   # dummy variable to hold RMS error

    def _actionProbs(self,obs):

        self.q_mean=[]
        self.q_cov=[]
        i=0
        for act in self.env.actions:
            self.K=[]
            for i in range(self.learn.ret_dict().shape[0]):

                self.K=np.append(self.K,self.learn.kernel(self.learn.ret_dict()[i],np.append(obs,act)))

            alpha=self.learn.u_tilde
            C=self.learn.C_tilde

            self.q_mean=np.append(self.q_mean,np.dot(self.K,alpha))                                    #  Mean Q value for every action
            self.q_cov=np.append(self.q_cov,self.learn.kernel(np.append(obs,act),np.append(obs,act))-np.dot(np.dot(self.K,C),self.K.T)) #q_covariance for every action
        return self.q_mean,self.q_cov


    def getAction(self, state):

        if (self.learn.ret_dict() is not None):
            q_meanlist,q_covlist = self._actionProbs(state)
            self.mean_dict[tuple(state)] = q_meanlist
            rand = random.random()

            if (rand > self.epsilon):
                max_index = np.argwhere(q_meanlist == np.amax(q_meanlist))
                max_index = max_index.flatten().tolist()
                action = self.env.actions[random.choice(max_index)]
            else:
                #action = random.choice(self.env.actions)                       : Uncomment this for Epsilon greedy policy
                cov_index = np.argwhere(q_covlist == np.amax(q_covlist))          #Comment these 3 for Epsilon greedy policy
                cov_index = cov_index.flatten().tolist()
                action = self.env.actions[random.choice(cov_index)]

        else:
            action = random.choice(self.env.actions)

        return action