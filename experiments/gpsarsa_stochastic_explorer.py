import numpy as np
import random
from collections import defaultdict


class Agent():
    def __init__(self,environment,learner):
        self.env=environment
        self.learn=learner
        self.mean_dict = defaultdict(lambda: [0.0, 0.0, 0.0])                   # dummy variable to hold RMS error
        self.scale_cov=4                                                # Scaling factor(N) for covariance exploration

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


    def getAction(self,state):
        action=None
        if (self.learn.ret_dict() is not None):
            q_meanlist, q_covlist = self._actionProbs(state)
            self.mean_dict[tuple(state)] = q_meanlist

            samples=[np.random.normal(q_meanlist[i],self.scale_cov*np.sqrt(q_covlist[i])) for i in range(len(self.env.actions))]   # Sample Q value from obtained Gaussian distribution for each action
            action=np.argmax(samples)                       # Choose action with Max sampled Q value

        else:
            action=random.choice(self.env.actions)

        self.lastaction = action

        return action
