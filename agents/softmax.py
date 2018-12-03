import numpy as np
import random

class Agent():
    def __init__(self, environment, learner):
        self.env = environment
        self.learn = learner
        self.temp=None                                          # Softmax temperature

    def getAction(self,init_state):
        prob=[]
        sum=0

        for a in self.env.actions:
            sum += np.exp(self.learn.q_fn[tuple(init_state)][a]/self.temp)

        for item in self.env.actions:
            prob.append(np.exp(self.learn.q_fn[tuple(init_state)][item]/self.temp)/sum)

        action=random.choices(self.env.actions,weights=prob)
        return action[0]