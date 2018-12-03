import numpy as np
import random

class Agent():
    def __init__(self,environment,learner):
        self.epsilon=0.1                                # Epsilon for policy
        self.env=environment
        self.learn=learner

    def getAction(self,init_state):  # Policy selection here
        rand=random.random()
        some=tuple(init_state)
        if rand>self.epsilon:
            max_index = np.argwhere(self.learn.q_fn[some] == np.max(self.learn.q_fn[some]))
            #print('Q values',max(self.learn.q_fn[some]),min(self.learn.q_fn[some]))
            max_index=max_index.flatten().tolist()
            action = self.env.actions[random.choice(max_index)]
        else:
            action=random.choice(self.env.actions)
        return action
