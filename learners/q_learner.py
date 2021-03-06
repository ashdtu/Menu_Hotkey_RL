from collections import defaultdict
import numpy as np


class Learner:
    """For online learning uncomment td error"""

    def __init__(self):
        self.q_fn=defaultdict(lambda:[0.0,0.0,0.0])
        self.alpha=0.7                                  # Learning rate
        self.gamma = 0.8                                # Discount factor
        self.test_q_fn={}
        self.test_conv=[]

    def learn(self,state,action,reward,next_state):
        td_error=reward+self.gamma*max(self.q_fn[tuple(next_state)])-self.q_fn[tuple(state)][action]          # offline learning
        #td_error = reward + self.gamma *(self.q_fn[tuple(next_state)][agent.getAction(next_state)]) - self.q_fn[tuple(state)][action]  #online learning
        temp = -self.q_fn[tuple(state)][action]
        self.q_fn[tuple(state)][action] = (1-self.alpha)*self.q_fn[tuple(state)][action] + self.alpha*td_error
        temp+=self.q_fn[tuple(state)][action]
        self.test_conv.append(temp**2)


    def calc_rms(self,steps):
        rms_value=np.sqrt(sum(self.test_conv)/steps)
        self.test_conv=[]
        return rms_value

    """
    def test_model(self):
        r=open(os.path.join(path,"Q_30_0.50_softmax_.pickle"),"rb")
        self.test_q_fn=pkl.load(r)
        
    For test stage uncomment this & remove learning step from Experiment 
    """