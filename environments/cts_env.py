import numpy as np

class Environment():
    def __init__(self):
        self.state=np.array([0.1,0.0])
        self.actions=np.array(range(3))
        self.plp_m=[0.212,0.247,2.865]                          # Power law parameters: Menu & Hotkey
        self.plp_h=[0.146,0.272,2.211]
        self.t_learning=0.50                                    # Time cost of learning: Menu learning action
        self.step=0


    def transition(self,init_state,action):
        mu,sigma=0,0.05
        eps=np.random.normal(mu,sigma)                                      # gaussian noise added to transition
        state_update={0:[0.1,0.025],1:[0.1,0.05],2:[-0.05,0.1]}             # transition model
        next_state=init_state+state_update.get(action)+eps
        next_state= [round(item, 2) for item in next_state]
        self.step += 1
        return next_state

    def getReward(self, init_state, action):
        reward=0
        if action==0:
            reward=10-self.getPLP(self.plp_m)
        elif action==1:
            reward=10-(self.getPLP(self.plp_m)+self.t_learning)
        elif action==2:
            reward=10-(self.getPLP(self.plp_h)+(1-init_state[1])*self.getPLP(self.plp_m))
        return reward

    def getPLP(self,param):
        return param[0]*np.exp(-param[1]*self.step)+param[2]

    def check_exit(self,inp):
        stop = False
        if (inp[0] >= 1.0 and inp[1] >= 1.0):                               # Termination condition of episode
            stop = True
        return stop

    def reset(self):
        self.state=np.array([0.1,0])
        self.step=0