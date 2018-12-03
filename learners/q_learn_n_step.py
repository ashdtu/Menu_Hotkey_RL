from collections import defaultdict
import numpy as np

class Learner():

    def __init__(self):
        self.q_fn=defaultdict(lambda:[0.0,0.0,0.0])
        self.alpha=0.2
        self.gamma = 0.8
        self.n=3                                    #  N-step backup
        self.test_conv = []

    def learn(self,data):
        td_error=0
        for i in range(len(data)):
            td_error=0
            temp=-self.q_fn[tuple(data[i][0])][data[i][1]]
            td_error=data[i][2]-self.gamma*self.q_fn[tuple(data[i][0])][data[i][1]]
            #print(td_error)
            for j in range(min(self.n,len(data)-i-1)):
                td_error+=pow(self.gamma,j+1)*self.q_fn[tuple(data[i+j+1][0])][data[i+j+1][1]]
            self.q_fn[tuple(data[i][0])][data[i][1]] = (1-self.alpha)*self.q_fn[tuple(data[i][0])][data[i][1]] + self.alpha*td_error
            temp+=self.q_fn[tuple(data[i][0])][data[i][1]]
            self.test_conv.append(temp**2)


    def calc_rms(self,steps):
        rms_value=np.sqrt(sum(self.test_conv)/steps)
        self.test_conv=[]
        return rms_value