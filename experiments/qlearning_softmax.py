from environments.discrete_env import Environment
from agents.softmax import Agent
from learners.q_learner import Learner

import numpy as np
import dill as pkl
import os

path="/home/ash/Menu_hotkey/discrete_results"


env = Environment()
learn = Learner()
agent = Agent(env,learn)
num_episodes=70000
max_steps=30
net_reward=[]
action_data=[]
rms_error=[]


def save_model(file):
    f = open(os.path.join(path, "Q_30_newsoftmax_" + str(file[0]) + "_" + str(file[1]) + "_annealed_" + ".pickle"),"wb")
    pkl.dump(learn.q_fn, f)
    pkl.dump(rms_error, f)



for agent.temp in [0.5,1,2]:                                            # Softmax temperature: Grid search
    for env.t_learning in [0.25,0.50,0.75]:                             # Learning cost : Grid Search

        print("Experiment:{},{}".format(env.t_learning,agent.temp))
        d=open(os.path.join(path,"Action_log_30_new_softmax_"+str(env.t_learning)+"_"+str(agent.temp)+"_annealed_"+".pickle"),"wb")
        for i in range(num_episodes):
            action_log = []
            env.reset()
            while (env.step < max_steps):
                prev_state = env.state
                if (env.check_exit(env.state)):
                    break
                act = agent.getAction(prev_state)
                reward = env.getReward(prev_state, act)
                next_state = env.transition(prev_state, act)
                next_state = np.clip(next_state, 0.0, 1.0)
                env.state = next_state

                action_log.append(act)
                learn.learn(prev_state, act, reward, next_state)
            rms_error.append(learn.calc_rms(env.step))


            action_data.append(action_log)

            if(i%5000==0 and i!=0):
                agent.temp-=0.1*agent.temp                                 # Temperature annealing per 5k episodes


        param=[env.t_learning,agent.temp]
        save_model(param)
        pkl.dump(action_data,d)

