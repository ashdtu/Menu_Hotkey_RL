from environments.discrete_env import Environment
from agents.epsilon_greedy import Agent
from learners.q_learner import Learner
import dill as pkl
import os
import numpy as np

path="/home/ash/Menu_hotkey/experiments/"

env = Environment()
learn = Learner()
agent = Agent(env,learn)
num_episodes = 70000
net_reward = []
t = 0
max_steps = [10, 20, 30]                                            #todo:add multi threading for experiments
# state_data=[]
action_data = []
# reward_data = []
rms_error = []
# s=open("State_log_30_0.50.pickle","wb")
# r=open("Reward_log_30_0.50.pickle","wb")
d = open(os.path.join(path, "newAction_log_30_0.50_q_learn.pickle"), "wb")


def save_model():
    f = open(os.path.join(path,"newQ_function_30_0.50_q_learn.pickle"), "wb")
    pkl.dump(learn.q_fn, f)
    pkl.dump(rms_error,f)


for i in range(num_episodes):
    #state_log = []
    action_log = []
    # reward_log=[]
    env.reset()

    # state_log.append(env.state)
    while (env.step < 30):
        prev_state = env.state
        if (env.check_exit(env.state)):                             # Check terminal condition
            break
        act = agent.getAction(prev_state)                           # Perform action
        reward = env.getReward(prev_state, act)                     # Get Reward
        next_state = env.transition(prev_state, act)                # Make transition
        next_state = np.clip(next_state, 0.0, 1.0)
        env.state = next_state

        #state_log.append(env.state)
        action_log.append(act)
        # reward_log.append(reward)
        learn.learn(prev_state, act, reward, next_state)

    rms_error.append(learn.calc_rms(env.step))
    action_data.append(action_log)
    # state_data.append(state_log)

    if (i % 1000 == 0 and i != 0):
        # avg_net.append(np.mean(net_reward[t:i]))
        # t+=500

        print("{},Episodes".format(i))
        print("Actions", action_log)


save_model()
pkl.dump(action_data,d)




