import numpy as np
import dill as pkl
import os
from environments.cts_env import Environment
from learners.gpsarsa_baseline import GP_SARSA
from agents.baseline_gpsarsa import Agent
from sklearn.metrics import mean_squared_error

path="/home/ash/Menu_hotkey/discrete_results"


env = Environment()
learner = GP_SARSA()
agent = Agent(env,learner)
num_episodes=800
max_steps=30
net_reward=[]
t=0
action_data=[]
rms_error=[]
d=open(os.path.join(path,"new2Action_log_0.50_baseline.pickle"),"wb")

def save_model():
    f = open(os.path.join(path, "new2Q_0.50_baseline.pickle"), "wb")
    pkl.dump(learner.inv, f)
    pkl.dump(learner.H,f)


try:
    for i in range(num_episodes):
        env.reset()
        state_log = []
        action_log = []
        reward_log = []
        state_log.append(env.state)

        while (env.step < max_steps):

            prev_state = env.state
            if (env.check_exit(env.state)):
                break
            act = agent.getAction(prev_state)
            reward = env.getReward(prev_state, act)
            next_state = env.transition(prev_state, act)
            next_state = np.clip(next_state, 0.0, 1.0)
            env.state = next_state

            state_log.append(env.state)
            action_log.append(act)
            reward_log.append(reward)

        dataset = zip(state_log, action_log, reward_log)
        temp = learner.inv
        learner.learn(dataset)
        #print('temp',learner.inv.shape)
        if (len(temp) == len(learner.inv)):
            rms_error.append(np.sqrt(mean_squared_error(temp, learner.inv)))
        else:
            temp = np.append(temp, np.zeros(len(learner.inv) - len(temp)))
            rms_error.append(np.sqrt(mean_squared_error(temp, learner.inv)))
        print("Actions",action_log)
        # print("Reward",reward_log,len(reward_log))

        action_data.append(action_log)

        if (i % 40 == 0 and i != 0):
            print("{},Episodes".format(i))
            print("Action log",action_log)



except KeyboardInterrupt:
    save_model()
    pkl.dump(action_data,d)










