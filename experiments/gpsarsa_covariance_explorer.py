import numpy as np
import dill as pkl
import os
from learners.gpsarsa_sparse import GP_SARSA_SPARSE
from environments.cts_env import Environment
from agents.gpsarsa_covariance_explorer import Agent


path = "/home/ash/Menu_hotkey/experiments"


env = Environment()                     # Environment initialisation
learner = GP_SARSA_SPARSE()             # Learner selection
agent = Agent(env,learner)              # Agent selection ( Greedy, softmax, covariance)
num_episodes = 800                      # Total episodes
max_steps=40                            # Max steps allowed per episode
net_reward = []
t = 0
action_data = []                        # Log action values during training
rms_error = []

d = open(os.path.join(path, "Model2Action_log_40_0.50_gpsarsa_covariance_epsilon_0.80.pickle"), "wb")

def save_model():
    f = open(os.path.join(path, "Model2Q_40_0.50_gpsarsa_covariance_epsilon_0.80.pickle"), "wb")
    #pkl.dump(learner.u_tilde, f)
    pkl.dump(rms_error, f)
    #pkl.dump(agent.mean_dict, f)


try:
    for i in range(num_episodes):
        env.reset()
        state_log = []
        action_log = []
        reward_log = []
        state_log.append(env.state)
        temp_dict = agent.mean_dict.copy()

        while (env.step < max_steps):
            prev_state = env.state
            if (env.check_exit(env.state)):                             # Check terminal condition
                break
            act = agent.getAction(prev_state)                           # Perform action
            reward = env.getReward(prev_state, act)                     # Get Reward
            next_state = env.transition(prev_state, act)                # Make transition
            next_state = np.clip(next_state, 0.0, 1.0)
            env.state = next_state

            state_log.append(env.state)
            action_log.append(act)
            reward_log.append(reward)

        dataset = zip(state_log, action_log, reward_log)
        learner.learn(dataset)                                                      # Episodic training: One learning step per episode
        rms_error.append(learner.calc_rms(temp_dict, agent.mean_dict))
        print("rms", rms_error[len(rms_error) - 1], len(agent.mean_dict.keys()))
        # print("States",state_log,len(state_log))
        # print("Actions",action_log)
        # print("Reward",reward_log,len(reward_log))
        action_data.append(action_log)

        if (i % 100 == 0 and i != 0):
            print("{},Episodes".format(i))
            agent.epsilon -= 0.05

        save_model()
        pkl.dump(action_data, d)


except KeyboardInterrupt:
    save_model()
    pkl.dump(action_data, d)







# sampling from discrete distribution

'''

total_commands=4
commands=range(total_commands)
freq=[2,1,3,4]
freq=[freq[i]/sum(freq) for i in range(len(freq))]
print(np.random.choice(commands,10,p=freq))

    def getAction(self,state):

        if (learner.ret_dict() is not None):
            q_meanlist = self._actionProbs(state)
            mean_dict[tuple(state)]=q_meanlist
            prob = []
            sum = 0
            for i in range(len(q_meanlist)):
                sum += np.exp(q_meanlist[i]/self.temp)

            for item in env.actions:
                prob.append(np.exp(q_meanlist[item] / self.temp) / sum)

            #print("Qvalues",q_meanlist)
            #print(prob)
            action = random.choices(env.actions, weights=prob)

        else:
            action=random.choices(env.actions)

        return action[0]


'''






