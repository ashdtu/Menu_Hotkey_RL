import numpy as np 
from matplotlib import pyplot as plt
import time
from collections import Counter
import dill as pkl
import os

path="/home/ash/Menu_hotkey/"
'''
def some(param):
    values=[]
    for i in range(1,21):
        values.append(param[2]+param[0]*np.exp(-param[1]*(i-1)))
    return values

plp_m=[0.212,0.247,2.865]
plp_h=[0.146,0.272,2.211]
points1=np.array(some(plp_m))
points2=np.array(some(plp_h))
points3=points1+0.5
hotkey_reward=[]
l=0
for i in range(20):
    hotkey_reward.append(10-(points2[i]+(1-l)*points1[i]))
    l+=0.1
    l=np.clip(l,0,1)

fig=plt.figure()
ax=fig.add_subplot(111)

ax.plot(range(20),10-points1,'r',label='Menu')
ax.plot(range(20),hotkey_reward,'g',label='HotKeys')
ax.plot(range(20),10-points3,'b',label='Menu learning')
ax.set_ylim(4,8)
ax.set_xlabel('Time steps')
ax.set_ylabel('Reward')
ax.set_title('Reward following single option(M,H,ML) ')
ax.legend(loc='lower right')
print('menu',10-points1)
print('hotkeys',hotkey_reward)
print('menu_learning',10-points3)
plt.grid()
plt.show()
#plt.legend()



import random

actions=['A','B','C']
q_fn=[1,0,1]
max_index = np.argwhere(q_fn == np.max(q_fn))
max_index=max_index.flatten().tolist()
print(actions[random.choice(max_index)])

from collections import defaultdict
some=defaultdict(lambda : [0.0,0.0,0.0])
state=np.array([1.0,0.0])
some[tuple(state)][1]=1
print(np.all(state==1.0))



import os
path="/home/ash/Menu_hotkey/"
f=open(os.path.join(path,"Action_log_30_0.50.pickle"),"rb")

action_log=pkl.load(f)
action_log=action_log[::-1]
action_log=action_log[:20]
print(action_log)

fig=plt.figure()
ax2=fig.add_subplot(111)
ax2.set_ylim(-1,3)
ax2.set_xlim(-1,30)
plt.ion()
t=0
for i in range(len(action_log)):
    X = range(len(action_log[i+t]))
    Y = action_log[i+t]
    animated_plot = plt.plot(X, Y)[0]
    for i in range(len(Y)):
        animated_plot.set_xdata(X[0:i])
        animated_plot.set_ydata(Y[0:i])
        plt.pause(0.05)
    t+=1
    plt.draw()
    ax2.clear()
    ax2.set_ylim(-1, 3)
    ax2.set_xlim(-1, 30)


#plt.show()
'''
#for item in action_log:
from matplotlib.colors import ListedColormap

def ret_act(action_log,length):
    new_action = []
    for some in action_log:
        some = np.array(some)
        if (len(some) == length):
            new_action.append(some)
    new_action = np.array(new_action)
    return new_action


f=open(os.path.join(path+"experiments/","newAction_log_30_0.50_q_learn.pickle"),"rb")
g=open(os.path.join(path+"Results_2/","Model2Action_log_40_0.50_gpsarsa_covariance_epsilon_0.40.pickle"),"rb")
#h=open(os.path.join(path,"Action_log_30_softmax_0.5_2.pickle"),"rb")
#e=open(os.path.join(path,"Action_log_15_1_traces.pickle"),"rb")

action_log1=pkl.load(f)
action_log_new=[]
for item in action_log1:
    some=item
    if(len(item)!=40):
        some=some+[-1]*(40-len(item))
    action_log_new.append(some)


action_log1=action_log1[50000:]
length=[len(i) for i in action_log1]

data = Counter(length)
print(data.most_common())
act1=ret_act(action_log1,30)

action_log2=pkl.load(g)
action_log2=action_log2[750:]
length=[len(i) for i in action_log2]
print(len(action_log1))

data = Counter(length)
print(data.most_common())
print(len(action_log2))

act2=ret_act(action_log2,40)
'''
action_log3=pkl.load(h)
action_log3=action_log3[69000:]
act3=ret_act(action_log3,30)

#action_log4=pkl.load(e)
#action_log4=action_log4[69000:]
#act4=ret_act(action_log4,15)
'''
import seaborn as sns
#sns.set()
fig, ax = plt.subplots()
sns.heatmap(data=act1,cbar=True)
plt.title('Actions vs Steps')
ax.set_ylabel('Episodes')
ax.set_xlabel('Steps')

fig2, ax2 = plt.subplots()
plt.title('Actions vs Steps')
sns.heatmap(data=act2,cbar=True,vmin=0,vmax=2)
ax2.set_ylabel('Episodes')
ax2.set_xlabel('Steps')
'''
fig3, ax3 = plt.subplots()
plt.title('Actions vs Steps')
sns.heatmap(data=act3,cbar=True,vmin=0,vmax=2)
ax3.set_ylabel('Episodes')
ax3.set_xlabel('Steps')
plt.show()

fig4, ax4 = plt.subplots()
plt.title('Actions vs Steps')

sns.heatmap(data=act4,cbar=True,vmin=0,vmax=2)
ax4.set_ylabel('Episodes')
ax4.set_xlabel('Steps')
'''
plt.show()
