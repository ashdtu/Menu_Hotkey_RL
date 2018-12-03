import numpy as np
import os
import matplotlib.pyplot as plt
import dill as pkl

path="/home/ash/Menu_hotkey/"

a=open(os.path.join(path+'experiments/','newQ_function_30_0.50_q_learn.pickle'),'rb')
b=open(os.path.join(path+'discrete_results/','Model2Q_40_0.50_gpsarsa_covariance_epsilon_0.40.pickle'),'rb')
lmao=pkl.load(b)
rms_greedy=pkl.load(b)
u_tilde=pkl.load(a)

avg_rms_cov=[]
avg_rms_greedy=[]
t=0
'''
for i in range(80):
    avg_rms_cov.append(np.mean(rms_error[t:t+10]))
    avg_rms_greedy.append(np.mean(rms_greedy[t:t+10]))
    t+=10
'''
plt.plot(avg_rms_cov,'r',label='Model1')
plt.plot(avg_rms_greedy,'b',label='Model 2')
plt.xlabel("Episodes")
plt.ylabel("Avg RMS Error")
plt.legend()
plt.show()
rms_error=pkl.load(a)
q_fn=pkl.load(a)

cov_fn=pkl.load(a)

states=list(q_fn.keys())
state_final=[]
for item in states:
    item=list(item)
    state_final.append(item)

state_final=np.array(state_final)
fig = plt.figure()
ax = fig.add_subplot(111)

values=list(q_fn.values())
q_0=[]   #Q value for Menu Action
values=np.array(values)
for row in values:
    q_0.append(row[0])
q_0=np.array(q_0)
q_1=[] #Q value for Learning
for row in values:
    q_1.append(row[1])
q_1=np.array(q_1)
q_2=[] # Q value for HotKey Action
for row in values:
    q_2.append(row[2])
q_2=np.array(q_2)
print("zeros_menu",sum(i==0 for i in q_0),len(q_0))
print("zeros_learning",sum(i==0 for i in q_1),len(q_1))
print("zeros_hotkey",sum(i==0 for i in q_2),len(q_2))

ax.set_xlabel('Hotkey knowledge  Kh')
ax.set_ylabel('Q_value')
ax.scatter(state_final[:,1],q_0,c='r',label='Menu',alpha=0.2)
ax.scatter(state_final[:,1],q_1,c='g',label='Menu learning',alpha=0.2)
ax.scatter(state_final[:,1],q_2,c='b',label='Hotkey',alpha=0.2)
ax.legend(loc=2)

plt.show()