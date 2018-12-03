import numpy as np
import dill as pkl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

path="/home/ash/Menu_hotkey/discrete_results"
a=open(os.path.join(path,'Q_30_0.50_traces_softmax.pickle'),'rb')
b=open(os.path.join(path,'Q_30_softmax_0.5_0.5.pickle'),'rb')
c=open(os.path.join(path,'Q_30_softmax_1_1.pickle'),'rb')
q_fn_soft=pkl.load(a)
q_fn_trace=pkl.load(b)
q_fn_q=pkl.load(c)
print(len(q_fn_trace.keys()))
rms_error_soft=pkl.load(a)
rms_error_trace=pkl.load(b)
rms_error_q=pkl.load(c)
t=0
count_soft=[]
count_trace=[]
count_q=[]
for j in range(350):
    count_soft.append(np.mean(rms_error_soft[t:t+200]))
    count_trace.append(np.mean(rms_error_trace[t:t+200]))
    count_q.append(np.mean(rms_error_q[t:t + 200]))
    t+=200

fig = plt.figure()
ax = plt.subplot(111)
plt.title('Mean RMS error vs Training episodes')
plt.plot(count_soft,'r',label='Cost=0.50')
plt.plot(count_trace,'g',label='Cost=0.75')
plt.plot(count_q,'b',label='Cost=1')
ax.legend()
plt.xlabel('Episodes')
plt.ylabel('RMS error')
plt.show()


'''
for item in some:
    item=list(item)
    m.append(item[0])
    h.append(item[1])

q_values=list(q_fn.values())
q_values=np.array(q_values)
#print(q_values)
#q_0=np.reshape(q_values[:,0],m.shape)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(h,q_values[:,1],c='r')
ax.scatter(h,q_values[:,2],c='b')
ax.scatter(h,q_values[:,0],c='g')
#plt.show()
'''