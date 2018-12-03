import matplotlib.pyplot as plt
import dill as pkl
import os
import numpy as np

"""Plot RMS Error/Reward/Dict size vs Episodes(or time)"""

def plot(temp):
    values=[]
    values.append(temp)
    for i in range(15):
        temp=temp-0.1*temp
        values.append(temp)
    return values

#plt.plot(plot(2))

path="/home/ash/Menu_hotkey/discrete_results"
a=open(os.path.join(path,'CTS_gpsarsa.pickle'),'rb')
rms_error=pkl.load(a)
#plt.plot(rms_error)
dict_size=pkl.load(a)
reward=pkl.load(a)
avg_rms=[]
sum=[]
t=0

for i in range(70):
    avg_rms.append(np.mean(rms_error[t:t+10]))
    sum.append(np.mean(reward[t:t+10]))
    t+=10

plt.figure()
plt.plot(avg_rms)
plt.xlabel("Episodes")
plt.ylabel("Mean RMS error")
plt.figure()
plt.plot(sum)
plt.xlabel("Episodes")
plt.ylabel("Avg Reward")
plt.figure()
plt.plot(dict_size)
plt.xlabel("Episodes")
plt.ylabel("Kernel dictionary size")
plt.show()
