A **reinforcement learning** package to determine computationally rational strategy of navigating **Menu-Hotkey interfaces**.  
> The package is intended to be used as a playground to test HCI models with general Reinforcement learning algorithms. Therefore, the package has been modularised into **4 main classes** to quickly experiment with model definitions and user strategies.  

**Agent**: Policy and decision making(Strategy)

**Environment**:
- defines Utility(Reward structure)
- Ecology (observation definition)
- Mechanism (Observation model,transition dynamics)

**Learner**: Learning algorithms 

**Experiment**: Model training/testing and overall structure 

---

**Learning Algorithms:**

A. Tabular model free methods (Discrete space):

- Q-learning
- SARSA
- N-step TD backup
- Eligibility traces with Q learning   

B. Bayesian RL :

- Gaussian Process-SARSA  (non-sparse version/computational constraints) 
- Episodic GP-SARSA 	      (sparsified dictionary/fast) 

Original GP-TD algortithm discussed here:  [Engel et. Al, 2005](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.81.6420&rep=rep1&type=pdf), Reinforcement learning with Gaussian processes.

Episodic GP-SARSA for dialogue managers: [Gasic et al](http://mi.eng.cam.ac.uk/~sjy/papers/gayo14.pdf), Gaussian processes for POMDP-based dialogue manager optimisation


**Policy(Agent):**
- Epsilon-greedy Exploration
- Softmax 
- Covariance based Exploration for Gaussian Process( Active learning)
- Stochastic exploration for Gaussian Process 

**Sample Environment**: A sample **continuous maze** type environement is provided to test new implemented RL algorithms where the goal is to navigate a maze with obstacles to reach goal position. THe envionment can be used as  a testing playground. 

---

### Getting started

To test a new HCI model with the above RL algorithms and policies, we can get started in a few lines of code. Modify the proposed reward structure, transiton model and episode terminal conditon in the **Environment Class** functions.  Then simply import the required policy,learner as below:

```
env = Environment()
learner = GP_SARSA()
agent = Agent(env,learner)

while(TerminalCondition!=True):

  action=agent.getAction(state)
  reward=env.getReward(state,action)
  next_State=env.transition(state, action)
  learner.learn(state,action,reward,next_State)
```

4 sample experiments have been demonstrated in the **Experiment directory** detailing the above.

---

**Further updates:**
- Adding multi kernel support from Gaussian process library [Gpy library](https://sheffieldml.github.io/GPy/) 
- Implementation of Deep-Q with PO-MDP model 

