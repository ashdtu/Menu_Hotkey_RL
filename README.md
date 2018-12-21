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
- Epsilon-greedy Explora
- Softmax 
- Covariance based Exploration for Gaussian Process( Active learning)
- Stochastic exploration for Gaussian Process 

