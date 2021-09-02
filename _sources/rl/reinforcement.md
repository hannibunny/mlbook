# Reinforcement Learning

## Introduction

### Basic Notions

In supervised Machine learning a set of training data $T=\lbrace (\mathbf{x}_i,r_i) \rbrace_{i=1}^N$ is given and the task is to learn a model $y=f(\mathbf{x})$, which maps any input $\mathbf{x}$ to the corresponding output $y$. This type of learning is also called *learning with teacher*. The teacher provides the training data in the form that he labels each input $\mathbf{x}_i$ with the corresponding output $r_i$ and the student (the supervised ML algorithm) must learn a general mapping from input to output. This is also called inductive reasoning. 

In reinforcement learning we speak of an **agent** (the AI), which acts in an **environment**. The **actions $A$** of the agent must be such that in the long-term the agent is successful in the sense that it approximates a pre-defined goal as close and as efficiently as possible.  The environment is modeled by it's **state $S$**. Depending on it's actions in the environment the agent may or may not receive a positive or negative **reward $R$** [^f1]. Reinforcement learning is also called *learning from a critic*, because the agent trials different actions and sporadically receives feedback from a critic, which is regarded by the agent for future action decisions.

Reinforcement Learning refers to **Sequential Decision Making**. This means that we model the agents behaviour over $time$. At each **discrete time-step $t$** the agent perceives the environment state $S_t$ and  possibly a reward $R_t$. Based on these perceptions it must then decide for an action $A_t$. This action influences the state of the environment and possibly triggers an award. The new state of the environment is denoted by $S_{t+1}$ and the new reward is denoted by $R_{t+1}$. In the long-term the agents decision-making-process must be such, that a pre-defined target-state of the environment is approximated as efficiently as possible. The proximity of a state $s$ to the target-state is measured by an **utility function $U(s)$**. 

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/RLAgent.png
---
align: center
width: 600pt
name:  rlagent
---
Interaction of an Agent with it's environment in a Markov Decision Process (MDP). Image source: {cite}`Sutton1998`.

```

### Examples

#### Board Games

Games, in particular board games like chess, checkers or Go are typical applications for reinforcement learning. The agent is the AI, which plays against a human player. The agents's goal is to win the game. In the most simple setting it perceives a non-zero reward only at the very end of the game. This reward is positive if the AI wins, otherwise it is negative. If the AI is in turn, it first perceives the current state, which is the current board-constellation. Then it has to decide for a new action. This action modifies the state on the board. This new state is the basis for the human players action, which in turn provides a new state to the AI and so on.

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/alphaGo.jpeg
---
align: center
width: 400pt
name:  alphaGo
---

Deepmind's AlphaGo ({cite}`Silver16`) combines Reinforcement Learning, Deep Learning and Monte Carlo Tree Search. AlphaGo was able to beat the world champion in Go, Lee Sedol, in 2016. 

```

#### Non-deterministic Navigation

Navigation or similarly Pathfinding can be solved by the A*-algorithm, if the environment is *deterministic*. This means that given the current state $s_t$ and a selected action $a_t$, the successive state $s_{t+1}$ can uniquely be determined. In a non-deterministic there exist different possible successive states for a given state-action-pair. In such non-deterministic environments Reinforcemt Learning can be applied to learn an optimum strategy. This stategy defines for each state the action, which is best, in the sense that the expected future cumulative reward (the utility) is maximized.

In the example depicted below the task is to find best path from the field $Start$ to the field in the upper right corner. The possible actions of the agent are to move *upwards, downwards, right* or *left*. The environment is observable in the sense that the agent knows it's current state. However, the environment is non-deterministic because for a known state-action-pair different successive states are possible. For this uncertainty the following probabilities are assumed to be known:

* Probability that state in the selected direction is actually reached is $P=0.8$.
* Probability for a $\pm 90°$ deviation is $P=0.1$ for each. 

If selected direction hits a wall, the agent remains in it's current state. A reward of $r_t=1$ is provided, if $a_t$ terminates in field $(4/3)$ (the upper right corner) and a reward of $r_t=-1$ is provided if $a_t$ terminates in field $(4/2)$. For any other action the reward (cost) is $r_t=-0.04$.


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/4-3welt.jpg
---
align: center
width: 200pt
name:  rlex1
---

Image source: {cite}`russel2010`: Pathfinding in a non-deterministic environment
```

For this pathfinding scenario, the optimum strategy, learned by the RL agent may look like in the picture below. The strategy is defined by the arrows in the states. These arrows determine in each state the action to take in order to maximize the expected future cumulative reward.

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/4-3weltstrategie1.jpg
---
align: center
width: 200pt
name:  rlex1strat
---

Image source: {cite}`russel2010`: Optimum strategy learned by the RL agent
```

#### Movement of simple Robots

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/krabbelroboter1Engl.png
---
align: center
width: 600pt
name:  rlex2
---
Image source: {cite}`ertel09`: Two simple crawling robots. Each robot has two joints. Each joint can be in two different positions. Hence, there exists 4 different states. The robot shall learn to control the joints such that it efficiently moves from left to right. Whenever the robot moves to the right it perceives a positive reward. Movement to the left is punished with a negative reward. As can easily be verified, a movement of the robot depends not only on a single action but on a state-action-pair.
```

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/krabbelroboter2Engl.png
---
align: center
width: 600pt
name:  rlex2ff
---
Image source: {cite}`ertel09`: State $s_t$, action $a_t$ and reward $r_t$ for 4 successive time-steps.

```


## Markov Decision Process (MDP)

Formally, Sequential Decision Making is usually described by a **Markov Decision Process (MDP)**. In a **finite MDP** the set of states $\mathcal{S}$, the set of actions $\mathcal{A}$ and the set of rewards $\mathcal{R}$ is finite. For MDPs the *Markovian property* is assumed, which states that for each given state-action-pair, the 

* probability distribution of the successive state $S_t$ and 
* the probability distribution of the successive reward $R_t$

depends only on the immediate preceding state and all states which lie further back need not be regarded. The function 

$$
p(s',r \mid s,a) = P(S_t=s',R_t=r \mid S_{t-1}=s, A_{t-1}=a) \quad \forall s`,s \in \mathcal{S}, r \in \mathcal{R} \mbox{ and } a \in \mathcal{A}
$$ (mdpfull)

describes the full dynamics of the MDP. By applying the marginalisation law (see [Basics of Probability Theory](https://hannibunny.github.io/probability/ProbabilityMultivariate.html)), 

* the **state-transition probability** can be calculated by

$$
p(s' \mid s,a) = P(S_t=s'\mid S_{t-1}=s, A_{t-1}=a) = \sum\limits_{r \in \mathcal{R}} p(s',r \mid s,a)
$$ (transmod)

* the **expected reward** can be calculated by

$$
r(s,a) = \mathbb{E}(R_t \mid S_{t-1}=s, A_{t-1}=a) = \sum\limits_{r \in \mathcal{R}} \sum\limits_{s' \in \mathcal{S}} p(s',r \mid s,a).
$$ (rewexp)

In a MDP the goal of an agent is always to maximize the expected value of the cumulative sum of received rewards. The cumulative sum of received future rewards at time $t$ is als called the **return $G_t$**:

$$
G_t= R_{t+1}+R_{t+2}+R_{t+3}+\cdots + R_{T},
$$ (returnfinite)

where $T$ is the final time step. This formula can only be applied in the cases, where a *final time step $T$* can be defined or the entire agent-environment interaction can be partitioned into subsequences, called *episodes*. Whenever, such a termination can not be defined, the return can not be calculated by equation {ref}`returnfinite`, because $T=\infty$ and the return could be infinite. In order to cope with this problem a *discounting rate $\gamma$* is applied to calculate the return  

$$
G_t= R_{t+1}+\gamma R_{t+2}+ \gamma^2  R_{t+3}+cdots + \cdots = \sum\limits_{k=0}^{\infty} \gamma^k R_{t+k+1},
$$ (returninfinite)

with $0< \gamma < 1$. In this way rewards in the far future are weighted less than rewards immediately ahead and the return value will be finite. Equation {ref}`returninfinite` can easily be turned in a recursive form as follows:

$$
G_t= R_{t+1}+\gamma G_{t+1}
$$ (returnrecursive)


* Further reading: {cite}`Sutton1998`, {cite}`russel2010`, {cite}`ertel09`
* [RL video tutorial by deeplizard](https://youtu.be/nyjbcRQ-uQ8)

[^f1]: *Reinforcement* is just another word for *reward*, hence learning from reinforcements.