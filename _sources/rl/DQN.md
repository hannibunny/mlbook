# Deep Reinforcement Learning

In the previous section Reinforcementlearning has been introduced. For solving the **control problem** with TD-Learning, Sarsa and Q-Learning has been described. These algorithms estimate $Q(S,A)$ -values for each possible state-action pair $(S,A)$. From the estimated $Q(S,A)$-values the policy can easily be derived by selecting in each state $S$ the action $A$, which has the maximum $Q(S,A)$-value in this state. In order to estimate the $Q(S,A)$ reliably, each state-action pair must be traversed multiple times. This is not feasible for many real-world problems, where the number of possible state-actions pair is very large. For example, in Backgammon alone the number of states is estimated to be in the range of $10^{20}$ and the number of state-action pairs is correspondingly higher.

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/Backgammon.jpg
---
align: center
width: 300pt
name:  backgammon
---
In Backgammon alone the number of states is estimated to be in the range of $10^{20}.$

```

The problem, that a multiple traversion of all state action pairs is not feasible, is solved by learning a regression model, i.e. a function, which maps Q-Values to each possible state-action pair. In this approach **Reinforcement-Learning and Supervised Learning are combined** in the way that e.g. Q-Learning is applied to provide $Q(S,A)$-values for a limited set of state-action pairs. Then a supervised ML-algorithm, e.g. a MLP, is applied to learn a function 

$$
f: (S,A) \rightarrow Q(S,a).
$$


## Deep Q-Network (DQN) - Under Construction

Before the invention of Deep Q-Network (DQN) in {cite}`MnihKSGAWR13`, either linear regression models or non-linear conventional neural networks (MLPs) have been applied for learning a good approximation. Training linear models turned out to be stabel, in the sense of convergence. However, they often lacked sufficient accuracy. On the other hand non-linear models turned out to have weak convergence behaviour. This is because ML-algorithms usually assume **i.i.d** (independent and identical distributed data). However, data provided by the reinforcement-algorithm is sequentially strongly correlated. Moreover, the distribution of data may change abruptly, because a relatively small adaption of the $Q(S,A)$-values may yield a totally different policy.

The publication *Human-level control through deep reinforcement learning* ({cite}`MnihKSGAWR13`) constituted a breakthrough in the combination of Reinforcement Learning and Deep Neural Networks. For some retro Atari computer games such as *Pong*, *Breakout*, *Seaquest* or *Space Invaders* , the authors managed to learn a good non-linear $Q(S,A)$-function with a Convolutional Neural Network (CNN).

In their experiments, the authors applied their DQN to a range of *Atari 2600* games implemented in the *Arcade Learning Environment*[^f1]. *Atari 2600* is a challenging RL testbed that presents agents with a high dimensional visual input: $210 \times 160$ RGB video at 60Hz. The only information provided to the DQN has been the screencasts of the computer games. 


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/atariClassics.PNG
---
align: center
width: 600pt
name:  atariscreenshots
---
Screenshots of some classical Atari computer games, for which DQN has been applied to learn a good $Q(S,A)$-function.

```

In this subsection the concept of DQN, as developed in {cite}`MnihKSGAWR13`, will be described.


```{figure} https://maucher.home.hdm-stuttgart.de/Pics/Q-Network.PNG
---
align: center
width: 600pt
name:  statetoactionvalue
---
Predicting a Q-value for each state-action pair at the input of a network is conceptionally the same as predicting for a given input state a Q-value for each possible action. Image Source: [http://ir.hit.edu.cn/~jguo/docs/notes/dqn-atari.pdf](http://ir.hit.edu.cn/~jguo/docs/notes/dqn-atari.pdf)

```

The input to the CNN represents the current game state. The output of the DQN has as much elements as the maximal number of actions available. Atari classical games have been controlled by a joystick, which provides 15 different postions. For each of these 15 actions one Q-value is estimated for the current input state.

The input to the network represents the current game state. In general a *state* should always contain as much as possible relevant information. In computer games **movement** is obviously a relevant information. Since movement can not be obtained by observing only a single frame of the screencast, four successive frames are applied to the input of the network. However, these frames are not directly passed to the first layer of the CNN, instead a **preprocessed version of 4 successive frames of the screencast** are used.  

```{figure} https://maucher.home.hdm-stuttgart.de/Pics/DQNcnn.PNG
---
align: center
width: 600pt
name:  dqnarchitecture
---
Architectur of the DQN CNN. Image Source: {cite}`MnihKSGAWR13`

```



```{figure} https://maucher.home.hdm-stuttgart.de/Pics/DQNcnnTable.PNG
---
align: center
width: 400pt
name:  dqnparams
---
Hyperparameters of the DQN CNN. Image Source: {cite}`MnihKSGAWR13`

```

# Under Constuction !!!

[^f1]: [Arcade Learning environment](https://github.com/mgbellemare/Arcade-Learning-Environment) 