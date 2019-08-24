---
title: 'TRPO:使策略梯度单调递增的杀器'
date: 2019-08-24 19:44:15
tags: [算法, 机器学习, 计算广告]
mathjax: true
categories: [强化学习]
---



## 从Q learning到策略梯度

在解决`MDP`问题的算法中，`Value Base`类算法的思路将关注点放在价值函数上，传统的`Q Learning`等算法是一个很好的例子。`Q Learning`通过与环境的交互，不断学习逼近`(状态, 行为)`价值函数$Q(s_t, a_t)$，而策略本身即选取使得在特定状态下价值函数最大的动作，即$a_t = \mathop{\arg\min}_{a}Q(s_t, a)$ ， 具体算法如图1所示。

![Q Learning算法](C:\Users\Conley\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu18.04onWindows_79rhkp1fndgsc\LocalState\rootfs\home\conley\github\embolismsoil.github.io\source\_posts\TRPO-使策略梯度单调递增的杀器\1566647970040.png)

其中$Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma \max _{a} Q\left(S^{\prime}, a\right)-Q(S, A)\right]$一步即时序差分法的价值函数逼近过程，具体原理详见。

Q learning算法已经能解决许多问题，但最致命的一点是: 在确定环境$s_t$下，策略选择的行动总是确定的，这对于很多场景来说，并不适用。例如玩剪刀石头布的时候，如果出拳的策略是一定的话，就很容易被对手察觉并击破。同时，Q learning也无法解决状态重名的问题。具体地说，状态重名是指在两个现实中的状态，在建模中表现出来的`state`是一样的，也就是$s_t$向量的每个维度都相等。如下图中格子世界的例子，如果状态被建模成二维向量，维度分别表示左右是否有墙阻挡，那么图中两个灰色格子的状态向量是一样的，于是他们在Q learning中学习到的策略会选择一样的行动，但矛盾的是: **如果选择向左走，对于第一个格子就是一次失败的决策。如果选择向右走，对于第二个格子来说就是一次失败的决策**。特别是如果使用$\epsilon-greedy$策略时，很可能在第一个灰格子会不停选择向左的行动，直到一次$\epsilon$概率的事件发生时，才有可能选择一次随机行为，从而有机会跳出这个坏处境。这时候还不如直接使用随机策略管用。

![格子世界](C:\Users\Conley\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu18.04onWindows_79rhkp1fndgsc\LocalState\rootfs\home\conley\github\embolismsoil.github.io\source\_posts\TRPO-使策略梯度单调递增的杀器\1566648579995.png)

针对上述种种缺点，策略梯度法应运而生。

首先，我们需要明确的是，强化学习的最终目的是最大化价值函数。Q learning算法的思路比较绕，Q learning并没有直接去最大化价值函数，而是思考: 在给定状态$s_t$下，做出动作$a_t$会得到什么样的回报？ 得到答案之后，每次都贪婪地选择回报最大的那个动作。 可是为什么我们不直接思考: 在给定状态下，做出什么样的动作，才能让回报最大化？ 策略梯度就是这样一个直接的算法。

具体地说，策略梯度算法将策略建模成为$\pi_{\theta}(s,a)$，表示在$s$状态下选择$a$动作的概率，其中$\theta$为参数。并且将负回报函数作为损失函数，应用梯度下降法将期望奖励最大化。定义为

$$J(\theta)=\sum_{s} d(s) \sum_{a} \pi_{\theta}(s, a) \mathcal{R}(s,a) \tag{1}$$

这样，(1)式对参数$\theta$求梯度得到

$$\begin{aligned} \nabla_{\theta} J(\theta) &=\sum_{\mathbf{s} \in S} d(s) \sum_{a \in A} \pi_{\theta}(s, a) \nabla_{\theta} \log \pi_{\theta}(s, a) \mathcal{R}_{s, a} \\ &=\mathbb{E}_{\pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(s, a) \mathcal{R}(s,a)\right] \end{aligned}  \tag{2}$$

式子(2)的期望通过均值代替得到

$$\nabla_{\theta}J(\theta)=\frac{1}{N}\sum{\nabla_{\theta}\log\pi_{\theta}(s, a)\mathcal{R}(s,a)} \tag{3}$$

于是我们得到了蒙特卡洛策略梯度算法

![蒙特卡洛策略梯度](C:\Users\Conley\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu18.04onWindows_79rhkp1fndgsc\LocalState\rootfs\home\conley\github\embolismsoil.github.io\source\_posts\TRPO-使策略梯度单调递增的杀器\1566661134557.png)

