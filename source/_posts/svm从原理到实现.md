---
title: svm从原理到实现
date: 2018-05-03 15:40:40
tags: [算法, 机器学习]
mathjax: true
categories:
    - 机器学习
    - 算法
---

> **引言**：svm是一种二分类算法，它通过寻找特征空间中的一个超平面，将样本划分开来，通过一定的策略选择最优的超平面，以期获得较好的准确性能和泛化性能。

<!-- more -->

## 一、svm的模型
在分类问题中，有一种思路是**在样本空间中寻找一个超平面将不同类的样本划分开**，依据此超平面，在新数据到来时，我们只需要看这个新的数据点落在超平面的哪一侧即可知道该数据的类别。
![](svm_model.gif)
用数学可以形式化地将这个超平面表示为
$$
W^TX+b=0
$$
如果给定一个训练样本$$$(X_0, y_0)$$$
但问题是能够将样本划分开的超平面可能不止一个，那我们要如何选择最好的超平面呢？从数学的角度来讲，我们该如何
$$
\begin{eqnarray}
\nabla\cdot\vec{E} &=& \frac{\rho}{\epsilon_0} \\
\nabla\cdot\vec{B} &=& 0 \\
\nabla\times\vec{E} &=& -\frac{\partial B}{\partial t} \\
\nabla\times\vec{B} &=& \mu_0\left(\vec{J}+\epsilon_0\frac{\partial E}{\partial t} \right)
\end{eqnarray}
$$
## 二、svm的策略

## 三、svm的数学推导
## 四、svm的算法
