```toc 
style: number
min_depth: 2
max_depth: 6
title: Lecture 2 Linear Regression and Gradient Descent
allow_inconsistent_headings: true
varied_style: true
```

> Outline:
> - Linear Regression
> - Batch/stochastics gradient descent
> - Normal equation


## Linear Regression

#### Design a Learning Algorithm: LMS (Least Mean Squares)

###### First Step: How to represent the hypothesis H?

>$h(x) = \Theta_{0} + \Theta_{1}X_{1} + \Theta_{2}X_{2} + ...$
>where $X_{1}= size,\  X_{2}= \#bedrooms,\ \ \Theta =\ parameters$
>
>To simplify the notation, we have hypothesis for House-Price problem:
>$$\begin{align}
 &h(x)=\sum\limits_{j=0}^{n}\Theta_{j}X_{j}\hspace{1cm}  where\ X_{0}= 1\\
 &\Theta = \begin{bmatrix} \Theta_{0}\\ \Theta_{1} \\ \Theta_{2}\end{bmatrix}
\hspace{0.5cm} X = \begin{bmatrix} X_{0}\\ X_{1} \\ X_{2}\end{bmatrix}\space where\ X_{1}= size,\  X_{2}= \#bedrooms
 \end{align}$$
>The job of the algorithm is to choose $\Theta$ to make good predictions about your prices of houses.
> We define:
> 
> 	- $m = \#training\ examples$ (the number of rows in the table, or the number of training set)
> 	- $n =\ \#features$
> 	- The data table is a matrix of size $m\times n$ , the particular tuple is $m_{i}\times n_{j}$
> 	- $X =\ Inputs\ / \ features$
> 	- $Y =\ Outputs\ / \ target\ variable$
> 	- $(X, Y) =\ Training\ Examples$
> 	- $(X^{i}, Y^{i}) =\ i^{th}\ training\ example$
> 
> 
>We choose $\Theta$ such that $h(X)\approx y$ for training examples, or we want to minimize the difference between the prediction and y:
>$$Minimize\ \ \frac{1}{2}\sum\limits_{i=1}^{m}(h_{\theta}(X^{i}) - y^{i})^2$$So the ***cost function*** for linear regression is:
>$$J(\Theta)=\frac{1}{2}\sum\limits_{i=1}^{m}(h_{\theta}(X^{i}) - y^{i})^2$$
---

## Gradient Descent

#### Batch Gradient Descent
When you sum up all the training data sets, the algorithm is called Batch Gradient Descent.
**Main disadvantage** of batch gradient descent:
	Imagine you have a large amount of dataset, when making one update to your parameters, in order to take a single step of gradient descent, you need to calculate this sum $\sum\limits_{i=1}^{m}$ for $n$ times.
	If the descent needs hundreds of time to converge, then you'll be scanning through your entire data-set hundreds of times.

The algorithm mainly tooks two steps:
	- State with some $\Theta$ (e.g. $\Theta = \vec{0}=[0, 0, 0, ...]$)
	- Keep changing $\Theta$ to reduce $J(\Theta)$ , there is only one global minimum for linear regression
$$\Theta_{j\ i}=\Theta_{j} - \alpha\cdot\frac{\partial}{\partial\ \Theta_{j}}(J(\Theta))\hspace{1cm} where\ j=0,1,2,...,\ \alpha=learning\ rate$$
> In practice, we set $\alpha = 0.01$
> Assume $m = 1$, such that there is only one house for training dataset:
$$\begin{align}

 \frac{\partial}{\partial\ \Theta_{j}}J(\Theta)  
   &= \frac{\partial}{\partial\ \Theta_{j}}\left(\frac{1}{2}(h_{\theta}(X) - y)^{2}\right)\\
   &= 2\cdot\frac{1}{2}(h_\theta(X)-Y)\cdot\frac{\partial}{\partial\ \Theta_{j}}\Big(h_\theta(X)-Y\Big)\\
   &=  (h_\theta(X)-Y)\cdot \frac{\partial}{\partial\ \Theta_{j}}\Big(\Theta_{0}X_{0} + \Theta_{1}X_{1} + \Theta_{2}X_{2}\ + ...+\ \Theta_{n}X_{n}\ -Y\Big)\\
   &= \Big(h_\theta(X)-Y\Big)\cdot X_{j}\\
   \\
   Overall\ \ \Theta_{j\ i} &= \Theta_{j}-\alpha\cdot\sum\limits_{i=1}^m\Big(h_\theta(X^{i})-Y^{i}\Big)\cdot X_{j}^{i}
   \ \ \ (for\ j=0,1,...,n)
   
   \end{align}$$
---
#### Stochastic Gradient Descent
$$\begin{align}
 Repeat&\{\\
 &For\ i=1\ to\ m\{\\
 &\hspace{1cm}\Theta_{j\ i}=\Theta_{j} - \alpha\cdot\Big(h_\theta(X^{i})-Y^{i}\Big)\cdot X_{j}^{i}\\
 & \hspace{1cm}//\ Update\ for\ every\ j\\
 &\hspace{1cm}\}\\
 &\}
\end{align}$$

In Stochastic Gradient Descent algorithm, instead of scanning through all million examples before you update the parameters $\Theta$ even a little bit, it use the derivative of just one single example. 

---

## Normal Equation (The gradient descent algorithm works only for linear regression)

Refer to the lecture notes: ![[cs229-lec2.pdf#page=8]]



