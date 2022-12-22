```toc 
style: number
min_depth: 2
max_depth: 6
title: Lecture 3 Locally Weighted & Logistic Regression
allow_inconsistent_headings: true
varied_style: true
```

> Outline:
> - Locally weighted regression
> - Probability interpretation
> - Logistic regression
> - Newton's method


## Locally Weighted Regression

>We can distinguish machine learning techniques into two types:
>1. *Parametric learning algorithm*: Fit **fixed** set of parameters ($\Theta_{i}$) to data (e.g. linear regression)
>2. N*on-parametric learning algorithm*: Amount of data/parameters **grows** (linearly) with the size of data (e.g. Locally weighted regression)

Fit $\Theta$ to minimize: $$\frac{1}{2}\sum\limits_{i=1}^{m}w^{i}(y^{i}-\Theta^{\tau
}x^{i})^2$$ Where $w^{i}$ is a "weight" function:$$w^{i}=exp\left(-\frac{(x^{i}-x)^2}{2\tau^2}\right)=e^{\left(-\frac{(x^{i}-x)^2}{2\tau^2}\right)}$$
If $(x^{i}-x)$ is small, $w^{i}\approx 1$
If $(x^{i}-x)$ is small, $w^{i}\approx 0$

> $x$ - the position you want to make a prediction
> $x^i$ - the input x for your $i^{th}$ training data example
> $\tau$ - the bandwidth, to choose how many nearby examples to use in order to fit the straight line

By adding a weight function, it allows the algorithm to only pay attention to the data that close to $x$ and then predict a linear function

---

## Probability Interpretation
Why might the least-squares cost function J, be a reasonable choice?
Assumptions
	Assume $y_{i}=\Theta^{\tau}x^{i}+\epsilon^{i}$ where $\epsilon^{i}$ is "error" or un-modelled effects, with random noise.
	Assume $\epsilon^{i}\sim\mathcal{N}(0,\sigma^2)$, so the noise is normal distribution.
	So the density function of $\epsilon^{i}$ (Gaussian distribution) is: $$p(\epsilon^{i})=\frac{1}{\sqrt{2\pi\sigma}}exp\left(-\frac{(\epsilon^i)^2}{2\sigma^{2}}\right)$$
	A huge assumption is that the $\epsilon^{i}$ is **IID** **(independently and identically distributed)**: the error term of one house is independent of the other house.

Those assumptions imply:![[Pasted image 20221221183933.png | 400]]
( $p(y^{i}|x^{i}; \theta)$ - the probability of y given $x$ parameterized by $\theta$)
OR: $$(y^{i}|x^{i};\theta)\sim\mathcal{N}(\theta^{\tau}x^{i}, \sigma^2)$$**Given $x$ and $\theta$, what's the probability of a particular house's price: it's going to be Gaussian with mean = $\theta^{\tau}x^{i}$ and the variance = $\sigma^2$.**
![[Pasted image 20221221185840.png|600]]
![[Pasted image 20221221185902.png|600]]

----

## Logistic Regression
One of the most popular classification algorithm, that:$$h_\theta\in[0,1]\hspace{1cm} y\in\{0, 1\}$$
Hypothesis function: $$h_{\theta}(x)=g(\theta^{\tau}x)=\frac{1}{1+e^{-\theta^{\tau}x}}$$
where g(z) is the sigmoid/logistic function: $$g(z)=\frac{1}{1+e^{-z}}$$
Let us assume that, the probability of y given by x parameterized by $\theta$ is:
$$
\begin{cases} 
P(y=1|x;\theta)=h_\theta(x) \\ \\

P(y=0|x;\theta)=1-h_\theta(x)
\end{cases}
$$
This can be written by (the RHS only calculate one term because y is either 0 or 1): $$p(y|x;\theta)=(h_\theta(x))^{y}(1-h_\theta(x))^{1-y}$$
Then we write this as the likelihood of parameter: $$
\begin{align}
L(\theta) &=p(\vec{y}|x;\theta) \\
&=\prod_{i=1}^{m}p(y^{i}|x^{i};\theta) \\
& =\prod_{i=1}^{m}(h_\theta(x^{i}))^{y^{i}}(1-h_\theta(x^{i}))^{1-y^{i}}
\end{align}$$
To make the algorithm easier to maximize: $$\begin{align}
\mathscr{l}(\theta) & =log\ L(\theta) \\
& = \sum\limits_{i=1}^{m}\ y^{i}\ log\left[h_\theta(x^{i})\right]\ +\ (1-y^{i})\ log\left[(1-h_\theta(x^{i}))\right]
\end{align}$$
Now, last step is to choose $\theta$ to maximize $\mathscr{l}(\theta)$.
We use **batch gradient ascent** $$\theta_{ji}=\theta_{j}+\alpha\frac{\partial}{\partial \theta_{j}}\mathscr{l}(\theta)$$
> We take gradient descent with loss function, while taking gradient ascent with likelihood function
![[Pasted image 20221221195406.png]]

So, overall:$$\theta_{ji}=\theta_{j} + \alpha \sum\limits_{i=1}^{m}(y^{i}-h_\theta(x^{i}))x_{j}^{i}$$
This is actually the same algorithm as linear regression.

---

## Newton's Methods
A method to optimize $\theta$ differ from gradient descent, which is sometimes quicker and fewer iterations.
The disadvantage of Newton's Method is that in high-dimensional problems, when $\theta$ is a vector, then each step of Newton's Method is much more expensive.
![[cs229-lec2.pdf#page=8]]



