```toc 
style: number
min_depth: 2
max_depth: 6
title: Lecture 5 GDA & Naive Bayes
allow_inconsistent_headings: true
varied_style: true
```


## Gaussian Discriminant Analysis (GDA)
Gaussian Discriminant Analysis is **a Generative Learning Algorithm** and in order to capture the distribution of each class, it tries to fit a Gaussian Distribution to every class of the data separately.

Suppose $x \in \mathbb{R}^{n}$ (drop $x_{0}=1$ conventionally)
Assume $P(x|y)$ is Gaussian, if z is due to Gaussian, with some mean vector $\vec{\mu}$ and some [[covariance]] matrix $\Sigma$ :
	$z \sim \mathcal{N}(\vec{\mu}, \Sigma)$ 
		$z \in \mathbb{R}^{n}$ 
		$\vec{\mu} \in \mathbb{R}^{n}$ 
		$\Sigma \in \mathbb{R}^{n \times n}$ 
	$\mathbb{E}(z)=\mu$ 
	$$\begin{align}
	Cov(z)&=\mathbb{E}[(z-\mu)(z-\mu)^\tau]\\
	&=\mathbb{E}(z\times z^{\tau}) - \mathbb{E}(z)\times \mathbb{E}(z)^\tau
	\end{align}$$
> *Multivariate Normal Distribution* ![[Pasted image 20230109143406.png | 500]]
 So the Multivariate Gaussian Density has two parameters: $\mu$ and $\Sigma$, they control the mean and the variance of this density.

Examples of how a Gaussian Distribution looks like (with $\mu = [0, 0]^{\tau}$): ![[Pasted image 20230109152053.png]]
$$\Sigma_{1} = \begin{bmatrix}1 &0 \\ 0 &1 \end{bmatrix}
\hspace{3cm} \Sigma_{2} = \begin{bmatrix}0.6 &0 \\ 0 &0.6 \end{bmatrix}
\hspace{3cm} \Sigma_{3} = \begin{bmatrix}2 &0 \\ 0 &2 \end{bmatrix}$$
![[Pasted image 20230109152643.png]]
$$\Sigma_{1} = \begin{bmatrix}1 &0 \\ 0 &1 \end{bmatrix}
\hspace{3cm} \Sigma_{2} = \begin{bmatrix}1 &0.5 \\ 0.5 &1 \end{bmatrix}
\hspace{3cm} \Sigma_{3} = \begin{bmatrix}1 &0.8 \\ 0.8 &1 \end{bmatrix}$$
The contours of the above distributions:
![[Pasted image 20230109153447.png]]
$$\Sigma_{1} = \begin{bmatrix}1 &0 \\ 0 &1 \end{bmatrix}
\hspace{3cm} \Sigma_{2} = \begin{bmatrix}1 &0.5 \\ 0.5 &1 \end{bmatrix}
\hspace{3cm} \Sigma_{3} = \begin{bmatrix}1 &0.8 \\ 0.8 &1 \end{bmatrix}$$
![[Pasted image 20230109153518.png]]
$$\Sigma_{1} = \begin{bmatrix}1 &-0.5 \\ -0.5 &1 \end{bmatrix}
\hspace{2.5cm} \Sigma_{2} = \begin{bmatrix}1 &-0.8 \\ -0.8 &1 \end{bmatrix}
\hspace{2.5cm} \Sigma_{3} = \begin{bmatrix}3 &0.8 \\ 0.8 &3 \end{bmatrix}$$
When $\Sigma = I$, the $\mu$ indicates:
![[Pasted image 20230109153849.png]]
$$\mu_{1} = \begin{bmatrix}\ 1\ \\ 0 \end{bmatrix}
\hspace{3cm} \mu_{2} = \begin{bmatrix}\ -0.5\ \\ 0 \end{bmatrix}
\hspace{3cm} \mu_{1} = \begin{bmatrix}\ -1\ \\ -1.5 \end{bmatrix}$$

#### GDA Model
For GDA model:
> $$\begin{align} p(y) &= \phi^{y}(1-\phi)^{1-y} \\
	 p(x|y=0) &= {\frac{1}{(2\pi)^\frac{d}{2}|\Sigma|^\frac{1}{2}}exp\Big{(} -\frac{1}{2}(x-\mu_{0})^{T}\Sigma^{-1}(x-\mu_{0})\Big{)}}\\
	 p(x|y=1) &= {\frac{1}{(2\pi)^\frac{d}{2}|\Sigma|^\frac{1}{2}}exp\Big{(} -\frac{1}{2}(x-\mu_{1})^{T}\Sigma^{-1}(x-\mu_{1})\Big{)}}
\end{align}$$
 Parameters:  
	$\mu_{0} \in \mathbb{R}^{n}$
	$\mu_{1} \in \mathbb{R}^n$
	$\Sigma \in \mathbb{R}^{n\times{n}}$
	$\phi\in\mathbb{R}$

> Usual Training Set: $$\{(x^{i}, y^{i})\}_{i=1}^{m}$$
> Objective: maximize the joint likelihood $$\begin{align} l(\phi, \mu_{0}, \mu_{1}, \Sigma) &= log \prod_{i=1}^{n}p(x^{(i)}, y^{(i)}; \ \phi,  \mu_{0}, \mu_{1}, \Sigma)\\
	& = log \prod_{i=1}^{n}p(x^{(i)}| y^{(i)}; \mu_{0}, \mu_{1}, \Sigma)p(y^{(i)};\phi) \\
	& = log \prod_{i=1}^{n}p(x^{(i)}| y^{(i)})p(y^{(i)})
	\end{align}$$![[Pasted image 20230109164153.png]]
	Prediction (using [[arg max notation]]):
	Because $p(x)$ is a constant, so we do not need it for argmax -  $$\begin{align} argmax_{y} p(y|x) &= argmax_{y} \frac{p(x|y)p(y)}{p(x)} \\ & = argmax_{y} p(x|y)p(y)
\end{align}$$




 
---

## Generative VS. Discriminative Comparison

#### Discriminative:
> Learn $P(y|x)$ (the probability of y given x) or learn:
> $$h_\theta(x)= \begin{cases}0 \\ 1 \end{cases} 
  \text{\hspace{1cm} some function mapping from x -> y directly}$$

#### Generative:
> Learn $P(x|y)$ (what's the feature x like given the class y) and also
> learn $P(y)$ (also called class prior)


---

## Naive Bayes

#### Bayes Rule
![[Pasted image 20230106202836.png]]


___
