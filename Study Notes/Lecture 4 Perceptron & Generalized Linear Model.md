```toc 
style: number
min_depth: 2
max_depth: 6
title: Lecture 4 Perceptron & Generalized Linear Model
allow_inconsistent_headings: true
varied_style: true
```

## Perceptron
A very different type of algorithm than logistic regression and least squares linear regression.

---

## Exponential Family
We say that a class of distributions is in the exponential family if it can be written in the form:$$p(y;\eta)=b(y)exp(\eta^{\tau}T(y)-a(\eta)) = b(y)\times\frac{exp(\eta^{\tau}T(y))}{e^{a(\eta)}}$$
> $y$ - Data
> $\eta$ - **Natural parameter** (also called the canonical parameter) of the distribution
> $T(y)$ - is the **sufficient statistic** (for the distributions we consider, it will often be the case that T(y) = y)
> $b(y)$ - **Base measure**
> $a(\eta)$ - **log partition function**
> 	$e^{-a\eta}$ plays the role of a normalization constant, that makes sure the distribution $p(y;\eta)$ integrates over y to 1.
> 
> A fixed choice of T, a and b defines a family (or set) of distributions that is parameterized by $\eta$; as we vary $\eta$, we then get different distributions within this family.

#### Bernoulli Distribution (used to model Binary Data)
![[Pasted image 20221227194231.png]]
 ![[Pasted image 20221227194700.png]]

#### Gaussian Distribution (with fixed variance)
![[Pasted image 20221227194805.png]]


#### Properties of Exponential Families (Why we use it)
 
1. If we perform Maximum Likelihood (MLE) on the exponential family, then MLE with respect to $\eta$ is concave, the Minimum cost is therefore convex.
2. Expectation of y: $E[y;\eta] = \frac{\partial}{\partial n}a(\eta)$
3. Variance of y: $Var[y;\eta] = \frac{\partial^{2}}{\partial n^{2}}a(\eta)$


---

## Generalized Linear Models

 We can build a lot of powerful models by choosing an appropriate exponential family and plugging it onto a linear model.
 By doing this, we need three assumptions/design choices:
 1. $y|x;\theta \sim Exponential\ Family(\eta)$
 2. $\eta = \theta^{\tau}x,\ \ \theta\in \mathbb{R}^{n},\ x\in\mathbb{R}^{n}$
 3. In the test, the output of a given x should be: $E[y|x;\theta]$, that $$h_\theta(x)=E[y|x;\theta]$$![[Pasted image 20221227201458.png]]
>Tips of picking appropriate exponential family:
>	Real number - Gaussian
>	Binary - Bernoulli
>	Count - Poisson
>	$\mathbb{R}^{+}$ - Gamma, Exponential

#### GLM Training

Learning Update Rule (Batch method): $$\theta_{ji}=\theta_{j} + \alpha \sum\limits_{i=1}^{m}(y^{i}-h_\theta(x^{i}))x_{j}^{i}$$
The update rule is the same for each linear model, no matter it is regression or classification.

#### More Terminologies

$\eta$ - **Natural parameter** (also called the canonical parameter)
$\mu=E[y;\eta]=g(\eta)$ - Canonical Response Function
$\eta=g^{-1}(\mu)$ - Canonical Link Function
$g(\eta) = \frac{\partial}{\partial n}a(\eta)$

We have 3 parameterization:
> $\theta$ - model parameter
> $\eta$ - natural parameter
> $\phi$ - Canonical Parameter (for Bernoulli Distribution)
> $\mu \sigma^{2}$ - Canonical Parameter (for Gaussian Distribution)
> $\lambda$ - Canonical Parameter (for Poisson Distribution)
![[Pasted image 20221227203628.png]]

Taking logistic regression as an example (Binary data so use Bernoulli):
$$h_{\theta}(x)=E[y|x;\theta]=\phi=\frac{1}{1+e^{-\eta}}=\frac{1}{1+e^{-\theta^{\tau}x}}$$

---

## Softmax Regression (Multiclass Classification)

The response variable y can take on any one of k values, so $y\in\{1,2,...,k\}$ 
So label y is a one-hot vector, indicating which class x corresponds to, and we have k $\theta_{class}$ for  each class instead of one.
https://www.youtube.com/watch?v=iZTeva0WSTQ&t=638s (From 1:15:09)
![[Pasted image 20221227210324.png]]


#### Cross-Entropy
Goal: We use Cross-Entropy to minimize the difference between $\hat{p}(y)$ and $p(y)$.

![[Pasted image 20221227210707.png]]
![[Pasted image 20221227210759.png]]
Then we do degree descent with respect to the parameters.