# Toy Problems

Comparing the performance of Muon to other optimizers like Adam on relatively simple toy problems. 


## Orthogonal Proscrutes Problem
Consider the constrained optimization problem
$$
\text{minimize}\|A \Omega - B \|_F \quad \text{subject to } \Omega^T\Omega = I
$$
where $\Omega \in \mathbb{R}^{d\times d} , A, B \in \mathbb{R}^{n \times d}$

We consider projected descent methods. Namely after each optimizer step, project onto the subspace of orthogonal matrices. 

We compare the performance of Adam and Muon here, measuring the distance between $f(\Omega) and f(\Omega^*)$ where $\Omega^*$ is the global minimizer.

We note that ignoring the orthogonality constraint gives just a matrix regression problem.

## Matrix Quadratic Regression
As discussed in [https://arxiv.org/abs/2505.21799] 

Consider the problem
$$
f(X) := \frac{1}{2}\| A X B -C \|_F^2
$$
for $X \in \mathbb{R}^{m\times n}$ and $A \in \mathbb{R}^{p\times m},  \in \mathbb{R}^{n\times q}, C \in \mathbb{R}^{p\times q}$

As in the paper, we choose $(m, n, p, q) = (500, 100, 1000, 250)$. This ensures f is strongly convex.