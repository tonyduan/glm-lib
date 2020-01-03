### Generalized Linear Models in Python

Last update: January 2020.

---

A generalized linear model fits an exponential family distribution with a linear model. The resulting optimization problem is convex.

**Exponential Family Distributions**

We assume the response is generated from a distribution parameterized by $\eta$,
$$
p_\eta(y) = b(y)\exp(\eta^\top T(y) - a(\eta)).
$$


Here $\eta$ is the natural parameter, $T(y)$ is the sufficient statistic, and $a(\eta)$ is the log partition function. Using a linear model with a canonical link function, we suppose 
$$
\eta = \theta x \implies \nabla_\theta \eta_i = x,\ \nabla^2_\theta \eta_i = xx^\top.
$$


Note that here $\eta \in\mathbb{R}^p, \theta\in\mathbb{R}^{p,n},x\in\mathbb{R}^n$. 

**Fisher Scoring**

The model is fit using maximum likelihood. We take a natural gradient step using the Fisher information in $\theta$,

$$
\theta \leftarrow \theta - \mathcal{I}_\theta^{-1}\nabla_\theta \left(-\log p_\theta(y)\right).
$$


Notice that by the chain rule, we have the following score and Hessian.
$$
\begin{align*}
\nabla_\theta -\log p_\theta(y) & = \nabla_\eta-\log p_\eta(y) x^\top\\
& =(\mathbb{E}_\eta[T(Y)] - T(y))x^\top \\
\nabla^2_{\theta_i}-\log p_\theta(y) & = \nabla_{\eta_i}^2 -\log p_\eta(y) xx^\top\\
& = \mathrm{Var}_\eta[T_i(Y)]xx^\top
\end{align*}
$$


The Fisher information matrix with respect to the $i$-th row of $\theta$ is then the expected value of a constant, so
$$
\mathbb{E}_\eta[\nabla^2_{\theta_i} -\log p_\theta(y)] = \mathrm{Var}_\eta[T_i(Y)]x x^\top
$$
This coincides with a Newton-Raphson step, since we are using the canonical link function.



**Link Functions**

A link function relates the expected value of the response to the natural parameters,
$$
\mathbb{E}_{\eta}[y] = g^{-1}(\eta).
$$

In our case we only ever use the canonical link function for each distribution.

#### References

[1] Nelder, J.A., and Wedderburn, R.W.M. (1972). Generalized Linear Models. Journal of the Royal Statistical Society. Series A (General) *135*, 370–384.

---

#### Appendix

Here we list useful results of exponential families.

**Result 1.** $T(y)$ are sufficient statistics. 
$$
p_\eta(y_1,\cdots,y_n) = b(y_1)\cdots b(y_n)\exp\left(\eta^\top \sum_{i=1}^n T(y_i)-na(\eta)\right)
$$


**Result 2.** The gradients of the log partition function always yield moments of the sufficient statistics. First recall that
$$
a(\eta) = \log \int_y b(y)\exp \left(\eta^\top T(y)\right)dy.
$$
Now observe that
$$
\begin{align*}
\frac{\partial}{\partial \eta_i} a(\eta) & = \frac{\int_y b(y)\exp \left(\eta^\top T(y)\right)T_i(y)dy}{\int_y b(y)\exp \left(\eta^\top T(y)\right)dy}\\
& = \int_y b(y)\exp(\eta^\top T(y) - a(\eta))T_i(y)dy\\
& = \mathbb{E}_\eta[T_i(Y)].
\end{align*}
$$
Similarly it can be shown that 
$$
\frac{\partial}{\partial \eta_i\eta_j}a(\eta) = \mathrm{Cov}[T_i(Y), T_j(Y)].
$$


**Result 3.** The negative log-likelihood of an exponential family distribution is always convex with respect to the natural parameters. This is because the Hessian is positive semi-definite.
$$
\begin{align*}
-\log p_\eta(y) & = a(\eta) -\eta^\top T(y) -\log b(y)\\
\nabla_\eta -\log p_\eta(y) & = \nabla_\eta a(\eta) - T(y)\\
                            & = \mathbb{E}_\eta[T(Y)] - T(y)\\
\nabla^2_\eta -\log p_\eta(y) & = \nabla^2_\eta a(\eta) = \text{Var}_\eta[T(Y)] \succeq 0.
\end{align*}
$$
