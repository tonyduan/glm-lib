### Generalized Linear Models in Python

Last update: January 2020.

---

A generalized linear model fits an exponential family distribution with a linear model. The resulting optimization problem is convex.

**Exponential Family Distributions**

We assume the response is generated from a distribution parameterized by <img alt="$\eta$" src="svgs/1d0496971a2775f4887d1df25cea4f7e.svg" align="middle" width="8.751954749999989pt" height="14.15524440000002pt"/>,
<p align="center"><img alt="$$&#10;p_\eta(y) = b(y)\exp(\eta^\top T(y) - a(\eta)).&#10;$$" src="svgs/3cfdd6ff06cb4ac341316ede046b486e.svg" align="middle" width="236.8150059pt" height="19.4813124pt"/></p>


Here <img alt="$\eta$" src="svgs/1d0496971a2775f4887d1df25cea4f7e.svg" align="middle" width="8.751954749999989pt" height="14.15524440000002pt"/> is the natural parameter, <img alt="$T(y)$" src="svgs/3c325d194ef326ae90b038090df1a962.svg" align="middle" width="33.32395109999999pt" height="24.65753399999998pt"/> is the sufficient statistic, and <img alt="$a(\eta)$" src="svgs/a7fa43c120ea21d26f73c627e514119d.svg" align="middle" width="30.22652489999999pt" height="24.65753399999998pt"/> is the log partition function. Using a linear model with a canonical link function, we suppose 
<p align="center"><img alt="$$&#10;\eta = \theta x \implies \nabla_\theta \eta_i = x,\ \nabla^2_\theta \eta_i = xx^\top.&#10;$$" src="svgs/80145fbf2d16523fb5abf06d8e37730e.svg" align="middle" width="262.99594035pt" height="18.84197535pt"/></p>


Note that here <img alt="$\eta \in\mathbb{R}^p, \theta\in\mathbb{R}^{p,n},x\in\mathbb{R}^n$" src="svgs/05edb6d2c6f090903e0bd98a5a2d37cd.svg" align="middle" width="172.17507285pt" height="22.831056599999986pt"/>. 

**Fisher Scoring**

The model is fit using maximum likelihood. We take a natural gradient step using the Fisher information in <img alt="$\theta$" src="svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg" align="middle" width="8.17352744999999pt" height="22.831056599999986pt"/>,

<p align="center"><img alt="$$&#10;\theta \leftarrow \theta - \mathcal{I}_\theta^{-1}\nabla_\theta \left(-\log p_\theta(y)\right).&#10;$$" src="svgs/f7caafc068c590813973f4238b88db75.svg" align="middle" width="210.42856395pt" height="19.40624895pt"/></p>


Notice that by the chain rule, we have the following score and Hessian.
<p align="center"><img alt="$$&#10;\begin{align*}&#10;\nabla_\theta -\log p_\theta(y) &amp; = \nabla_\eta-\log p_\eta(y) x^\top\\&#10;&amp; =(\mathbb{E}_\eta[T(Y)] - T(y))x^\top \\&#10;\nabla^2_{\theta_i}-\log p_\theta(y) &amp; = \nabla_{\eta_i}^2 -\log p_\eta(y) xx^\top\\&#10;&amp; = \mathrm{Var}_\eta[T_i(Y)]xx^\top&#10;\end{align*}&#10;$$" src="svgs/781e3ee1b5ae66b53628d618a1990828.svg" align="middle" width="280.95289035pt" height="101.67868589999999pt"/></p>


The Fisher information matrix with respect to the <img alt="$i$" src="svgs/77a3b857d53fb44e33b53e4c8b68351a.svg" align="middle" width="5.663225699999989pt" height="21.68300969999999pt"/>-th row of <img alt="$\theta$" src="svgs/27e556cf3caa0673ac49a8f0de3c73ca.svg" align="middle" width="8.17352744999999pt" height="22.831056599999986pt"/> is then the expected value of a constant, so
<p align="center"><img alt="$$&#10;\mathbb{E}_\eta[\nabla^2_{\theta_i} -\log p_\theta(y)] = \mathrm{Var}_\eta[T_i(Y)]x x^\top&#10;$$" src="svgs/5683629d35e2c617e09f9f6713fd7656.svg" align="middle" width="270.09361995pt" height="20.485810949999998pt"/></p>
This coincides with a Newton-Raphson step, since we are using the canonical link function.



**Link Functions**

A link function relates the expected value of the response to the natural parameters,
<p align="center"><img alt="$$&#10;\mathbb{E}_{\eta}[y] = g^{-1}(\eta).&#10;$$" src="svgs/f0b779510648cfd06ee28a1c9a3b9d94.svg" align="middle" width="110.8483134pt" height="18.905967299999997pt"/></p>

In our case we only ever use the canonical link function for each distribution.

#### References

[1] Nelder, J.A., and Wedderburn, R.W.M. (1972). Generalized Linear Models. Journal of the Royal Statistical Society. Series A (General) *135*, 370–384.

---

#### Appendix

Here we list useful results of exponential families.

**Result 1.** <img alt="$T(y)$" src="svgs/3c325d194ef326ae90b038090df1a962.svg" align="middle" width="33.32395109999999pt" height="24.65753399999998pt"/> are sufficient statistics. 
<p align="center"><img alt="$$&#10;p_\eta(y_1,\cdots,y_n) = b(y_1)\cdots b(y_n)\exp\left(\eta^\top \sum_{i=1}^n T(y_i)-na(\eta)\right)&#10;$$" src="svgs/f36b1803f7b7c4779ee1db4fc4d3206d.svg" align="middle" width="423.55476734999996pt" height="49.315569599999996pt"/></p>


**Result 2.** The gradients of the log partition function always yield moments of the sufficient statistics. First recall that
<p align="center"><img alt="$$&#10;a(\eta) = \log \int_y b(y)\exp \left(\eta^\top T(y)\right)dy.&#10;$$" src="svgs/d8a37a7864e45ba5e4a8ca9d466265d7.svg" align="middle" width="247.72468379999998pt" height="39.58940535pt"/></p>
Now observe that
<p align="center"><img alt="$$&#10;\begin{align*}&#10;\frac{\partial}{\partial \eta_i} a(\eta) &amp; = \frac{\int_y b(y)\exp \left(\eta^\top T(y)\right)T_i(y)dy}{\int_y b(y)\exp \left(\eta^\top T(y)\right)dy}\\&#10;&amp; = \int_y b(y)\exp(\eta^\top T(y) - a(\eta))T_i(y)dy\\&#10;&amp; = \mathbb{E}_\eta[T_i(Y)].&#10;\end{align*}&#10;$$" src="svgs/f7cb6da610b446d81ca41bb34c3dc9bd.svg" align="middle" width="323.502729pt" height="119.23320585pt"/></p>
Similarly it can be shown that 
<p align="center"><img alt="$$&#10;\frac{\partial}{\partial \eta_i\eta_j}a(\eta) = \mathrm{Cov}[T_i(Y), T_j(Y)].&#10;$$" src="svgs/2a1b5ef99c7e9147ce92b02161baf56b.svg" align="middle" width="225.59923155pt" height="38.5152603pt"/></p>


**Result 3.** The negative log-likelihood of an exponential family distribution is always convex with respect to the natural parameters. This is because the Hessian is positive semi-definite.
<p align="center"><img alt="$$&#10;\begin{align*}&#10;-\log p_\eta(y) &amp; = a(\eta) -\eta^\top T(y) -\log b(y)\\&#10;\nabla_\eta -\log p_\eta(y) &amp; = \nabla_\eta a(\eta) - T(y)\\&#10;                            &amp; = \mathbb{E}_\eta[T(Y)] - T(y)\\&#10;\nabla^2_\eta -\log p_\eta(y) &amp; = \nabla^2_\eta a(\eta) = \text{Var}_\eta[T(Y)] \succeq 0.&#10;\end{align*}&#10;$$" src="svgs/9bc1d1d8b870ae2dc879be425180f30b.svg" align="middle" width="314.5818885pt" height="97.09048965pt"/></p>
