---
title: "Model-Based Offline Reinfoecement Learning"
layout: post
date: 2021-07-07 22:44
image: /assets/images/[Markdowm Image.jpg
headerImage: false
use_math: true
tag:
- Reinforcement Learning
star: false
category: blog
author: joseongmin
description: Model Based Offline Reinforment Learning
---

# MOReL: Model-Based Offline Reinforcement Learning (2020) 

## Goal

<img width="553" alt="asdfasdf" src="https://user-images.githubusercontent.com/76901622/130322238-92e35cd3-3075-4a5b-9fb4-448b602cbf11.png">

1. MOReL learns a pessimistic MDP (P-MDP) from the dataset and uses it for policy sea
2. P-MDP partitions the state-action space into known (green) and unknown (orange) regions, and also forces a transition to a low reward
   absorbing state (HALT) from unknown regions. Blue dots denote the support in the dataset.

## Algorithm

<img width="1063" alt="qwer" src="https://user-images.githubusercontent.com/76901622/130322264-122c7b01-7c44-4a27-98a5-8c7ef3af4926.png">

1. **Learning the dynamics model**

The first step involves using the offline dataset to learn an approximate dynamics model $\hat{P}(\cdot \mid s, a)$. Since the offline dataset may not span the entire state space, the learned model may not be globally accurate. Naïve MBRL approach that directly plans with the learned model may overestimate rewards in unfamiliar parts of the state space, resulting in a highly sub-optimal policy. We overcome this with the next step.

2. **Unknown state-action detector (USAD)**

Like hypothesis testing, partition known and unknown regions based on the accuracy of learned model as follows.

<img width="800" alt="zvb" src="https://user-images.githubusercontent.com/76901622/130322302-28586160-c920-4254-9c92-e67de0467f47.png">

- $D_{T V}(\hat{P}(\cdot \mid s, a), P(\cdot \mid s, a))$ denotes the total variation distance between $\hat{P}(\cdot \mid s, a)$ and $P(\cdot \mid s, a)$

- Two factors contribute to USAD’s effectiveness

      - data availability: having sufficient data points “close” to the query
      - quality of representations: certain representations, like those based on physics, can lead to better

  generalization guarantees.

3. **Construct Pessimistic MDP construction**

The $(\alpha, \kappa)$-pessimistic MDP is described by $\hat{\mathcal{M}}_{p}:=\left\{S \cup H A L T, A, r_{p}, \hat{P}_{p}, \hat{\rho}_{0}, \gamma\right\} .$ Here, $S$ and $A$ are states and actions in the MDP $\mathcal{M}$. HALT is an additional absorbing state we introduce into the state space of $\hat{\mathcal{M}}_{p}$. $\hat{\rho}_{0}$ is the initial state distribution learned from the dataset $\mathcal{D} \cdot \gamma$ is the discount factor (same as $\left.\mathcal{M}\right)$. The modified reward and transition dynamics are given by:

<img width="844" alt="sghf" src="https://user-images.githubusercontent.com/76901622/130322351-56542f54-204e-4db8-b076-5ace4f852cb5.png">

$\delta\left(s^{\prime}=\mathrm{HALT}\right)$ is the Dirac delta function, which forces the MDP to transition to the absorbing state HALT. For unknown state-action pairs, use a reward of $-\kappa$, while all known state-actions receive the same reward as in the environment. The P-MDP heavily punishes policies that visit unknown states, thereby providing a safeguard against distribution shift and model exploitation.

4. **Planning**

Perform planning in the P-MDP defined above. For simplicity, we assume a planning oracle that returns an $\epsilon_{\pi}$ -sub-optimal policy in the P-MDP. A number of algorithms based on MPC, search-based planning, dynamic programming, or policy optimization can be used to approximately realize this.

## Benchmark results(Toy Gym)

<img width="671" alt="sfhj" src="https://user-images.githubusercontent.com/76901622/130322386-a153f428-bebd-4388-9857-e8e7725b6007.png">

## Conclusion

1. **Importance of pessimistic MDP**

- Compare MOReL with a naive MBRL approach that first learns a dynamics model using the offline data without any safeguards against model inaccuracy.

<img width="832" alt="sgnf" src="https://user-images.githubusercontent.com/76901622/130322407-2ca9f1e1-fd67-4bdf-a1fe-73a1a0983ef3.png">

- The naive MBRL approach already works well, achieving results comparable to prior algorithms like BCQ and BEAR. However, MOReL clearly exhibits more stable and monotonic learning progress.

- Furthermore, in the case of naive MBRL, we observe that performance can quickly degrade after a few hundred steps of policy improvement

2. **Transfer from pessimistic MDP to environment**

- MOReL suggest that the value of a policy in the P-MDP cannot substantially exceed the value in the
  environment.

- This makes in the P-MDP an approximate lower bound on the true performance, and a good surrogate of True MDP for optimization.

- Author observe that the value in the true environment closely correlates with the value in P-MDP. In par- ticular, the P-MDP value never substantially exceeds the true performance, suggesting that the pessimism helps to avoid model exploitation.

- But MOReL constructs terminating states based on a hard threshold on uncertainty.

# MOPO: Model-based Offline Policy Optimization (2020) 

## Goal

1. Design an offline model-based reinforcement learning algorithm that can take actions that are not strictly within the support of the behavioral distribution.

2. Balance the return and risk since models will become increasingly inaccurate further from the behavioral distri- bution (vanilla model-based policy optimization)

- the potential gain in performance by escaping the behavioral distribution and finding a better policy
- the risk of overfitting to the errors of the dynamics at regions far away from the behavioral distribution.

3. To achieve the optimal balance, bound the return from below by the return of a constructed model MDP penal- ized by the uncertainty of the dynamics and maximize conservative estimation of the return by an off-the-shelf reinforcement learning algorithm (MOPO)

## Preliminaries

- $T\left(s^{\prime} \mid s, a\right)=$ the transition dynamics (True dynamics)
- $\eta_{M}(\pi):=\underset{\pi, T, \mu_{0}}{\mathbb{E}}\left[\sum_{t=0}^{\infty} \gamma^{t} r\left(s_{t}, a_{t}\right)\right] .=$ expected discounted return (goal is maximizing this)
- $\eta_{\widehat{M}}(\pi)=$ A natural estimator for the true return $\eta_{M}(\pi)$
- Behavioral distribution $=$ sampled from distribution $\mathcal{D}_{\text {env }}$
- $\widehat{T}$ defines a model $\operatorname{MDP} \widehat{M}=\left(\mathcal{S}, \mathcal{A}, \widehat{T}, r, \mu_{0}, \gamma\right)$
- $\mathbb{P}_{\widehat{T}, t}^{\pi}(s)=$ the probability of being in state $s$ at time step $t$ if actions are sampled according to $\pi$ and transitions according to $\widehat{T}$
- $\rho_{\widehat{T}}^{\pi}(s, a)=$ the discounted occupancy measure of policy $\pi$ under dynamics $\widehat{T}: \rho_{\widehat{T}}^{\pi}(s, a):=\pi(a \mid s) \sum_{t=0}^{\infty} \gamma^{t} \mathbb{P}_{\widehat{T}, t}^{\pi}(s)$

## Algorithm

1. **Quantifying the uncertainty: from the dynamics to the total return**

Let $G_{\widehat{M}}^{\pi}(s, a):=\underset{s^{\prime} \sim \widehat{T}(s, a)}{\mathbb{E}}\left[V_{M}^{\pi}\left(s^{\prime}\right)\right]-\underset{s^{\prime} \sim T(s, a)}{\mathbb{E}}\left[V_{M}^{\pi}\left(s^{\prime}\right)\right]$, we can get

<img width="723" alt="shfg" src="https://user-images.githubusercontent.com/76901622/130322605-aaba76d4-1774-4d9f-a84c-f5f7f00a5698.png">

By definition, $G \frac{\pi}{\widehat{M}}(s, a)$ measures the difference between $M$ and $\widehat{M}$ under the test function $V^{\pi}$. By equation(1), it governs the differences between the performances of $\pi$ in the two MDPs. If we could estimate $G_{\widehat{M}}^{\pi}(s, a)$ or bound it from above, then we could use the RHS of (1) as an upper bound for the estimation error of $\eta_{M}(\pi)$.

Moreover, equation (2) suggests that a policy that obtains high reward in the estimated MDP while also minimizing $G_{\widehat{M}}^{\pi}(s, a)$ will obtain high reward in the real MDP.

However, computing $G_{\widehat{M}}^{\pi}(s, a)$ remains elusive because it depends on the unknown function $V_{M}^{\pi} .$ Leveraging properties of $V_{M}^{\pi}$, we will replace $G_{\widehat{M}}^{\pi}(s, a)$ by an upper bound that depends solely on the error of the dynamics $\widehat{T}$

Given an admissible error estimator, we define the uncertainty-penalized reward $\tilde{r}(s, a):=r(s, a)-\lambda u(s, a)$ where $\lambda:=\gamma c$, and the uncertainty-penalized MDP $\widetilde{M}=\left(\mathcal{S}, \mathcal{A}, \widehat{T}, \tilde{r}, \mu_{0}, \gamma\right)$. We observe that $\widetilde{M}$ is conservative in that the return under it bounds from below the true return:

<img width="946" alt="dhgj" src="https://user-images.githubusercontent.com/76901622/130322676-f1130a3c-c38c-40a6-afe5-4b62b0cb4cae.png">

2. **Policy optimization on uncertainty-penalized MDPs**

Optimize the policy on the uncertainty-penalized MDP $\widetilde{M}$ in Algorithm 1 .

<img width="765" alt="agdh" src="https://user-images.githubusercontent.com/76901622/130322701-711b3644-6e9c-40fc-badf-0cdaef20703c.png">

## Benchmark results (Toy Gym)

<img width="895" alt="rsyj" src="https://user-images.githubusercontent.com/76901622/130322736-0f8caf60-df92-4b4c-802b-9c4ae481b8bd.png">

- Unlike MOReL, which constructs terminating states based on a hard threshold on uncertainty, MOPO uses a soft reward penalty to incorporate uncertainty. ( potential benefit of a soft penalty is that the policy is allowed to take a few risky actions and then return to the confident area near the behavioral distribution without being terminated. )

- Using uncertainty penalized MDP, which is constructed where the reward is given by $\widetilde{r}(\mathbf{s}, \mathbf{a})=\hat{r}(\mathbf{s}, \mathbf{a})-$ $\lambda u(\mathbf{s}, \mathbf{a})$ and the learned dynamics model, MOPO learns a policy in this "uncertainty-penalized" MDP $\widetilde{\mathcal{M}}=$ $\left(\mathcal{S}, \mathcal{A}, \widehat{T}, \widetilde{r}, \mu_{0}, \gamma\right)$ which has the property that $J(\widetilde{\mathcal{M}}, \pi) \leq J(\mathcal{M}, \pi) \forall \pi .$ By constructing and optimizing such a lower bound, offline model-based RL algorithms (MOPO and MOReL) avoid the aforementioned pitfalls like model-bias and distribution shift.

- However, strong reliance on uncertainty quantification is challenging for complex datasets or deep neural network models (MOReL and MOPO)

### Refernece

[1] R. Kidambi. MOReL: Model-based Offline Reinforcement Learning. 2020.

[2] T. Yu. MOPO: Model-based Offline Policy Optimization. 2020.
