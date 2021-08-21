---
title: "Introduction of Offline Reinfoecement Learning"
layout: post
date: 2021-06-21 22:44
image: /assets/images/[Markdowm Image.jpg
headerImage: false
tag:
- Reinforcement Learning
star: false
category: blog
author: joseongmin
description: Definition of Offline Reinforment Learning
---

# What is Offline(Batched) reinforcement learning

**Definition 1.1** Offline reinforcement learning means RL algorithms that utilize previously collected data *D*, without additional online data collection [1]

![offline_frame](https://user-images.githubusercontent.com/76901622/130313887-5e7f1d37-062e-4104-a89d-0fcc71066a34.jpg)






## Why Offline RL

- Compared to supervised learning, Online RL utilizes a feedback loop based on trial and error that requires interaction during learning. [2]

- In many settings, this sort of online interaction is impractical, either because data collection is expensive and dangerous (e.g. autonomous driving, or healthcare)[4]
  
- Furthermore, even in domains where online interaction is feasible, weight still prefer to utilize previously collected data instead (e.g. if the domain is complex and effective generalization requires large datasets.)[5]




## Example Senario in Optimal Exectution Agent

**Limits of making simulator** Order execution can be viewed as interactive sequential decision making problem. However, since the goal of Optimal Execution Algorithm is to interact successfully with real humans (in market), collecting trials requires interacting in market, which may be prohibitively expensive at the scale needed to train effective execution agents. So **the RL Agent needs simulator to train agents**, but it has lots limits. However, offline data collected directly from past execution in real market can replace simulator [6]

**Decision Making in execution** Conventional active reinforcement learning may be prohibitively dangerous in market - even utilizing a fully trained policy to execute. Therefore, offline RL might be the viable path to apply reinforcement learning in such settings. Offline data would then be obtained from past execution or select "actions" from historical data.

**Generalization of execution policy** There is many stocks of diverse prices in market. Therefore, we want to learn policies for a variety of stocks. ( e.g. the agent who orders Samsung Electronics well should also order LG Electronics well.) In that case, each skill by itself might require a very large amount of interaction, as we would need to collect enough data to earn the skill which generalizes effectively to all the situations (e.g. all the different stocks) in which the agent might need to perform it. With offline RL, we could instead imagine including all of the data the agent has ever collected for all of its previously learned skills in the data buffer for each new skill that it learns. In this way, offline RL can effectively utilize multi-task data.



# Challenges of Offline RL

1. Offline RL relies entirely on the static dataset *D*, without exploration : nothing to address this challenge [[1]]

2. Offline RL makes challenge when making and answer Counterfactual queries : to learn a policy that something differently from the pattern of behavior observed in the dataset *D* : forgo the goal of finding the optimal policy, and instead aim to find the best possible policy using the fixed offline dataset [[1]]

3. Recent studies have observed that direct use of RL algorithms originally developed for the online or interactive paradigm leads to poor results in the offline RL setting

      * **Distribution shift issue** : function approximator (policy, value function, model) trained one distribution should be evaluated on a **different distribution** without further interaction



# How to address distribution shift issue currently

## Paradigm in Model free 

  1. Model-free approach is fail in the offline RL setting, due to large extrapolation error when the Q-function is evaluated on **out-of-distribution actions (distribution shift)**, which can lead to unstable learning and divergence [1]
  
  2. Constraining the learned policy to the behavior policy induced by the dataset to overcome this [4]

  3. This method is limited to behaviors within the data manifold and make difficult to generalize [1], [5]


## Paradigm in Model based

   1. Model-based approach is fail in the offline RL setting, due to distribution shift and model-bias [3]

   2. Model error on out-of-distribution states that often drives exploration and corrective feedback in the online setting can be detrimental when interaction is not allowed in offline setting [3], [5]

   3. Using uncertainty quantification to overcome them ( MOReL, MOPO ) [3], [5]



### Refernece
[1] A. Kumar. Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. 2020.

[2] R. Kidambi. MOReL: Model-based Offline Reinforcement Learning. 2020.

[3] T. Yu. MOPO: Model-based Offline Policy Optimization. 2020.

[4] A. Kumar. Conservative Q-Learning for Offline Reinforcement Learning. 2020.

[5] T. Yu. COMBO: Conservative Offline Model-Based Policy Optimization. 2021.

[6] B. Ning. Double Deep Q-Learning for Optimal Execution







[1]: https://arxiv.org/pdf/2005.01643.pdf 

[2]: https://papers.nips.cc/paper/2020/hash/f7efa4f864ae9b88d43527f4b14f750f-Abstract.html 

[3]: https://proceedings.neurips.cc/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html 

[4]: https://proceedings.neurips.cc/paper/2020/file/0d2b2061826a5df3221116a5085a6052-Paper.pdf 

[5]: https://arxiv.org/abs/2102.08363 

[6]: https://arxiv.org/abs/1812.06600 

