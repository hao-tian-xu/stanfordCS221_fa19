

# **<u>hw1. foundations</u>**

### Problem 1: Optimization and probability

a. minimize $f(\theta)=\frac{1}{2} \sum_{i=1}^{n} w_{i}\left(\theta-x_{i}\right)^{2}$

$\to\quad f'(\theta) = {1\over 2} \sum_{i=1}^n 2w_i(\theta-x_i) = 0$

$\to\quad \theta = {\sum_{i=1}^n w_ix_i\over \sum_{i=1}^n w_i}$

b. $f(x) \geq g(x) \quad\forall x$

## *Thinking*

- Problem 3 focuses on some auxiliary functions related to machine learning, and also practices dynamic programming



- 问题3主要是一些和机器学习相关的辅助函数，也练习了动态规划 dynamic programming

# **<u>hw2. sentiment</u>**

### Problem 3: Sentiment Classification

- d. context matters
- f. n = 6: 0.2706
  - use multiple n together: more contexual information and word information

## *Thinking*

- Problem 3 focuses on feature extraction, and stochastic gradient descent for linear equations
- Problem 4 focuses on the k-means algorithm in unsupervised learning
  
  
- 问题3主要是特征提取，和线性方程的随机梯度下降 stochastic gradient descent SGD
- 问题4主要是无监督学习中的k均值算法

# **<u>hw3. reconstruct</u>**

## *Thinking*

- focuses on  the definition of the search problem using UCS
  - Definition of state space: the current state can determine the future optimal cost
    - So when using bigramCost, the current state needs to contain the previous word



- 主要练习了使用ucs的搜索问题的定义
  - 状态空间的定义：当前状态能够决定未来的最优成本
    - 因此使用bigramCost时，当前状态需要包含前一个单词

# <u>**hw4. blackjack**</u>

## *Thinking*

- 3a practiced the definition of the MDP problem (i.e., actions correspond to random outcomes)
- 4a practiced the code implementation of Q-Learning
- 4b compared the performance of Value Iteration and Q-Learning in different scale state spaces
  - actually examined the performance of Q-Learning in large scale state spaces, since the former requires known probabilities of rewards and action outcomes
  - Q-Learning is more difficult to explore in large scale spaces
- 4c practiced Function Approximation, where the key is feature extraction (again related to the extraction of feature vectors in machine learning)



- 3a练习了MDP问题的定义（即行动对应随机结果）
- 4a练习了Q-Learning的代码实现
- 4b对比了Value Iteration和Q-Learning在不同尺度状态空间中的表现
  - 实际是看Q-Learning在大尺度状态空间中的表现，因为前者需要已知的reward和行动结果的概率
  - Q-Learning在大尺度空间中的探索较为困难
- 4c练习了Function Approximation，关键是特征提取（再次联系了机器学习中的特征向量的提取）



# **<u>hw5. pacman</u>**

## Problem 1: Minimax

### a.

minimax:

$V_{\operatorname{minmax}}(s)= \begin{cases}\operatorname{Utility}(s) & \operatorname{IsEnd}(s) \\ \max _{a \in \operatorname{Actions}(s)} V_{\operatorname{minmax}}(\operatorname{Succ}(s, a)) & \operatorname{Player}(s)=\operatorname{agent} \\ \min _{a \in \operatorname{Actions}(s)} V_{\operatorname{minmax}}(\operatorname{Succ}(s, a)) & \operatorname{Player}(s)=\mathrm{opp}\end{cases}$

minimax with evaluation function:

$V_{\text {minmax }}(s, d)= \begin{cases}\text {Utility }(s) & \operatorname{lsEnd}(s) \\ \operatorname{Eval}(s) & d=0 \\ \max _{a \in \operatorname{Actions}(s)} V_{\operatorname{minmax}}(\operatorname{Succ}(s, a), d) & \operatorname{Player}(s)=\operatorname{agent} \\ \min _{a \in \operatorname{Actinns}(s)} V_{\operatorname{minmax}}(\operatorname{Succ}(s, a), d-1) & \operatorname{Player}(s)=\mathrm{opp}\end{cases}$

muti-agent minimax with evaluation function

$V_{\text {minmax }}(s, d)= \begin{cases}\text {Utility }(s) & \operatorname{lsEnd}(s) \\ \operatorname{Eval}(s) & d=0 \\ \max _{a \in \operatorname{Actions}(s)} V_{\operatorname{minmax}}(\operatorname{Succ}(s, a), d) & \operatorname{Player}(s)=0 \\ \min _{a \in \operatorname{Actinns}(s)} V_{\operatorname{minmax}}(\operatorname{Succ}(s, a), d) & 0 \lt \operatorname{Player}(s) \lt n \\ \min _{a \in \operatorname{Actinns}(s)} V_{\operatorname{minmax}}(\operatorname{Succ}(s, a), d-1) & \operatorname{Player}(s) = n\end{cases}$



## Problem 3: Expectimax

### a.

expectimax:

$V_{\text {exptmax }}(s)= \begin{cases}\text { Utility }(s) & \text { IsEnd }(s) \\ \max _{a \in \operatorname{Actions}(s)} V_{\text {exptmax }}(\operatorname{Succ}(s, a)) & \text { Player }(s)=\text { agent } \\ \sum_{a \in \text { Actions }(s)} \pi_{\text {opp }}(s, a) V_{\text {exptmax }}(\operatorname{Succ}(s, a)) & \text { Player }(s)=\text { opp }\end{cases}$

muti-agent expectimax with evaluation function:

$V_{\text {minmax }}(s, d)= \begin{cases}
\text {Utility }(s) & \operatorname{lsEnd}(s) \\
\operatorname{Eval}(s) & d=0 \\
\max _{a \in \operatorname{Actions}(s)} V_{\operatorname{exptmax}}(\operatorname{Succ}(s, a), d) & \operatorname{Player}(s)=0 \\
\sum_{a \in \text {Actions}(s)} \pi_{\text {opp}}(s, a) V_{\text {exptmax}}(\operatorname{Succ}(s, a), d) & 0 \lt \operatorname{Player}(s) \lt n \\
\sum_{a \in \text {Actions}(s)} \pi_{\text {opp}}(s, a) V_{\text {exptmax}}(\operatorname{Succ}(s, a), d-1) & \operatorname{Player}(s) = n
\end{cases}$



## *Thinking*

- Game learning: essentially it is still a search algorithm, mainly with the addition of different strategies for different layers (the number is the total number of players)
- The idea of using proven algorithms to solve problems is the key



- 游戏学习：本质上来说还是搜索算法，主要是加入了不同层的不同策略（数量为玩家总数）
- 使用成熟算法解决问题的思路是关键学习到的内容



# **<u>hw6. scheduling</u>**

- In Problem 1, you will implement two of the three <u>heuristics</u> you learned from the lectures that will make CSP solving much faster. 
- In problem 2, you will add a helper function to <u>reduce n-ary factors to unary and binary factors</u>. 
- Lastly, in Problem 3, you will create <u>the course scheduling CSP</u> and solve it using the code from previous parts.

## *Thinking*

- Practiced problem definition and algorithm implementation related to the constraint satisfaction problem CSP
- 3b: Whether to classify by elements of the algorithm or by elements of the problem greatly affects the brevity and clarity of the code
  - As in 3b, classifying the code according to the elements of the problem makes the code short and clear



- 练习了约束满足问题CSP相关的问题定义和算法实现
  - 3b：按照算法的要素进行分类还是按照问题的要素进行分类会很大程度影响代码的简洁和清晰度
    - 如3b中按照问题的要素进行分类使得代码变得短小清晰



# **<u>hw7. car</u>**

- Problem 1 (written) will give you some practice with probabilistic inference on a simple Bayesian network.
- In Problems 2 and 3 (code), you will implement `ExactInference`, which computes a full probability distribution of another car's location over tiles `(row, col)`.
- In Problem 4 (code), you will implement `ParticleFilter`, which works with particle-based representation of this same distribution.
- Problem 5 (written) gives you a chance to extend your probability analyses to a slightly more realistic scenario where there are multiple other cars and we can't automatically distinguish between them.

## *Thinking*

- Practiced the algorithm implementation of Bayesian networks to obtain the posterior conditional probabilities by Bayesian inference
- Practiced the implementation of particle filtering algorithm to further reduce the computational complexity
  - proposal: selection based on observations
  - reweight, resample



- 练习了贝叶斯网络中，通过贝叶斯推断得到后验条件概率的算法实现
- 练习了粒子滤波的算法实现，从而进一步减少计算复杂度
  - proposal：根据观测选择
  - reweight，resample



# **<u>hw8. logic</u>**

## *Thinking*

- The main exercise is to describe problems using formal logic, which can be cross-referenced with the part of CNF in 6.009
- 主要练习了使用形式逻辑描述问题，可以和6.009中CNF的部分互相参考

















