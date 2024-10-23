corpus = ["""
# DP-HyPO: An Adaptive Private Hyperparameter Optimization Framework 

Hua Wang<br>Department of Statistics and Data Science<br>University of Pennsylvania<br>Philadelphia, PA 19104<br>wanghua@wharton.upenn.edu

Sheng Gao<br>Department of Statistics and Data Science<br>University of Pennsylvania<br>Philadelphia, PA 19104<br>shenggao@wharton.upenn.edu

Huanyu Zhang<br>Meta Platforms, Inc.<br>New York, NY 10003<br>huanyuzhang@meta.com

Weijie J. Su

Department of Statistics and Data Science

University of Pennsylvania

Philadelphia, PA 19104

suw@wharton.upenn.edu

Milan Shen<br>Meta Platforms, Inc.<br>Menlo Park, CA 94025<br>milanshen@gmail.com


#### Abstract

Hyperparameter optimization, also known as hyperparameter tuning, is a widely recognized technique for improving model performance. Regrettably, when training private ML models, many practitioners often overlook the privacy risks associated with hyperparameter optimization, which could potentially expose sensitive information about the underlying dataset. Currently, the sole existing approach to allow privacy-preserving hyperparameter optimization is to uniformly and randomly select hyperparameters for a number of runs, subsequently reporting the best-performing hyperparameter. In contrast, in non-private settings, practitioners commonly utilize "adaptive" hyperparameter optimization methods such as Gaussian process-based optimization, which select the next candidate based on information gathered from previous outputs. This substantial contrast between private and non-private hyperparameter optimization underscores a critical concern. In our paper, we introduce DP-HyPO, a pioneering framework for "adaptive" private hyperparameter optimization, aiming to bridge the gap between private and non-private hyperparameter optimization. To accomplish this, we provide a comprehensive differential privacy analysis of our framework. Furthermore, we empirically demonstrate the effectiveness of DP-HyPO on a diverse set of real-world datasets.


## 1 Introduction

In recent decades, modern deep learning has demonstrated remarkable advancements in various applications. Nonetheless, numerous training tasks involve the utilization of sensitive information pertaining to individuals, giving rise to substantial concerns regarding privacy [29, 6]. To address these concerns, the concept of differential privacy (DP) was introduced by [12, 13]. DP provides a mathematically rigorous framework for quantifying privacy leakage, and it has gained widespread
acceptance as the most reliable approach for formally evaluating the privacy guarantees of machine learning algorithms.

When training deep learning models, the most popular method to ensure privacy is noisy (stochastic) gradient descent (DP-SGD) [4,35]. DP-SGD typically resembles non-private gradient-based methods; however, it incorporates gradient clipping and noise injection. More specifically, each individual gradient is clipped to ensure a bounded $/ell_{2}$ norm. Gaussian noise is then added to the average gradient which is utilized to update the model parameters. These adjustments guarantee a bounded sensitivity of each update, thereby enforcing DP through the introduction of additional noise.

In both non-private and private settings, hyperparameter optimization (HPO) plays a crucial role in achieving optimal model performance. Commonly used methods for HPO include grid search (GS), random search (RS), and Bayesian optimization (BO). GS and RS approaches are typically non-adaptive, as they select the best hyperparameter from a predetermined or randomly selected set. While these methods are straightforward to implement, they can be computationally expensive and inefficient when dealing with large search spaces. As the dimensionality of hyperparameters increases, the number of potential trials may grow exponentially. To address this challenge, adaptive HPO methods such as Bayesian optimization have been introduced [34, 14, 40]. BO leverages a probabilistic model that maps hyperparameters to objective metrics, striking a balance between exploration and exploitation. BO quickly emerged as the default method for complex HPO tasks, offering improved efficiency and effectiveness compared to non-adaptive methods.

While HPO is a well-studied problem, the integration of a DP constraint into HPO has received little attention. Previous works on DP machine learning often neglect to account for the privacy cost associated with HPO [1, 39, 42, 42]. These works either assume that the best parameters are known in advance or rely on a supplementary public dataset that closely resembles the private dataset distribution, which is not feasible in most real-world scenarios.

Only recently have researchers turned to the important concept of honest HPO [28], where the privacy cost during HPO cannot be overlooked. Private HPO poses greater challenges compared to the non-private case for two primary reasons. First, learning with DP-SGD introduces additional hyperparameters (e.g., clipping norm, the noise scale, and stopping time), which hugely adds complexity to the search for optimal hyperparameters. Second, DP-SGD is more sensitive to the selection of hyperparameter combinations, with its performance largely influenced by this choice [28, 10, 31].

To tackle this challenging question, previous studies such as [24, 32] propose running the base algorithm with different hyperparameters a random number of times. They demonstrate that this approach significantly benefits privacy accounting, contrary to the traditional scaling of privacy guarantees with the square root of the number of runs (based on the composition properties from [19]). While these papers make valuable contributions, their approaches only allow for uniformly random subsampling from a finite and pre-fixed set of candidate hyperparameters at each run. As a result, any advanced technique from HPO literature that requires adaptivity is either prohibited or necessitates a considerable privacy cost (polynomially dependent on the number of runs), creating a substantial gap between non-private and private HPO methods.

Given these considerations, a natural question arises: Can private hyperparameter optimization be adaptive, without a huge privacy cost? In this paper, we provide an affirmative answer to this question.

### 1.1 Our Contributions

- We introduce the pioneering adaptive private hyperparameter optimization framework, DP-HyPO, which enables practitioners to adapt to previous runs and focus on potentially superior hyperparameters. DP-HyPO permits the flexible use of non-DP adaptive hyperparameter optimization methods, such as Gaussian process, for enhanced efficiency, while avoiding the substantial privacy costs due to composition. In contrast to the nonadaptive methods presented in [32, 24], our adaptive framework, DP-HyPO, effectively bridges the gap between private and non-private hyperparameter optimization. Importantly, our framework not only encompasses the aforementioned non-adaptive methods as special cases, but also seamlessly integrates virtually all conceivable adaptive methods into the framework.
- We provide sharp DP guarantees for the adaptive private hyperparameter optimization. Specifically, when the training procedure is executed multiple times, with each iteration being DP on its own, outputting the best repetition is DP ensured by the composition property. However, applying composition results in excessively loose privacy guarantees. Prior work in [24,32] presents bounds that are either independent of the number of repetitions or depend logarithmically on it. Nevertheless, these results require that the hyperparameter selection for each iteration follows a uniform sampling distribution. In contrast, DP-HyPO allows arbitrary adaptive sampling distributions based on previous runs. Utilizing the Rényi DP framework, we offer a strict generalization of those uniform results by providing an accurate characterization of the Rényi divergence between the adaptive sampling distributions of neighboring datasets, without any stability assumptions.
- Empirically, we observe that the Gaussian process-based DP-HyPO algorithm outperforms its uniform counterpart across several practical scenarios. Generally, practitioners can integrate any non-private adaptive HPO methods into the DP-HyPO framework, opening up a vast range of adaptive private HPO algorithm possibilities. Furthermore, DP-HyPO grants practitioners the flexibility to determine the privacy budget allocation for adaptivity, empowering them to balance between the adaptivity and privacy loss when confronting various hyperparameter optimization challenges.


## 2 Preliminaries

### 2.1 Differential Privacy and Hyperparameter Optimization

Differential Privacy is a mathematically rigorous framework for quantifying privacy leakage. A DP algorithm promises that an adversary with perfect information about the entire private dataset in use except for a single individual - would find it hard to distinguish between its presence or absence based on the output of the algorithm [12]. Formally, for $/varepsilon>0$, and $0 /leq /delta<1$, we consider a (randomized) algorithm $M: /mathcal{Z}^{n} /rightarrow /mathcal{Y}$ that takes as input a dataset.

Definition 2.1 (Differential privacy). A randomized algorithm $M$ is $(/varepsilon, /delta)$-DP if for any neighboring dataset $D, D^{/prime} /in /mathcal{Z}^{n}$ differing by an arbitrary sample, and for any event $E$, we have

$$
/mathbb{P}[M(D) /in E] /leqslant /mathrm{e}^{/varepsilon} /cdot /mathbb{P}/left[M/left(D^{/prime}/right) /in E/right]+/delta
$$

Here, $/varepsilon$ and $/delta$ are privacy parameters that characterize the privacy guarantee of algorithm $M$. One of the fundamental properties of DP is composition. When multiple DP algorithms are sequentially composed, the resulting algorithm remains private. The total privacy cost of the composition scales approximately with the square root of the number of compositions [19].

We now formalize the problem of hyperparameter optimization with DP guarantees, which builds upon the finite-candidate framework presented in [24, 32]. Specifically, we consider a set of base DP algorithms $M_{/lambda}: /mathcal{Z}^{n} /rightarrow /mathcal{Y}$, where $/lambda /in /Lambda$ represents a set of hyperparameters of interest, $/mathcal{Z}^{n}$ is the domain of datasets, and $/mathcal{Y}$ denotes the range of the algorithms. This set $/Lambda$ may be any infinite set, e.g., the cross product of the learning rate $/eta$ and clipping norm $R$ in DP-SGD. We require that the set $/Lambda$ is a measure space with an associated measure $/mu$. Common choices for $/mu$ include the counting measure or Lebesgue measure. We make a mild assumption that $/mu(/Lambda)</infty$.

Based on the previous research [32], we make two simplifying assumptions. First, we assume that there is a total ordering on the range $/mathcal{Y}$, which allows us to compare two selected models based on their "performance measure", denoted by $q$. Second, we assume that, for hyperparameter optimization purposes, we output the trained model, the hyperparameter, and the performance measure. Specifically, for any input dataset $D$ and hyperparameter $/lambda$, the return value of $M_{/lambda}$ is $(x, q) /sim M_{/lambda}(D)$, where $x$ represents the combination of the model parameters and the hyperparameter $/lambda$, and $q$ is the (noisy) performance measure of the model.

### 2.2 Related Work

In this section, we focus on related work concerning private HPO, while deferring the discussion on non-private HPO to Appendix F

Historically, research in DP machine learning has neglected the privacy cost associated with HPO [1, 39, 42]. It is only recently that researchers have begun to consider the honest HPO setting [28], in which the cost is taken into account.

A direct approach to addressing this issue involves composition-based analysis. If each training run of a hyperparameter satisfies DP, the entire HPO procedure also complies with DP through composition across all attempted hyperparameter values. However, the challenge with this method is that the privacy guarantee derived from accounting can be excessively loose, scaling polynomially with the number of runs.

Chaudhuri et al. [7] were the first to enhance the DP bounds for HPO by introducing additional stability assumptions on the learning algorithms. [24] made significant progress in enhancing DP bounds for HPO without relying on any stability properties of the learning algorithms. They proposed a simple procedure where a hyperparameter was randomly selected from a uniform distribution for each training run. This selection process was repeated a random number of times according to a geometric distribution, and the best model obtained from these runs was outputted. They showed that this procedure satisfied $(3 /varepsilon, 0)$-DP as long as each training run of a hyperparameter was $(/varepsilon, 0)$-DP Building upon this, [32] extended the procedure to accommodate negative binomial or Poisson distributions for the repeated uniform selection. They also offered more precise Rényi DP guarantees for this extended procedure. Furthermore, [8] explored a generalization of the procedure for top- $k$ selection, considering $(/varepsilon, /delta)$-DP guarantees.

In a related context, [28] explored a setting that appeared superficially similar to ours, as their title mentioned "adaptivity." However, their primary focus was on improving adaptive optimizers such as DP-Adam, which aimed to reduce the necessity of hyperparameter tuning, rather than the adaptive HPO discussed in this paper. Notably, in terms of privacy accounting, their approach only involved composing the privacy cost of each run without proposing any new method.

Another relevant area of research is DP selection, which encompasses well-known methods such as the exponential mechanism [25] and the sparse vector technique [13], along with subsequent studies. However, this line of research always assumes the existence of a low-sensitivity score function for each candidate, which is an unrealistic assumption for hyperparameter optimization.

## 3 DP-HyPO: General Framework for Private Hyperparameter Optimization

The obvious approach to the problem of differentially private hyperparameter optimization would be to run each base algorithm and simply return the best one. However, running such an algorithm on large hyperparameter space is not feasible due to the privacy cost growing linearly in the worst case.

While [24, 32] have successfully reduced the privacy cost for hyperparameter optimization from linear to constant, there are still two major drawbacks. First, none of the previous methods considers the case when the potential number of hyperparameter candidates is infinite, which is common in most hyperparameter optimization scenarios. In fact, we typically start with a range of hyperparameters that we are interested in, rather than a discrete set of candidates. Furthermore, prior methods are limited to the uniform sampling scheme over the hyperparameter domain $/Lambda$. In practice, this setting is unrealistic since we want to "adapt" the selection based on previous results. For instance, one could use Gaussian process to adaptively choose the next hyperparameter for evaluation, based on all the previous outputs. However, no adaptive hyperparameter optimization method has been proposed or analyzed under the DP constraint. In this paper, we bridge this gap by introducing the first DP adaptive hyperparameter optimization framework.

### 3.1 DP-HyPO Framework

To achieve adaptive hyperparameter optimization with differential privacy, we propose the DP-HyPO framework. Our approach keeps an adaptive sampling distribution $/pi$ at each iteration that reflects accumulated information.

Let $Q(D, /pi)$ be the procedure that randomly draws a hyperparameter $/lambda$ from the distribution ${ }^{1}$ $/pi /in /mathcal{D}(/Lambda)$, and then returns the output from $M_{/lambda}(D)$. We allow the sampling distribution to depend on both the dataset and previous outputs, and we denote as $/pi^{(j)}$ the sampling distribution at the $j$-th

/footnotetext{
${ }^{1}$ Here, $/mathcal{D}(/Lambda)$ represents the space of probability densities on $/Lambda$.
iteration on dataset $D$. Similarly, the sampling distribution at the $j$-th iteration on the neighborhood dataset $D^{/prime}$ is denoted as $/pi^{/prime(j)}$.

We now present the DP-HyPO framework, denoted as $/mathcal{A}/left(D, /pi^{(0)}, /mathcal{T}, C, c/right)$, in Framework 1 The algorithm takes a prior distribution $/pi^{(0)} /in /mathcal{D}(/Lambda)$ as input, which reflects arbitrary prior knowledge about the hyperparameter space. Another input is the distribution $/mathcal{T}$ of the total repetitions of training runs. Importantly, we require it to be a random variable rather than a fixed number to preserve privacy. The last two inputs are $C$ and $c$, which are upper and lower bounds of the density of any posterior sampling distributions. A finite $C$ and a positive $c$ are required to bound the privacy cost of the entire framework.

```
Framework 1 DP-HyPO $/mathcal{A}/left(D, /pi^{(0)}, /mathcal{T}, C, c/right)$
    Initialize $/pi^{(0)}$, a prior distribution over $/Lambda$.
    Initialize the result set $A=/{/}$
    Draw $T /sim /mathcal{T}$
    for $j=0$ to $T-1$ do
        $(x, q) /sim Q/left(D, /pi^{(j)}/right)$
        $A=A /cup/{(x, q)/}$
        Update $/pi^{(j+1)}$ based on $A$ according to any adaptive algorithm such that for all $/lambda /in /Lambda$,
```$c /leq /frac{/pi^{(j+1)}(/lambda)}{/pi^{(0)}(/lambda)} /leq C$end forOutput $(x, q)$ from $A$ with the highest $q$

Note that we intentionally leave the update rule for $/pi^{(j+1)}$ unspecified in Framework 1 to reflect the fact that any adaptive update rule that leverages information from previous runs can be used. However, for a non-private adaptive HPO update rule, the requirement of bounded adaptive density $c /leq /frac{/pi^{(j+1)}(/lambda)}{/pi^{(0)}(/lambda)} /leq C$ may be easily violated. In Section 3.2 We provide a simple projection technique to privatize any non-private update rules. In Section 4, we provide an instantiation of DP-HyPO using Gaussian process.

We now state our main privacy results for this framework in terms of Rényi Differential Privacy (RDP) [27]. RDP is a privacy measure that is more general than the commonly used $(/varepsilon, /delta)$-DP and provides tighter privacy bounds for composition. We defer its exact definition to Definition A.2 in the appendix.

We note that different distributions of the number of selections (iterations), $/mathcal{T}$, result in very different privacy guarantees. Here, we showcase the key idea for deriving the privacy guarantee of DP-HyPO framework by considering a special case when $/mathcal{T}$ follows a truncated negative binomial distribution ${ }^{2}$ $/operatorname{NegBin}(/theta, /gamma)$ (the same assumption as in [32]). In fact, as we show in the proof of Theorem 1 in Appendix A the privacy bounds only depend on $/mathcal{T}$ directly through its probability generating function, and therefore one can adapt the proof to obtain the corresponding privacy guarantees for other probability families, for example, the Possion distribution considered in [32]. From here and on, unless otherwise specified, we will stick with $/mathcal{T}=/operatorname{NegBin}(/theta, /gamma)$ for simplicity. We also assume for simplicity that the prior distribution $/pi^{(0)}$ is a uniform distribution over $/Lambda$. We provide more detailed discussion of handling informed prior other than uniform distribution in Appendix D

Theorem 1. Suppose that $T$ follows truncated negative Binomial distribution $T /sim /operatorname{NegBin}(/theta, /gamma)$. Let $/theta /in(-1, /infty), /gamma /in(0,1)$, and $0<c /leq C$. Suppose for all $M_{/lambda}: /mathcal{Z}^{n} /rightarrow /mathcal{Y}$ over $/lambda /in /Lambda$, the base algorithms satisfy $(/alpha, /varepsilon)$-RDP and $(/hat{/alpha}, /hat{/varepsilon})$-RDP for some $/varepsilon, /hat{/varepsilon} /geq 0, /alpha /in(1, /infty)$, and $/hat{/alpha} /in[1, /infty)$. Then the DP-HyPO algorithm $/mathcal{A}/left(D, /pi^{(0)}, /operatorname{NegBin}(/theta, /gamma), C, c/right)$ satisfies $/left(/alpha, /varepsilon^{/prime}/right)$-RDP where

$$
/varepsilon^{/prime}=/varepsilon+(1+/theta) /cdot/left(1-/frac{1}{/hat{/alpha}}/right) /hat{/varepsilon}+/left(/frac{/alpha}{/alpha-1}+1+/theta/right) /log /frac{C}{c}+/frac{(1+/theta) /cdot /log (1 / /gamma)}{/hat{/alpha}}+/frac{/log /mathbb{E}[T]}{/alpha-1}
$$


/footnotetext{
${ }^{2}$ Truncated negative binomial distribution is a direct generalization of the geometric distribution. See Appendix B for its definition.
}

To prove Theorem 1, one of our main technical contributions is Lemma A.4, which quantifies the Rényi divergence of the sampling distribution at each iteration between the neighboring datasets. We then leverage this crucial result and the probability generating function of $/mathcal{T}$ to bound the Rényi divergence in the output of $/mathcal{A}$. We defer the detailed proof to Appendix A .

Next, we present the case with pure DP guarantees. Recall the fact that $(/varepsilon, 0)$-DP is equivalent to $(/infty, /varepsilon)$-RDP [27]. When both $/alpha$ and $/hat{/alpha}$ tend towards infinity, we easily obtain the following theorem in terms of $(/varepsilon, 0)$-DP.

Theorem 2. Suppose that $T$ follows truncated negative Binomial distribution $T /sim /operatorname{NegBin}(/theta, /gamma)$. Let $/theta /in(-1, /infty)$ and $/gamma /in(0,1)$. If all the base algorithms $M_{/lambda}$ satisfies $(/varepsilon, 0)-D P$, then the DP-HyPO algorithm $/mathcal{A}/left(D, /pi^{(0)}, /operatorname{NegBin}(/theta, /gamma), C, c/right)$ satisfies $/left((2+/theta)/left(/varepsilon+/log /frac{C}{c}/right), 0/right)-D P$.

Theorem 1 and Theorem 2 provide practitioners the freedom to trade off between allocating more DP budget to enhance the base algorithm or to improve adaptivity. In particular, a higher value of $/frac{C}{c}$ signifies greater adaptivity, while a larger $/varepsilon$ improves the performance of base algorithms.

/subsection*{3.1.1 Uniform Optimization Method as a Special Case}

We present the uniform hyperparameter optimization method [32, 23] in Algorithm 2, which is a special case of our general DP-HyPO Framework with $C=c=1$. Essentially, this algorithm never updates the sampling distribution $/pi$.

```

Algorithm 2 Uniform Hyperparameter Optimization $/mathcal{U}(D, /theta, /gamma, /Lambda)$
Let $/pi=/operatorname{Unif}(/{1, /ldots,|/Lambda|/})$, and $A=/{/}$
Draw $T /sim /operatorname{NegBin}(/theta, /gamma)$
for $j=0$ to $T-1$ do
$(x, q) /sim Q(D, /pi)$
$A=A /cup/{(x, q)/}$
end for
Output $(x, q)$ from $A$ with the highest $q$

```

Our results in Theorem 1 and Theorem 2] generalize the main technical results of [32, 24]. Specifically, when $C=c=1$ and $/Lambda$ is a finite discrete set, our Theorem 1 precisely recovers Theorem 2 in [32]. Furthermore, when we set $/theta=1$, the truncated negative binomial distribution reduces to the geometric distribution, and our Theorem 2 recovers Theorem 3.2 in [24] .

/subsection*{3.2 Practical Recipe to Privatize HPO Algorithms}

In the DP-HyPO framework, we begin with a prior and adaptively update it based on the accumulated information. However, for privacy purposes, we require the density $/pi^{(j)}$ to be bounded by some constants $c$ and $C$, which is due to the potential privacy leakage when updating $/pi^{(j)}$ based on the history. It is crucial to note that this distribution $/pi^{(j)}$ can be significantly different from the distribution $/pi^{/prime(j)}$ if we were given a different input dataset $D^{/prime}$. Therefore, we require the probability mass/density function to satisfy $/frac{c}{/mu(/Lambda)} /leq /pi^{(j)}(/lambda) /leq /frac{C}{/mu(/Lambda)}$ for all $/lambda /in /Lambda$ to control the privacy loss due to adaptivity.

This requirement is not automatically satisfied and typically necessitates modifications to current non-private HPO methods. To address this challenge, we propose a general recipe to modify any non-private method. The idea is quite straightforward: throughout the algorithm, we maintain a non-private version of the distribution density $/pi^{(j)}$. When sampling from the space $/Lambda$, we perform a projection from $/pi^{(j)}$ to the space consisting of bounded densities. Specifically, we define the space of essentially bounded density functions by $S_{C, c}=/left/{f /in /Lambda^{/mathbb{R}^{+}}/right.$: ess sup $f /leq /frac{C}{/mu(/Lambda)}$, ess inf $f /geq$ $/left./frac{c}{/mu(/Lambda)}, /int_{/alpha /in /Lambda} f(/alpha) /mathrm{d} /alpha=1/right/}$. For such a space to be non-empty, we require that $c /leq 1 /leq C$, where $/mu$ is the measure on $/Lambda$. This condition is well-defined as we assume $/mu(/Lambda)</infty$.

To privatize $/pi^{(j)}$ at the $j$-th iteration, we project it into the space $S_{C, c}$, by solving the following convex functional programming problem:

$$
/begin{align*}
/min _{f} & /left/|f-/pi^{(j)}/right/|_{2}  /tag{3.1}//
/text { s.t. } & f /in S_{C, c}
/end{align*}
$$

Note that this is a convex program since $S_{C, c}$ is convex and closed. We denote the output from this optimization problem by $/mathcal{P}_{S_{C, c}}/left(/pi^{(j)}/right)$. Theoretically, problem 3.1 allows the hyperparameter space $/Lambda$ to be general measurable space with arbitrary topological structure. However, empirically, practitioners need to discretize $/Lambda$ to some extent to make the convex optimization computationally feasible. Compared to the previous work, our formulation provides the most general characterization of the problem and allows pratitioners to adaptively and iteratively choose a proper discretization as needed. Framework 1 tolerates a much finer level of discretization than the previous method, as the performance of latter degrades fast when the number of candidates increases. We also provide examples using CVX to solve this problem in Section 4.2 In Appendix C, we discuss about its practical implementation, and the connection to information projection.

/section*{4 Application: DP-HyPO with Gaussian Process}

In this section, we provide an instantiation of DP-HyPO using Gaussian process (GP) [38]. GPs are popular non-parametric Bayesian models frequently employed for hyperparameter optimization. At the meta-level, GPs are trained to generate surrogate models by establishing a probability distribution over the performance measure $q$. While traditional GP implementations are not private, we leverage the approach introduced in Section 3.2 to design a private version that adheres to the bounded density contraint.

We provide the algorithmic description in Section 4.1 and the empircal evaluation in Section 4.2

/subsection*{4.1 Algorithm Description}

The following Algorithm $(/mathcal{A G P})$ is a private version of Gaussian process for hyperparameter tuning. In Algorithm 3, we utilize GP to construct a surrogate model that generates probability distributions

![](https://cdn.mathpix.com/cropped/2024_08_12_735d0bc2ca9785fb0a60g-07.jpg?height=50&width=1040&top_left_y=1593&top_left_x=369)
    Initialize $/pi^{(0)}=/operatorname{Unif}(/Lambda)$, and $A=/{/}$
    Draw $T /sim /operatorname{NegBin}(/theta, /gamma)$
    for $t=0$ to $T-1$ do
        Truncate the density of current $/pi^{(t)}$ to be bounded into the range of $[c, C]$ by projecting to
    $S_{C, c}$.
```

$$
/tilde{/pi}^{(t)}=/mathcal{P}_{S_{C, c}}/left(/pi^{(t)}/right)
$$

Sample $(x, q) /sim Q/left(D, /tilde{/pi}^{(j)}/right)$, and update $A=A /cup/{(x, q)/}$

Update mean estimation and variance estimation of the Gaussian process $/mu_{/lambda}, /sigma_{/lambda}^{2}$, and get the score as $s_{/lambda}=/mu_{/lambda}+/tau /sigma_{/lambda}$.

Update true (untruncated) posterior $/pi^{(t+1)}$ with softmax, by $/pi^{(t+1)}(/lambda)=/frac{/exp /left(/beta /cdot s_{/lambda}/right)}{/int_{/lambda^{/prime} /in /Lambda} /exp /left(/beta /cdot s_{/lambda}^{/prime}/right)}$.

end for

Output $(x, q)$ from $A$ with the highest $q$

for the performance measure $q$. By estimating the mean and variance, we assign a "score" to each hyperparameter $/lambda$, known as the estimated upper confidence bound (UCB). The weight factor $/tau$ controls the balance between exploration and exploitation, where larger weights prioritize exploration by assigning higher scores to hyperparameters with greater uncertainty.

To transform these scores into a sampling distribution, we apply the softmax function across all hyperparameters, incorporating the parameter $/beta$ as the inverse temperature. A higher value of $/beta$ signifies increased confidence in the learned scores for each hyperparameter.

### 4.2 Empirical Evaluations

We now evaluate the performance of our GP-based DP-HyPO (referred to as "GP") in various settings. Since DP-HyPO is the first adaptive private hyperparameter optimization method of its kind, we compare it to the special case of Uniform DP-HyPO (Algorithm 2, referred to as "Uniform", as proposed in [24, 32]. In this demonstration, we consider two pragmatic privacy configurations: the white-box setting and the black-box setting, contingent on whether adaptive HPO algorithms incur extra privacy cost. In the white-box scenario (Section 4.2.1 and 4.2.2), we conduct experiments involving training deep learning models on both the MNIST dataset and CIFAR-10 dataset. Conversely, when considering the black-box setting (Section 4.2.3), our attention shifts to a real-world Federated Learning (FL) task from the industry. These scenarios provide meaningful insights into the effectiveness and applicability of our GP-based DP-HyPO approach.

### 4.2.1 MNIST Simulation

We begin with the white-box scenario, in which the data curator aims to provide overall protection to the published model. In this context, to accommodate adaptive HPO algorithms, it becomes necessary to reduce the budget allocated to the base algorithm.

In this section, we consider the MNIST dataset, where we employ DP-SGD to train a standard CNN. The base algorithms in this case are different DP-SGD models with varying hyperparameters, and we evaluate each base algorithm based on its accuracy. Our objective is to identify the best hyperparameters that produce the most optimal model within a given total privacy budget.

Specifically, we consider two variable hyperparameters: the learning rate $/eta$ and clipping norm $R$, while keeping the other parameters fixed. We ensure that both the GP algorithm and the Uniform algorithm operate under the same total privacy budget, guaranteeing a fair comparison.

Due to constraints on computational resources, we conduct a semi-real simulation using the MNIST dataset. For both base algorithms (with different noise multipliers), we cache the mean accuracy of 5 independently trained models for each discretized hyperparameter and treat that as a proxy for the "actual accuracy" of the hyperparameter. Each time we sample the accuracy of a hyperparameter, we add a Gaussian noise with a standard deviation of 0.1 to the cached mean. We evaluate the performance of the output model based on the "actual accuracy" corresponding to the selected hyperparameter. Further details on the simulation and parameter configuration can be found in Appendix E. 1

In the left panel of Figure 1, we demonstrated the comparison of performance of the Uniform and GP methods with total privacy budget $/varepsilon=15^{3}$ and $/delta=1 e-5$. The accuracy reported is the actual accuracy of the output hyperparameter. From the figure, we see that when $T$ is very small $(T<8)$, GP method is slightly worse than Uniform method as GP spends $/log (C / c)$ budget less than Uniform method for each base algorithm (the cost of adaptivity). However, we see that after a short period of exploration, GP consistently outperform Uniform, mostly due to the power of being adaptive. The superiority of GP is further demonstrated in Table 1, aggregating over geometric distribution.

### 4.2.2 CIFAR-10 Simulation

When examining the results from MNIST, a legitimate critique arises: our DP-Hypo exhibits only marginal superiority over its uniform counterpart, which questions the assertion that adaptivity holds significant value. Our conjecture is that the hyperparameter landscape of MNIST is relatively uncomplicated, which limits the potential benefits of adaptive algorithms.

To test the hypothesis, we conduct experiments on the CIFAR-10 dataset, with a setup closely mirroring the previous experiment: we employ the same CNN model for training, and optimize the same set of hyperparameters, which are the learning rate $/eta$ and clipping norm $R$. The primary difference lies in how we generate the hyperparameter landscape. Given that a single run on CIFAR-10 is considerably more time-consuming than on MNIST, conducting multiple runs for every hyperparameter combination is unfeasible. To address this challenge, we leverage BoTorch [3],

[^0]![](https://cdn.mathpix.com/cropped/2024_08_12_735d0bc2ca9785fb0a60g-09.jpg?height=359&width=1387&top_left_y=238&top_left_x=369)

Figure 1: Left: The accuracy of the output hyperparameter in MNIST semi-real simulation, with $/varepsilon=15, /delta=0.00001$. Middle: The accuracy of the output hyperparameter in CIFAR-10, with $/varepsilon=12, /delta=0.00001$. Right: The loss of the output hyperparameter in FL. Error bars stands for $95 /%$ confidence. Curves for GP are calculated by averaging 400 independent runs, and curves for Uniform are calculated by averaging 10000 independent runs. For a clearer demonstration, we compare the performance for each fixed value of $T$, and recognize that the actual performance is a weighted average across different values of $T$.

an open-sourced library for HPO, to generate the landscape. Since we operate in the white-box setting, where the base algorithms have distinct privacy budgets for the uniform and adaptive scenarios, we execute 50 runs and generate the landscape for each case, including the mean $/left(/mu_{/lambda}/right)$ and standard error $/left(/sigma_{/lambda}/right)$ of accuracy for each hyperparameter combination $/lambda$. When the algorithm (GP or Uniform) visits a specific $/lambda$, our oracle returns a noisy score $q(/lambda)$ drawn from a normal distribution of $N/left(/mu_{/lambda}, /sigma_{/lambda}/right)$. A more detailed description of our landscapes and parameter configuration can be found in Appendix E. 2

In the middle of Figure 1, we showcase a performance comparison between the Uniform and GP methods with a total privacy budget of $/varepsilon=12$ and $/delta=1 e-5$. Clearly, GP consistently outperforms the Uniform method, with the largest performance gap occurring when the number of runs is around 10 .

### 4.2.3 Federated Learning

In this section, we move to the black-box setting, where the privacy budget allocated to the base algorithm remains fixed, while we allow extra privacy budget for HPO. That being said, the adaptivity can be achieved without compromising the utility of the base algorithm.

We explore another real-world scenario: a Federated Learning (FL) task conducted on a proprietary dataset ${ }_{4}^{4}$ from industry. Our aim is to determine the optimal learning rates for the central server (using AdaGrad) and the individual users (using SGD). To simulate this scenario, we once again rely on the landscape generated by BoTorch [3], as shown in Figure 3]in Appendix E. 3 .

Under the assumption that base algorithms are black-box models with fixed privacy costs, we proceed with HPO while varying the degree of adaptivity. The experiment results are visualized in the right panel of Figure 1, and Table 2 presents the aggregated performance data.

We consistently observe that GP outperforms Uniform in the black-box setting. Furthermore, our findings suggest that allocating a larger privacy budget to the GP method facilitates the acquisition of adaptive information, resulting in improved performance in HPO. This highlights the flexibility of GP in utilizing privacy resources effectively.

| Geometric $(/gamma)$ | 0.001 | 0.002 | 0.003 | 0.005 | 0.01 | 0.02 | 0.025 | 0.03 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GP | 0.946 | 0.948 | 0.948 | 0.947 | 0.943 | 0.937 | 0.934 | 0.932 |
| Uniform | 0.943 | 0.945 | 0.945 | 0.944 | 0.940 | 0.935 | 0.932 | 0.929 |

Table 1: Accuracy of MNIST using Geometric Distribution with various different values of $/gamma$ for Uniform and GP methods. Each number is the mean of 200 runs.

[^1]| Geometric $(/gamma)$ | 0.001 | 0.002 | 0.003 | 0.005 | 0.01 | 0.02 | 0.025 | 0.03 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GP $(/mathrm{C}=1.25)$ | 0.00853 | 0.0088 | 0.00906 | 0.00958 | 0.0108 | 0.0129 | 0.0138 | 0.0146 |
| GP $(/mathrm{C}=1.33)$ | 0.00821 | 0.00847 | 0.00872 | 0.00921 | 0.0104 | 0.0123 | 0.0132 | 0.0140 |
| GP $(/mathrm{C}=1.5)$ | 0.00822 | 0.00848 | 0.00872 | 0.00920 | 0.0103 | 0.0123 | 0.0131 | 0.0130 |
| Uniform | 0.0104 | 0.0106 | 0.0109 | 0.0113 | 0.0123 | 0.0141 | 0.0149 | 0.0156 |

Table 2: Loss of FL using Geometric Distribution with various different values of $/gamma$ for Uniform and GP methods with different choice of $C$ and $c=1 / C$. Each number is the mean of 200 runs.

## 5 Conclusion

In conclusion, this paper presents a novel framework, DP-HyPO. As the first adaptive HPO framework with sharp DP guarantees, DP-HyPO effectively bridges the gap between private and non-private HPO. Our work encompasses the random search method by [24, 32] as a special case, while also granting practitioners the ability to adaptively learn better sampling distributions based on previous runs. Importantly, DP-HyPO enables the conversion of any non-private adaptive HPO algorithm into a private one. Our framework proves to be a powerful tool for professionals seeking optimal model performance and robust DP guarantees.

The DP-HyPO framework presents two interesting future directions. One prospect involves an alternative HPO specification which is practically more favorable. Considering the extensive literature on HPO, there is a significant potential to improve the empirical performance by leveraging more advanced HPO methods. Secondly, there is an interest in establishing a theoretical utility guarantee for DP-HyPO. By leveraging similar proof methodologies to those in Theorem 3.3 in [24], it is feasible to provide basic utility guarantees for the general DP-HyPO, or for some specific configurations within DP-HyPO.

## 6 Acknowledgements

The authors would like to thank Max Balandat for his thoughtful comments and insights that helped us improve the paper.

## References

[1] Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In Proceedings of the 2016 ACM SIGSAC conference on computer and communications security, pages 308-318, 2016.

[2] Martin S Andersen, Joachim Dahl, Lieven Vandenberghe, et al. Cvxopt: A python package for convex optimization. Available at cvxopt. org, 54, 2013.

[3] Maximilian Balandat, Brian Karrer, Daniel Jiang, Samuel Daulton, Ben Letham, Andrew G Wilson, and Eytan Bakshy. Botorch: A framework for efficient monte-carlo bayesian optimization. Advances in neural information processing systems, 33:21524-21538, 2020.

[4] Raef Bassily, Adam Smith, and Abhradeep Thakurta. Private empirical risk minimization: Efficient algorithms and tight error bounds. In 2014 IEEE 55th annual symposium on foundations of computer science, pages 464-473. IEEE, 2014.

[5] James Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. Algorithms for hyperparameter optimization. Advances in neural information processing systems, 24, 2011.

[6] Nicholas Carlini, Chang Liu, Úlfar Erlingsson, Jernej Kos, and Dawn Song. The secret sharer: Evaluating and testing unintended memorization in neural networks. In USENIX Security Symposium, volume 267, 2019.

[7] Kamalika Chaudhuri and Staal A Vinterbo. A stability-based validation procedure for differentially private machine learning. Advances in Neural Information Processing Systems, 26, 2013.

[8] Edith Cohen, Xin Lyu, Jelani Nelson, Tamás Sarlós, and Uri Stemmer. Generalized private selection and testing with high confidence. arXiv preprint arXiv:2211.12063, 2022.

[9] Imre Csiszár and Frantisek Matus. Information projections revisited. IEEE Transactions on Information Theory, 49(6):1474-1490, 2003.

[10] Soham De, Leonard Berrada, Jamie Hayes, Samuel L Smith, and Borja Balle. Unlocking high-accuracy differentially private image classification through scale. arXiv preprint arXiv:2204.13650, 2022.

[11] Jinshuo Dong, Aaron Roth, and Weijie J Su. Gaussian differential privacy. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(1):3-37, 2022.

[12] Cynthia Dwork, Frank McSherry, Kobbi Nissim, and Adam Smith. Calibrating noise to sensitivity in private data analysis. In Theory of Cryptography: Third Theory of Cryptography Conference, TCC 2006, New York, NY, USA, March 4-7, 2006. Proceedings 3, pages 265-284. Springer, 2006.

[13] Cynthia Dwork, Moni Naor, Omer Reingold, Guy N Rothblum, and Salil Vadhan. On the complexity of differentially private data release: efficient algorithms and hardness results. In Proceedings of the forty-first annual ACM symposium on Theory of computing, pages 381-390, 2009 .

[14] Matthias Feurer and Frank Hutter. Hyperparameter optimization. Automated machine learning: Methods, systems, challenges, pages 3-33, 2019.

[15] Yonatan Geifman and Ran El-Yaniv. Deep active learning with a neural architecture search. Advances in Neural Information Processing Systems, 32, 2019.

[16] Xin He, Kaiyong Zhao, and Xiaowen Chu. Automl: A survey of the state-of-the-art. KnowledgeBased Systems, 212:106622, 2021.

[17] Andrew Hundt, Varun Jain, and Gregory D Hager. sharpdarts: Faster and more accurate differentiable architecture search. arXiv preprint arXiv:1903.09900, 2019.

[18] Frank Hutter, Holger H Hoos, and Kevin Leyton-Brown. Sequential model-based optimization for general algorithm configuration. In Learning and Intelligent Optimization: 5th International Conference, LION 5, Rome, Italy, January 17-21, 2011. Selected Papers 5, pages 507-523. Springer, 2011.

[19] Peter Kairouz, Sewoong Oh, and Pramod Viswanath. The composition theorem for differential privacy. In International conference on machine learning, pages 1376-1385. PMLR, 2015.

[20] Kirthevasan Kandasamy, Willie Neiswanger, Jeff Schneider, Barnabas Poczos, and Eric P Xing. Neural architecture search with bayesian optimisation and optimal transport. Advances in neural information processing systems, 31, 2018.

[21] Rajiv Khanna, Joydeep Ghosh, Rusell Poldrack, and Oluwasanmi Koyejo. Information projection and approximate inference for structured sparse variables. In Artificial Intelligence and Statistics, pages 1358-1366. PMLR, 2017.

[22] Liam Li and Ameet Talwalkar. Random search and reproducibility for neural architecture search. In Uncertainty in artificial intelligence, pages 367-377. PMLR, 2020

[23] Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar. Hyperband: A novel bandit-based approach to hyperparameter optimization. The Journal of Machine Learning Research, 18(1):6765-6816, 2017.

[24] Jingcheng Liu and Kunal Talwar. Private selection from private candidates. In Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing, pages 298-309, 2019.

[25] Frank McSherry and Kunal Talwar. Mechanism design via differential privacy. In 48th Annual IEEE Symposium on Foundations of Computer Science (FOCS'07), pages 94-103. IEEE, 2007.

[26] Hector Mendoza, Aaron Klein, Matthias Feurer, Jost Tobias Springenberg, and Frank Hutter. Towards automatically-tuned neural networks. In Workshop on automatic machine learning, pages $58-65$. PMLR, 2016.

[27] Ilya Mironov. Rényi differential privacy. In 2017 IEEE 30th computer security foundations symposium ( $C S F$ ), pages 263-275. IEEE, 2017.

[28] Shubhankar Mohapatra, Sajin Sasy, Xi He, Gautam Kamath, and Om Thakkar. The role of adaptive optimizers for honest private hyperparameter selection. In Proceedings of the aaai conference on artificial intelligence, volume 36, pages 7806-7813, 2022.

[29] Milad Nasr, Reza Shokri, and Amir Houmansadr. Comprehensive privacy analysis of deep learning: Passive and active white-box inference attacks against centralized and federated learning. In 2019 IEEE symposium on security and privacy $(S P)$, pages 739-753. IEEE, 2019.

[30] Renato Negrinho, Matthew Gormley, Geoffrey J Gordon, Darshan Patil, Nghia Le, and Daniel Ferreira. Towards modular and programmable architecture search. Advances in neural information processing systems, 32, 2019.

[31] Ashwinee Panda, Xinyu Tang, Vikash Sehwag, Saeed Mahloujifar, and Prateek Mittal. Dp-raft: A differentially private recipe for accelerated fine-tuning. arXiv preprint arXiv:2212.04486, 2022.

[32] Nicolas Papernot and Thomas Steinke. Hyperparameter tuning with renyi differential privacy. In International Conference on Learning Representations, 2021.

[33] Carl Edward Rasmussen. Gaussian processes in machine learning. In Advanced Lectures on Machine Learning: ML Summer Schools 2003, Canberra, Australia, February 2-14, 2003, Tübingen, Germany, August 4-16, 2003, Revised Lectures, pages 63-71. Springer, 2004.

[34] Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P Adams, and Nando De Freitas. Taking the human out of the loop: A review of bayesian optimization. Proceedings of the IEEE, 104(1):148-175, 2015.

[35] Shuang Song, Kamalika Chaudhuri, and Anand D Sarwate. Stochastic gradient descent with differentially private updates. In 2013 IEEE global conference on signal and information processing, pages 245-248. IEEE, 2013.

[36] Salil Vadhan. The complexity of differential privacy. Tutorials on the Foundations of Cryptography: Dedicated to Oded Goldreich, pages 347-450, 2017.

[37] Hua Wang, Sheng Gao, Huanyu Zhang, Milan Shen, and Weijie J Su. Analytical composition of differential privacy via the edgeworth accountant. arXiv preprint arXiv:2206.04236, 2022.

[38] Christopher KI Williams and Carl Edward Rasmussen. Gaussian processes for machine learning, volume 2. MIT press Cambridge, MA, 2006.

[39] Da Yu, Huishuai Zhang, Wei Chen, Jian Yin, and Tie-Yan Liu. Large scale private learning via low-rank reparametrization. In International Conference on Machine Learning, pages 12208-12218. PMLR, 2021.

[40] Tong Yu and Hong Zhu. Hyper-parameter optimization: A review of algorithms and applications. arXiv preprint arXiv:2003.05689, 2020.

[41] Arber Zela, Aaron Klein, Stefan Falkner, and Frank Hutter. Towards automated deep learning: Efficient joint neural architecture and hyperparameter search. arXiv preprint arXiv:1807.06906, 2018.

[42] Huanyu Zhang, Ilya Mironov, and Meisam Hejazinia. Wide network learning with differential privacy. arXiv preprint arXiv:2103.01294, 2021.

## A Proofs of the technical results

## A. 1 Proof of Main Results

First, we define Rényi divergence as follows.

Definition A. 1 (Rényi Divergences). Let $P$ and $Q$ be probability distributions on a common space $/Omega$. Assume that $P$ is absolutely continuous with respect to $Q$ - i.e., for all measurable $E /subset /Omega$, if $Q(E)=0$, then $P(E)=0$. Let $P(x)$ and $Q(x)$ denote the densities of $P$ and $Q$ respectively. The KL divergence from $P$ to $Q$ is defined as

$$
/mathrm{D}_{1}(P /| Q):=/underset{X /leftarrow P}{/mathbb{E}}/left[/log /left(/frac{P(X)}{Q(X)}/right)/right]=/int_{/Omega} P(x) /log /left(/frac{P(x)}{Q(x)}/right) /mathrm{d} x
$$

The max divergence from $P$ to $Q$ is defined as

$$
/mathrm{D}_{/infty}(P /| Q):=/sup /left/{/log /left(/frac{P(E)}{Q(E)}/right): P(E)>0/right/}
$$

For $/alpha /in(1, /infty)$, the Rényi divergence from $P$ to $Q$ of order $/alpha$ is defined as

$$
/begin{aligned}
/mathrm{D}_{/alpha}(P /| Q) & :=/frac{1}{/alpha-1} /log /left(/underset{X /leftarrow P}{/mathbb{E}}/left[/left(/frac{P(X)}{Q(X)}/right)^{/alpha-1}/right]/right) //
& =/frac{1}{/alpha-1} /log /left(/underset{X /leftarrow Q}{/mathbb{E}}/left[/left(/frac{P(X)}{Q(X)}/right)^{/alpha}/right]/right) //
& =/frac{1}{/alpha-1} /log /left(/int_{Q} P(x)^{/alpha} Q(x)^{1-/alpha} /mathrm{d} x/right)
/end{aligned}
$$

We now present the definition of Rényi DP (RDP) in [27].

Definition A. 2 (Rényi Differential Privacy). A randomized algorithm $M: /mathcal{X}^{n} /rightarrow /mathcal{Y}$ is $(/alpha, /varepsilon)$-Rényi differentially private if, for all neighbouring pairs of inputs $D, D^{/prime} /in /mathcal{X}^{n}, /mathrm{D}_{/alpha}/left(M(x) /| M/left(x^{/prime}/right)/right) /leq /varepsilon$.

We define some additional notations for the sake of the proofs. In algorithm 1 , for any $1 /leq j /leq T$, and neighboring dataset $D$ and $D^{/prime}$, we define the following notations for any $y=(x, q) /in /mathcal{Y}$, the totally ordered range set.

$$
/begin{array}{rlll}
P_{j}(y)=/mathbb{P}_{/tilde{y} /sim Q/left(D, /pi^{(j)}/right)}(/tilde{y}=y) & /text { and } & P_{j}^{/prime}(y)=/mathbb{P}_{/tilde{y} /sim Q/left(D^{/prime}, /pi^{/prime(j)}/right)}(/tilde{y}=y) //
P_{j}(/leq y)=/mathbb{P}_{/tilde{y} /sim Q/left(D, /pi^{(j)}/right)}(/tilde{y} /leq y) & /text { and } & P_{j}^{/prime}(/leq y)=/mathbb{P}_{/tilde{y} /sim Q/left(D^{/prime}, /pi^{/prime(j)}/right)}(/tilde{y} /leq y) //
P_{j}(<y)=/mathbb{P}_{/tilde{y} /sim Q/left(D, /pi^{(j)}/right)}(/tilde{y}<y) & /text { and } & P_{j}^{/prime}(<y)=/mathbb{P}_{/tilde{y} /sim Q/left(D^{/prime}, /pi^{/prime(j)}/right)}(/tilde{y}<y)
/end{array}
$$

By these definitions, we have $P_{j}(/leq y)=P_{j}(<y)+P_{j}(y)$, and $P_{j}^{/prime}(/leq y)=P_{j}^{/prime}(<y)+P_{j}^{/prime}(y)$. And additionally, we have

$$
/begin{align*}
/frac{P_{j}(y)}{P_{j}^{/prime}(y)}=/frac{/int_{/lambda /in /Lambda} /mathbb{P}/left(M_{/lambda}(D)=y/right) /pi^{(j)}(/lambda) d /lambda}{/int_{/lambda /in /Lambda} /mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right) /pi^{/prime(j)}(/lambda) d /lambda} & /leq /sup _{/lambda /in /Lambda} /frac{/mathbb{P}/left(M_{/lambda}(D)=y/right) /pi^{(j)}(/lambda)}{/mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right) /pi^{/prime(j)}(/lambda)} //
& /leq /frac{C}{c} /cdot /sup _{/lambda /in /Lambda} /frac{/mathbb{P}/left(M_{/lambda}(D)=y/right)}{/mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right)} /tag{A.1}
/end{align*}
$$

Here, the first inequality follows from the simple property of integration, and the second inequality follows from the fact that $/pi^{(j)}$ has bounded density between $c$ and $C$. Similarly, we have

$$
/begin{equation*}
/frac{P_{j}(/leq y)}{P_{j}^{/prime}(/leq y)} /leq /frac{C}{c} /cdot /sup _{/lambda /in /Lambda} /frac{/mathbb{P}/left(M_{/lambda}(D) /leq y/right)}{/mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right) /leq y/right)} /tag{A.2}
/end{equation*}
$$

and

$$
/begin{equation*}
/frac{P_{j}(<y)}{P_{j}^{/prime}(<y)} /leq /frac{C}{c} /cdot /sup _{/lambda /in /Lambda} /frac{/mathbb{P}/left(M_{/lambda}(D)<y/right)}{/mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)<y/right)} /tag{A.3}
/end{equation*}
$$

Note that $D$ and $D^{/prime}$ are neighboring datasets, and $M_{/lambda}$ satisfies some DP guarantees. So the ratio $/frac{/mathbb{P}/left(M_{/lambda}(D) /in E/right)}{/mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right) /in E/right)}$ for any event $E$ can be bounded.

For simplicity, we define the inner product of a distribution $/pi$ with the vector $M(D)=/left(/mathbb{P}/left(M_{/lambda}(D)=/right./right.$ y) $: /lambda /in /Lambda$ ) as

$$
/begin{equation*}
/pi /cdot /boldsymbol{M}(D):=/int_{/lambda /in /Lambda} /mathbb{P}/left(M_{/lambda}(D)=y/right) /pi(/lambda) d /lambda /tag{A.4}
/end{equation*}
$$

Now, we define additional notations to bound the probabilities. Recall $S_{C, s}$ is given by $/left/{f /in /Lambda^{/mathbb{R}^{+}}/right.$: ess sup $f /leq C$, ess inf $f /geq c, /int_{/alpha /in /Lambda} f(/alpha) d /alpha=1$./}. It is straightforward to see this is a compact set as it is the intersection of three compact sets. We define

$$
/begin{equation*}
P^{+}(y):=/sup _{/pi /in S_{C, c}} /int_{/lambda /in /Lambda} /mathbb{P}/left(M_{/lambda}(D)=y/right) /pi^{(j)}(/lambda) d /lambda=/pi^{+} /cdot /boldsymbol{M}(D) /tag{A.5}
/end{equation*}
$$

where $/pi^{+}$is the distribution that achieves the supreme in the compact set $S_{C, c}$. Similarly, we define $P^{/prime-}(y)$ for $D^{/prime}$ as given by

$$
/begin{equation*}
P^{/prime-}(y):=/inf _{/pi /in S_{C, c}} /int_{/lambda /in /Lambda} /mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right) /cdot /pi^{/prime(j)}(/lambda) d /lambda=/pi^{/prime-} /cdot /boldsymbol{M} /tag{A.6}
/end{equation*}
$$

Similarly, we can define $P^{/prime+}(y)$ and $P^{-}(y)$ accordingly. From the definition, we know that

$$
/begin{equation*}
P^{-}(y) /leq P_{j}(y) /leq P^{+}(y) /quad /text { and } /quad P^{/prime-}(y) /leq P_{j}^{/prime}(y) /leq P^{/prime+}(y) /tag{A.7}
/end{equation*}
$$

We also have

$$
/begin{equation*}
/frac{P^{+}(y)}{P^{/prime-}(y)}=/frac{/pi^{*} /cdot /boldsymbol{M}(D)}{/pi^{/prime-} /cdot /boldsymbol{M}/left(D^{/prime}/right)} /leq /sup _{/lambda} /frac{/mathbb{P}/left(M_{/lambda}(D)=y/right)}{/mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right)} /cdot /frac{C}{c} /tag{A.8}
/end{equation*}
$$

It is similar to define

$$
/begin{aligned}
P^{+}(/leq y):=/sup _{/pi /in S_{C, c}} /int_{/lambda /in /Lambda} /mathbb{P}/left(M_{/lambda}(D) /leq y/right) /quad /text { and } /quad P^{/prime+}(/leq y):=/sup _{/pi /in S_{C, c}} /int_{/lambda /in /Lambda} /mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right) /leq y/right) //
P^{-}(/leq y):=/inf _{/pi /in S_{C, c}} /int_{/lambda /in /Lambda} /mathbb{P}/left(M_{/lambda}(D) /leq y/right) /quad /text { and } /quad P^{/prime-}(/leq y):=/inf _{/pi /in S_{C, c}} /int_{/lambda /in /Lambda} /mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right) /leq y/right) //
P^{+}(<y):=/sup _{/pi /in S_{C, c}} /int_{/lambda /in /Lambda} /mathbb{P}/left(M_{/lambda}(D)<y/right) /quad /text { and } /quad P^{/prime+}(<y):=/sup _{/pi /in S_{C, c}} /int_{/lambda /in /Lambda} /mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)<y/right) //
P^{-}(<y):=/inf _{/pi /in S_{C, c}} /int_{/lambda /in /Lambda} /mathbb{P}/left(M_{/lambda}(D)<y/right) /quad /text { and } /quad P^{/prime-}(<y):=/inf _{/pi /in S_{C, c}} /int_{/lambda /in /Lambda} /mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)<y/right)
/end{aligned}
$$

Following the exact same proof, we have

$$
/begin{align*}
P^{-}(/leq y) /leq P_{j}(/leq y) /leq P^{+}(/leq y) /quad /text { and } /quad P^{/prime-}(/leq y) /leq P_{j}^{/prime}(/leq y) /leq P^{/prime+}(/leq y)  /tag{A.9}//
P^{-}(<y) /leq P_{j}(<y) /leq P^{+}(<y) /quad /text { and } /quad P^{/prime-}(<y) /leq P_{j}^{/prime}(<y) /leq P^{/prime+}(<y)  /tag{A.10}//
/frac{P^{+}(/leq y)}{P^{/prime-}(/leq y)} /leq /sup _{/lambda} /frac{/mathbb{P}/left(M_{/lambda}(D) /leq y/right)}{/mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right) /leq y/right)} /cdot /frac{C}{c} /quad /text { and } /quad /frac{P^{+}(<y)}{P^{/prime-}(<y)} /leq /sup _{/lambda} /frac{/mathbb{P}/left(M_{/lambda}(D)<y/right)}{/mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)<y/right)} /cdot /frac{C}{c} /tag{A.11}
/end{align*}
$$

It is also straightforward to verify from the definition that

$$
/begin{align*}
P^{+}(/leq y) & =P^{+}(<y)+P^{+}(y) /quad /text { and } /quad P^{/prime+}(/leq y)=P^{/prime+}(<y)+P^{/prime+}(y)  /tag{A.12}//
P^{+}-(/leq y) & =P^{-}(<y)+P^{-}(y) /quad /text { and } /quad P^{/prime-}(/leq y)=P^{/prime-}(<y)+P^{/prime-}(y) /tag{A.13}
/end{align*}
$$

Lemma A.3. Suppose if $a_{/lambda}, b_{/lambda}$ are non-negative and $c_{/lambda}, c_{/lambda}^{/prime}$ are positive for all $/lambda$. Then we have

$$
/frac{/sum_{/lambda} a_{/lambda} c_{/lambda}}{/sum_{/lambda} b_{/lambda} c_{/lambda}^{/prime}} /leq /frac{/sum_{/lambda} a_{/lambda}}{/sum_{/lambda} b_{/lambda}} /cdot /sup _{/lambda, /lambda^{/prime}}/left|/frac{c_{/lambda}}{c_{/lambda}^{/prime}}/right|
$$

Proof of Lemma A.3. This lemma is pretty straight forward by comparing the coefficient for each term in the full expansion. Specifically, we re-write the inequality as

$$
/begin{equation*}
/sum_{/lambda} a_{/lambda} c_{/lambda} /sum_{/lambda^{/prime}} b_{/lambda}^{/prime} /leq /sum_{/lambda} a_{/lambda} /sum_{/lambda^{/prime}} b_{/lambda}^{/prime} c_{/lambda}^{/prime} /cdot /sup _{/lambda, /lambda^{/prime}}/left|/frac{c_{/lambda}}{c_{/lambda}^{/prime}}/right| /tag{A.14}
/end{equation*}
$$

For each term $a_{/lambda} b_{/lambda}^{/prime}$, its coefficient on the left hand side of A.14 is $c_{/lambda}$, but its coefficient on the right hand side of A.14) is $c_{/lambda}^{/prime} /cdot /sup _{/lambda, /lambda^{/prime}}/left|/frac{c_{/lambda}}{c_{/lambda}^{/prime}}/right|$. Since we always have $c_{/lambda}^{/prime} /cdot /sup _{/lambda, /lambda^{/prime}}/left|/frac{c_{/lambda}}{c_{/lambda}^{/prime}}/right| /geq c_{/lambda}$, and $a_{/lambda} b_{/lambda}^{/prime} /geq 0$, we know the inequality A.14 holds.

Next, in order to present our results in terms of RDP guarantees, we prove the following lemma.

Lemma A.4. The Rényi divergence between $P^{+}$and $P^{-}$is be bounded as follows:

$$
/mathrm{D}_{/alpha}/left(P^{+} /| P^{/prime-}/right) /leq /frac{/alpha}{/alpha-1} /log /frac{C}{c}+/sup _{/lambda /in /Lambda} /mathrm{D}_{/alpha}/left(M_{/lambda}(D) /| M_{/lambda}/left(D^{/prime}/right)/right)
$$

Proof of Lemma A.4. We write that

$$
/begin{equation*}
e^{(/alpha-1) /mathrm{D}_{/alpha}/left(P^{+} /| P^{/prime-}/right)}=/sum_{y /in /mathcal{Y}} P^{+}(y)^{/alpha} /cdot P^{/prime-}(y)^{1-/alpha}=/sum_{y /in /mathcal{Y}} /frac{/left(/sum_{/lambda} /pi^{+}(/lambda) /mathbb{P}/left(M_{/lambda}(D)=y/right)/right)^{/alpha}}{/left(/sum_{/lambda} /pi^{/prime-}(/lambda) /mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right)/right)^{/alpha-1}} /tag{A.15}
/end{equation*}
$$

Here, $/pi^{+}$and $/pi^{/prime-}$ are defined in A.5 and A.6, so they are essentially $/pi_{y}^{+}$and $/pi_{y}^{/prime-}$ as they depend on the value of $y$. Therefore, we need to "remove" this dependence on $y$ to leverage the RDP guarantees for each base algorithm $M_{/lambda}$. We accomplish this task by bridging via $/pi$, the uniform density on $/Lambda$ (that is $/pi(/lambda)=/pi/left(/lambda^{/prime}/right)$ for any $/lambda, /lambda^{/prime} /in /Lambda$ ). Specifically, we define $a_{/lambda}=/pi(/lambda) /mathbb{P}/left(M_{/lambda}(D)=y/right)$, $b_{/lambda}=/pi(/lambda) /mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right), c_{/lambda}=/frac{/pi_{y}^{+}(/lambda)}{/pi(/lambda)}$, and $c_{/lambda}^{/prime}=/frac{/pi_{y}^{/prime-}(/lambda)}{/pi(/lambda)}$. We see that

$$
/begin{equation*}
/sup _{/lambda, /lambda^{/prime}}/left|/frac{c_{/lambda}}{c_{/lambda}^{/prime}}/right|=/sup _{/lambda, /lambda^{/prime}}/left|/frac{/pi_{y}^{+}(/lambda) / /pi(/lambda)}{/pi_{y}^{/prime-}/left(/lambda^{/prime}/right) / /pi/left(/lambda^{/prime}/right)}/right|=/sup _{/lambda, /lambda^{/prime}}/left|/frac{/left./pi_{y}^{+}(/lambda)/right)}{/pi_{y}^{/prime-}/left(/lambda^{/prime}/right)}/right| /leq C / c /tag{A.16}
/end{equation*}
$$

since $/pi$ is the uniform, and $/pi_{y}^{+}$and $/pi_{y}^{/prime-}$ belongs to $S_{C, c}$. We now apply Lemma A.3 with the above notations for each $y$ to A.15, and we have

$$
/begin{aligned}
& /sum_{y /in /mathcal{Y}} /frac{/left(/sum_{/lambda} /pi^{+}(/lambda) /mathbb{P}/left(M_{/lambda}(D)=y/right)/right)^{/alpha}}{/left(/sum_{/lambda} /pi^{/prime-}(/lambda) /mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right)/right)^{/alpha-1}} //
= & /sum_{y /in /mathcal{Y}} /frac{/left(/sum_{/lambda} /pi(/lambda) /mathbb{P}/left(M_{/lambda}(D)=y/right) /cdot /frac{/pi^{+}(/lambda)}{/pi(/lambda)}/right)^{/alpha-1}/left(/sum_{/lambda} /pi(/lambda) /mathbb{P}/left(M_{/lambda}(D)=y/right) /cdot /frac{/pi^{+}(/lambda)}{/pi(/lambda)}/right)}{/left(/sum_{/lambda} /pi(/lambda) /mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right) /cdot /frac{/pi^{/prime-}(/lambda)}{/pi(/lambda)}/right)^{/alpha-1}} //
= & /sum_{y /in /mathcal{Y}} /frac{/left(/sum_{/lambda} a_{/lambda} /cdot c_{/lambda}/right)^{/alpha-1}/left(/sum_{/lambda} /pi(/lambda) /mathbb{P}/left(M_{/lambda}(D)=y/right) /cdot /frac{/pi^{+}(/lambda)}{/pi(/lambda)}/right)}{/left(/sum_{/lambda} b_{/lambda} /cdot c_{/lambda}^{/prime}/right)^{/alpha-1}} //
/leq & /sum_{y /in /mathcal{Y}^{/lambda, /lambda^{/prime}}} /sup _{/lambda_{/lambda}}/left|/frac{c_{/lambda}}{c_{/lambda}^{/prime}}/right|^{/alpha-1} /frac{/left(/sum_{/lambda} a_{/lambda}/right)^{/alpha-1}/left(/sum_{/lambda} /pi(/lambda) /mathbb{P}/left(M_{/lambda}(D)=y/right) /cdot /frac{/pi^{+}(/lambda)}{/pi(/lambda)}/right)}{/left(/sum_{/lambda} b_{/lambda}/right)^{/alpha-1}} //
= & /sum_{y /in /mathcal{Y}^{/lambda, /lambda^{/prime}}} /sup _{/lambda_{/lambda}}/left|/frac{c_{/lambda}}{c_{/lambda}^{/prime}}/right|^{/alpha-1} /frac{/left(/sum_{/lambda} a_{/lambda}/right)^{/alpha-1}/left(/sum_{/lambda} a_{/lambda} /cdot c_{/lambda}/right)}{/left(/sum_{/lambda} b_{/lambda}/right)^{/alpha-1}} //
/leq & /sum_{y /in /mathcal{Y}} /sup _{/lambda, /lambda^{/prime}}/left|/frac{c_{/lambda}}{c_{/lambda}^{/prime}}/right|^{/alpha-1} /frac{/left(/sum_{/lambda} a_{/lambda}/right)^{/alpha-1}/left(/sum_{/lambda} a_{/lambda}/right) /cdot /sup _{/lambda} c_{/lambda}}{/left(/sum_{/lambda} b_{/lambda}/right)^{/alpha-1}} //
/leq & /sum_{y /in /mathcal{Y}}/left(/frac{C}{c}/right)^{/alpha-1} /frac{/left(/sum_{/lambda} a_{/lambda}/right)^{/alpha-1}/left(/sum_{/lambda} a_{/lambda}/right) /cdot/left(/frac{C}{c}/right)}{/left(/sum_{/lambda} b_{/lambda}/right)^{/alpha-1}} //
= & /sum_{y /in /mathcal{Y}}/left(/frac{C}{c}/right)^{/alpha} /cdot /frac{/left(/sum_{/lambda} /pi(/lambda) /mathbb{P}/left(M_{/lambda}(D)=y/right)/right)^{/alpha}}{/left(/sum_{/lambda} /pi(/lambda) /mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right)/right)^{/alpha-1}}
/end{aligned}
$$

The first inequality is due to Lemma A.3 the second inequality is because $a_{/lambda}$ are non-negative, and the last inequality is because of A.16) and the fact that both $/pi^{+}(/lambda)$ and $/pi(/lambda)$ are defined in $/mathbf{S}_{C, c}$, and thus their ratio is upper bounded by $/frac{C}{c}$ for any $/lambda$.

Now we only need to prove that for any fixed distribution $/pi$ that doesn't depend on value $y$, we have

$$
/begin{equation*}
/sum_{y /in /mathcal{Y}} /frac{/left(/sum_{/lambda} /pi(/lambda) /mathbb{P}/left(M_{/lambda}(D)=y/right)/right)^{/alpha}}{/left(/sum_{/lambda} /pi(/lambda) /mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right)/right)^{/alpha-1}} /leq /sup _{/lambda /in /Lambda} e^{(/alpha-1) /mathrm{D}_{/alpha}/left(M_{/lambda}(D) /| M_{/lambda}/left(D^{/prime}/right)/right)} /tag{A.17}
/end{equation*}
$$

With this result, we immediately know the result holds for uniform distribution $/pi$ as a special case. To prove this result, we first observe that the function $f(u, v)=u^{/alpha} v^{1-/alpha}$ is a convex function. This is because the Hessian of $f$ is

$$
/left(/begin{array}{cc}
/alpha(/alpha-1) u^{/alpha-2} v^{1-/alpha} & -/alpha(/alpha-1) u^{/alpha-1} v^{-/alpha} //
-/alpha(/alpha-1) u^{/alpha-1} v^{-/alpha} & /alpha(/alpha-1) u^{/alpha} v^{-/alpha-1}
/end{array}/right)
$$

which is easy to see to be positive semi-definite. And now, consider any distribution $/pi$, denote $u(/lambda)=/mathbb{P}/left(M_{/lambda}(D)=y/right)$ and $v(/lambda)=/mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right)$ by Jensen's inequality, we have

$$
f/left(/sum_{/lambda} /pi(/lambda) u(/lambda), /sum_{/lambda} /pi(/lambda) v(/lambda)/right) /leq /sum_{/lambda} /pi(/lambda) f(u(/lambda), v(/lambda))
$$

By adding the summation over $y$ on both side of the above inequality, we have

$$
/begin{aligned}
/sum_{y /in /mathcal{Y}} /frac{/left(/sum_{/lambda} /pi(/lambda) /mathbb{P}/left(M_{/lambda}(D)=y/right)/right)^{/alpha}}{/left(/sum_{/lambda} /pi(/lambda) /mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right)/right)^{/alpha-1}} & /leq /sum_{y /in /mathcal{Y}} /sum_{/lambda} /pi(/lambda) /frac{/mathbb{P}/left(M_{/lambda}(D)=y/right)^{/alpha}}{/mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right)^{/alpha-1}} //
& =/sum_{/lambda} /sum_{y /in /mathcal{Y}} /pi(/lambda) /frac{/mathbb{P}/left(M_{/lambda}(D)=y/right)^{/alpha}}{/mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right)^{/alpha-1}} //
& /leq /sup _{/lambda} /sum_{y /in /mathcal{Y}} /frac{/mathbb{P}/left(M_{/lambda}(D)=y/right)^{/alpha}}{/mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right)^{/alpha-1}}
/end{aligned}
$$

The first equality is due to Fubini's theorem. And the second inequality is straight forward as one observe $/pi(/lambda)$ only depends on $/lambda$. This concludes the proof as we know that

$$
/begin{aligned}
e^{(/alpha-1) /mathrm{D}_{/alpha}/left(P^{+} /| P^{/prime-}/right)} & /leq/left(/frac{C}{c}/right)^{/alpha} /sup _{/lambda} /sum_{y /in /mathcal{Y}} /frac{/mathbb{P}/left(M_{/lambda}(D)=y/right)^{/alpha}}{/mathbb{P}/left(M_{/lambda}/left(D^{/prime}/right)=y/right)^{/alpha-1}} //
& =/left(/frac{C}{c}/right)^{/alpha} /sup _{/lambda} e^{(/alpha-1) /mathrm{D}_{/alpha}/left(M_{/lambda}(D) /| M_{/lambda}/left(D^{/prime}/right)/right.}
/end{aligned}
$$

or equivalently,

$$
/mathrm{D}_{/alpha}/left(P^{+} /| P^{/prime-}/right) /leq /frac{/alpha}{/alpha-1} /log /frac{C}{c}+/sup _{/lambda /in /Lambda} /mathrm{D}_{/alpha}/left(M_{/lambda}(D) /| M_{/lambda}/left(D^{/prime}/right)/right)
$$

We now present our crucial technical lemma for adaptive hyperparameter tuing with any distribution on the number of repetitions $T$. This is a generalization from [32].

Lemma A.5. Fix $/alpha>1$. Let $T$ be a random variable supported on $/mathbb{N}_{/geq 0}$. Let $f:[0,1] /rightarrow /mathbb{R}$ be the probability generating function of $K$, that is, $f(x)=/sum_{k=0}^{/infty} /mathbb{P}[T=k] x^{k}$.

Let $M_{/lambda}$ and $M_{/lambda}^{/prime}$ be the base algorithm for $/lambda /in /Lambda$ on $/mathcal{Y}$ on $D$ and $D^{/prime}$ respectively. Define $A_{1}:=$ $/mathcal{A}/left(D, /pi^{(0)}, /mathcal{T}, C, c/right)$, and $A_{2}:=/mathcal{A}/left(D^{/prime}, /pi^{(0)}, /mathcal{T}, C, c/right)$. Then

$$
/mathrm{D}_{/alpha}/left(A_{1} /| A_{2}/right) /leq /sup _{/lambda} /mathrm{D}_{/alpha}/left(M_{/lambda} /| M_{/lambda}^{/prime}/right)+/frac{/alpha}{/alpha-1} /log /frac{C}{c}+/frac{1}{/alpha-1} /log /left(f^{/prime}(q)^{/alpha} /cdot f^{/prime}/left(q^{/prime}/right)^{1-/alpha}/right)
$$

where applying the same postprocessing to the bounding probabilities $P^{+}$and $/mathrm{P}^{/prime-}$ gives probabilities $q$ and $q^{/prime}$ respectively. This means that, there exist a function set $g: /mathcal{Y} /rightarrow[0,1]$ such that $q=$ $/underset{X /leftarrow P^{+}}{/mathbb{E}}[g(X)]$ and $q^{/prime}=/underset{X^{/prime} /leftarrow P^{/prime-}}{/mathbb{E}}/left[g/left(X^{/prime}/right)/right]$.

Proof of Lemma A.5. We consider the event that $A_{1}$ outputs $y$. By definition, we have

$$
/begin{aligned}
& A_{1}(y)=/sum_{k=1}^{/infty} /mathbb{P}(T=k)/left[/prod_{j=1}^{k} P_{j}(/leq y)-/prod_{i=1}^{k} P_{j}(<y)/right] //
& =/sum_{k=1}^{/infty} /mathbb{P}(T=k)/left[/sum_{i=1}^{k} P_{i}(y) /prod_{j=1}^{i-1} P_{j}(<y) /cdot /prod_{j=i+1}^{k} P_{j}(/leq y)/right] //
& /leq /sum_{k=1}^{/infty} /mathbb{P}(T=k)/left[/sum_{i=1}^{k} P^{+}(y) /prod_{j=1}^{i-1} P^{+}(<y) /cdot /prod_{j=i+1}^{k} P^{+}(/leq y)/right] //
& =/sum_{k=1}^{/infty} /mathbb{P}(T=k)/left[/sum_{i=1}^{k} P^{+}(y) /cdot P^{+}(<y)^{i-1} /cdot P^{+}(/leq y)^{k-i}/right] //
& =/sum_{k=1}^{/infty} /mathbb{P}(T=k)/left[P^{+}(/leq y)^{k}-P^{+}(<y)^{k}/right]
/end{aligned}
$$

![](https://cdn.mathpix.com/cropped/2024_08_12_735d0bc2ca9785fb0a60g-17.jpg?height=74&width=1110&top_left_y=944&top_left_x=556)

The second equality is by partitioning on the events of the first time of getting $y$, we use $i$ to index such a time. The third inequality is using (A.7), A.9, and A.10). The third to last equality is by A.12) and algebra. The second to last equality is by definition of the probability generating function $f$. The last equality follows from definition of integral.

Similarly, we have

![](https://cdn.mathpix.com/cropped/2024_08_12_735d0bc2ca9785fb0a60g-17.jpg?height=118&width=1408&top_left_y=1296&top_left_x=364)

The rest part of the proof is standard and follows similarly as in [32]. Specifically, we have

$$
/begin{aligned}
& e^{(/alpha-1) /mathrm{D}_{/alpha}/left(A_{1} /| A_{2}/right)} //
& =/sum_{y /in /mathcal{Y}} A_{1}(y)^{/alpha} /cdot A_{2}(y)^{1-/alpha}
/end{aligned}
$$

![](https://cdn.mathpix.com/cropped/2024_08_12_735d0bc2ca9785fb0a60g-17.jpg?height=101&width=1349&top_left_y=1695&top_left_x=388)

$$
/begin{aligned}
& /leq /sum_{y /in /mathcal{Y}} P^{+}(y)^{/alpha} /cdot P^{/prime-}(y)^{1-/alpha} /cdot /underset{/substack{X /leftarrow/left[P^{+}(<y), P^{+}(/leq y)/right] //
X^{/prime} /leftarrow/left[P^{/prime-}(<y), P^{/prime-}(/leq y)/right]}}{/mathbb{E}}/left[f^{/prime}(X)^{/alpha} /cdot f^{/prime}/left(X^{/prime}/right)^{1-/alpha}/right] //
& /leq/left(/frac{C}{c}/right)^{/alpha} /sup _{/lambda} e^{(/alpha-1) /mathrm{D}_{/alpha}/left(M_{/lambda}(D) /| M_{/lambda}/left(D^{/prime}/right)/right)} /cdot /max _{y /in /mathcal{Y}} /underset{/substack{X /leftarrow/left[P^{+}(<y), P^{+}(/leq y)/right] //
X^{/prime} /leftarrow/left[P^{/prime-}(<y), P^{/prime-}(/leq y)/right]}}{/mathbb{E}}/left[f^{/prime}(X)^{/alpha} /cdot f^{/prime}/left(X^{/prime}/right)^{1-/alpha}/right] .
/end{aligned}
$$

The last inequality follows from Lemma A.4. The second inequality follows from the fact that, for any $/alpha /in /mathbb{R}$, the function $h:(0, /infty)^{2} /rightarrow(0, /infty)$ given by $h(u, v)=u^{/alpha} /cdot v^{1-/alpha}$ is convex. Therefore, $/mathbb{E}[U]^{/alpha} /mathbb{E}[V]^{1-/alpha}=h(/mathbb{E}[(U, V)]) /leq /mathbb{E}[h(U, V)]=/mathbb{E}/left[U^{/alpha} /cdot V^{1-/alpha}/right]$ all positive random variables $(U, V)$. Note that $X$ and $X^{/prime}$ are required to be uniform separately, but their joint distribution can be arbitrary. As in [32], we will couple them so that $/frac{X-P^{+}(<y)}{P^{+}(y)}=/frac{X^{/prime}-P^{/prime-}(<y)}{P^{/prime-}(y)}$. In particular, this implies that, for each $y /in /mathcal{Y}$, there exists some $t /in[0,1]$ such that

$$
/underset{/substack{X /leftarrow/left[P^{+}(<y), P^{+}(/leq y)/right] // X^{/prime} /leftarrow/left[P^{/prime-}(<y), P^{/prime}-(/leq y)/right]}}{/mathbb{E}}/left[f^{/prime}(X)^{/alpha} /cdot f^{/prime}/left(X^{/prime}/right)^{1-/alpha}/right] /leq f^{/prime}/left(P^{+}(<y)+t /cdot P^{+}(y)/right)^{/alpha} /cdot f^{/prime}/left(P^{/prime-}(<y)+t /cdot P^{/prime-}(y)/right)^{1-/alpha}
$$

Therefore, we have

$/mathrm{D}_{/alpha}/left(A_{1} /| A_{2}/right) /leq /sup _{/lambda} /mathrm{D}_{/alpha}/left(M_{/lambda} /| M_{/lambda}^{/prime}/right)+/frac{/alpha}{/alpha-1} /log /frac{C}{c}$

$$
+/frac{1}{/alpha-1} /log /left(/max _{/substack{y /in /mathcal{Y} // t /in[0,1]}} f^{/prime}/left(P^{+}(<y)+t /cdot P^{+}(y)/right)^{/alpha} /cdot f^{/prime}/left(P^{/prime-}(<y)+t /cdot P^{/prime-}(y)/right)^{1-/alpha}/right)
$$

To prove the result, we simply fix $y_{*} /in /mathcal{Y}$ and $t_{*} /in[0,1]$ achieving the maximum above and define

$$
g(y):=/left/{/begin{array}{cl}
1 & /text { if } y<y_{*} //
t_{*} & /text { if } y=y_{*} //
0 & /text { if } y>y_{*}
/end{array}/right.
$$

The result directly follows by setting $q=/underset{X /leftarrow P^{+}}{/mathbb{E}}[g(X)]$ and $q^{/prime}=/underset{X^{/prime} /leftarrow P^{/prime-}}{/mathbb{E}}/left[g/left(X^{/prime}/right)/right]$.

Now we can prove Theorem 1, given the previous technical lemma. The proof share similarity to the proof of Theorem 2 in [32] with the key difference from the different form in Lemma A.5. We demonstrate this proof as follows for completeness.

Proof of Theorem 1 . We first specify the probability generating function of the truncated negative binomial distribution

$$
f(x)=/underset{T /sim /operatorname{NegBin}(/theta, /gamma)}{/mathbb{E}}/left[x^{T}/right]= /begin{cases}/frac{(1-(1-/gamma) x)^{-/theta}-1}{/gamma^{-/theta}-1} & /text { if } /theta /neq 0 // /frac{/log (1-1-/gamma) x)}{/log (/gamma)} & /text { if } /theta=0/end{cases}
$$

Therefore,

$$
/begin{aligned}
f^{/prime}(x) & =(1-(1-/gamma) x)^{-/theta-1} /cdot /begin{cases}/frac{/theta /cdot(1-/gamma)}{/gamma-/theta-1} & /text { if } /theta /neq 0 //
/frac{1-/gamma}{/log (1 / /gamma)} & /text { if } /theta=0/end{cases} //
& =(1-(1-/gamma) x)^{-/theta-1} /cdot /gamma^{/theta+1} /cdot /mathbb{E}[T]
/end{aligned}
$$

By Lemma A.5, for appropriate values $q, q^{/prime} /in[0,1]$ and for all $/alpha>1$ and all $/hat{/alpha}>1$, we have

$$
/begin{aligned}
& /mathrm{D}_{/alpha}/left(A_{1} /| A_{2}/right) //
& /leq /sup _{/lambda} /mathrm{D}_{/alpha}/left(M_{/lambda} /| M_{/lambda}^{/prime}/right)+/frac{/alpha}{/alpha-1} /log /frac{C}{c}+/frac{1}{/alpha-1} /log /left(f^{/prime}(q)^{/alpha} /cdot f^{/prime}/left(q^{/prime}/right)^{1-/alpha}/right) //
& /leq /varepsilon+/frac{/alpha}{/alpha-1} /log /frac{C}{c}+/frac{1}{/alpha-1} /log /left(/gamma^{/theta+1} /cdot /mathbb{E}[T] /cdot(1-(1-/gamma) q)^{-/alpha(/theta+1)} /cdot/left(1-(1-/gamma) q^{/prime}/right)^{-(1-/alpha)(/theta+1)}/right) //
& =/varepsilon+/frac{/alpha}{/alpha-1} /log /frac{C}{c} //
& /quad+/frac{1}{/alpha-1} /log /left(/gamma^{/theta+1} /cdot /mathbb{E}[T] /cdot/left((/gamma+(1-/gamma)(1-q))^{1-/hat{/alpha}} /cdot/left(/gamma+(1-/gamma)/left(1-q^{/prime}/right)/right)^{/hat{/alpha}}/right)^{/nu} /cdot(/gamma+(1-/gamma)(1-q))^{u}/right)
/end{aligned}
$$

(Here, we let $/hat{/alpha} /nu=(/alpha-1)(1+/theta)$ and $(1-/hat{/alpha}) /nu+u=-/alpha(/theta+1)$ )

$/leq /varepsilon+/frac{/alpha}{/alpha-1} /log /frac{C}{c}+/frac{1}{/alpha-1} /log /left(/gamma^{/theta+1} /cdot /mathbb{E}[T] /cdot/left(/gamma+(1-/gamma) /cdot e^{(/hat{/alpha}-1) /mathrm{D}_{/hat{/alpha}}/left(P^{+} /| P^{-}/right)}/right)^{/nu} /cdot(/gamma+(1-/gamma)(1-q))^{u}/right)$

(Here, $1-q$ and $1-q^{/prime}$ are postprocessings of some $P^{+}$and $P^{/prime-}$ respectively and $e^{(/hat{/alpha}-1) /mathrm{D}_{/hat{/alpha}}(/cdot /| /cdot)}$ is convex )

$/leq /varepsilon+/frac{/alpha}{/alpha-1} /log /frac{C}{c}+/frac{1}{/alpha-1} /log /left(/gamma^{/theta+1} /cdot /mathbb{E}[T] /cdot/left(/gamma+(1-/gamma) /cdot e^{(/hat{/alpha}-1) /sup _{/lambda} /mathrm{D}_{/hat{/alpha}}/left(M_{/lambda} /| M_{/lambda}^{/prime}/right)+/hat{/alpha} /log /frac{C}{c}}/right)^{/nu} /cdot(/gamma+(1-/gamma)(1-q))^{u}/right)$ (By Lemma A.4

![](https://cdn.mathpix.com/cropped/2024_08_12_735d0bc2ca9785fb0a60g-19.jpg?height=88&width=1752&top_left_y=1062&top_left_x=371)

![](https://cdn.mathpix.com/cropped/2024_08_12_735d0bc2ca9785fb0a60g-19.jpg?height=87&width=1457&top_left_y=1149&top_left_x=372)

$($ Here $/gamma /leq /gamma+(1-/gamma)(1-q)$ and $u /leq 0)$

$$
/begin{aligned}
= & /varepsilon+/frac{/alpha}{/alpha-1} /log /frac{C}{c}+/frac{/nu}{/alpha-1} /log /left(/gamma+(1-/gamma) /cdot e^{(/hat{/alpha}-1) /sup _{/lambda} /mathrm{D}_{/hat{/alpha}}/left(M_{/lambda} /| M_{/lambda}^{/prime}/right)+/hat{/alpha} /log /frac{C}{c}}/right)+/frac{1}{/alpha-1} /log /left(/gamma^{/theta+1} /cdot /mathbb{E}[T] /cdot /gamma^{u}/right) //
= & /varepsilon+/frac{/alpha}{/alpha-1} /log /frac{C}{c}+/frac{/nu}{/alpha-1}/left((/hat{/alpha}-1) /sup _{/lambda} /mathrm{D}_{/hat{/alpha}}/left(M_{/lambda} /| M_{/lambda}^{/prime}/right)+/hat{/alpha} /log /frac{C}{c}/right. //
& /left.+/log /left(1-/gamma /cdot/left(1-e^{-(/hat{/alpha}-1) /sup _{/lambda} /mathrm{D}_{/hat{/alpha}}/left(M_{/lambda} /| M_{/lambda}^{/prime}/right)+/hat{/alpha} /log /frac{C}{c}}/right)/right)/right)+/frac{1}{/alpha-1} /log /left(/gamma^{u+/theta+1} /cdot /mathbb{E}[T]/right) //
= & /varepsilon+/frac{/alpha}{/alpha-1} /log /frac{C}{c}+(1+/theta)/left(1-/frac{1}{/hat{/alpha}}/right) /sup _{/lambda} /mathrm{D}_{/hat{/alpha}}/left(M_{/lambda} /| M_{/lambda}^{/prime}/right)+(1+/theta) /log /frac{C}{c} //
& +/frac{1+/theta}{/hat{/alpha}} /log /left(1-/gamma /cdot/left(1-e^{-(/hat{/alpha}-1) /sup _{/lambda} /mathrm{D}_{/hat{/alpha}}/left(M_{/lambda} /| M_{/lambda}^{/prime}/right)+/hat{/alpha} /log /frac{C}{c}}/right)/right)+/frac{/log (/mathbb{E}[T])}{/alpha-1}+/frac{1+/theta}{/hat{/alpha}} /log (1 / /gamma)
/end{aligned}
$$

(Here we have $/nu=/frac{(/alpha-1)(1+/theta)}{/hat{/alpha}}$ and $u=-(1+/theta)/left(/frac{/alpha-1}{/hat{/alpha}}+1/right)$ )

$$
=/varepsilon+/frac{/alpha}{/alpha-1} /log /frac{C}{c}+(1+/theta)/left(1-/frac{1}{/hat{/alpha}}/right) /sup _{/lambda} /mathrm{D}_{/hat{/alpha}}/left(M_{/lambda} /| M_{/lambda}^{/prime}/right)+(1+/theta) /log /frac{C}{c}
$$

$$
+/frac{1+/theta}{/hat{/alpha}} /log /left(/frac{1}{/gamma}-1+e^{-(/hat{/alpha}-1) /sup _{/lambda} /mathrm{D}_{/hat{/alpha}}/left(M_{/lambda} /| M_{/lambda}^{/prime}/right)-/hat{/alpha} /log /frac{C}{c}}/right)+/frac{/log (/mathbb{E}[T])}{/alpha-1}
$$

$/leq /varepsilon+/frac{/alpha}{/alpha-1} /log /frac{C}{c}+(1+/theta)/left(1-/frac{1}{/hat{/alpha}}/right) /hat{/varepsilon}+(1+/theta) /log /frac{C}{c}+/frac{1+/theta}{/hat{/alpha}} /log /left(/frac{1}{/gamma}/right)+/frac{/log (/mathbb{E}[T])}{/alpha-1}$,

which completes the proof.

## B Truncated Negative Binomial Distribution

We introduce the definition of truncated negative binomial distribution [32] in this section.

Definition B.1. (Truncated Negative Binomial Distribution [32]). Let $/gamma /in(0,1)$ and $/theta /in(-1, /infty)$.

Define a distribution $/operatorname{NegBin}(/theta, /gamma)$ on $/mathbb{N}^{+}$as follows:

- If $/theta /neq 0$ and $T$ is drawn from $/operatorname{NegBin}(/theta, /gamma)$, then

$$
/forall k /in /mathbb{N} /quad /mathbb{P}[T=k]=/frac{(1-/gamma)^{k}}{/gamma^{-/theta}-1} /cdot /prod_{/ell=0}^{k-1}/left(/frac{/ell+/theta}{/ell+1}/right)
$$

and $/mathbb{E}[T]=/frac{/theta /cdot(1-/gamma)}{/gamma /cdot/left(1-/gamma^{/theta}/right)}$. Note that when $/theta=1$, it reduces to the geometric distribution with parameter $/gamma$.

- If $/theta=0$ and $T$ is drawn from $/operatorname{NegBin}(0, /gamma)$, then

$$
/mathbb{P}[T=k]=/frac{(1-/gamma)^{k}}{k /cdot /log (1 / /gamma)}
$$

and $/mathbb{E}[T]=/frac{1 / /gamma-1}{/log (1 / /gamma)}$.

## C Privatization of Sampling Distribution

## C. 1 General Functional Projection Framework

In section 3.2, we define the projection onto a convex set $S_{C, c}$ as an optimization in terms of $/ell_{2}$ loss. More generally, we can perform the following general projection at the $j$-th iteration by considering an additional penalty term, with a constant $/nu$ :

$$
/begin{align*}
/min _{f} & /left/|f-/pi^{(j)}/right/|_{2}+/nu K L/left(/pi^{(j)}, f/right)  /tag{C.1}//
/text { s.t. } & f /in S_{C, c} .
/end{align*}
$$

When $/nu=0$, we recover the original $/ell_{2}$ projection. Moreover, it's worth noting that our formulation has implications for the information projection literature [9, 21]. Specifically, as the penalty term parameter $/nu$ approaches infinity, the optimization problem evolves into a minimization of KL divergence, recovering the objective function of information projection (in this instance, moment projection). However, the constraint sets in the literature of information projection are generally much simpler than our set $S_{C, c}$, making it infeasible to directly borrow methods from its field. To the best of our knowledge, our framework is the first to address this specific problem in functional projection and establish a connection to information projection in the DP community.

## C. 2 Practical Implementation of Functional Projection

Optimization program $/sqrt{3.1}$ is essentially a functional programming since $f$ is a function on $/Lambda$. However, when $/Lambda$ represents a non-discrete parameter space, such functional minimization is typically difficult to solve analytically. Even within the literature of information projection, none of the methods considers our constraint set $S_{C, c}$, which can be viewed as the intersections of uncountable single-point constraints on $f$. To obtain a feasible solution to the optimization problem, we leverage the idea of discretization. Instead of viewing (3.1) as a functional projection problem, we manually discretize $/Lambda$ and solve 3.1) as a minimization problem over a discrete set. Note that such approximation is unavoidable in numerical computations since computers can only manage discrete functions, even when we solve the functional projection analytically. Moreover, we also have the freedom of choosing the discretization grid without incurring extra privacy loss since the privacy cost is independent of the size of parameter space. By converting $S_{C, c}$ into a set of finite constraints, we are able to solve the discrete optimization problem efficiently using CVXOPT [2].

## D DP-HyPO with General Prior Distribution

In the main manuscript, we assume $/pi^{(0)}$ follows a uniform distribution over the parameter space $/Lambda$ for simplicity. In practice, informed priors can be used when we want to integrate knowledge about the parameter space into sampling distribution, which is common in the Bayesian optimization framework. We now present the general DP-HyPO framework under the informed prior distribution.

To begin with, we define the space of essentially bounded density functions with respect to $/pi^{(0)}$ as

$$
S_{C, c}/left(/pi^{(0)}/right)=/left/{f /in /Lambda^{/mathbb{R}^{+}}: /operatorname{ess} /sup f / /pi^{(0)} /leq C, /operatorname{ess} /inf f / /pi^{(0)} /geq c, /int_{/alpha /in /Lambda} f(/alpha) /mathrm{d} /alpha=1, f /ll /pi^{(0)}/right/}
$$

When $/pi^{(0)}=/frac{1}{/mu(/lambda)}$, we recover the original definition of $S_{C, c}$. Note that here $f /ll /pi^{(0)}$ means that $f$ is absolute continuous with respect to the prior distribution $/pi^{(0)}$ and this ensures that $S_{C, c}/left(/pi^{(0)}/right)$ is non-empty. Note that such condition is automatically satisfied when $/pi^{(0)}$ is the uniform prior over the entire parameter space.

To define the projection of a density at the $j$-th iteration, $/pi^{(j)}$, into the space $S_{C, c}/left(/pi^{(0)}/right)$, we consider the following functional programming problem:

$$
/begin{aligned}
/min _{f} & /left/|f-/pi^{(j)}/right/|_{2} //
/text { s.t. } & f /in S_{C, c}/left(/pi^{(0)}/right)
/end{aligned}
$$

which is a direct generalization of Equation 3.1). As before, $S_{C, c}/left(/pi^{(0)}/right)$ is also convex and closed and the optimization program can be solved efficiently via discretization on $/Lambda$.

## E Experiment Details

## E. 1 MNIST Simulation

We now provide the detailed description of the experiment in Section 4.2.1. As specified therein, we consider two variable hyperparameters: the learning rate $/eta$ and clipping norm $R$, while keeping all the other hyperparameters fixed. We set the training batch size to be 256, and the total number of epoch to be 10. The value of $/sigma$ is determined based on the allocated $/varepsilon$ budget for each base algorithm. Specifically, $/sigma=0.71$ for GP and $/sigma=0.64$ for Uniform. For demonstration purposes, we set $C$ to 2 and $c$ to 0.75 in the GP method, so each base algorithm of Uniform has $/log C / c$ more privacy budget than base algorithms in GP method. In Algorithm 3, we set $/tau$ to 0.1 and $/beta$ to 1 . To facilitate the implementation of both methods, we discretize the learning rates and clipping norms as specified in the following setting to allow simple implementation of sampling and projection for Uniform and GP methods.

Setting E.1. we set a log-spaced grid discretization on $/eta$ in the range $[0.0001,10]$ with a multiplicative factor of $/sqrt[3]{10}$, resulting in 16 observations for $/eta$. We also set a linear-spaced grid discretization on $R$ in the range $[0.3,6]$ with an increment of 0.3 , resulting in 20 observations for $R$. This gives a total of 320 hyperparameters over the search region.

We specify the network structure we used in the simulation as below. It is the standard CNN in Tensorflow Privacy and Opacus

```
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32*4*4, 32)
        self.fc2 = nn.Linear(32, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 1)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 1)
        x = x.view(-1, 32*4*4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

Despite the simple nature of MNIST, the simulation of training CNN with the two methods over each different fixed $T$ still take significant computation resources. Due to the constraints on computational resources, we conduct a semi-real simulation using the MNIST dataset. We cache the mean accuracy of 5 independently trained models for each discretized hyperparameter and treat that as a proxy for the
"actual accuracy" of the hyperparameter. Each time we sample the accuracy of a hyperparameter, we add Gaussian noise with a standard deviation of 0.1 to the cached mean. We evaluate the performance of the output model based on the "actual accuracy" corresponding to the selected hyperparameter.

## E. 2 CIFAR-10 Simulation

We also provide a description of the experiment in Section 4.2.2. We set the training batch size to be 256 , and the total number of epoch to be 10 . The value of $/sigma$ is determined based on the allocated $/varepsilon$ budget for each base algorithm. Specifically, $/sigma=0.65$ for GP and $/sigma=0.6$ for Uniform. Regarding our GP method, we adopt the same set of hyperparameters as used in our MNIST experiments, which include $C=2, c=0.75, /tau=0.1$, and $/beta=1$. As usual, we discretize the learning rates and clipping norms as specified in the following Setting.

Setting E.2. we set a log-spaced grid discretization on $/eta$ in the range $[0.0001,1]$ with a multiplicative factor of $10^{0.1}$, resulting in 50 observations for $/eta$. We also set a linear-spaced grid discretization on $R$ in the range $[0,100]$ with an increment of 2 , resulting in 50 observations for $R$. This gives a total of 2500 hyperparameter combinations over the search region.

We follow the same CNN model architecture with our MNIST experiments.

In Figure 2, we provide the hyperparameter landscape for $/sigma=0.65$, as generated by BoTorch [3].

Figure 2: Mean and standard error of the accuracy of DP-SGD over the two hyperparameters for $/sigma=0.65$. The learning rate (log-scale) ranges from 0.00001 (left) to 1 (right) while the clipping norm ranges from 0 (top) to 100 (bottom). The landscape for $/sigma=0.6$ is similar, with a better accuracy.

## E. 3 Federated Learning Simulation

![](https://cdn.mathpix.com/cropped/2024_08_12_735d0bc2ca9785fb0a60g-22.jpg?height=362&width=895&top_left_y=1895&top_left_x=604)

Figure 3: Mean and Standard Error of the loss of the FL over the two hyperparameters.

We now provide the detailed description of the experiment in Section 4.2.3. As specified therein, we considered a FL task on a proprietary datase ${ }^{5}$ Our objective is to determine the optimal learning

/footnotetext{
${ }^{5} /mathrm{We}$ are unable to report a lot of detail about the proprietary dataset due to confidentiality.
rates for the central server (using AdaGrad) and the individual users (using SGD). To simulate this scenario, we utilize the landscape generated by BoTorch [3], as illustrated in Figure 33 and consider it as our reference landscape for both mean and standard deviation of the loss for each hyperparameter. When the algorithm (GP or Uniform) visits a specific hyperparameter $/lambda$, our oracle returns a noisy score $q(/lambda)$ drawn from a normal distribution $N/left(/mu_{/lambda}, /sigma_{/lambda}/right)$. Figure 3 displays a heatmap that presents the mean $/left(/mu_{/lambda}/right)$ and standard error $/left(/sigma_{/lambda}/right)$ structure of the loss over these two hyperparameters, providing insights into the landscape's characteristics.

## F Additional Related Work

In this section, we delve into a more detailed review of the pertinent literature.

We begin with non-private Hyperparameter Optimization, a critical topic in the realm of Automated Machine Learning (AutoML) [16]. The fundamental inquiry revolves around the generation of highperforming models within a specific search space. In historical context, two types of optimizations have proven significant in addressing this inquiry: architecture optimization and hyperparameter optimization. Architecture optimization pertains to model-specific parameters such as the number of neural network layers and their interconnectivity, while hyperparameter optimization concerns training-specific parameters, including the learning rate and minibatch size. In our paper, we incorporate both types of optimizations within our HPO framework. Practically speaking, $/Lambda$ can encompass various learning rates and network architectures for selection. For HPO, elementary methods include grid search and random search [22, 17, 15]. Progressing beyond non-adaptive random approaches, surrogate model-based optimization presents an adaptive method, leveraging information from preceding results to construct a surrogate model of the objective function [26, 41, 20, 30]. These methods predominantly employs Bayesian optimization techniques, including Gaussian process [33], Random Forest [18], and tree-structured Parzen estimator [5].

Another important topic in this paper is Differential Privacy (DP). DP offers a mathematically robust framework for measuring privacy leakage. A DP algorithm promises that an adversary with perfect information about the entire private dataset in use - except for a single individual - would find it hard to distinguish between its presence or absence based on the output of the algorithm [12].

Historically, DP machine learning research has overlooked the privacy cost associated with HPO [1, 39, 42]. The focus has only recently shifted to the "honest HPO" setting, where this cost is factored in [28]. Addressing this issue directly involves employing a composition-based analysis. If each training run of a hyperparameter upholds DP, then the overall HPO procedure adheres to DP through composition across all attempted hyperparameter values. A plethora of literature on the composition of DP mechanisms attempts to quantify a better DP guarantee of the composition. Vadhan et al. [36] demonstrated that though $(/varepsilon, /delta)$-DP possesses a simple mathematical form, deriving the precise privacy parameters of a composition is /#-P hard. Despite this obstacle, numerous advanced techniques are available to calculate a reasonably accurate approximation of the privacy parameters, such as Moments Accountant [1], GDP Accountant [11], and Edgeworth Accountant [37]. The efficacy of these accountants is attributed to the fact that it is easier to reason about the privacy guarantees of compositions within the framework of Rényi differential privacy [27] or $f$-differential privacy [11]. These methods have found widespread application in DP machine learning. For instance, when training deep learning models, one of the most commonly adopted methods to ensure DP is via noisy stochastic gradient descent (noisy SGD) [4, 35], which uses Moments Accountant to better quantify the privacy guarantee.

Although using composition for HPO is a simple and straightforward approach, it carries with it a significant challenge. The privacy guarantee derived from composition accounting can be excessively loose, scaling polynomially with the number of runs. Chaudhuri et al. [7] were the first to enhance the DP bounds for HPO by introducing additional stability assumptions on the learning algorithms. [24] made significant progress in enhancing DP bounds for HPO without relying on any stability properties of the learning algorithms. They proposed a simple procedure where a hyperparameter was randomly selected from a uniform distribution for each training run. This selection process was repeated a random number of times according to a geometric distribution, and the best model obtained from these runs was outputted. They showed that this procedure satisfied $(3 /varepsilon, 0)$-DP as long as each training run of a hyperparameter was $(/varepsilon, 0)$-DP. Building upon this, [32] extended the procedure to accommodate negative binomial or Poisson distributions for the repeated uniform selection. They also
offered more precise Rényi DP guarantees for this extended procedure. Furthermore, [8] explored a generalization of the procedure for top- $k$ selection, considering $(/varepsilon, /delta)$-DP guarantees.

In a related context, [28] explored a setting that appeared superficially similar to ours, as their title mentioned "adaptivity." However, their primary focus was on improving adaptive optimizers such as DP-Adam, which aimed to reduce the necessity of hyperparameter tuning, rather than the adaptive HPO discussed in this paper. Notably, in terms of privacy accounting, their approach only involved composing the privacy cost of each run without proposing any new method.

Another relevant area of research is DP selection, which encompasses well-known methods such as the exponential mechanism [25] and the sparse vector technique [13], along with subsequent studies. However, this line of research always assumes the existence of a low-sensitivity score function for each candidate, which is an unrealistic assumption for hyperparameter optimization.


[^0]:    ${ }^{3}$ The $/varepsilon$ values are seemingly very large. Nonetheless, the reported privacy budget encompasses the overall cost of the entire HPO, which is typically overlooked in the existing literature. Given that HPO roughly incurs three times the privacy cost of the base algorithm, an $/varepsilon$ as high as 15 could be reported as only 5 in many other works.

[^1]:    ${ }^{4}$ We have to respect confidentiality constraints that limit our ability to provide extensive details about this dataset.
          
# A Differential Equation for Modeling Nesterov's Accelerated Gradient Method: Theory and Insights 

Weijie Su

SUW@WHARTON.UPENN.EDU

Department of Statistics

University of Pennsylvania

Philadelphia, PA 19104, USA

Stephen Boyd

BOYD@STANFORD.EDU

Department of Electrical Engineering

Stanford University

Stanford, CA 94305, USA

Emmanuel J. Candès

CANDES@STANFORD.EDU

Departments of Statistics and Mathematics

Stanford University

Stanford, CA 94305, USA

Editor: Yoram Singer


#### Abstract

We derive a second-order ordinary differential equation (ODE) which is the limit of Nesterov's accelerated gradient method. This ODE exhibits approximate equivalence to Nesterov's scheme and thus can serve as a tool for analysis. We show that the continuous time ODE allows for a better understanding of Nesterov's scheme. As a byproduct, we obtain a family of schemes with similar convergence rates. The ODE interpretation also suggests restarting Nesterov's scheme leading to an algorithm, which can be rigorously proven to converge at a linear rate whenever the objective is strongly convex.


Keywords: Nesterov's accelerated scheme, convex optimization, first-order methods, differential equation, restarting

## 1. Introduction

In many fields of machine learning, minimizing a convex function is at the core of efficient model estimation. In the simplest and most standard form, we are interested in solving

$$
/operatorname{minimize} f(x)
$$

where $f$ is a convex function, smooth or non-smooth, and $x /in /mathbb{R}^{n}$ is the variable. Since Newton, numerous algorithms and methods have been proposed to solve the minimization problem, notably gradient and subgradient descent, Newton's methods, trust region methods, conjugate gradient methods, and interior point methods (see e.g. Polyak, 1987; Boyd and Vandenberghe, 2004; Nocedal and Wright, 2006; Ruszczyński, 2006; Boyd et al., 2011; Shor, 2012; Beck, 2014, for expositions).

First-order methods have regained popularity as data sets and problems are ever increasing in size and, consequently, there has been much research on the theory and practice
of accelerated first-order schemes. Perhaps the earliest first-order method for minimizing a convex function $f$ is the gradient method, which dates back to Euler and Lagrange. Thirty years ago, however, in a seminal paper Nesterov proposed an accelerated gradient method (Nesterov, 1983), which may take the following form: starting with $x_{0}$ and $y_{0}=x_{0}$, inductively define

$$
/begin{align*}
& x_{k}=y_{k-1}-s /nabla f/left(y_{k-1}/right) //
& y_{k}=x_{k}+/frac{k-1}{k+2}/left(x_{k}-x_{k-1}/right) /tag{1}
/end{align*}
$$

For any fixed step size $s /leq 1 / L$, where $L$ is the Lipschitz constant of $/nabla f$, this scheme exhibits the convergence rate

$$
/begin{equation*}
f/left(x_{k}/right)-f^{/star} /leq O/left(/frac{/left/|x_{0}-x^{/star}/right/|^{2}}{s k^{2}}/right) /tag{2}
/end{equation*}
$$

Above, $x^{/star}$ is any minimizer of $f$ and $f^{/star}=f/left(x^{/star}/right)$. It is well-known that this rate is optimal among all methods having only information about the gradient of $f$ at consecutive iterates (Nesterov, 2004). This is in contrast to vanilla gradient descent methods, which have the same computational complexity but can only achieve a rate of $O(1 / k)$. This improvement relies on the introduction of the momentum term $x_{k}-x_{k-1}$ as well as the particularly tuned coefficient $(k-1) /(k+2) /approx 1-3 / k$. Since the introduction of Nesterov's scheme, there has been much work on the development of first-order accelerated methods, see Nesterov $(2004,2005,2013)$ for theoretical developments, and Tseng (2008) for a unified analysis of these ideas. Notable applications can be found in sparse linear regression (Beck and Teboulle, 2009; Qin and Goldfarb, 2012), compressed sensing (Becker et al., 2011) and, deep and recurrent neural networks (Sutskever et al., 2013).

In a different direction, there is a long history relating ordinary differential equation (ODEs) to optimization, see Helmke and Moore (1996), Schropp and Singer (2000), and Fiori (2005) for example. The connection between ODEs and numerical optimization is often established via taking step sizes to be very small so that the trajectory or solution path converges to a curve modeled by an ODE. The conciseness and well-established theory of ODEs provide deeper insights into optimization, which has led to many interesting findings. Notable examples include linear regression via solving differential equations induced by linearized Bregman iteration algorithm (Osher et al., 2014), a continuous-time Nesterov-like algorithm in the context of control design (Dürr and Ebenbauer, 2012; Dürr et al., 2012), and modeling design iterative optimization algorithms as nonlinear dynamical systems (Lessard et al., 2014).

In this work, we derive a second-order ODE which is the exact limit of Nesterov's scheme by taking small step sizes in (1); to the best of our knowledge, this work is the first to use ODEs to model Nesterov's scheme or its variants in this limit. One surprising fact in connection with this subject is that a first-order scheme is modeled by a second-order ODE. This ODE takes the following form:

$$
/begin{equation*}
/ddot{X}+/frac{3}{t} /dot{X}+/nabla f(X)=0 /tag{3}
/end{equation*}
$$

for $t>0$, with initial conditions $X(0)=x_{0}, /dot{X}(0)=0$; here, $x_{0}$ is the starting point in Nesterov's scheme, $/dot{X} /equiv /mathrm{d} X / /mathrm{d} t$ denotes the time derivative or velocity and similarly
$/ddot{X} /equiv /mathrm{d}^{2} X / /mathrm{d} t^{2}$ denotes the acceleration. The time parameter in this ODE is related to the step size in (1) via $t /approx k /sqrt{s}$. Expectedly, it also enjoys inverse quadratic convergence rate as its discrete analog,

$$
f(X(t))-f^{/star} /leq O/left(/frac{/left/|x_{0}-x^{/star}/right/|^{2}}{t^{2}}/right)
$$

Approximate equivalence between Nesterov's scheme and the ODE is established later in various perspectives, rigorous and intuitive. In the main body of this paper, examples and case studies are provided to demonstrate that the homogeneous and conceptually simpler ODE can serve as a tool for understanding, analyzing and generalizing Nesterov's scheme.

In the following, two insights of Nesterov's scheme are highlighted, the first one on oscillations in the trajectories of this scheme, and the second on the peculiar constant 3 appearing in the ODE.

### 1.1 From Overdamping to Underdamping

In general, Nesterov's scheme is not monotone in the objective function value due to the introduction of the momentum term. Oscillations or overshoots along the trajectory of iterates approaching the minimizer are often observed when running Nesterov's scheme. Figure 1 presents typical phenomena of this kind, where a two-dimensional convex function is minimized by Nesterov's scheme. Viewing the ODE as a damping system, we obtain interpretations as follows.

Small $t$. In the beginning, the damping ratio $3 / t$ is large. This leads the ODE to be an overdamped system, returning to the equilibrium without oscillating;

Large $t$. As $t$ increases, the ODE with a small $3 / t$ behaves like an underdamped system, oscillating with the amplitude gradually decreasing to zero.

As depicted in Figure 1a, in the beginning the ODE curve moves smoothly towards the origin, the minimizer $x^{/star}$. The second interpretation "Large $t$ " provides partial explanation for the oscillations observed in Nesterov's scheme at later stage. Although our analysis extends farther, it is similar in spirit to that carried in O'Donoghue and Candès (2013). In particular, the zoomed Figure 1b presents some butterfly-like oscillations for both the scheme and ODE. There, we see that the trajectory constantly moves away from the origin and returns back later. Each overshoot in Figure 1b causes a bump in the function values, as shown in Figure 1c. We observe also from Figure 1c that the periodicity captured by the bumps are very close to that of the ODE solution. In passing, it is worth mentioning that the solution to the ODE in this case can be expressed via Bessel functions, hence enabling quantitative characterizations of these overshoots and bumps, which are given in full detail in Section 3.

### 1.2 A Phase Transition

The constant 3, derived from $(k+2)-(k-1)$ in (3), is not haphazard. In fact, it is the smallest constant that guarantees $O/left(1 / t^{2}/right)$ convergence rate. Specifically, parameterized by a constant $r$, the generalized ODE

$$
/ddot{X}+/frac{r}{t} /dot{X}+/nabla f(X)=0
$$

![](https://cdn.mathpix.com/cropped/2024_08_12_d19cd0b096d3ee24c53cg-04.jpg?height=481&width=1529&top_left_y=291&top_left_x=298)

Figure 1: Minimizing $f=2 /times 10^{-2} x_{1}^{2}+5 /times 10^{-3} x_{2}^{2}$, starting from $x_{0}=(1,1)$. The black and solid curves correspond to the solution to the ODE. In (c), for the x-axis we use the identification between time and iterations, $t=k /sqrt{s}$.

can be translated into a generalized Nesterov's scheme that is the same as the original (1) except for $(k-1) /(k+2)$ being replaced by $(k-1) /(k+r-1)$. Surprisingly, for both generalized ODEs and schemes, the inverse quadratic convergence is guaranteed if and only if $r /geq 3$. This phase transition suggests there might be deep causes for acceleration among first-order methods. In particular, for $r /geq 3$, the worst case constant in this inverse quadratic convergence rate is minimized at $r=3$.

Figure 2 illustrates the growth of $t^{2}/left(f(X(t))-f^{/star}/right)$ and $s k^{2}/left(f/left(x_{k}/right)-f^{/star}/right)$, respectively, for the generalized ODE and scheme with $r=1$, where the objective function is simply $f(x)=/frac{1}{2} x^{2}$. Inverse quadratic convergence fails to be observed in both Figures 2a and 2b, where the scaled errors grow with $t$ or iterations, for both the generalized ODE and scheme.

![](https://cdn.mathpix.com/cropped/2024_08_12_d19cd0b096d3ee24c53cg-04.jpg?height=494&width=1533&top_left_y=1496&top_left_x=296)

Figure 2: Minimizing $f=/frac{1}{2} x^{2}$ by the generalized ODE and scheme with $r=1$, starting from $x_{0}=1$. In (b), the step size $s=10^{-4}$.

### 1.3 Outline and Notation

The rest of the paper is organized as follows. In Section 2, the ODE is rigorously derived from Nesterov's scheme, and a generalization to composite optimization, where $f$ may be non-smooth, is also obtained. Connections between the ODE and the scheme, in terms of trajectory behaviors and convergence rates, are summarized in Section 3. In Section

4, we discuss the effect of replacing the constant 3 in (3) by an arbitrary constant on the convergence rate. A new restarting scheme is suggested in Section 5, with linear convergence rate established and empirically observed.

Some standard notations used throughout the paper are collected here. We denote by $/mathcal{F}_{L}$ the class of convex functions $f$ with $L$-Lipschitz continuous gradients defined on $/mathbb{R}^{n}$, i.e., $f$ is convex, continuously differentiable, and satisfies

$$
/|/nabla f(x)-/nabla f(y)/| /leq L/|x-y/|
$$

for any $x, y /in /mathbb{R}^{n}$, where $/|/cdot/|$ is the standard Euclidean norm and $L>0$ is the Lipschitz constant. Next, $/mathcal{S}_{/mu}$ denotes the class of $/mu$-strongly convex functions $f$ on $/mathbb{R}^{n}$ with continuous gradients, i.e., $f$ is continuously differentiable and $f(x)-/mu/|x/|^{2} / 2$ is convex. We set $/mathcal{S}_{/mu, L}=$ $/mathcal{F}_{L} /cap /mathcal{S}_{/mu}$.

## 2. Derivation

First, we sketch an informal derivation of the ODE (3). Assume $f /in /mathcal{F}_{L}$ for $L>0$. Combining the two equations of (1) and applying a rescaling gives

$$
/begin{equation*}
/frac{x_{k+1}-x_{k}}{/sqrt{s}}=/frac{k-1}{k+2} /frac{x_{k}-x_{k-1}}{/sqrt{s}}-/sqrt{s} /nabla f/left(y_{k}/right) . /tag{4}
/end{equation*}
$$

Introduce the Ansatz $x_{k} /approx X(k /sqrt{s})$ for some smooth curve $X(t)$ defined for $t /geq 0$. Put $k=t / /sqrt{s}$. Then as the step size $s$ goes to zero, $X(t) /approx x_{t / /sqrt{s}}=x_{k}$ and $X(t+/sqrt{s}) /approx$ $x_{(t+/sqrt{s}) / /sqrt{s}}=x_{k+1}$, and Taylor expansion gives

$/left(x_{k+1}-x_{k}/right) / /sqrt{s}=/dot{X}(t)+/frac{1}{2} /ddot{X}(t) /sqrt{s}+o(/sqrt{s}), /quad/left(x_{k}-x_{k-1}/right) / /sqrt{s}=/dot{X}(t)-/frac{1}{2} /ddot{X}(t) /sqrt{s}+o(/sqrt{s})$

and $/sqrt{s} /nabla f/left(y_{k}/right)=/sqrt{s} /nabla f(X(t))+o(/sqrt{s})$. Thus (4) can be written as

$$
/begin{align*}
/dot{X}(t)+/frac{1}{2} /ddot{X}(t) /sqrt{s} & +o(/sqrt{s}) //
& =/left(1-/frac{3 /sqrt{s}}{t}/right)/left(/dot{X}(t)-/frac{1}{2} /ddot{X}(t) /sqrt{s}+o(/sqrt{s})/right)-/sqrt{s} /nabla f(X(t))+o(/sqrt{s}) /tag{5}
/end{align*}
$$

By comparing the coefficients of $/sqrt{s}$ in (5), we obtain

$$
/ddot{X}+/frac{3}{t} /dot{X}+/nabla f(X)=0
$$

The first initial condition is $X(0)=x_{0}$. Taking $k=1$ in (4) yields

$$
/left(x_{2}-x_{1}/right) / /sqrt{s}=-/sqrt{s} /nabla f/left(y_{1}/right)=o(1) /text {. }
$$

Hence, the second initial condition is simply $/dot{X}(0)=0$ (vanishing initial velocity).

One popular alternative momentum coefficient is $/theta_{k}/left(/theta_{k-1}^{-1}-1/right)$, where $/theta_{k}$ are iteratively defined as $/theta_{k+1}=/left(/sqrt{/theta_{k}^{4}+4 /theta_{k}^{2}}-/theta_{k}^{2}/right) / 2$, starting from $/theta_{0}=1$ (Nesterov, 1983; Beck and

Teboulle, 2009). Simple analysis reveals that $/theta_{k}/left(/theta_{k-1}^{-1}-1/right)$ asymptotically equals $1-3 / k+$ $O/left(1 / k^{2}/right)$, thus leading to the same ODE as (1).

Classical results in ODE theory do not directly imply the existence or uniqueness of the solution to this ODE because the coefficient $3 / t$ is singular at $t=0$. In addition, $/nabla f$ is typically not analytic at $x_{0}$, which leads to the inapplicability of the power series method for studying singular ODEs. Nevertheless, the ODE is well posed: the strategy we employ for showing this constructs a series of ODEs approximating (3), and then chooses a convergent subsequence by some compactness arguments such as the Arzelá-Ascoli theorem. Below, $C^{2}/left((0, /infty) ; /mathbb{R}^{n}/right)$ denotes the class of twice continuously differentiable maps from $(0, /infty)$ to $/mathbb{R}^{n}$; similarly, $C^{1}/left([0, /infty) ; /mathbb{R}^{n}/right)$ denotes the class of continuously differentiable maps from $[0, /infty)$ to $/mathbb{R}^{n}$.

Theorem 1 For any $f /in /mathcal{F}_{/infty}:=/cup_{L>0} /mathcal{F}_{L}$ and any $x_{0} /in /mathbb{R}^{n}$, the $O D E$ (3) with initial conditions $X(0)=x_{0}, /dot{X}(0)=0$ has a unique global solution $X /in C^{2}/left((0, /infty) ; /mathbb{R}^{n}/right) /cap C^{1}/left([0, /infty) ; /mathbb{R}^{n}/right)$.

The next theorem, in a rigorous way, guarantees the validity of the derivation of this ODE. The proofs of both theorems are deferred to the appendices.

Theorem 2 For any $f /in /mathcal{F}_{/infty}$, as the step size $s /rightarrow 0$, Nesterov's scheme (1) converges to the $O D E$ (3) in the sense that for all fixed $T>0$,

$$
/lim _{s /rightarrow 0} /max _{0 /leq k /leq /frac{T}{/sqrt{s}}}/left/|x_{k}-X(k /sqrt{s})/right/|=0
$$

### 2.1 Simple Properties

We collect some elementary properties that are helpful in understanding the ODE.

Time Invariance. If we adopt a linear time transformation, $/tilde{t}=c t$ for some $c>0$, by the chain rule it follows that

$$
/frac{/mathrm{d} X}{/mathrm{~d} /tilde{t}}=/frac{1}{c} /frac{/mathrm{d} X}{/mathrm{~d} t}, /frac{/mathrm{d}^{2} X}{/mathrm{~d} /tilde{t}^{2}}=/frac{1}{c^{2}} /frac{/mathrm{d}^{2} X}{/mathrm{~d} t^{2}}
$$

This yields the ODE parameterized by $/tilde{t}$,

$$
/frac{/mathrm{d}^{2} X}{/mathrm{~d} /tilde{t}^{2}}+/frac{3}{/tilde{t}} /frac{/mathrm{d} X}{/mathrm{~d} /tilde{t}}+/nabla f(X) / c^{2}=0
$$

Also note that minimizing $f / c^{2}$ is equivalent to minimizing $f$. Hence, the ODE is invariant under the time change. In fact, it is easy to see that time invariance holds if and only if the coefficient of $/dot{X}$ has the form $C / t$ for some constant $C$.

Rotational Invariance. Nesterov's scheme and other gradient-based schemes are invariant under rotations. As expected, the ODE is also invariant under orthogonal transformation. To see this, let $Y=Q X$ for some orthogonal matrix $Q$. This leads to $/dot{Y}=Q /dot{X}, /ddot{Y}=Q /ddot{X}$ and $/nabla_{Y} f=Q /nabla_{X} f$. Hence, denoting by $Q^{T}$ the transpose of $Q$, the ODE in the new coordinate system reads $Q^{T} /ddot{Y}+/frac{3}{t} Q^{T} /dot{Y}+Q^{T} /nabla_{Y} f=0$, which is of the same form as (3) once multiplying $Q$ on both sides.

Initial Asymptotic. Assume sufficient smoothness of $X$ such that $/lim _{t /rightarrow 0} /ddot{X}(t)$ exists. The mean value theorem guarantees the existence of some $/xi /in(0, t)$ that satisfies $/dot{X}(t) / t=$ $(/dot{X}(t)-/dot{X}(0)) / t=/ddot{X}(/xi)$. Hence, from the ODE we deduce $/ddot{X}(t)+3 /ddot{X}(/xi)+/nabla f(X(t))=0$.

Taking the limit $t /rightarrow 0$ gives $/ddot{X}(0)=-/nabla f/left(x_{0}/right) / 4$. Hence, for small $t$ we have the asymptotic form:

$$
X(t)=-/frac{/nabla f/left(x_{0}/right) t^{2}}{8}+x_{0}+o/left(t^{2}/right)
$$

This asymptotic expansion is consistent with the empirical observation that Nesterov's scheme moves slowly in the beginning.

### 2.2 ODE for Composite Optimization

It is interesting and important to generalize the ODE to minimizing $f$ in the composite form $f(x)=g(x)+h(x)$, where the smooth part $g /in /mathcal{F}_{L}$ and the non-smooth part $h$ : $/mathbb{R}^{n} /rightarrow(-/infty, /infty]$ is a structured general convex function. Both Nesterov (2013) and Beck and Teboulle (2009) obtain $O/left(1 / k^{2}/right)$ convergence rate by employing the proximal structure of $h$. In analogy to the smooth case, an ODE for composite $f$ is derived in the appendix.

## 3. Connections and Interpretations

In this section, we explore the approximate equivalence between the ODE and Nesterov's scheme, and provide evidence that the ODE can serve as an amenable tool for interpreting and analyzing Nesterov's scheme. The first subsection exhibits inverse quadratic convergence rate for the ODE solution, the next two address the oscillation phenomenon discussed in Section 1.1, and the last subsection is devoted to comparing Nesterov's scheme with gradient descent from a numerical perspective.

### 3.1 Analogous Convergence Rate

The original result from Nesterov (1983) states that, for any $f /in /mathcal{F}_{L}$, the sequence $/left/{x_{k}/right/}$ given by (1) with step size $s /leq 1 / L$ satisfies

$$
/begin{equation*}
f/left(x_{k}/right)-f^{/star} /leq /frac{2/left/|x_{0}-x^{/star}/right/|^{2}}{s(k+1)^{2}} /tag{6}
/end{equation*}
$$

Our next result indicates that the trajectory of (3) closely resembles the sequence $/left/{x_{k}/right/}$ in terms of the convergence rate to a minimizer $x^{/star}$. Compared with the discrete case, this proof is shorter and simpler.

Theorem 3 For any $f /in /mathcal{F}_{/infty}$, let $X(t)$ be the unique global solution to (3) with initial conditions $X(0)=x_{0}, /dot{X}(0)=0$. Then, for any $t>0$,

$$
/begin{equation*}
f(X(t))-f^{/star} /leq /frac{2/left/|x_{0}-x^{/star}/right/|^{2}}{t^{2}} /tag{7}
/end{equation*}
$$

Proof Consider the energy functional ${ }^{1}$ defined as $/mathcal{E}(t)=t^{2}/left(f(X(t))-f^{/star}/right)+2 /| X+t /dot{X} / 2-$ $x^{/star} /|^{2}$, whose time derivative is

$$
/dot{/mathcal{E}}=2 t/left(f(X)-f^{/star}/right)+t^{2}/langle/nabla f, /dot{X}/rangle+4/left/langle X+/frac{t}{2} /dot{X}-x^{/star}, /frac{3}{2} /dot{X}+/frac{t}{2} /ddot{X}/right/rangle
$$

1. We may also view this functional as the negative entropy. Similarly, for the gradient flow $/dot{X}+/nabla f(X)=0$, an energy function of form $/mathcal{E}_{/text {gradient }}(t)=t/left(f(X(t))-f^{/star}/right)+/left/|X(t)-x^{/star}/right/|^{2} / 2$ can be used to derive the bound $f(X(t))-f^{/star} /leq /frac{/left/|x_{0}-x^{/star}/right/|^{2}}{2 t}$.

Substituting $3 /dot{X} / 2+t /ddot{X} / 2$ with $-t /nabla f(X) / 2$, the above equation gives

$$
/dot{/mathcal{E}}=2 t/left(f(X)-f^{/star}/right)+4/left/langle X-x^{/star},-t /nabla f(X) / 2/right/rangle=2 t/left(f(X)-f^{/star}/right)-2 t/left/langle X-x^{/star}, /nabla f(X)/right/rangle /leq 0
$$

where the inequality follows from the convexity of $f$. Hence by monotonicity of $/mathcal{E}$ and non-negativity of $2/left/|X+t /dot{X} / 2-x^{/star}/right/|^{2}$, the gap satisfies

$$
f(X(t))-f^{/star} /leq /frac{/mathcal{E}(t)}{t^{2}} /leq /frac{/mathcal{E}(0)}{t^{2}}=/frac{2/left/|x_{0}-x^{/star}/right/|^{2}}{t^{2}}
$$

Making use of the approximation $t /approx k /sqrt{s}$, we observe that the convergence rate in (6) is essentially a discrete version of that in (7), providing yet another piece of evidence for the approximate equivalence between the ODE and the scheme.

We finish this subsection by showing that the number 2 appearing in the numerator of the error bound in (7) is optimal. Consider an arbitrary $f /in /mathcal{F}_{/infty}(/mathbb{R})$ such that $f(x)=x$ for $x /geq 0$. Starting from some $x_{0}>0$, the solution to (3) is $X(t)=x_{0}-t^{2} / 8$ before hitting the origin. Hence, $t^{2}/left(f(X(t))-f^{/star}/right)=t^{2}/left(x_{0}-t^{2} / 8/right)$ has a maximum $2 x_{0}^{2}=2/left|x_{0}-0/right|^{2}$ achieved at $t=2 /sqrt{x_{0}}$. Therefore, we cannot replace 2 by any smaller number, and we can expect that this tightness also applies to the discrete analog (6).

### 3.2 Quadratic $f$ and Bessel Functions

For quadratic $f$, the ODE (3) admits a solution in closed form. This closed form solution turns out to be very useful in understanding the issues raised in the introduction.

Let $f(x)=/frac{1}{2}/langle x, A x/rangle+/langle b, x/rangle$, where $A /in /mathbb{R}^{n /times n}$ is a positive semidefinite matrix and $b$ is in the column space of $A$ because otherwise this function can attain $-/infty$. Then a simple translation in $x$ can absorb the linear term $/langle b, x/rangle$ into the quadratic term. Since both the ODE and the scheme move within the affine space perpendicular to the kernel of $A$, without loss of generality, we assume that $A$ is positive definite, admitting a spectral decomposition $A=Q^{T} /Lambda Q$, where $/Lambda$ is a diagonal matrix formed by the eigenvalues. Replacing $x$ with $Q x$, we assume $f=/frac{1}{2}/langle x, /Lambda x/rangle$ from now on. Now, the ODE for this function admits a simple decomposition of form

$$
/ddot{X}_{i}+/frac{3}{t} /dot{X}_{i}+/lambda_{i} X_{i}=0, /quad i=1, /ldots, n
$$

with $X_{i}(0)=x_{0, i}, /dot{X}_{i}(0)=0$. Introduce $Y_{i}(u)=u X_{i}/left(u / /sqrt{/lambda_{i}}/right)$, which satisfies

$$
u^{2} /ddot{Y}_{i}+u /dot{Y}_{i}+/left(u^{2}-1/right) Y_{i}=0
$$

This is Bessel's differential equation of order one. Since $Y_{i}$ vanishes at $u=0$, we see that $Y_{i}$ is a constant multiple of $J_{1}$, the Bessel function of the first kind of order one. ${ }^{2}$ It has an analytic expansion:

$$
J_{1}(u)=/sum_{m=0}^{/infty} /frac{(-1)^{m}}{(2 m)!!(2 m+2)!!} u^{2 m+1}
$$

[^0]which gives the asymptotic expansion

$$
J_{1}(u)=(1+o(1)) /frac{u}{2}
$$

when $u /rightarrow 0$. Requiring $X_{i}(0)=x_{0, i}$, hence, we obtain

$$
/begin{equation*}
X_{i}(t)=/frac{2 x_{0, i}}{t /sqrt{/lambda_{i}}} J_{1}/left(t /sqrt{/lambda_{i}}/right) /tag{8}
/end{equation*}
$$

For large $t$, the Bessel function has the following asymptotic form (see e.g. Watson, 1995):

$$
/begin{equation*}
J_{1}(t)=/sqrt{/frac{2}{/pi t}}(/cos (t-3 /pi / 4)+O(1 / t)) . /tag{9}
/end{equation*}
$$

This asymptotic expansion yields (note that $f^{/star}=0$ )

$$
/begin{equation*}
f(X(t))-f^{/star}=f(X(t))=/sum_{i=1}^{n} /frac{2 x_{0, i}^{2}}{t^{2}} J_{1}/left(t /sqrt{/lambda_{i}}/right)^{2}=O/left(/frac{/left/|x_{0}-x^{/star}/right/|^{2}}{t^{3} /sqrt{/min /lambda_{i}}}/right) /tag{10}
/end{equation*}
$$

On the other hand, (9) and (10) give a lower bound:

$$
/begin{align*}
/limsup _{t /rightarrow /infty} t^{3}/left(f(X(t))-f^{/star}/right) & /geq /lim _{t /rightarrow /infty} /frac{1}{t} /int_{0}^{t} u^{3}/left(f(X(u))-f^{/star}/right) /mathrm{d} u //
& =/lim _{t /rightarrow /infty} /frac{1}{t} /int_{0}^{t} /sum_{i=1}^{n} 2 x_{0, i}^{2} u J_{1}/left(u /sqrt{/lambda_{i}}/right)^{2} /mathrm{~d} u  /tag{11}//
& =/sum_{i=1}^{n} /frac{2 x_{0, i}^{2}}{/pi /sqrt{/lambda_{i}}} /geq /frac{2/left/|x_{0}-x^{/star}/right/|^{2}}{/pi /sqrt{L}}
/end{align*}
$$

where $L=/|A/|_{2}$ is the spectral norm of $A$. The first inequality follows by interpreting $/lim _{t /rightarrow /infty} /frac{1}{t} /int_{0}^{t} u^{3}/left(f(X(u))-f^{/star}/right) /mathrm{d} u$ as the mean of $u^{3}/left(f(X(u))-f^{/star}/right)$ on $(0, /infty)$ in certain sense.

In view of (10), Nesterov's scheme might possibly exhibit $O/left(1 / k^{3}/right)$ convergence rate for strongly convex functions. This convergence rate is consistent with the second inequality in Theorem 6. In Section 4.3, we prove the $O/left(1 / t^{3}/right)$ rate for a generalized version of (3). However, (11) rules out the possibility of a higher order convergence rate.

Recall that the function considered in Figure 1 is $f(x)=0.02 x_{1}^{2}+0.005 x_{2}^{2}$, starting from $x_{0}=(1,1)$. As the step size $s$ becomes smaller, the trajectory of Nesterov's scheme converges to the solid curve represented via the Bessel function. While approaching the minimizer $x^{/star}$, each trajectory displays the oscillation pattern, as well-captured by the zoomed Figure 1b. This prevents Nesterov's scheme from achieving better convergence rate. The representation (8) offers excellent explanation as follows. Denote by $T_{1}, T_{2}$, respectively, the approximate periodicities of the first component $/left|X_{1}/right|$ in absolute value and the second $/left|X_{2}/right|$. By (9), we get $T_{1}=/pi / /sqrt{/lambda_{1}}=5 /pi$ and $T_{2}=/pi / /sqrt{/lambda_{2}}=10 /pi$. Hence, as the amplitude gradually decreases to zero, the function $f=2 x_{0,1}^{2} J_{1}/left(/sqrt{/lambda_{1}} t/right)^{2} / t^{2}+2 x_{0,2}^{2} J_{1}/left(/sqrt{/lambda_{2}} t/right)^{2} / t^{2}$ has a major cycle of $10 /pi$, the least common multiple of $T_{1}$ and $T_{2}$. A careful look at Figure 1c reveals that within each major bump, roughly, there are $10 /pi / T_{1}=2$ minor peaks.

### 3.3 Fluctuations of Strongly Convex $f$

The analysis carried out in the previous subsection only applies to convex quadratic functions. In this subsection, we extend the discussion to one-dimensional strongly convex functions. The Sturm-Picone theory (see e.g. Hinton, 2005) is extensively used all along the analysis.

Let $f /in /mathcal{S}_{/mu, L}(/mathbb{R})$. Without loss of generality, assume $f$ attains minimum at $x^{/star}=0$. Then, by definition $/mu /leq f^{/prime}(x) / x /leq L$ for any $x /neq 0$. Denoting by $X$ the solution to the ODE (3), we consider the self-adjoint equation,

$$
/begin{equation*}
/left(t^{3} Y^{/prime}/right)^{/prime}+/frac{t^{3} f^{/prime}(X(t))}{X(t)} Y=0 /tag{12}
/end{equation*}
$$

which, apparently, admits a solution $Y(t)=X(t)$. To apply the Sturm-Picone comparison theorem, consider

$$
/left(t^{3} Y^{/prime}/right)^{/prime}+/mu t^{3} Y=0
$$

for a comparison. This equation admits a solution $/widetilde{Y}(t)=J_{1}(/sqrt{/mu} t) / t$. Denote by $/tilde{t}_{1}</tilde{t}_{2}<$ ... all the positive roots of $J_{1}(t)$, which satisfy (see e .g. Watson, 1995)

$$
3.8317=/tilde{t}_{1}-/tilde{t}_{0}>/tilde{t}_{2}-/tilde{t}_{3}>/tilde{t}_{3}-/tilde{t}_{4}>/cdots>/pi
$$

where $/tilde{t}_{0}=0$. Then, it follows that the positive roots of $/tilde{Y}$ are $/tilde{t}_{1} / /sqrt{/mu}, /tilde{t}_{2} / /sqrt{/mu}, /ldots$ Since $t^{3} f^{/prime}(X(t)) / X(t) /geq /mu t^{3}$, the Sturm-Picone comparison theorem asserts that $X(t)$ has a root in each interval $/left[/tilde{t}_{i} / /sqrt{/mu}, /tilde{t}_{i+1} / /sqrt{/mu}/right]$.

To obtain a similar result in the opposite direction, consider

$$
/begin{equation*}
/left(t^{3} Y^{/prime}/right)^{/prime}+L t^{3} Y=0 /tag{13}
/end{equation*}
$$

Applying the Sturm-Picone comparison theorem to (12) and (13), we ensure that between any two consecutive positive roots of $X$, there is at least one $/tilde{t}_{i} / /sqrt{L}$. Now, we summarize our findings in the following. Roughly speaking, this result concludes that the oscillation frequency of the ODE solution is between $O(/sqrt{/mu})$ and $O(/sqrt{L})$.

Theorem 4 Denote by $0<t_{1}<t_{2}</cdots$ all the roots of $X(t)-x^{/star}$. Then these roots satisfy, for all $i /geq 1$,

$$
t_{1}</frac{7.6635}{/sqrt{/mu}}, t_{i+1}-t_{i}</frac{7.6635}{/sqrt{/mu}}, t_{i+2}-t_{i}>/frac{/pi}{/sqrt{L}}
$$

### 3.4 Nesterov's Scheme Compared with Gradient Descent

The ansatz $t /approx k /sqrt{s}$ in relating the ODE and Nesterov's scheme is formally confirmed in Theorem 2. Consequently, for any constant $t_{c}>0$, this implies that $x_{k}$ does not change much for a range of step sizes $s$ if $k /approx t_{c} / /sqrt{s}$. To empirically support this claim, we present an example in Figure 3a, where the scheme minimizes $f(x)=/|y-A x/|^{2} / 2+/|x/|_{1}$ with $y=(4,2,0)$ and $A(:, 1)=(0,2,4), A(:, 2)=(1,1,1)$ starting from $x_{0}=(2,0)$ (here $A(:, j)$ is the $j$ th column of $A)$. From this figure, we are delighted to observe that $x_{k}$ with the same $t_{c}$ are very close to each other.

This interesting square-root scaling has the potential to shed light on the superiority of Nesterov's scheme over gradient descent. Roughly speaking, each iteration in Nesterov's scheme amounts to traveling $/sqrt{s}$ in time along the integral curve of (3), whereas it is known that the simple gradient descent $x_{k+1}=x_{k}-s /nabla f/left(x_{k}/right)$ moves $s$ along the integral curve of $/dot{X}+/nabla f(X)=0$. We expect that for small $s$ Nesterov's scheme moves more in each iteration since $/sqrt{s}$ is much larger than $s$. Figure 3 b illustrates and supports this claim, where the function minimized is $f=/left|x_{1}/right|^{3}+5/left|x_{2}/right|^{3}+0.001/left(x_{1}+x_{2}/right)^{2}$ with step size $s=0.05$ (The coordinates are appropriately rotated to allow $x_{0}$ and $x^{/star}$ lie on the same horizontal line). The circles are the iterates for $k=1,10,20,30,45,60,90,120,150,190,250,300$. For Nesterov's scheme, the seventh circle has already passed $t=15$, while for gradient descent the last point has merely arrived at $t=15$.

![](https://cdn.mathpix.com/cropped/2024_08_12_d19cd0b096d3ee24c53cg-11.jpg?height=583&width=736&top_left_y=874&top_left_x=304)

(a) Square-root scaling of $s$.

![](https://cdn.mathpix.com/cropped/2024_08_12_d19cd0b096d3ee24c53cg-11.jpg?height=590&width=736&top_left_y=865&top_left_x=1085)

(b) Race between Nesterov's and gradient.

Figure 3: In (a), the circles, crosses and triangles are $x_{k}$ evaluated at $k=/lceil 1 / /sqrt{s}/rceil,/lceil 2 / /sqrt{s}/rceil$ and $/lceil 3 / /sqrt{s}/rceil$, respectively. In (b), the circles are iterations given by Nesterov's scheme or gradient descent, depending on the color, and the stars are $X(t)$ on the integral curves for $t=5,15$.

A second look at Figure 3b suggests that Nesterov's scheme allows a large deviation from its limit curve, as compared with gradient descent. This raises the question of the stable step size allowed for numerically solving the $/operatorname{ODE}$ (3) in the presence of accumulated errors. The finite difference approximation by the forward Euler method is

$$
/begin{equation*}
/frac{X(t+/Delta t)-2 X(t)+X(t-/Delta t)}{/Delta t^{2}}+/frac{3}{t} /frac{X(t)-X(t-/Delta t)}{/Delta t}+/nabla f(X(t))=0 /tag{14}
/end{equation*}
$$

which is equivalent to

$$
/begin{equation*}
X(t+/Delta t)=/left(2-/frac{3 /Delta t}{t}/right) X(t)-/Delta t^{2} /nabla f(X(t))-/left(1-/frac{3 /Delta t}{t}/right) X(t-/Delta t) /tag{15}
/end{equation*}
$$

Assuming $f$ is sufficiently smooth, we have $/nabla f(x+/delta x) /approx /nabla f(x)+/nabla^{2} f(x) /delta x$ for small perturbations $/delta x$, where $/nabla^{2} f(x)$ is the Hessian of $f$ evaluated at $x$. Identifying $k=t / /Delta t$,
the characteristic equation of this finite difference scheme is approximately

$$
/begin{equation*}
/operatorname{det}/left(/lambda^{2}-/left(2-/Delta t^{2} /nabla^{2} f-/frac{3 /Delta t}{t}/right) /lambda+1-/frac{3 /Delta t}{t}/right)=0 /tag{16}
/end{equation*}
$$

The numerical stability of (14) with respect to accumulated errors is equivalent to this: all the roots of (16) lie in the unit circle (see e.g. Leader, 2004). When $/nabla^{2} f /preceq L I_{n}$ (i.e. $L I_{n}-$ $/nabla^{2} f$ is positive semidefinite), if $/Delta t / t$ small and $/Delta t<2 / /sqrt{L}$, we see that all the roots of (16) lie in the unit circle. On the other hand, if $/Delta t>2 / /sqrt{L}$, (16) can possibly have a root $/lambda$ outside the unit circle, causing numerical instability. Under our identification $s=/Delta t^{2}$, a step size of $s=1 / L$ in Nesterov's scheme (1) is approximately equivalent to a step size of $/Delta t=1 / /sqrt{L}$ in the forward Euler method, which is stable for numerically integrating (14).

As a comparison, note that the finite difference scheme of the $/mathrm{ODE} /dot{X}(t)+/nabla f(X(t))=0$, which models gradient descent with updates $x_{k+1}=x_{k}-s /nabla f/left(x_{k}/right)$, has the characteristic equation $/operatorname{det}/left(/lambda-/left(1-/Delta t /nabla^{2} f/right)/right)=0$. Thus, to guarantee $-I_{n} /preceq 1-/Delta t /nabla^{2} f /preceq I_{n}$ in worst case analysis, one can only choose $/Delta t /leq 2 / L$ for a fixed step size, which is much smaller than the step size $2 / /sqrt{L}$ for (14) when $/nabla f$ is very variable, i.e., $L$ is large.

## 4. The Magic Constant 3

Recall that the constant 3 appearing in the coefficient of $/dot{X}$ in (3) originates from $(k+$ $2)-(k-1)=3$. This number leads to the momentum coefficient in (1) taking the form $(k-1) /(k+2)=1-3 / k+O/left(1 / k^{2}/right)$. In this section, we demonstrate that 3 can be replaced by any larger number, while maintaining the $O/left(1 / k^{2}/right)$ convergence rate. To begin with, let us consider the following ODE parameterized by a constant $r$ :

$$
/begin{equation*}
/ddot{X}+/frac{r}{t} /dot{X}+/nabla f(X)=0 /tag{17}
/end{equation*}
$$

with initial conditions $X(0)=x_{0}, /dot{X}(0)=0$. The proof of Theorem 1 , which seamlessly applies here, guarantees the existence and uniqueness of the solution $X$ to this ODE.

Interpreting the damping ratio $r / t$ as a measure of friction ${ }^{3}$ in the damping system, our results say that more friction does not end the $O/left(1 / t^{2}/right)$ and $O/left(1 / k^{2}/right)$ convergence rate. On the other hand, in the lower friction setting, where $r$ is smaller than 3, we can no longer expect inverse quadratic convergence rate, unless some additional structures of $f$ are imposed. We believe that this striking phase transition at 3 deserves more attention as an interesting research challenge.

### 4.1 High Friction

Here, we study the convergence rate of (17) with $r>3$ and $f /in /mathcal{F}_{/infty}$. Compared with (3), this new ODE as a damping suffers from higher friction. Following the strategy adopted in the proof of Theorem 3, we consider a new energy functional defined as

$$
/mathcal{E}(t)=/frac{2 t^{2}}{r-1}/left(f(X(t))-f^{/star}/right)+(r-1)/left/|X(t)+/frac{t}{r-1} /dot{X(t)}-x^{/star}/right/|^{2}
$$

3. In physics and engineering, damping may be modeled as a force proportional to velocity but opposite in direction, i.e. resisting motion; for instance, this force may be used as an approximation to the friction caused by drag. In our model, this force would be proportional to $-/frac{r}{t} /dot{X}$ where $/dot{X}$ is velocity and $/frac{r}{t}$ is the damping coefficient.

By studying the derivative of this functional, we get the following result.

Theorem 5 The solution $X$ to (17) satisfies

$$
f(X(t))-f^{/star} /leq /frac{(r-1)^{2}/left/|x_{0}-x^{/star}/right/|^{2}}{2 t^{2}}, /quad /int_{0}^{/infty} t/left(f(X(t))-f^{/star}/right) /mathrm{d} t /leq /frac{(r-1)^{2}/left/|x_{0}-x^{/star}/right/|^{2}}{2(r-3)}
$$

Proof Noting $r /dot{X}+t /ddot{X}=-t /nabla f(X)$, we get $/dot{/mathcal{E}}$ equal to

$$
/begin{align*}
& /frac{4 t}{r-1}/left(f(X)-f^{/star}/right)+/frac{2 t^{2}}{r-1}/langle/nabla f, /dot{X}/rangle+2/left/langle X+/frac{t}{r-1} /dot{X}-x^{/star}, r /dot{X}+t /ddot{X}/right/rangle //
&=/frac{4 t}{r-1}/left(f(X)-f^{/star}/right)-2 t/left/langle X-x^{/star}, /nabla f(X)/right/rangle /leq-/frac{2(r-3) t}{r-1}/left(f(X)-f^{/star}/right) /tag{18}
/end{align*}
$$

where the inequality follows from the convexity of $f$. Since $f(X) /geq f^{/star}$, the last display implies that $/mathcal{E}$ is non-increasing. Hence

$$
/frac{2 t^{2}}{r-1}/left(f(X(t))-f^{/star}/right) /leq /mathcal{E}(t) /leq /mathcal{E}(0)=(r-1)/left/|x_{0}-x^{/star}/right/|^{2}
$$

yielding the first inequality of this theorem. To complete the proof, from (18) it follows that

$$
/int_{0}^{/infty} /frac{2(r-3) t}{r-1}/left(f(X)-f^{/star}/right) /mathrm{d} t /leq-/int_{0}^{/infty} /frac{/mathrm{d} /mathcal{E}}{/mathrm{d} t} /mathrm{~d} t=/mathcal{E}(0)-/mathcal{E}(/infty) /leq(r-1)/left/|x_{0}-x^{/star}/right/|^{2}
$$

as desired for establishing the second inequality.

The first inequality is the same as (7) for the ODE (3), except for a larger constant $(r-1)^{2} / 2$. The second inequality measures the error $f(X(t))-f^{/star}$ in an average sense, and cannot be deduced from the first inequality.

Now, it is tempting to obtain such analogs for the discrete Nesterov's scheme as well. Following the formulation of Beck and Teboulle (2009), we wish to minimize $f$ in the composite form $f(x)=g(x)+h(x)$, where $g /in /mathcal{F}_{L}$ for some $L>0$ and $h$ is convex on $/mathbb{R}^{n}$ possibly assuming extended value $/infty$. Define the proximal subgradient

$$
G_{s}(x) /triangleq /frac{x-/operatorname{argmin}_{z}/left(/|z-(x-s /nabla g(x))/|^{2} /(2 s)+h(z)/right)}{s}
$$

Parametrizing by a constant $r$, we propose the generalized Nesterov's scheme,

$$
/begin{align*}
& x_{k}=y_{k-1}-s G_{s}/left(y_{k-1}/right) //
& y_{k}=x_{k}+/frac{k-1}{k+r-1}/left(x_{k}-x_{k-1}/right) /tag{19}
/end{align*}
$$

starting from $y_{0}=x_{0}$. The discrete analog of Theorem 5 is below.

Theorem 6 The sequence $/left/{x_{k}/right/}$ given by (19) with $0<s /leq 1 / L$ satisfies

$$
f/left(x_{k}/right)-f^{/star} /leq /frac{(r-1)^{2}/left/|x_{0}-x^{/star}/right/|^{2}}{2 s(k+r-2)^{2}}, /quad /sum_{k=1}^{/infty}(k+r-1)/left(f/left(x_{k}/right)-f^{/star}/right) /leq /frac{(r-1)^{2}/left/|x_{0}-x^{/star}/right/|^{2}}{2 s(r-3)}
$$

The first inequality suggests that the generalized Nesterov's schemes still achieve $O/left(1 / k^{2}/right)$ convergence rate. However, if the error bound satisfies $f/left(x_{k^{/prime}}/right)-f^{/star} /geq c / k^{/prime 2}$ for some arbitrarily small $c>0$ and a dense subsequence $/left/{k^{/prime}/right/}$, i.e., $/left|/left/{k^{/prime}/right/} /cap/{1, /ldots, m/}/right| /geq /alpha m$ for all $m /geq 1$ and some $/alpha>0$, then the second inequality of the theorem would be violated. To see this, note that if it were the case, we would have $/left(k^{/prime}+r-1/right)/left(f/left(x_{k^{/prime}}/right)-f^{/star}/right) /gtrsim /frac{1}{k^{/prime}}$; the sum of the harmonic series $/frac{1}{k^{/prime}}$ over a dense subset of $/{1,2, /ldots/}$ is infinite. Hence, the second inequality is not trivial because it implies the error bound is, in some sense, $O/left(1 / k^{2}/right)$ suboptimal.

Now we turn to the proof of this theorem. It is worth pointing out that, though based on the same idea, the proof below is much more complicated than that of Theorem 5.

Proof Consider the discrete energy functional,

$$
/mathcal{E}(k)=/frac{2(k+r-2)^{2} s}{r-1}/left(f/left(x_{k}/right)-f^{/star}/right)+(r-1)/left/|z_{k}-x^{/star}/right/|^{2}
$$

where $z_{k}=(k+r-1) y_{k} /(r-1)-k x_{k} /(r-1)$. If we have

$$
/begin{equation*}
/mathcal{E}(k)+/frac{2 s[(r-3)(k+r-2)+1]}{r-1}/left(f/left(x_{k-1}/right)-f^{/star}/right) /leq /mathcal{E}(k-1) /tag{20}
/end{equation*}
$$

then it would immediately yield the desired results by summing (20) over $k$. That is, by recursively applying $(20)$, we see

$$
/begin{aligned}
& /mathcal{E}(k)+/sum_{i=1}^{k} /frac{2 s[(r-3)(i+r-2)+1]}{r-1}/left(f/left(x_{i-1}/right)-f^{/star}/right) //
& /leq /mathcal{E}(0)=/frac{2(r-2)^{2} s}{r-1}/left(f/left(x_{0}/right)-f^{/star}/right)+(r-1)/left/|x_{0}-x^{/star}/right/|^{2}
/end{aligned}
$$

which is equivalent to

$$
/begin{equation*}
/mathcal{E}(k)+/sum_{i=1}^{k-1} /frac{2 s[(r-3)(i+r-1)+1]}{r-1}/left(f/left(x_{i}/right)-f^{/star}/right) /leq(r-1)/left/|x_{0}-x^{/star}/right/|^{2} /tag{21}
/end{equation*}
$$

Noting that the left-hand side of (21) is lower bounded by $2 s(k+r-2)^{2}/left(f/left(x_{k}/right)-f^{/star}/right) /(r-1)$, we thus obtain the first inequality of the theorem. Since $/mathcal{E}(k) /geq 0$, the second inequality is verified via taking the limit $k /rightarrow /infty$ in (21) and replacing $(r-3)(i+r-1)+1$ by $(r-3)(i+r-1)$

We now establish (20). For $s /leq 1 / L$, we have the basic inequality,

$$
/begin{equation*}
f/left(y-s G_{s}(y)/right) /leq f(x)+G_{s}(y)^{T}(y-x)-/frac{s}{2}/left/|G_{s}(y)/right/|^{2} /tag{22}
/end{equation*}
$$

for any $x$ and $y$. Note that $y_{k-1}-s G_{s}/left(y_{k-1}/right)$ actually coincides with $x_{k}$. Summing of $(k-1) /(k+r-2) /times(22)$ with $x=x_{k-1}, y=y_{k-1}$ and $(r-1) /(k+r-2) /times(22)$ with $x=x^{/star}, y=y_{k-1}$ gives

$$
/begin{aligned}
f/left(x_{k}/right) & /leq /frac{k-1}{k+r-2} f/left(x_{k-1}/right)+/frac{r-1}{k+r-2} f^{/star} //
& +/frac{r-1}{k+r-2} G_{s}/left(y_{k-1}/right)^{T}/left(/frac{k+r-2}{r-1} y_{k-1}-/frac{k-1}{r-1} x_{k-1}-x^{/star}/right)-/frac{s}{2}/left/|G_{s}/left(y_{k-1}/right)/right/|^{2} //
& =/frac{k-1}{k+r-2} f/left(x_{k-1}/right)+/frac{r-1}{k+r-2} f^{/star}+/frac{(r-1)^{2}}{2 s(k+r-2)^{2}}/left(/left/|z_{k-1}-x^{/star}/right/|^{2}-/left/|z_{k}-x^{/star}/right/|^{2}/right)
/end{aligned}
$$

where we use $z_{k-1}-s(k+r-2) G_{s}/left(y_{k-1}/right) /(r-1)=z_{k}$. Rearranging the above inequality and multiplying by $2 s(k+r-2)^{2} /(r-1)$ gives the desired (20).

In closing, we would like to point out this new scheme is equivalent to setting $/theta_{k}=$ $(r-1) /(k+r-1)$ and letting $/theta_{k}/left(/theta_{k-1}^{-1}-1/right)$ replace the momentum coefficient $(k-1) /(k+r-1)$. Then, the equal sign " $=$ " in the update $/theta_{k+1}=/left(/sqrt{/theta_{k}^{4}+4 /theta_{k}^{2}}-/theta_{k}^{2}/right) / 2$ has to be replaced by an inequality sign " $/geq$ ". In examining the proof of Theorem 1(b) in Tseng (2010), we can get an alternative proof of Theorem 6 .

### 4.2 Low Friction

Now we turn to the case $r<3$. Then, unfortunately, the energy functional approach for proving Theorem 5 is no longer valid, since the left-hand side of (18) is positive in general. In fact, there are counterexamples that fail the desired $O/left(1 / t^{2}/right)$ or $O/left(1 / k^{2}/right)$ convergence rate. We present such examples in continuous time. Equally, these examples would also violate the $O/left(1 / k^{2}/right)$ convergence rate in the discrete schemes, and we forego the details.

Let $f(x)=/frac{1}{2}/|x/|^{2}$ and $X$ be the solution to (17). Then, $Y=t^{/frac{r-1}{2}} X$ satisfies

$$
t^{2} /ddot{Y}+t /dot{Y}+/left(t^{2}-(r-1)^{2} / 4/right) Y=0
$$

With the initial condition $Y(t) /approx t^{/frac{r-1}{2}} x_{0}$ for small $t$, the solution to the above Bessel equation in a vector form of order $(r-1) / 2$ is $Y(t)=2^{/frac{r-1}{2}} /Gamma((r+1) / 2) J_{(r-1) / 2}(t) x_{0}$. Thus,

$$
X(t)=/frac{2^{/frac{r-1}{2}} /Gamma((r+1) / 2) J_{(r-1) / 2}(t)}{t^{/frac{r-1}{2}}} x_{0}
$$

For large $t$, the Bessel function $J_{(r-1) / 2}(t)=/sqrt{2 /(/pi t)}(/cos (t-(r-1) /pi / 4-/pi / 4)+O(1 / t))$. Hence,

$$
f(X(t))-f^{/star}=O/left(/left/|x_{0}-x^{/star}/right/|^{2} / t^{r}/right)
$$

where the exponent $r$ is tight. This rules out the possibility of inverse quadratic convergence of the generalized ODE and scheme for all $f /in /mathcal{F}_{L}$ if $r<2$. An example with $r=1$ is plotted in Figure 2.

Next, we consider the case $2 /leq r<3$ and let $f(x)=|x|$ (this also applies to multivariate $f=/|x/|){ }^{4}$ Starting from $x_{0}>0$, we get $X(t)=x_{0}-/frac{t^{2}}{2(1+r)}$ for $t /leq /sqrt{2(1+r) x_{0}}$. Requiring continuity of $X$ and $/dot{X}$ at the change point 0 , we get

$$
X(t)=/frac{t^{2}}{2(1+r)}+/frac{2/left(2(1+r) x_{0}/right)^{/frac{r+1}{2}}}{/left(r^{2}-1/right) t^{r-1}}-/frac{r+3}{r-1} x_{0}
$$

for $/sqrt{2(1+r) x_{0}}<t /leq /sqrt{2 c^{/star}(1+r) x_{0}}$, where $c^{/star}$ is the positive root other than 1 of $(r-$ 1) $c+4 c^{-/frac{r-1}{2}}=r+3$. Repeating this process solves for $X$. Note that $t^{1-r}$ is in the null

[^1]space of $/ddot{X}+r /dot{X} / t$ and satisfies $t^{2} /times t^{1-r} /rightarrow /infty$ as $t /rightarrow /infty$. For illustration, Figure 4 plots $t^{2}/left(f(X(t))-f^{/star}/right)$ and $s k^{2}/left(f/left(x_{k}/right)-f^{/star}/right)$ with $r=2,2.5$, and $r=4$ for comparison ${ }^{5}$. It is clearly that inverse quadratic convergence does not hold for $r=2,2.5$, that is, (2) does not hold for $r<3$. Interestingly, in Figures 4 a and 4d, the scaled errors at peaks grow linearly, whereas for $r=2.5$, the growth rate, though positive as well, seems sublinear.

![](https://cdn.mathpix.com/cropped/2024_08_12_d19cd0b096d3ee24c53cg-16.jpg?height=752&width=1524&top_left_y=575&top_left_x=298)

Figure 4: Scaled errors $t^{2}/left(f(X(t))-f^{/star}/right)$ and $s k^{2}/left(f/left(x_{k}/right)-f^{/star}/right)$ of generalized ODEs and schemes for minimizing $f=|x|$. In (d), the step size $s=10^{-6}$, in (e), $s=10^{-7}$, and in (f), $s=10^{-6}$.

However, if $f$ possesses some additional property, inverse quadratic convergence is still guaranteed, as stated below. In that theorem, $f$ is assumed to be a continuously differentiable convex function.

Theorem 7 Suppose $1<r<3$ and let $X$ be a solution to the $O D E$ (17). If $/left(f-f^{/star}/right)^{/frac{r-1}{2}}$ is also convex, then

$$
f(X(t))-f^{/star} /leq /frac{(r-1)^{2}/left/|x_{0}-x^{/star}/right/|^{2}}{2 t^{2}}
$$

Proof Since $/left(f-f^{/star}/right)^{/frac{r-1}{2}}$ is convex, we obtain

$$
/left(f(X(t))-f^{/star}/right)^{/frac{r-1}{2}} /leq/left/langle X-x^{/star}, /nabla/left(f(X)-f^{/star}/right)^{/frac{r-1}{2}}/right/rangle=/frac{r-1}{2}/left(f(X)-f^{/star}/right)^{/frac{r-3}{2}}/left/langle X-x^{/star}, /nabla f(X)/right/rangle
$$

which can be simplified to $/frac{2}{r-1}/left(f(X)-f^{/star}/right) /leq/left/langle X-x^{/star}, /nabla f(X)/right/rangle$. This inequality combined with (18) leads to the monotonically decreasing of $/mathcal{E}(t)$ defined for Theorem 5. This completes the proof by noting $f(X)-f^{/star} /leq(r-1) /mathcal{E}(t) //left(2 t^{2}/right) /leq(r-1) /mathcal{E}(0) //left(2 t^{2}/right)=$ $(r-1)^{2}/left/|x_{0}-x^{/star}/right/|^{2} //left(2 t^{2}/right)$.

[^2]
### 4.3 Strongly Convex $f$

Strong convexity is a desirable property for optimization. Making use of this property carefully suggests a generalized Nesterov's scheme that achieves optimal linear convergence (Nesterov, 2004). In that case, even vanilla gradient descent has a linear convergence rate. Unfortunately, the example given in the previous subsection simply rules out such possibility for (1) and its generalizations (19). However, from a different perspective, this example suggests that $O/left(t^{-r}/right)$ convergence rate can be expected for (17). In the next theorem, we prove a slightly weaker statement of this kind, that is, a provable $O/left(t^{-/frac{2 r}{3}}/right)$ convergence rate is established for strongly convex functions. Bridging this gap may require new tools and more careful analysis.

Let $f /in /mathcal{S}_{/mu, L}/left(/mathbb{R}^{n}/right)$ and consider a new energy functional for $/alpha>2$ defined as

$$
/mathcal{E}(t ; /alpha)=t^{/alpha}/left(f(X(t))-f^{/star}/right)+/frac{(2 r-/alpha)^{2} t^{/alpha-2}}{8}/left/|X(t)+/frac{2 t}{2 r-/alpha} /dot{X}-x^{/star}/right/|^{2}
$$

When clear from the context, $/mathcal{E}(t ; /alpha)$ is simply denoted as $/mathcal{E}(t)$. For $r>3$, taking $/alpha=2 r / 3$ in the theorem stated below gives $f(X(t))-f^{/star} /lesssim/left/|x_{0}-x^{/star}/right/|^{2} / t^{/frac{2 r}{3}}$.

Theorem 8 For any $f /in /mathcal{S}_{/mu, L}/left(/mathbb{R}^{n}/right)$, if $2 /leq /alpha /leq 2 r / 3$ we get

$$
f(X(t))-f^{/star} /leq /frac{C/left/|x_{0}-x^{/star}/right/|^{2}}{/mu^{/frac{/alpha-2}{2}} t^{/alpha}}
$$

for any $t>0$. Above, the constant $C$ only depends on $/alpha$ and $r$.

Proof Note that $/dot{/mathcal{E}}(t ; /alpha)$ equals

$$
/begin{align*}
& /alpha t^{/alpha-1}/left(f(X)-f^{/star}/right)-/frac{(2 r-/alpha) t^{/alpha-1}}{2}/left/langle X-x^{/star}, /nabla f(X)/right/rangle+/frac{(/alpha-2)(2 r-/alpha)^{2} t^{/alpha-3}}{8}/left/|X-x^{/star}/right/|^{2} //
&+/frac{(/alpha-2)(2 r-/alpha) t^{/alpha-2}}{4}/left/langle/dot{X}, X-x^{/star}/right/rangle . /tag{23}
/end{align*}
$$

By the strong convexity of $f$, the second term of the right-hand side of (23) is bounded below as

$$
/frac{(2 r-/alpha) t^{/alpha-1}}{2}/left/langle X-x^{/star}, /nabla f(X)/right/rangle /geq /frac{(2 r-/alpha) t^{/alpha-1}}{2}/left(f(X)-f^{/star}/right)+/frac{/mu(2 r-/alpha) t^{/alpha-1}}{4}/left/|X-x^{/star}/right/|^{2} .
$$

Substituting the last display into (23) with the awareness of $r /geq 3 /alpha / 2$ yields

$$
/dot{/mathcal{E}} /leq-/frac{/left(2 /mu(2 r-/alpha) t^{2}-(/alpha-2)(2 r-/alpha)^{2}/right) t^{/alpha-3}}{8}/left/|X-x^{/star}/right/|^{2}+/frac{(/alpha-2)(2 r-/alpha) t^{/alpha-2}}{8} /frac{/mathrm{d}/left/|X-x^{/star}/right/|^{2}}{/mathrm{~d} t}
$$

Hence, if $t /geq t_{/alpha}:=/sqrt{(/alpha-2)(2 r-/alpha) /(2 /mu)}$, we obtain

$$
/dot{/mathcal{E}}(t) /leq /frac{(/alpha-2)(2 r-/alpha) t^{/alpha-2}}{8} /frac{/mathrm{d}/left/|X-x^{/star}/right/|^{2}}{/mathrm{~d} t}
$$

Integrating the last inequality on the interval $/left(t_{/alpha}, t/right)$ gives

$$
/begin{align*}
/mathcal{E}(t) /leq /mathcal{E}/left(t_{/alpha}/right)+/frac{(/alpha-2)(2 r-/alpha) t^{/alpha-2}}{8}/left/|X(t)-x^{/star}/right/|^{2}-/frac{(/alpha-2)(2 r-/alpha) t_{/alpha}^{/alpha-2}}{8}/left/|X/left(t_{/alpha}/right)-x^{/star}/right/|^{2} //
-/frac{1}{8} /int_{t_{/alpha}}^{t}(/alpha-2)^{2}(2 r-/alpha) u^{/alpha-3}/left/|X(u)-x^{/star}/right/|^{2} /mathrm{~d} u /leq /mathcal{E}/left(t_{/alpha}/right)+/frac{(/alpha-2)(2 r-/alpha) t^{/alpha-2}}{8}/left/|X(t)-x^{/star}/right/|^{2} //
/leq /mathcal{E}/left(t_{/alpha}/right)+/frac{(/alpha-2)(2 r-/alpha) t^{/alpha-2}}{4 /mu}/left(f(X(t))-f^{/star}/right) /tag{24}
/end{align*}
$$

Making use of (24), we apply induction on $/alpha$ to finish the proof. First, consider $2<$ $/alpha /leq 4$. Applying Theorem 5 , from (24) we get that $/mathcal{E}(t)$ is upper bounded by

$$
/begin{equation*}
/mathcal{E}/left(t_{/alpha}/right)+/frac{(/alpha-2)(r-1)^{2}(2 r-/alpha)/left/|x_{0}-x^{/star}/right/|^{2}}{8 /mu t^{4-/alpha}} /leq /mathcal{E}/left(t_{/alpha}/right)+/frac{(/alpha-2)(r-1)^{2}(2 r-/alpha)/left/|x_{0}-x^{/star}/right/|^{2}}{8 /mu t_{/alpha}^{4-/alpha}} /tag{25}
/end{equation*}
$$

Then, we bound $/mathcal{E}/left(t_{/alpha}/right)$ as follows.

$$
/begin{gather*}
/mathcal{E}/left(t_{/alpha}/right) /leq t_{/alpha}^{/alpha}/left(f/left(X/left(t_{/alpha}/right)/right)-f^{/star}/right)+/frac{(2 r-/alpha)^{2} t_{/alpha}^{/alpha-2}}{4}/left/|/frac{2 r-2}{2 r-/alpha} X/left(t_{/alpha}/right)+/frac{2 t_{/alpha}}{2 r-/alpha} /dot{X}/left(t_{/alpha}/right)-/frac{2 r-2}{2 r-/alpha} x^{/star}/right/|^{2} //
+/frac{(2 r-/alpha)^{2} t_{/alpha}^{/alpha-2}}{4}/left/|/frac{/alpha-2}{2 r-/alpha} X/left(t_{/alpha}/right)-/frac{/alpha-2}{2 r-/alpha} x^{/star}/right/|^{2} //
/leq(r-1)^{2} t_{/alpha}^{/alpha-2}/left/|x_{0}-x^{/star}/right/|^{2}+/frac{(/alpha-2)^{2}(r-1)^{2}/left/|x_{0}-x^{/star}/right/|^{2}}{4 /mu t_{/alpha}^{4-/alpha}} /tag{26}
/end{gather*}
$$

where in the second inequality we use the decreasing property of the energy functional defined for Theorem 5. Combining (25) and (26), we have

$/mathcal{E}(t) /leq(r-1)^{2} t_{/alpha}^{/alpha-2}/left/|x_{0}-x^{/star}/right/|^{2}+/frac{(/alpha-2)(r-1)^{2}(2 r+/alpha-4)/left/|x_{0}-x^{/star}/right/|^{2}}{8 /mu t_{/alpha}^{4-/alpha}}=O/left(/frac{/left/|x_{0}-x^{/star}/right/|^{2}}{/mu^{/frac{/alpha-2}{2}}}/right)$.

For $t /geq t_{/alpha}$, it suffices to apply $f(X(t))-f^{/star} /leq /mathcal{E}(t) / t^{3}$ to the last display. For $t<t_{/alpha}$, by Theorem $5, f(X(t))-f^{/star}$ is upper bounded by

$$
/begin{align*}
/frac{(r-1)^{2}/left/|x_{0}-x^{/star}/right/|^{2}}{2 t^{2}} & /leq /frac{(r-1)^{2} /mu^{/frac{/alpha-2}{2}}[(/alpha-2)(2 r-/alpha) /(2 /mu)]^{/frac{/alpha-2}{2}}}{2} /frac{/left/|x_{0}-x^{/star}/right/|^{2}}{/mu^{/frac{/alpha-2}{2}} t^{/alpha}}  /tag{27}//
& =O/left(/frac{/left/|x_{0}-x^{/star}/right/|^{2}}{/mu^{/frac{/alpha-2}{2}} t^{/alpha}}/right)
/end{align*}
$$

Next, suppose that the theorem is valid for some $/tilde{/alpha}>2$. We show below that this theorem is still valid for $/alpha:=/tilde{/alpha}+1$ if still $r /geq 3 /alpha / 2$. By the assumption, (24) further induces

$$
/mathcal{E}(t) /leq /mathcal{E}/left(t_{/alpha}/right)+/frac{(/alpha-2)(2 r-/alpha) t^{/alpha-2}}{4 /mu} /frac{/tilde{C}/left/|x_{0}-x^{/star}/right/|^{2}}{/mu^{/frac{/tilde{/alpha}-2}{2}} t^{/tilde{/alpha}}} /leq /mathcal{E}/left(t_{/alpha}/right)+/frac{/tilde{C}(/alpha-2)(2 r-/alpha)/left/|x_{0}-x^{/star}/right/|^{2}}{4 /mu^{/frac{/alpha-1}{2}} t_{/alpha}}
$$

for some constant $/tilde{C}$ only depending on $/tilde{/alpha}$ and $r$. This inequality with (26) implies

$$
/begin{aligned}
/mathcal{E}(t) & /leq(r-1)^{2} t_{/alpha}^{/alpha-2}/left/|x_{0}-x^{/star}/right/|^{2}+/frac{(/alpha-2)^{2}(r-1)^{2}/left/|x_{0}-x^{/star}/right/|^{2}}{4 /mu t_{/alpha}^{-/alpha-/alpha}}+/frac{/tilde{C}(/alpha-2)(2 r-/alpha)/left/|x_{0}-x^{/star}/right/|^{2}}{4 /mu^{/frac{/alpha-1}{2}} t_{/alpha}} //
& =O/left(/left/|x_{0}-x^{/star}/right/|^{2} / /mu^{/frac{/alpha-2}{2}}/right)
/end{aligned}
$$

which verify the induction for $t /geq t_{/alpha}$. As for $t<t_{/alpha}$, the validity of the induction follows from Theorem 5 , similarly to (27). Thus, combining the base and induction steps, the proof is completed.

It should be pointed out that the constant $C$ in the statement of Theorem 8 grows with the parameter $r$. Hence, simply increasing $r$ does not guarantee to give a better error bound. While it is desirable to expect a discrete analogy of Theorem 8, i.e., $O/left(1 / k^{/alpha}/right)$ convergence rate for (19), a complete proof can be notoriously complicated. That said, we mimic the proof of Theorem 8 for $/alpha=3$ and succeed in obtaining a $O/left(1 / k^{3}/right)$ convergence rate for the generalized Nesterov's schemes, as summarized in the theorem below.

Theorem 9 Suppose $f$ is written as $f=g+h$, where $g /in /mathcal{S}_{/mu, L}$ and $h$ is convex with possible extended value $/infty$. Then, the generalized Nesterov's scheme (19) with $r /geq 9 / 2$ and $s=1 / L$ satisfies

$$
f/left(x_{k}/right)-f^{/star} /leq /frac{C L/left/|x_{0}-x^{/star}/right/|^{2}}{k^{2}} /frac{/sqrt{L / /mu}}{k}
$$

where $C$ only depends on $r$.

This theorem states that the discrete scheme (19) enjoys the error bound $O/left(1 / k^{3}/right)$ without any knowledge of the condition number $L / /mu$. In particular, this bound is much better than that given in Theorem 6 if $k /gg /sqrt{L / /mu}$. The strategy of the proof is fully inspired by that of Theorem 8, though it is much more complicated and thus deferred to the Appendix. The relevant energy functional $/mathcal{E}(k)$ for this Theorem 9 is equal to

$$
/begin{align*}
& /frac{s(2 k+3 r-5)(2 k+2 r-5)(4 k+4 r-9)}{16}/left(f/left(x_{k}/right)-f^{/star}/right) //
& /quad+/frac{2 k+3 r-5}{16}/left/|2(k+r-1) y_{k}-(2 k+1) x_{k}-(2 r-3) x^{/star}/right/|^{2} . /tag{28}
/end{align*}
$$

### 4.4 Numerical Examples

We study six synthetic examples to compare (19) with the step sizes are fixed to be $1 / L$, as illustrated in Figure 5. The error rates exhibits similar patterns for all $r$, namely, decreasing while suffering from local bumps. A smaller $r$ introduces less friction, thus allowing $x_{k}$ moves towards $x^{/star}$ faster in the beginning. However, when sufficiently close to $x^{/star}$, more friction is preferred in order to reduce overshoot. This point of view explains what we observe in these examples. That is, across these six examples, (19) with a smaller $r$ performs slightly better in the beginning, but a larger $r$ has advantage when $k$ is large. It is an interesting question how to choose a good $r$ for different problems in practice.

## Su, Boyd and Candès

![](https://cdn.mathpix.com/cropped/2024_08_12_d19cd0b096d3ee24c53cg-20.jpg?height=1535&width=1560&top_left_y=284&top_left_x=280)

Figure 5: Comparisons of generalized Nesterov's schemes with different $r$.

Lasso with fat design. Minimize $f(x)=/frac{1}{2}/|A x-b/|^{2}+/lambda/|x/|_{1}$, in which $A$ a $100 /times 500$ random matrix with i.i.d. standard Gaussian $/mathcal{N}(0,1)$ entries, $b$ generated independently has i.i.d. $/mathcal{N}(0,25)$ entries, and the penalty $/lambda=4$. The plot is Figure 5 a

Lasso with square design. Minimize $f(x)=/frac{1}{2}/|A x-b/|^{2}+/lambda/|x/|_{1}$, where $A$ a $500 /times$ 500 random matrix with i.i.d. standard Gaussian entries, $b$ generated independently has i.i.d. $/mathcal{N}(0,9)$ entries, and the penalty $/lambda=4$. The plot is Figure 5 b .

Nonnegative least squares (NLS) with fat design. Minimize $f(x)=/|A x-b/|^{2}$ subject to $x /succeq 0$, with the same design $A$ and $b$ as in Figure 5a. The plot is Figure 5c.

Nonnegative least squares with sparse design. Minimize $f(x)=/|A x-b/|^{2}$ subject to $x /succeq 0$, in which $A$ is a $1000 /times 10000$ sparse matrix with nonzero probability $10 /%$ for each entry and $b$ is given as $b=A x^{0}+/mathcal{N}/left(0, I_{1000}/right)$. The nonzero entries of $A$ are independently Gaussian distributed before column normalization, and $x^{0}$ has 100 nonzero entries that are all equal to 4 . The plot is Figure 5d.

Logistic regression. Minimize $/sum_{i=1}^{n}-y_{i} a_{i}^{T} x+/log /left(1+/mathrm{e}^{a_{i}^{T} x}/right)$, in which $A=/left(a_{1}, /ldots, a_{n}/right)^{T}$ is a $500 /times 100$ matrix with i.i.d. $/mathcal{N}(0,1)$ entries. The labels $y_{i} /in/{0,1/}$ are generated by the logistic model: $/mathbb{P}/left(Y_{i}=1/right)=1 //left(1+/mathrm{e}^{-a_{i}^{T} x^{0}}/right)$, where $x^{0}$ is a realization of i.i.d. $/mathcal{N}(0,1 / 100)$. The plot is Figure 5e.

$/ell_{1}$-regularized logistic regression. Minimize $/sum_{i=1}^{n}-y_{i} a_{i}^{T} x+/log /left(1+/mathrm{e}^{a_{i}^{T} x}/right)+/lambda/|x/|_{1}$, in which $A=/left(a_{1}, /ldots, a_{n}/right)^{T}$ is a $200 /times 1000$ matrix with i.i.d. $/mathcal{N}(0,1)$ entries and $/lambda=5$. The labels $y_{i}$ are generated similarly as in the previous example, except for the ground truth $x^{0}$ here having 10 nonzero components given as i.i.d. $/mathcal{N}(0,225)$. The plot is Figure $5 f$.

## 5. Restarting

The example discussed in Section 4.2 demonstrates that Nesterov's scheme and its generalizations (19) are not capable of fully exploiting strong convexity. That is, this example suggests evidence that $O(1 / /mathrm{poly}(k))$ is the best rate achievable under strong convexity. In contrast, the vanilla gradient method achieves linear convergence $O/left((1-/mu / L)^{k}/right)$. This drawback results from too much momentum introduced when the objective function is strongly convex. The derivative of a strongly convex function is generally more reliable than that of non-strongly convex functions. In the language of ODEs, at later stage a too small $3 / t$ in (3) leads to a lack of friction, resulting in unnecessary overshoot along the trajectory. Incorporating the optimal momentum coefficient $/frac{/sqrt{L}-/sqrt{/mu}}{/sqrt{L}+/sqrt{/mu}}$ (This is less than $(k-1) /(k+2)$ when $k$ is large), Nesterov's scheme has convergence rate of $O/left((1-/sqrt{/mu / L})^{k}/right)$ (Nesterov, 2004), which, however, requires knowledge of the condition number $/mu / L$. While it is relatively easy to bound the Lipschitz constant $L$ by the use of backtracking, estimating the strong convexity parameter $/mu$, if not impossible, is very challenging.

Among many approaches to gain acceleration via adaptively estimating $/mu / L$ (see Nesterov, 2013), O'Donoghue and Candès (2013) proposes a procedure termed as gradient restarting for Nesterov's scheme in which (1) is restarted with $x_{0}=y_{0}:=x_{k}$ whenever $f/left(x_{k+1}/right)>f/left(x_{k}/right)$. In the language of ODEs, this restarting essentially keeps $/langle/nabla f, /dot{X}/rangle$ negative, and resets $3 / t$ each time to prevent this coefficient from steadily decreasing along the trajectory. Although it has been empirically observed that this method significantly boosts convergence, there is no general theory characterizing the convergence rate.

In this section, we propose a new restarting scheme we call the speed restarting scheme. The underlying motivation is to maintain a relatively high velocity $/dot{X}$ along the trajectory, similar in spirit to the gradient restarting. Specifically, our main result, Theorem 10, ensures linear convergence of the continuous version of the speed restarting. More generally, our contribution here is merely to provide a framework for analyzing restarting schemes rather than competing with other schemes; it is beyond the scope of this paper to get optimal constants in these results. Throughout this section, we assume $f /in /mathcal{S}_{/mu, L}$ for some $0</mu /leq L$. Recall that function $f /in /mathcal{S}_{/mu, L}$ if $f /in /mathcal{F}_{L}$ and $f(x)-/mu/|x/|^{2} / 2$ is convex.

## Su, BOYd ANd CANDÈs

### 5.1 A New Restarting Scheme

We first define the speed restarting time. For the ODE (3), we call

$$
T=T/left(x_{0} ; f/right)=/sup /left/{t>0: /forall u /in(0, t), /frac{/mathrm{d}/|/dot{X}(u)/|^{2}}{/mathrm{~d} u}>0/right/}
$$

the speed restarting time. In words, $T$ is the first time the velocity $/|/dot{X}/|$ decreases. Back to the discrete scheme, it is the first time when we observe $/left/|x_{k+1}-x_{k}/right/|</left/|x_{k}-x_{k-1}/right/|$. This definition itself does not directly imply that $0<T</infty$, which is proven later in Lemmas 13 and 25 . Indeed, $f(X(t))$ is a decreasing function before time $T$; for $t /leq T$,

$$
/frac{/mathrm{d} f(X(t))}{/mathrm{d} t}=/langle/nabla f(X), /dot{X}/rangle=-/frac{3}{t}/|/dot{X}/|^{2}-/frac{1}{2} /frac{/mathrm{d}/|/dot{X}/|^{2}}{/mathrm{~d} t} /leq 0
$$

The speed restarted ODE is thus

$$
/begin{equation*}
/ddot{X}(t)+/frac{3}{t_{/mathrm{sr}}} /dot{X}(t)+/nabla f(X(t))=0 /tag{29}
/end{equation*}
$$

where $t_{/mathrm{sr}}$ is set to zero whenever $/langle/dot{X}, /ddot{X}/rangle=0$ and between two consecutive restarts, $t_{/text {sr }}$ grows just as $t$. That is, $t_{/mathrm{sr}}=t-/tau$, where $/tau$ is the latest restart time. In particular, $t_{/mathrm{sr}}=0$ at $t=0$. Letting $X^{/text {sr }}$ be the solution to (29), we have the following observations.

- $X^{/mathrm{sr}}(t)$ is continuous for $t /geq 0$, with $X^{/mathrm{sr}}(0)=x_{0}$;
- $X^{/text {sr }}(t)$ satisfies (3) for $0<t<T_{1}:=T/left(x_{0} ; f/right)$.
- Recursively define $T_{i+1}=T/left(X^{/mathrm{sr}}/left(/sum_{j=1}^{i} T_{j}/right) ; f/right)$ for $i /geq 1$, and $/widetilde{X}(t):=X^{/mathrm{sr}}/left(/sum_{j=1}^{i} T_{j}+t/right)$ satisfies the $/operatorname{ODE}(3)$, with $/widetilde{X}(0)=X^{/text {sr }}/left(/sum_{j=1}^{i} T_{j}/right)$, for $0<t<T_{i+1}$.

The theorem below guarantees linear convergence of $X^{/mathrm{sr}}$. This is a new result in the literature (O'Donoghue and Candès, 2013; Monteiro et al., 2012). The proof of Theorem 10 is based on Lemmas 12 and 13 , where the first guarantees the rate $f/left(X^{/mathrm{sr}}/right)-f^{/star}$ decays by a constant factor for each restarting, and the second confirms that restartings are adequate. In these lemmas we all make a convention that the uninteresting case $x_{0}=x^{/star}$ is excluded.

Theorem 10 There exist positive constants $c_{1}$ and $c_{2}$, which only depend on the condition number $L / /mu$, such that for any $f /in /mathcal{S}_{/mu, L}$, we have

$$
f/left(X^{/mathrm{sr}}(t)/right)-f^{/star} /leq /frac{c_{1} L/left/|x_{0}-x^{/star}/right/|^{2}}{2} /mathrm{e}^{-c_{2} t /sqrt{L}}
$$

Before turning to the proof, we make a remark that this linear convergence of $X^{/text {sr }}$ remains to hold for the generalized ODE (17) with $r>3$. Only minor modifications in the proof below are needed, such as replacing $u^{3}$ by $u^{r}$ in the definition of $I(t)$ in Lemma 25 .

### 5.2 Proof of Linear Convergence

First, we collect some useful estimates. Denote by $M(t)$ the supremum of $/|/dot{X}(u)/| / u$ over $u /in(0, t]$ and let

$$
I(t):=/int_{0}^{t} u^{3}/left(/nabla f(X(u))-/nabla f/left(x_{0}/right)/right) /mathrm{d} u
$$

It is guaranteed that $M$ defined above is finite, for example, see the proof of Lemma 18 . The definition of $M$ gives a bound on the gradient of $f$,

$$
/left/|/nabla f(X(t))-/nabla f/left(x_{0}/right)/right/| /leq L/left/|/int_{0}^{t} /dot{X}(u) /mathrm{d} u/right/| /leq L /int_{0}^{t} u /frac{/|/dot{X}(u)/|}{u} /mathrm{~d} u /leq /frac{L M(t) t^{2}}{2}
$$

Hence, it is easy to see that $I$ can also be bounded via $M$,

$$
/|I(t)/| /leq /int_{0}^{t} u^{3}/left/|/nabla f(X(u))-/nabla f/left(x_{0}/right)/right/| /mathrm{d} u /leq /int_{0}^{t} /frac{L M(u) u^{5}}{2} /mathrm{~d} u /leq /frac{L M(t) t^{6}}{12}
$$

To fully facilitate these estimates, we need the following lemma that gives an upper bound of $M$, whose proof is deferred to the appendix.

Lemma 11 For $t</sqrt{12 / L}$, we have

$$
M(t) /leq /frac{/left/|/nabla f/left(x_{0}/right)/right/|}{4/left(1-L t^{2} / 12/right)}
$$

Next we give a lemma which claims that the objective function decays by a constant through each speed restarting.

Lemma 12 There is a universal constant $C>0$ such that

$$
f(X(T))-f^{/star} /leq/left(1-/frac{C /mu}{L}/right)/left(f/left(x_{0}/right)-f^{/star}/right)
$$

Proof By Lemma 11, for $t</sqrt{12 / L}$ we have

$$
/left/|/dot{X}(t)+/frac{t}{4} /nabla f/left(x_{0}/right)/right/|=/frac{1}{t^{3}}/|I(t)/| /leq /frac{L M(t) t^{3}}{12} /leq /frac{L/left/|/nabla f/left(x_{0}/right)/right/| t^{3}}{48/left(1-L t^{2} / 12/right)}
$$

which yields

$$
/begin{equation*}
0 /leq /frac{t}{4}/left/|/nabla f/left(x_{0}/right)/right/|-/frac{L/left/|/nabla f/left(x_{0}/right)/right/| t^{3}}{48/left(1-L t^{2} / 12/right)} /leq/|/dot{X}(t)/| /leq /frac{t}{4}/left/|/nabla f/left(x_{0}/right)/right/|+/frac{L/left/|/nabla f/left(x_{0}/right)/right/| t^{3}}{48/left(1-L t^{2} / 12/right)} /tag{30}
/end{equation*}
$$

Hence, for $0<t<4 /(5 /sqrt{L})$ we get

$$
/begin{aligned}
/frac{/mathrm{d} f(X)}{/mathrm{d} t}=-/frac{3}{t}/|/dot{X}/|^{2}-/frac{1}{2} /frac{/mathrm{d}}{/mathrm{d} t}/|/dot{X}/|^{2} & /leq-/frac{3}{t}/|/dot{X}/|^{2} //
& /leq-/frac{3}{t}/left(/frac{t}{4}/left/|/nabla f/left(x_{0}/right)/right/|-/frac{L/left/|/nabla f/left(x_{0}/right)/right/| t^{3}}{48/left(1-L t^{2} / 12/right)}/right)^{2} /leq-C_{1} t/left/|/nabla f/left(x_{0}/right)/right/|^{2}
/end{aligned}
$$

where $C_{1}>0$ is an absolute constant and the second inequality follows from Lemma 25 in the appendix. Consequently,

$$
f(X(4 /(5 /sqrt{L})))-f/left(x_{0}/right) /leq /int_{0}^{/frac{4}{5 /sqrt{L}}}-C_{1} u/left/|/nabla f/left(x_{0}/right)/right/|^{2} /mathrm{~d} u /leq-/frac{C /mu}{L}/left(f/left(x_{0}/right)-f^{/star}/right)
$$

where $C=16 C_{1} / 25$ and in the last inequality we use the $/mu$-strong convexity of $f$. Thus we have

$$
f/left(X/left(/frac{4}{5 /sqrt{L}}/right)/right)-f^{/star} /leq/left(1-/frac{C /mu}{L}/right)/left(f/left(x_{0}/right)-f^{/star}/right)
$$

To complete the proof, note that $f(X(T)) /leq f(X(4 /(5 /sqrt{L})))$ by Lemma 25 .

With each restarting reducing the error $f-f^{/star}$ by a constant a factor, we still need the following lemma to ensure sufficiently many restartings.

Lemma 13 There is a universal constant $/tilde{C}$ such that

$$
T /leq /frac{4 /exp (/tilde{C} L / /mu)}{5 /sqrt{L}}
$$

Proof For $4 /(5 /sqrt{L}) /leq t /leq T$, we have $/frac{/mathrm{d} f(X)}{/mathrm{d} t} /leq-/frac{3}{t}/|/dot{X}(t)/|^{2} /leq-/frac{3}{t}/|/dot{X}(4 /(5 /sqrt{L}))/|^{2}$, which implies

$$
f(X(T))-f/left(x_{0}/right) /leq-/int_{/frac{4}{5 /sqrt{L}}}^{T} /frac{3}{t}/|/dot{X}(4 /(5 /sqrt{L}))/|^{2} /mathrm{~d} t=-3/|/dot{X}(4 /(5 /sqrt{L}))/|^{2} /log /frac{5 T /sqrt{L}}{4}
$$

Hence, we get an upper bound for $T$,

$$
T /leq /frac{4}{5 /sqrt{L}} /exp /left(/frac{f/left(x_{0}/right)-f(X(T))}{3/|/dot{X}(4 /(5 /sqrt{L}))/|^{2}}/right) /leq /frac{4}{5 /sqrt{L}} /exp /left(/frac{f/left(x_{0}/right)-f^{/star}}{3/|/dot{X}(4 /(5 /sqrt{L}))/|^{2}}/right)
$$

Plugging $t=4 /(5 /sqrt{L})$ into (30) gives $/|/dot{X}(4 /(5 /sqrt{L}))/| /geq /frac{C_{1}}{/sqrt{L}}/left/|/nabla f/left(x_{0}/right)/right/|$ for some universal constant $C_{1}>0$. Hence, from the last display we get

$$
T /leq /frac{4}{5 /sqrt{L}} /exp /left(/frac{L/left(f/left(x_{0}/right)-f^{/star}/right)}{3 C_{1}^{2}/left/|/nabla f/left(x_{0}/right)/right/|^{2}}/right) /leq /frac{4}{5 /sqrt{L}} /exp /frac{L}{6 C_{1}^{2} /mu}
$$

Now, we are ready to prove Theorem 10 by applying Lemmas 12 and 13.

Proof Note that Lemma 13 asserts, by time $t$ at least $m:=/left/lfloor 5 t /sqrt{L} /mathrm{e}^{-/tilde{C} L / /mu} / 4/right/rfloor$ restartings have occurred for $X^{/text {sr }}$. Hence, recursively applying Lemma 12, we have

$$
/begin{aligned}
f/left(X^{/mathrm{sr}}(t)/right)-f^{/star} & /leq f/left(X^{/mathrm{sr}}/left(T_{1}+/cdots+T_{m}/right)/right)-f^{/star} //
& /leq(1-C /mu / L)/left(f/left(X^{/mathrm{sr}}/left(T_{1}+/cdots+T_{m-1}/right)/right)-f^{/star}/right) //
& /leq /cdots /leq /cdots //
& /leq(1-C /mu / L)^{m}/left(f/left(x_{0}/right)-f^{/star}/right) /leq /mathrm{e}^{-C /mu m / L}/left(f/left(x_{0}/right)-f^{/star}/right) //
& /leq c_{1} /mathrm{e}^{-c_{2} t /sqrt{L}}/left(f/left(x_{0}/right)-f^{/star}/right) /leq /frac{c_{1} L/left/|x_{0}-x^{/star}/right/|^{2}}{2} /mathrm{e}^{-c_{2} t /sqrt{L}}
/end{aligned}
$$

where $c_{1}=/exp (C /mu / L)$ and $c_{2}=5 C /mu e^{-/tilde{C} /mu / L} /(4 L)$.

In closing, we remark that we believe that estimate in Lemma 12 is tight, while not for Lemma 13. Thus we conjecture that for a large class of $f /in /mathcal{S}_{/mu, L}$, if not all, $T=O(/sqrt{L} / /mu)$. If this is true, the exponent constant $c_{2}$ in Theorem 10 can be significantly improved.

### 5.3 Numerical Examples

Below we present a discrete analog to the restarted scheme. There, $k_{/min }$ is introduced to avoid having consecutive restarts that are too close. To compare the performance of the restarted scheme with the original (1), we conduct four simulation studies, including both smooth and non-smooth objective functions. Note that the computational costs of the restarted and non-restarted schemes are the same.

```
Algorithm 1 Speed Restarting Nesterov's Scheme
    input: $x_{0} /in /mathbb{R}^{n}, y_{0}=x_{0}, x_{-1}=x_{0}, 0<s /leq 1 / L, k_{/max } /in /mathbb{N}^{+}$and $k_{/min } /in /mathbb{N}^{+}$
    $j /leftarrow 1$
    for $k=1$ to $k_{/text {max }}$ do
        $x_{k} /leftarrow /operatorname{argmin}_{x}/left(/frac{1}{2 s}/left/|x-y_{k-1}+s /nabla g/left(y_{k-1}/right)/right/|^{2}+h(x)/right)$
        $y_{k} /leftarrow x_{k}+/frac{j-1}{j+2}/left(x_{k}-x_{k-1}/right)$
        if $/left/|x_{k}-x_{k-1}/right/|</left/|x_{k-1}-x_{k-2}/right/|$ and $j /geq k_{/min }$ then
            $j /leftarrow 1$
        else
            $j /leftarrow j+1$
        end if
    end for
```

Quadratic. $f(x)=/frac{1}{2} x^{T} A x+b^{T} x$ is a strongly convex function, in which $A$ is a $500 /times 500$ random positive definite matrix and $b$ a random vector. The eigenvalues of $A$ are between 0.001 and 1. The vector $b$ is generated as i.i.d. Gaussian random variables with mean 0 and variance 25 .

## Log-sum-exp.

$$
f(x)=/rho /log /left[/sum_{i=1}^{m} /exp /left(/left(a_{i}^{T} x-b_{i}/right) / /rho/right)/right]
$$

where $n=50, m=200, /rho=20$. The matrix $A=/left(a_{i j}/right)$ is a random matrix with i.i.d. standard Gaussian entries, and $b=/left(b_{i}/right)$ has i.i.d. Gaussian entries with mean 0 and variance 2. This function is not strongly convex.

Matrix completion. $f(X)=/frac{1}{2}/left/|X_{/text {obs }}-M_{/text {obs }}/right/|_{F}^{2}+/lambda/|X/|_{*}$, in which the ground truth $M$ is a rank- 5 random matrix of size $300 /times 300$. The regularization parameter is set to $/lambda=0.05$. The 5 singular values of $M$ are $1, /ldots, 5$. The observed set is independently sampled among the $300 /times 300$ entries so that $10 /%$ of the entries are actually observed.

Lasso in $/ell_{1}$-constrained form with large sparse design. $f(x)=/frac{1}{2}/|A x-b/|^{2} /quad$ s.t. $/|x/|_{1} /leq$ $/delta$, where $A$ is a $5000 /times 50000$ random sparse matrix with nonzero probability $0.5 /%$ for each

## Su, Boyd and Candès

![](https://cdn.mathpix.com/cropped/2024_08_12_d19cd0b096d3ee24c53cg-26.jpg?height=2063&width=1565&top_left_y=283&top_left_x=280)

Figure 6: Numerical performance of speed restarting ( srN ), gradient restarting ( grN ), the original Nesterov's scheme (oN) and the proximal gradient (PG).
entry and $b$ is generated as $b=A x^{0}+z$. The nonzero entries of $A$ independently follow the Gaussian distribution with mean 0 and variance 0.04 . The signal $x^{0}$ is a vector with 250 nonzeros and $z$ is i.i.d. standard Gaussian noise. The parameter $/delta$ is set to $/left/|x^{0}/right/|_{1}$.

Sorted $/ell_{1}$ penalized estimation. $f(x)=/frac{1}{2}/|A x-b/|^{2}+/sum_{i=1}^{p} /lambda_{i}|x|_{(i)}$, where $|x|_{(1)} /geq /cdots /geq$ $|x|_{(p)}$ are the order statistics of $|x|$. This is a recently introduced testing and estimation procedure (Bogdan et al., 2015). The design $A$ is a $1000 /times 10000$ Gaussian random matrix, and $b$ is generated as $b=A x^{0}+z$ for 20 -sparse $x^{0}$ and Gaussian noise $z$. The penalty sequence is set to $/lambda_{i}=1.1 /Phi^{-1}(1-0.05 i /(2 p))$.

Lasso. $f(x)=/frac{1}{2}/|A x-b/|^{2}+/lambda/|x/|_{1}$, where $A$ is a $1000 /times 500$ random matrix and $b$ is given as $b=A x^{0}+z$ for 20 -sparse $x^{0}$ and Gaussian noise $z$. We set $/lambda=1.5 /sqrt{2 /log p}$.

$/ell_{1}$-regularized logistic regression. $f(x)=/sum_{i=1}^{n}-y_{i} a_{i}^{T} x+/log /left(1+/mathrm{e}^{a_{i}^{T} x}/right)+/lambda/|x/|_{1}$, where the setting is the same as in Figure 5f. The results are presented in Figure 6g

Logistic regression with large sparse design. $f(x)=/sum_{i=1}^{n}-y_{i} a_{i}^{T} x+/log /left(1+/mathrm{e}^{a_{i}^{T} x}/right)$, in which $A=/left(a_{1}, /ldots, a_{n}/right)^{T}$ is a $10^{7} /times 20000$ sparse random matrix with nonzero probability $0.1 /%$ for each entry, so there are roughly $2 /times 10^{8}$ nonzero entries in total. To generate the labels $y$, we set $x^{0}$ to be i.i.d. $/mathcal{N}(0,1 / 4)$. The plot is Figure 6h.

In these examples, $k_{/min }$ is set to be 10 and the step sizes are fixed to be $1 / L$. If the objective is in composite form, the Lipschitz bound applies to the smooth part. Figure 6 presents the performance of the speed restarting scheme, the gradient restarting scheme, the original Nesterov's scheme and the proximal gradient method. The objective functions include strongly convex, non-strongly convex and non-smooth functions, violating the assumptions in Theorem 10. Among all the examples, it is interesting to note that both speed restarting scheme empirically exhibit linear convergence by significantly reducing bumps in the objective values. This leaves us an open problem of whether there exists provable linear convergence rate for the gradient restarting scheme as in Theorem 10. It is also worth pointing out that compared with gradient restarting, the speed restarting scheme empirically exhibits more stable linear convergence rate.

## 6. Discussion

This paper introduces a second-order ODE and accompanying tools for characterizing Nesterov's accelerated gradient method. This ODE is applied to study variants of Nesterov's scheme and is capable of interpreting some empirically observed phenomena, such as oscillations along the trajectories. Our approach suggests (1) a large family of generalized Nesterov's schemes that are all guaranteed to converge at the rate $O/left(1 / k^{2}/right)$, and (2) a restarting scheme provably achieving a linear convergence rate whenever $f$ is strongly convex.

In this paper, we often utilize ideas from continuous-time ODEs, and then apply these ideas to discrete schemes. The translation, however, involves parameter tuning and tedious calculations. This is the reason why a general theory mapping properties of ODEs into corresponding properties for discrete updates would be a welcome advance. Indeed, this would allow researchers to only study the simpler and more user-friendly ODEs.

As evidenced by many examples, the viewpoint of regarding the ODE as a surrogate for Nesterov's scheme would allow a new perspective for studying accelerated methods in optimization. The discrete scheme and the ODE are closely connected by the exact
mapping between the coefficients of momentum (e.g. $(k-1) /(k+2)$ ) and velocity (e.g. $3 / t)$. The derivations of generalized Nesterov's schemes and the speed restarting scheme are both motivated by trying a different velocity coefficient, in which the surprising phase transition at 3 is observed. Clearly, such alternatives are endless, and we expect this will lead to findings of many discrete accelerated schemes. In a different direction, a better understanding of the trajectory of the ODEs, such as curvature, has the potential to be helpful in deriving appropriate stopping criteria for termination, and choosing step size by backtracking.

## Acknowledgments

W. S. was partially supported by a General Wang Yaowu Stanford Graduate Fellowship. S. B. was partially supported by DARPA XDATA. E. C. was partially supported by AFOSR under grant FA9550-09-1-0643, by NSF under grant CCF-0963835, and by the Math + X Award from the Simons Foundation. We would like to thank Carlos Sing-Long, Zhou Fan, and Xi Chen for helpful discussions about parts of this paper. We would also like to thank the associate editor and two reviewers for many constructive comments that improved the presentation of the paper.

## Appendix A. Proof of Theorem 1

The proof is divided into two parts, namely, existence and uniqueness.

Lemma 14 For any $f /in /mathcal{F}_{/infty}$ and any $x_{0} /in /mathbb{R}^{n}$, the $O D E$ (3) has at least one solution $X$ in $C^{2}(0, /infty) /cap C^{1}[0, /infty)$.

Below, some preparatory lemmas are given before turning to the proof of this lemma. To begin with, for any $/delta>0$ consider the smoothed ODE

$$
/begin{equation*}
/ddot{X}+/frac{3}{/max (/delta, t)} /dot{X}+/nabla f(X)=0 /tag{31}
/end{equation*}
$$

with $X(0)=x_{0}, /dot{X}(0)=0$. Denoting by $Z=/dot{X}$, then (31) is equivalent to

$$
/frac{/mathrm{d}}{/mathrm{d} t}/binom{X}{Z}=/binom{Z}{-/frac{3}{/max (/delta, t)} Z-/nabla f(X)}
$$

with $X(0)=x_{0}, Z(0)=0$. As functions of $(X, Z)$, both $Z$ and $/left.-3 Z / /max (/delta, t)-/nabla f(X)/right)$ are $/max (1, L)+3 / /delta$-Lipschitz continuous. Hence by standard ODE theory, (31) has a unique global solution in $C^{2}[0, /infty)$, denoted by $X_{/delta}$. Note that $/ddot{X}_{/delta}$ is also well defined at $t=0$. Next, introduce $M_{/delta}(t)$ to be the supremum of $/left/|/dot{X}_{/delta}(u)/right/| / u$ over $u /in(0, t]$. It is easy to see that $M_{/delta}(t)$ is finite because $/left/|/dot{X}_{/delta}(u)/right/| / u=/left(/left/|/dot{X}_{/delta}(u)-/dot{X}_{/delta}(0)/right/|/right) / u=/left/|/ddot{X}_{/delta}(0)/right/|+o(1)$ for small $u$. We give an upper bound for $M_{/delta}(t)$ in the following lemma.

Lemma 15 For $/delta</sqrt{6 / L}$, we have

$$
M_{/delta}(/delta) /leq /frac{/left/|/nabla f/left(x_{0}/right)/right/|}{1-L /delta^{2} / 6}
$$

The proof of Lemma 15 relies on a simple lemma.

Lemma 16 For any $u>0$, the following inequality holds

$$
/left/|/nabla f/left(X_{/delta}(u)/right)-/nabla f/left(x_{0}/right)/right/| /leq /frac{1}{2} L M_{/delta}(u) u^{2}
$$

Proof By Lipschitz continuity,

$/left/|/nabla f/left(X_{/delta}(u)/right)-/nabla f/left(x_{0}/right)/right/| /leq L/left/|X_{/delta}(u)-x_{0}/right/|=/left/|/int_{0}^{u} /dot{X}_{/delta}(v) /mathrm{d} v/right/| /leq /int_{0}^{u} v /frac{/left/|/dot{X}_{/delta}(v)/right/|}{v} /mathrm{~d} v /leq /frac{1}{2} L M_{/delta}(u) u^{2}$.

Next, we prove Lemma 15

Proof For $0<t /leq /delta$, the smoothed ODE takes the form

$$
/ddot{X}_{/delta}+/frac{3}{/delta} /dot{X}_{/delta}+/nabla f/left(X_{/delta}/right)=0
$$

which yields

$/dot{X}_{/delta} /mathrm{e}^{3 t / /delta}=-/int_{0}^{t} /nabla f/left(X_{/delta}(u)/right) /mathrm{e}^{3 u / /delta} /mathrm{d} u=-/nabla f/left(x_{0}/right) /int_{0}^{t} /mathrm{e}^{3 u / /delta} /mathrm{d} u-/int_{0}^{t}/left(/nabla f/left(X_{/delta}(u)/right)-/nabla f/left(x_{0}/right)/right) /mathrm{e}^{3 u / /delta} /mathrm{d} u$.

Hence, by Lemma 16

$$
/begin{aligned}
/frac{/left/|/dot{X}_{/delta}(t)/right/|}{t} & /leq /frac{1}{t} /mathrm{e}^{-3 t / /delta}/left/|/nabla f/left(x_{0}/right)/right/| /int_{0}^{t} /mathrm{e}^{3 u / /delta} /mathrm{d} u+/frac{1}{t} /mathrm{e}^{-3 t / /delta} /int_{0}^{t} /frac{1}{2} L M_{/delta}(u) u^{2} /mathrm{e}^{3 u / /delta} /mathrm{d} u //
& /leq/left/|/nabla f/left(x_{0}/right)/right/|+/frac{L M_{/delta}(/delta) /delta^{2}}{6}
/end{aligned}
$$

Taking the supremum of $/left/|/dot{X}_{/delta}(t)/right/| / t$ over $0<t /leq /delta$ and rearranging the inequality give the desired result.

Next, we give an upper bound for $M_{/delta}(t)$ when $t>/delta$.

Lemma 17 For $/delta</sqrt{6 / L}$ and $/delta<t</sqrt{12 / L}$, we have

$$
M_{/delta}(t) /leq /frac{/left(5-L /delta^{2} / 6/right)/left/|/nabla f/left(x_{0}/right)/right/|}{4/left(1-L /delta^{2} / 6/right)/left(1-L t^{2} / 12/right)}
$$

Proof For $t>/delta$, the smoothed ODE takes the form

$$
/ddot{X}_{/delta}+/frac{3}{t} /dot{X}_{/delta}+/nabla f/left(X_{/delta}/right)=0
$$

which is equivalent to

$$
/frac{/mathrm{d} t^{3} /dot{X}_{/delta}(t)}{/mathrm{d} t}=-t^{3} /nabla f/left(X_{/delta}(t)/right)
$$

Hence, by integration, $t^{3} /dot{X}_{/delta}(t)$ is equal to

$$
-/int_{/delta}^{t} u^{3} /nabla f/left(X_{/delta}(u)/right) /mathrm{d} u+/delta^{3} /dot{X}_{/delta}(/delta)=-/int_{/delta}^{t} u^{3} /nabla f/left(x_{0}/right) /mathrm{d} u-/int_{/delta}^{t} u^{3}/left(/nabla f/left(X_{/delta}(u)/right)-/nabla f/left(x_{0}/right)/right) /mathrm{d} u+/delta^{3} /dot{X}_{/delta}(/delta)
$$

Therefore by Lemmas 16 and 15 , we get

$$
/begin{aligned}
/frac{/left/|/dot{X}_{/delta}(t)/right/|}{t} & /leq /frac{t^{4}-/delta^{4}}{4 t^{4}}/left/|/nabla f/left(x_{0}/right)/right/|+/frac{1}{t^{4}} /int_{/delta}^{t} /frac{1}{2} L M_{/delta}(u) u^{5} /mathrm{~d} u+/frac{/delta^{4}}{t^{4}} /frac{/left/|/dot{X}_{/delta}(/delta)/right/|}{/delta} //
& /leq /frac{1}{4}/left/|/nabla f/left(x_{0}/right)/right/|+/frac{1}{12} L M_{/delta}(t) t^{2}+/frac{/left/|/nabla f/left(X_{0}/right)/right/|}{1-L /delta^{2} / 6}
/end{aligned}
$$

where the last expression is an increasing function of $t$. So for any $/delta<t^{/prime}<t$, it follows that

$$
/frac{/left/|/dot{X}_{/delta}/left(t^{/prime}/right)/right/|}{t^{/prime}} /leq /frac{1}{4}/left/|/nabla f/left(x_{0}/right)/right/|+/frac{1}{12} L M_{/delta}(t) t^{2}+/frac{/left/|/nabla f/left(x_{0}/right)/right/|}{1-L /delta^{2} / 6}
$$

which also holds for $t^{/prime} /leq /delta$. Taking the supremum over $t^{/prime} /in(0, t)$ gives

$$
M_{/delta}(t) /leq /frac{1}{4}/left/|/nabla f/left(x_{0}/right)/right/|+/frac{1}{12} L M_{/delta}(t) t^{2}+/frac{/left/|/nabla f/left(X_{0}/right)/right/|}{1-L /delta^{2} / 6}
$$

The desired result follows from rearranging the inequality.

Lemma 18 The function class $/mathcal{F}=/left/{X_{/delta}:[0, /sqrt{6 / L}] /rightarrow /mathbb{R}^{n} /mid /delta=/sqrt{3 / L} / 2^{m}, m=0,1, /ldots/right/}$ is uniformly bounded and equicontinuous.

Proof By Lemmas 15 and 17 , for any $t /in[0, /sqrt{6 / L}], /delta /in(0, /sqrt{3 / L})$ the gradient is uniformly bounded as

$/left/|/dot{X}_{/delta}(t)/right/| /leq /sqrt{6 / L} M_{/delta}(/sqrt{6 / L}) /leq /sqrt{6 / L} /max /left/{/frac{/left/|/nabla f/left(x_{0}/right)/right/|}{1-/frac{1}{2}}, /frac{5/left/|/nabla f/left(x_{0}/right)/right/|}{4/left(1-/frac{1}{2}/right)/left(1-/frac{1}{2}/right)}/right/}=5 /sqrt{6 / L}/left/|/nabla f/left(x_{0}/right)/right/|$.

Thus it immediately implies that $/mathcal{F}$ is equicontinuous. To establish the uniform boundedness, note that

$$
/left/|X_{/delta}(t)/right/| /leq/left/|X_{/delta}(0)/right/|+/int_{0}^{t}/left/|/dot{X}_{/delta}(u)/right/| /mathrm{d} u /leq/left/|x_{0}/right/|+30/left/|/nabla f/left(x_{0}/right)/right/| / L
$$

We are now ready for the proof of Lemma 14 .

Proof By the Arzelá-Ascoli theorem and Lemma 18, $/mathcal{F}$ contains a subsequence converging uniformly on $[0, /sqrt{6 / L}]$. Denote by $/left/{X_{/delta_{m_{i}}}/right/}_{i /in /mathbb{N}}$ the convergent subsequence and $/breve{X}$ the limit. Above, $/delta_{m_{i}}=/sqrt{3 / L} / 2^{m_{i}}$ decreases as $i$ increases. We will prove that $/breve{X}$ satisfies (3) and the initial conditions $/breve{X}(0)=x_{0}, /dot{/grave{X}}(0)=0$.

Fix an arbitrary $t_{0} /in(0, /sqrt{6 / L})$. Since $/left/|/dot{X}_{/delta_{m_{i}}}/left(t_{0}/right)/right/|$ is bounded, we can pick a subsequence of $/dot{X}_{/delta_{m_{i}}}/left(t_{0}/right)$ which converges to a limit, denoted by $X_{t_{0}}^{D}$. Without loss of generality, assume the subsequence is the original sequence. Denote by $/tilde{X}$ the local solution to (3) with $X/left(t_{0}/right)=$ $/breve{X}/left(t_{0}/right)$ and $/dot{X}/left(t_{0}/right)=X_{t_{0}}^{D}$. Now recall that $X_{/delta_{m_{i}}}$ is the solution to (3) with $X/left(t_{0}/right)=X_{/delta_{m_{i}}}/left(t_{0}/right)$ and $/dot{X}/left(t_{0}/right)=/dot{X}_{/delta_{m_{i}}}/left(t_{0}/right)$ when $/delta_{m_{i}}<t_{0}$. Since both $X_{/delta_{m_{i}}}/left(t_{0}/right)$ and $/dot{X}_{/delta_{m_{i}}}/left(t_{0}/right)$ approach $/dot{X}/left(t_{0}/right)$ and $X_{t_{0}}^{D}$, respectively, there exists $/epsilon_{0}>0$ such that

$$
/sup _{t_{0}-/epsilon_{0}<t<t_{0}+/epsilon_{0}}/left/|X_{/delta_{m_{i}}}(t)-/tilde{X}(t)/right/| /rightarrow 0
$$

as $i /rightarrow /infty$. However, by definition we have

$$
/sup _{t_{0}-/epsilon_{0}<t<t_{0}+/epsilon_{0}}/left/|X_{/delta_{m_{i}}}(t)-/breve{X}(t)/right/| /rightarrow 0
$$

Therefore $/breve{X}$ and $/tilde{X}$ have to be identical on $/left(t_{0}-/epsilon_{0}, t_{0}+/epsilon_{0}/right)$. So $/breve{X}$ satisfies (3) at $t_{0}$. Since $t_{0}$ is arbitrary, we conclude that $/breve{X}$ is a solution to (3) on $(0, /sqrt{6 / L})$. By extension, $/breve{X}$ can be a global solution to $(3)$ on $(0, /infty)$. It only leaves to verify the initial conditions to complete the proof.

The first condition $/breve{X}(0)=x_{0}$ is a direct consequence of $X_{/delta_{m_{i}}}(0)=x_{0}$. To check the second, pick a small $t>0$ and note that

$$
/begin{aligned}
/frac{/|/breve{X}(t)-/breve{X}(0)/|}{t}=/lim _{i /rightarrow /infty} /frac{/left/|X_{/delta_{m_{i}}}(t)-X_{/delta_{m_{i}}}(0)/right/|}{t}= & /lim _{i /rightarrow /infty}/left/|/dot{X}_{/delta_{m_{i}}}/left(/xi_{i}/right)/right/| //
& /leq /limsup _{i /rightarrow /infty} t M_{/delta_{m_{i}}}(t) /leq 5 t /sqrt{6 / L}/left/|/nabla f/left(x_{0}/right)/right/|
/end{aligned}
$$

where $/xi_{i} /in(0, t)$ is given by the mean value theorem. The desired result follows from taking $t /rightarrow 0$.

Next, we aim to prove the uniqueness of the solution to (3).

Lemma 19 For any $f /in /mathcal{F}_{/infty}$, the $O D E$ (3) has at most one local solution in a neighborhood of $t=0$.

Suppose on the contrary that there are two solutions, namely, $X$ and $Y$, both defined on $(0, /alpha)$ for some $/alpha>0$. Define $/tilde{M}(t)$ to be the supremum of $/|/dot{X}(u)-/dot{Y}(u)/|$ over $u /in[0, t)$. To proceed, we need a simple auxiliary lemma.

Lemma 20 For any $t /in(0, /alpha)$, we have

$$
/|/nabla f(X(t))-/nabla f(Y(t))/| /leq /operatorname{Lt} /tilde{M}(t)
$$

Proof By Lipschitz continuity of the gradient, one has

$$
/begin{aligned}
&/|/nabla f(X(t))-/nabla f(Y(t))/| /leq L/|X(t)-Y(t)/|=L /| /int_{0}^{t} /dot{X}(u)-/dot{Y}(u) /mathrm{d} u+X(0)-Y(0) /| //
& /leq L /int_{0}^{t}/|/dot{X}(u)-/dot{Y}(u)/| /mathrm{d} u /leq L t /tilde{M}(t)
/end{aligned}
$$

Now we prove Lemma 19

Proof Similar to the proof of Lemma 17, we get

$$
t^{3}(/dot{X}(t)-/dot{Y}(t))=-/int_{0}^{t} u^{3}(/nabla f(X(u))-/nabla f(Y(u))) /mathrm{d} u
$$

Applying Lemma 20 gives

$$
t^{3}/|/dot{X}(t)-/dot{Y}(t)/| /leq /int_{0}^{t} L u^{4} /tilde{M}(u) /mathrm{d} u /leq /frac{1}{5} L t^{5} /tilde{M}(t)
$$

which can be simplified as $/|/dot{X}(t)-/dot{Y}(t)/| /leq L t^{2} /tilde{M}(t) / 5$. Thus, for any $t^{/prime} /leq t$ it is true that $/left/|/dot{X}/left(t^{/prime}/right)-/dot{Y}/left(t^{/prime}/right)/right/| /leq L t^{2} /tilde{M}(t) / 5$. Taking the supremum of $/left/|/dot{X}/left(t^{/prime}/right)-/dot{Y}/left(t^{/prime}/right)/right/|$ over $t^{/prime} /in(0, t)$ gives $/tilde{M}(t) /leq L t^{2} /tilde{M}(t) / 5$. Therefore $/tilde{M}(t)=0$ for $t</min (/alpha, /sqrt{5 / L})$, which is equivalent to saying $/dot{X}=/dot{Y}$ on $/left[0, /min (/alpha, /sqrt{5 / L})/right.$. With the same initial value $X(0)=Y(0)=x_{0}$ and the same gradient, we conclude that $X$ and $Y$ are identical on $(0, /min (/alpha, /sqrt{5 / L})$, a contradiction.

Given all of the aforementioned lemmas, Theorem 1 follows from a combination of Lemmas 14 and 19

## Appendix B. Proof of Theorem 2

Identifying $/sqrt{s}=/Delta t$, the comparison between (4) and (15) reveals that Nesterov's scheme is a discrete scheme for numerically integrating the ODE (3). However, its singularity of the damping coefficient at $t=0$ leads to the nonexistence of off-the-shelf ODE theory for proving Theorem 2. To address this difficulty, we use the smoothed ODE (31) to approximate the original one; then bound the difference between Nesterov's scheme and the forward Euler scheme of (31), which may take the following form:

$$
/begin{align*}
X_{k+1}^{/delta} & =X_{k}^{/delta}+/Delta t Z_{k}^{/delta} //
Z_{k+1}^{/delta} & =/left(1-/frac{3 /Delta t}{/max /{/delta, k /Delta t/}}/right) Z_{k}^{/delta}-/Delta t /nabla f/left(X_{k}^{/delta}/right) /tag{32}
/end{align*}
$$

with $X_{0}^{/delta}=x_{0}$ and $Z_{0}^{/delta}=0$.

Lemma 21 With step size $/Delta t=/sqrt{s}$, for any $T>0$ we have

$$
/max _{1 /leq k /leq /frac{T}{/sqrt{s}}}/left/|X_{k}^{/delta}-x_{k}/right/| /leq C /delta^{2}+o_{s}(1)
$$

for some constant $C$.

Proof Let $z_{k}=/left(x_{k+1}-x_{k}/right) / /sqrt{s}$. Then Nesterov's scheme is equivalent to

$$
/begin{align*}
& x_{k+1}=x_{k}+/sqrt{s} z_{k} //
& z_{k+1}=/left(1-/frac{3}{k+3}/right) z_{k}-/sqrt{s} /nabla f/left(x_{k}+/frac{2 k+3}{k+3} /sqrt{s} z_{k}/right) /tag{33}
/end{align*}
$$

Denote by $a_{k}=/left/|X_{k}^{/delta}-x_{k}/right/|, /quad b_{k}=/left/|Z_{k}^{/delta}-z_{k}/right/|$, whose initial values are $a_{0}=0$ and $b_{0}=$ $/left/|/nabla f/left(x_{0}/right)/right/| /sqrt{s}$. The idea of this proof is to bound $a_{k}$ via simultaneously estimating $a_{k}$ and $b_{k}$. By comparing (32) and (33), we get the iterative relationship for $a_{k}: a_{k+1} /leq a_{k}+/sqrt{s} b_{k}$. Denoting by $S_{k}=b_{0}+b_{1}+/cdots+b_{k}$, this yields

$$
/begin{equation*}
a_{k} /leq /sqrt{s} S_{k-1} /tag{34}
/end{equation*}
$$

Similarly, for sufficiently small $s$ we get

$$
/begin{aligned}
b_{k+1} & /leq/left|1-/frac{3}{/max /{/delta / /sqrt{s}, k/}}/right| b_{k}+L /sqrt{s} a_{k}+/left(/left|/frac{3}{k+3}-/frac{3}{/max /{/delta / /sqrt{s}, k/}}/right|+2 L s/right)/left/|z_{k}/right/| //
& /leq b_{k}+L /sqrt{s} a_{k}+/left(/left|/frac{3}{k+3}-/frac{3}{/max /{/delta / /sqrt{s}, k/}}/right|+2 L s/right)/left/|z_{k}/right/|
/end{aligned}
$$

To upper bound $/left/|z_{k}/right/|$, denoting by $C_{1}$ the supremum of $/sqrt{2 L/left(f/left(y_{k}/right)-f^{/star}/right)}$ over all $k$ and $s$, we have

$$
/left/|z_{k}/right/| /leq /frac{k-1}{k+2}/left/|z_{k-1}/right/|+/sqrt{s}/left/|/nabla f/left(y_{k}/right)/right/| /leq/left/|z_{k-1}/right/|+C_{1} /sqrt{s}
$$

which gives $/left/|z_{k}/right/| /leq C_{1}(k+1) /sqrt{s}$. Hence,

$$
/left(/left|/frac{3}{k+3}-/frac{3}{/max /{/delta / /sqrt{s}, k/}}/right|+2 L s/right)/left/|z_{k}/right/| /leq/left/{/begin{array}{l}
C_{2} /sqrt{s}, /quad k /leq /frac{/delta}{/sqrt{s}} //
/frac{C_{2} /sqrt{s}}{k}</frac{C_{2} s}{/delta}, /quad k>/frac{/delta}{/sqrt{s}}
/end{array}/right.
$$

Making use of (34) gives

$$
b_{k+1} /leq/left/{/begin{array}{l}
b_{k}+L s S_{k-1}+C_{2} /sqrt{s}, /quad k /leq /delta / /sqrt{s}  /tag{35}//
b_{k}+L s S_{k-1}+/frac{C_{2} s}{/delta}, /quad k>/delta / /sqrt{s}
/end{array}/right.
$$

By induction on $k$, for $k /leq /delta / /sqrt{s}$ it holds that

$b_{k} /leq /frac{C_{1} L s+C_{2}+/left(C_{1}+C_{2}/right) /sqrt{L s}}{2 /sqrt{L}}(1+/sqrt{L s})^{k-1}-/frac{C_{1} L s+C_{2}-/left(C_{1}+C_{2}/right) /sqrt{L s}}{2 /sqrt{L}}(1-/sqrt{L s})^{k-1}$.

Hence,

$S_{k} /leq /frac{C_{1} L s+C_{2}+/left(C_{1}+C_{2}/right) /sqrt{L s}}{2 L /sqrt{s}}(1+/sqrt{L s})^{k}+/frac{C_{1} L s+C_{2}-/left(C_{1}+C_{2}/right) /sqrt{L s}}{2 L /sqrt{s}}(1-/sqrt{L s})^{k}-/frac{C_{2}}{L /sqrt{s}}$.

Letting $k^{/star}=/lfloor/delta / /sqrt{s}/rfloor$, we get

$$
/limsup _{s /rightarrow 0} /sqrt{s} S_{k^{/star}-1} /leq /frac{C_{2} /mathrm{e}^{/delta /sqrt{L}}+C_{2} /mathrm{e}^{-/delta /sqrt{L}}-2 C_{2}}{2 L}=O/left(/delta^{2}/right)
$$

which allows us to conclude that

$$
/begin{equation*}
a_{k} /leq /sqrt{s} S_{k-1}=O/left(/delta^{2}/right)+o_{s}(1) /tag{36}
/end{equation*}
$$

for all $k /leq /delta / /sqrt{s}$.

Next, we bound $b_{k}$ for $k>k^{/star}=/lfloor/delta / /sqrt{s}/rfloor$. To this end, we consider the worst case of (35), that is,

$$
b_{k+1}=b_{k}+L s S_{k-1}+/frac{C_{2} s}{/delta}
$$

for $k>k^{/star}$ and $S_{k^{/star}}=S_{k^{/star}+1}=C_{3} /delta^{2} / /sqrt{s}+o_{s}(1 / /sqrt{s})$ for some sufficiently large $C_{3}$. In this case, $C_{2} s / /delta<s S_{k-1}$ for sufficiently small $s$. Hence, the last display gives

$$
b_{k+1} /leq b_{k}+(L+1) s S_{k-1}
$$

By induction, we get

$$
S_{k} /leq /frac{C_{3} /delta^{2} / /sqrt{s}+o_{s}(1 / /sqrt{s})}{2}/left((1+/sqrt{(L+1) s})^{k-k^{/star}}+(1-/sqrt{(L+1) s})^{k-k^{/star}}/right)
$$

Letting $k^{/diamond}=/lfloor T / /sqrt{s}/rfloor$, we further get

$$
/limsup _{s /rightarrow 0} /sqrt{s} S_{k^{/diamond}} /leq /frac{C_{3} /delta^{2}/left(/mathrm{e}^{(T-/delta) /sqrt{L+1}}+/mathrm{e}^{-(T-/delta) /sqrt{L+1}}/right)}{2}=O/left(/delta^{2}/right)
$$

which yields

$$
a_{k} /leq /sqrt{s} S_{k-1}=O/left(/delta^{2}/right)+o_{s}(1)
$$

for $k^{/star}<k /leq k^{/diamond}$. Last, combining (36) and the last display, we get the desired result.

Now we turn to the proof of Theorem 2 .

Proof Note the triangular inequality

$$
/left/|x_{k}-X(k /sqrt{s})/right/| /leq/left/|x_{k}-X_{k}^{/delta}/right/|+/left/|X_{k}^{/delta}-X_{/delta}(k /sqrt{s})/right/|+/left/|X_{/delta}(k /sqrt{s})-X(k /sqrt{s})/right/|
$$

where $X_{/delta}(/cdot)$ is the solution to the smoothed ODE (31). The proof of Lemma 14 implies that, we can choose a sequence $/delta_{m} /rightarrow 0$ such that

$$
/sup _{0 /leq t /leq T}/left/|X_{/delta_{m}}(t)-X(t)/right/| /rightarrow 0
$$

The second term $/left/|X_{k}^{/delta_{m}}-X_{/delta_{m}}(k /sqrt{s})/right/|$ will uniformly vanish as $s /rightarrow 0$ and so does the first term $/left/|x_{k}-X_{k}^{/delta_{m}}/right/|$ if first $s /rightarrow 0$ and then $/delta_{m} /rightarrow 0$. This completes the proof.

## Appendix C. ODE for Composite Optimization

In analogy to (3) for smooth $f$ in Section 2, we develop an ODE for composite optimization,

$$
/begin{equation*}
/operatorname{minimize} f(x)=g(x)+h(x) /tag{37}
/end{equation*}
$$

where $g /in /mathcal{F}_{L}$ and $h$ is a general convex function possibly taking on the value $+/infty$. Provided it is easy to evaluate the proximal of $h$, Beck and Teboulle (2009) propose a proximal
gradient version of Nesterov's scheme for solving (37). It is to repeat the following recursion for $k /geq 1$,

$$
/begin{aligned}
& x_{k}=y_{k-1}-s G_{t}/left(y_{k-1}/right) //
& y_{k}=x_{k}+/frac{k-1}{k+2}/left(x_{k}-x_{k-1}/right)
/end{aligned}
$$

where the proximal subgradient $G_{s}$ has been defined in Section 4.1. If the constant step size $s /leq 1 / L$, it is guaranteed that (Beck and Teboulle, 2009)

$$
f/left(x_{k}/right)-f^{/star} /leq /frac{2/left/|x_{0}-x^{/star}/right/|^{2}}{s(k+1)^{2}}
$$

which in fact is a special case of Theorem 6 .

Compared to the smooth case, it is not as clear to define the driving force as $/nabla f$ in (3). At first, it might be a good try to define

$$
G(x)=/lim _{s /rightarrow 0} G_{s}(x)=/lim _{s /rightarrow 0} /frac{x-/operatorname{argmin}_{z}/left(/|z-(x-s /nabla g(x))/|^{2} /(2 s)+h(z)/right)}{s}
$$

if it exists. However, as implied in the proof of Theorem 24 stated below, this definition fails to capture the directional aspect of the subgradient. To this end, we define the subgradients through the following lemma.

Lemma 22 ('Rockafellar, 1997) For any convex function $f$ and any $x, p /in /mathbb{R}^{n}$, the directional derivative $/lim _{t /rightarrow 0+}(f(x+s p)-f(x)) / s$ exists, and can be evaluated as

$$
/lim _{s /rightarrow 0+} /frac{f(x+s p)-f(x)}{s}=/sup _{/xi /in /partial f(x)}/langle/xi, p/rangle
$$

Note that the directional derivative is semilinear in $p$ because

$$
/sup _{/xi /in /partial f(x)}/langle/xi, c p/rangle=c /sup _{/xi /in /partial f(x)}/langle/xi, p/rangle
$$

for any $c>0$.

Definition $23 A$ Borel measurable function $G(x, p ; f)$ defined on $/mathbb{R}^{n} /times /mathbb{R}^{n}$ is said to be $a$ directional subgradient of $f$ if

$$
/begin{aligned}
& G(x, p) /in /partial f(x) //
& /langle G(x, p), p/rangle=/sup _{/xi /in /partial f(x)}/langle/xi, p/rangle
/end{aligned}
$$

for all $x, p$.

Convex functions are naturally locally Lipschitz, so $/partial f(x)$ is compact for any $x$. Consequently there exists $/xi /in /partial f(x)$ which maximizes $/langle/xi, p/rangle$. So Lemma 22 guarantees the existence of a directional subgradient. The function $G$ is essentially a function defined on $/mathbb{R}^{n} /times /mathbb{S}^{n-1}$ in that we can define

$$
G(x, p)=G(x, p //|p/|)
$$

and $G(x, 0)$ to be any element in $/partial f(x)$. Now we give the main theorem. However, note that we do not guarantee the existence of solution to (38).

Theorem 24 Given a convex function $f(x)$ with directional subgradient $G(x, p ; f)$, assume that the second order $O D E$

$$
/begin{equation*}
/ddot{X}+/frac{3}{t} /dot{X}+G(X, /dot{X})=0, /quad X(0)=x_{0}, /dot{X}(0)=0 /tag{38}
/end{equation*}
$$

admits a solution $X(t)$ on $[0, /alpha)$ for some $/alpha>0$. Then for any $0<t</alpha$, we have

$$
f(X(t))-f^{/star} /leq /frac{2/left/|x_{0}-x^{/star}/right/|_{2}^{2}}{t^{2}}
$$

Proof It suffices to establish that $/mathcal{E}$, first defined in the proof of Theorem 3 , is monotonically decreasing. The difficulty comes from that $/mathcal{E}$ may not be differentiable in this setting. Instead, we study $(/mathcal{E}(t+/Delta t)-/mathcal{E}(t)) / /Delta t$ for small $/Delta t>0$. In $/mathcal{E}$, the second term $2 /| X+$ $t /dot{X} / 2-x^{/star} /|^{2}$ is differentiable, with derivative $4/left/langle X+/frac{t}{2} /dot{X}-x^{/star}, /frac{3}{2} /dot{X}+/frac{t}{2} /ddot{X}/right/rangle$. Hence,

$$
/begin{align*}
& 2/left/|X(t+/Delta t)+/frac{t}{2} /dot{X}(t+/Delta t)-x^{/star}/right/|^{2}-2/left/|X(t)+/frac{t}{2} /dot{X}(t)-x^{/star}/right/|^{2} //
& =4/left/langle X+/frac{t}{2} /dot{X}-x^{/star}, /frac{3}{2} /dot{X}+/frac{t}{2} /ddot{X}/right/rangle /Delta t+o(/Delta t)  /tag{39}//
& =-t^{2}/langle/dot{X}, G(X, /dot{X})/rangle /Delta t-2 t/left/langle X-x^{/star}, G(X, /dot{X})/right/rangle /Delta t+o(/Delta t)
/end{align*}
$$

For the first term, note that

$$
/begin{aligned}
(t+/Delta t)^{2}/left(f(X(t+/Delta t))-f^{/star}/right)-t^{2}/left(f(X(t))-f^{/star}/right) & =2 t/left(f(X(t+/Delta t))-f^{/star}/right) /Delta t+ //
& t^{2}(f(X(t+/Delta t))-f(X(t)))+o(/Delta t)
/end{aligned}
$$

Since $f$ is locally Lipschitz, $o(/Delta t)$ term does not affect the function in the limit,

$$
/begin{equation*}
f(X(t+/Delta t))=f(X+/Delta t /dot{X}+o(/Delta t))=f(X+/Delta t /dot{X})+o(/Delta t) /tag{40}
/end{equation*}
$$

By Lemma 22, we have the approximation

$$
/begin{equation*}
f(X+/Delta t /dot{X})=f(X)+/langle/dot{X}, G(X, /dot{X})/rangle /Delta t+o(/Delta t) /tag{41}
/end{equation*}
$$

Combining all of $(39),(40)$ and $(41)$, we obtain

$$
/begin{gathered}
/mathcal{E}(t+/Delta t)-/mathcal{E}(t)=2 t/left(f(X(t+/Delta t))-f^{/star}/right) /Delta t+t^{2}/langle/dot{X}, G(X, /dot{X})/rangle /Delta t-t^{2}/langle/dot{X}, G(X, /dot{X})/rangle /Delta t //
-2 t/left/langle X-x^{/star}, G(X, /dot{X})/right/rangle /Delta t+o(/Delta t) //
=2 t/left(f(X)-f^{/star}/right) /Delta t-2 t/left/langle X-x^{/star}, G(X, /dot{X})/right/rangle /Delta t+o(/Delta t) /leq o(/Delta t)
/end{gathered}
$$

where the last inequality follows from the convexity of $f$. Thus,

$$
/limsup _{/Delta t /rightarrow 0+} /frac{/mathcal{E}(t+/Delta t)-/mathcal{E}(t)}{/Delta t} /leq 0
$$

which along with the continuity of $/mathcal{E}$, concludes that $/mathcal{E}(t)$ is a non-increasing function of $t$.

We give a simple example as follows. Consider the Lasso problem

$$
/operatorname{minimize} /quad /frac{1}{2}/|y-A x/|^{2}+/lambda/|x/|_{1}
$$

Any directional subgradients admits the form $G(x, p)=-A^{T}(y-A x)+/lambda /operatorname{sgn}(x, p)$, where

$$
/operatorname{sgn}(x, p)_{i}= /begin{cases}/operatorname{sgn}/left(x_{i}/right), & x_{i} /neq 0 // /operatorname{sgn}/left(p_{i}/right), & x_{i}=0, p_{i} /neq 0 // /in[-1,1], & x_{i}=0, p_{i}=0/end{cases}
$$

To encourage sparsity, for any index $i$ with $x_{i}=0, p_{i}=0$, we let

$$
G(x, p)_{i}=/operatorname{sgn}/left(A_{i}^{T}(A x-y)/right)/left(/left|A_{i}^{T}(A x-y)/right|-/lambda/right)_{+}
$$

## Appendix D. Proof of Theorem 9

Proof Let $g$ be $/mu$-strongly convex and $h$ be convex. For $f=g+h$, we show that (22) can be strengthened to

$$
/begin{equation*}
f/left(y-s G_{s}(y)/right) /leq f(x)+G_{s}(y)^{T}(y-x)-/frac{s}{2}/left/|G_{s}(y)/right/|^{2}-/frac{/mu}{2}/|y-x/|^{2} /tag{42}
/end{equation*}
$$

Summing $(4 k-3) /times(42)$ with $x=x_{k-1}, y=y_{k-1}$ and $(4 r-6) /times(42)$ with $x=x^{/star}, y=y_{k-1}$ yields

$$
/begin{align*}
& (4 k+4 r-9) f/left(x_{k}/right) /leq(4 k-3) f/left(x_{k-1}/right)+(4 r-6) f^{/star} //
& /quad+G_{s}/left(y_{k-1}/right)^{T}/left[(4 k+4 r-9) y_{k-1}-(4 k-3) x_{k-1}-(4 r-6) x^{/star}/right] //
& -/frac{s(4 k+4 r-9)}{2}/left/|G_{s}/left(y_{k-1}/right)/right/|^{2}-/frac{/mu(4 k-3)}{2}/left/|y_{k-1}-x_{k-1}/right/|^{2}-/mu(2 r-3)/left/|y_{k-1}-x^{/star}/right/|^{2} //
& /leq(4 k-3) f/left(x_{k-1}/right)+(4 r-6) f^{/star}-/mu(2 r-3)/left/|y_{k-1}-x^{/star}/right/|^{2} //
& /quad /quad+G_{s}/left(y_{k-1}/right)^{T}/left[(4 k+4 r-9)/left(y_{k-1}-x^{/star}/right)-(4 k-3)/left(x_{k-1}-x^{/star}/right)/right] /tag{43}
/end{align*}
$$

which gives a lower bound on $G_{s}/left(y_{k-1}/right)^{T}/left[(4 k+4 r-9) y_{k-1}-(4 k-3) x_{k-1}-(4 r-6) x^{/star}/right]$. Denote by $/Delta_{k}$ the second term of $/tilde{/mathcal{E}}(k)$ in (28), namely,

$$
/Delta_{k} /triangleq /frac{k+d}{8}/left/|(2 k+2 r-2)/left(y_{k}-x^{/star}/right)-(2 k+1)/left(x_{k}-x^{/star}/right)/right/|^{2}
$$

where $d:=3 r / 2-5 / 2$. Then by (43), we get

$$
/begin{gathered}
/Delta_{k}-/Delta_{k-1}=-/frac{k+d}{8}/left/langle s(2 r+2 k-5) G_{s}/left(y_{k-1}/right)+/frac{k-2}{k+r-2}/left(x_{k-1}-x_{k-2}/right),(4 k+4 r-9)/left(y_{k-1}-x^{/star}/right)/right. //
/left.-(4 k-3)/left(x_{k-1}-x^{/star}/right)/right/rangle+/frac{1}{8}/left/|(2 k+2 r-4)/left(y_{k-1}-x^{/star}/right)-(2 k-1)/left(x_{k-1}-x^{/star}/right)/right/|^{2} //
/leq-/frac{s(k+d)(2 k+2 r-5)}{8}/left[(4 k+4 r-9)/left(f/left(x_{k}/right)-f^{/star}/right)/right. //
/left.-(4 k-3)/left(f/left(x_{k-1}/right)-f^{/star}/right)+/mu(2 r-3)/left/|y_{k-1}-x^{/star}/right/|^{2}/right] //
-/frac{(k+d)(k-2)}{8(k+r-2)}/left/langle x_{k-1}-x_{k-2},(4 k+4 r-9)/left(y_{k-1}-x^{/star}/right)-(4 k-3)/left(x_{k-1}-x^{/star}/right)/right/rangle //
+/frac{1}{8}/left/|2(k+r-2)/left(y_{k-1}-x^{/star}/right)-(2 k-1)/left(x_{k-1}-x^{/star}/right)/right/|^{2}
/end{gathered}
$$

Hence,

$$
/begin{align*}
& /Delta_{k}+/frac{s(k+d)(2 k+2 r-5)(4 k+4 r-9)}{8}/left(f/left(x_{k}/right)-f^{/star}/right) //
& /leq /Delta_{k-1}+/frac{s(k+d)(2 k+2 r-5)(4 k-3)}{8}/left(f/left(x_{k-1}/right)-f^{/star}/right) //
& /quad-/frac{s /mu(2 r-3)(k+d)(2 k+2 r-5)}{8}/left/|y_{k-1}-x^{/star}/right/|^{2}+/Pi_{1}+/Pi_{2} /tag{44}
/end{align*}
$$

where

$$
/begin{gathered}
/Pi_{1} /triangleq-/frac{(k+d)(k-2)}{8(k+r-2)}/left/langle x_{k-1}-x_{k-2},(4 k+4 r-9)/left(y_{k-1}-x^{/star}/right)-(4 k-3)/left(x_{k-1}-x^{/star}/right)/right/rangle //
/Pi_{2} /triangleq /frac{1}{8}/left/|2(k+r-2)/left(y_{k-1}-x^{/star}/right)-(2 k-1)/left(x_{k-1}-x^{/star}/right)/right/|^{2}
/end{gathered}
$$

By the iterations defined in (19), one can show that

$$
/begin{aligned}
/Pi_{1} & =-/frac{(2 r-3)(k+d)(k-2)}{8(k+r-2)}/left(/left/|x_{k-1}-x^{/star}/right/|^{2}-/left/|x_{k-2}-x^{/star}/right/|^{2}/right) //
& -/frac{(k-2)^{2}(4 k+4 r-9)(k+d)+(2 r-3)(k-2)(k+r-2)(k+d)}{8(k+r-2)^{2}}/left/|x_{k-1}-x_{k-2}/right/|^{2} //
/Pi_{2} & =/frac{(2 r-3)^{2}}{8}/left/|y_{k-1}-x^{/star}/right/|^{2}+/frac{(2 r-3)(2 k-1)(k-2)}{8(k+r-2)}/left(/left/|x_{k-1}-x^{/star}/right/|^{2}-/left/|x_{k-2}-x^{/star}/right/|^{2}/right) //
& +/frac{(k-2)^{2}(2 k-1)(2 k+4 r-7)+(2 r-3)(2 k-1)(k-2)(k+r-2)}{8(k+r-2)^{2}}/left/|x_{k-1}-x_{k-2}/right/|^{2}
/end{aligned}
$$

Although this is a little tedious, it is straightforward to check that $(k-2)^{2}(4 k+4 r-9)(k+d)+$ $(2 r-3)(k-2)(k+r-2)(k+d) /geq(k-2)^{2}(2 k-1)(2 k+4 r-7)+(2 r-3)(2 k-1)(k-2)(k+r-2)$ for any $k$. Therefore, $/Pi_{1}+/Pi_{2}$ is bounded as

$$
/Pi_{1}+/Pi_{2} /leq /frac{(2 r-3)^{2}}{8}/left/|y_{k-1}-x^{/star}/right/|^{2}+/frac{(2 r-3)(k-d-1)(k-2)}{8(k+r-2)}/left(/left/|x_{k-1}-x^{/star}/right/|^{2}-/left/|x_{k-2}-x^{/star}/right/|^{2}/right)
$$

which, together with the fact that $s /mu(2 r-3)(k+d)(2 k+2 r-5) /geq(2 r-3)^{2}$ when $k /geq$ $/sqrt{(2 r-3) /(2 s /mu)}$, reduces (44) to

$$
/begin{aligned}
& /Delta_{k}+/frac{s(k+d)(2 k+2 r-5)(4 k+4 r-9)}{8}/left(f/left(x_{k}/right)-f^{/star}/right) //
& /leq /Delta_{k-1}+/frac{s(k+d)(2 k+2 r-5)(4 k-3)}{8}/left(f/left(x_{k-1}/right)-f^{/star}/right) //
&+/frac{(2 r-3)(k-d-1)(k-2)}{8(k+r-2)}/left(/left/|x_{k-1}-x^{/star}/right/|^{2}-/left/|x_{k-2}-x^{/star}/right/|^{2}/right)
/end{aligned}
$$

This can be further simplified as

$$
/begin{equation*}
/tilde{/mathcal{E}}(k)+A_{k}/left(f/left(x_{k-1}/right)-f^{/star}/right) /leq /tilde{/mathcal{E}}(k-1)+B_{k}/left(/left/|x_{k-1}-x^{/star}/right/|^{2}-/left/|x_{k-2}-x^{/star}/right/|^{2}/right) /tag{45}
/end{equation*}
$$

for $k /geq /sqrt{(2 r-3) /(2 s /mu)}$, where $A_{k}=(8 r-36) k^{2}+/left(20 r^{2}-126 r+200/right) k+12 r^{3}-100 r^{2}+$ $288 r-281>0$ since $r /geq 9 / 2$ and $B_{k}=(2 r-3)(k-d-1)(k-2) /(8(k+r-2))$. Denote by $k^{/star}=/lceil/max /{/sqrt{(2 r-3) /(2 s /mu)}, 3 r / 2-3 / 2/}/rceil /asymp 1 / /sqrt{s /mu}$. Then $B_{k}$ is a positive increasing sequence if $k>k^{/star}$. Summing (45) from $k$ to $k^{/star}+1$, we obtain

$$
/begin{aligned}
& /mathcal{E}(k)+/sum_{i=k^{/star}+1}^{k} A_{i}/left(f/left(x_{i-1}/right)-f^{/star}/right) /leq /mathcal{E}/left(k^{/star}/right)+/sum_{i=k^{/star}+1}^{k} B_{i}/left(/left/|x_{i-1}-x^{/star}/right/|^{2}-/left/|x_{i-2}-x^{/star}/right/|^{2}/right) //
& =/mathcal{E}/left(k^{/star}/right)+B_{k}/left/|x_{k-1}-x^{/star}/right/|^{2}-B_{k^{/star}+1}/left/|x_{k^{/star}-1}-x^{/star}/right/|^{2}+/sum_{i=k^{/star}+1}^{k-1}/left(B_{j}-B_{j+1}/right)/left/|x_{j-1}-x^{/star}/right/|^{2} //
& /leq /mathcal{E}/left(k^{/star}/right)+B_{k}/left/|x_{k-1}-x^{/star}/right/|^{2}
/end{aligned}
$$

Similarly, as in the proof of Theorem 8 , we can bound $/mathcal{E}/left(k^{/star}/right)$ via another energy functional defined from Theorem 5

$$
/begin{align*}
& /mathcal{E}/left(k^{/star}/right) /leq /frac{s/left(2 k^{/star}+3 r-5/right)/left(k^{/star}+r-2/right)^{2}}{2}/left(f/left(x_{k^{/star}}/right)-f^{/star}/right) //
&+/frac{2 k^{/star}+3 r-5}{16}/left/|2/left(k^{/star}+r-1/right) y_{k^{/star}}-2 k^{/star} x_{k^{/star}}-2(r-1) x^{/star}-/left(x_{k^{/star}}-x^{/star}/right)/right/|^{2} //
& /leq /frac{s/left(2 k^{/star}+3 r-5/right)/left(k^{/star}+r-2/right)^{2}}{2}/left(f/left(x_{k^{/star}}/right)-f^{/star}/right) //
& /quad+/frac{2 k^{/star}+3 r-5}{8}/left/|2/left(k^{/star}+r-1/right) y_{k^{/star}}-2 k^{/star} x_{k^{/star}}-2(r-1) x^{/star}/right/|^{2} //
& /quad+/frac{2 k^{/star}+3 r-5}{8}/left/|x_{k^{/star}}-x^{/star}/right/|^{2} /leq /frac{(r-1)^{2}/left(2 k^{/star}+3 r-5/right)}{2}/left/|x_{0}-x^{/star}/right/|^{2} //
& /quad+/frac{(r-1)^{2}/left(2 k^{/star}+3 r-5/right)}{8 s /mu/left(k^{/star}+r-2/right)^{2}}/left/|x_{0}-x^{/star}/right/|^{2} /lesssim /frac{/left/|x_{0}-x^{/star}/right/|^{2}}{/sqrt{s /mu}} /tag{46}
/end{align*}
$$

## Su, Boyd and Candès

For the second term, it follows from Theorem 6 that

$$
/begin{align*}
B_{k}/left/|x_{k-1}-x^{/star}/right/|^{2} & /leq /frac{(2 r-3)(2 k-3 r+3)(k-2)}{8 /mu(k+r-2)}/left(f/left(x_{k-1}/right)-x^{/star}/right) //
& /leq /frac{(2 r-3)(2 k-3 r+3)(k-2)}{8 /mu(k+r-2)} /frac{(r-1)^{2}/left/|x_{0}-x^{/star}/right/|^{2}}{2 s(k+r-3)^{2}}  /tag{47}//
& /leq /frac{(2 r-3)(r-1)^{2}/left(2 k^{/star}-3 r+3/right)/left(k^{/star}-2/right)}{16 s /mu/left(k^{/star}+r-2/right)/left(k^{/star}+r-3/right)^{2}}/left/|x_{0}-x^{/star}/right/|^{2} /lesssim /frac{/left/|x_{0}-x^{/star}/right/|^{2}}{/sqrt{s /mu}} .
/end{align*}
$$

For $k>k^{/star}$, (46) together with (47) this gives

$$
/begin{aligned}
f/left(x_{k}/right)-f^{/star} & /leq /frac{16 /mathcal{E}(k)}{s(2 k+3 r-5)(2 k+2 r-5)(4 k+4 r-9)} //
& /leq /frac{16/left(/mathcal{E}/left(k^{/star}/right)+B_{k}/left/|x_{k-1}-x^{/star}/right/|^{2}/right)}{s(2 k+3 r-5)(2 k+2 r-5)(4 k+4 r-9)} /lesssim /frac{/left/|x_{0}-x^{/star}/right/|^{2}}{s^{/frac{3}{2}} /mu^{/frac{1}{2}} k^{3}}
/end{aligned}
$$

To conclusion, note that by Theorem 6 the gap $f/left(x_{k}/right)-f^{/star}$ for $k /leq k^{/star}$ is bounded by

$$
/frac{(r-1)^{2}/left/|x_{0}-x^{/star}/right/|^{2}}{2 s(k+r-2)^{2}}=/frac{(r-1)^{2} /sqrt{s /mu} k^{3}}{2(k+r-2)^{2}} /frac{/left/|x_{0}-x^{/star}/right/|^{2}}{s^{/frac{3}{2}} /mu^{/frac{1}{2}} k^{3}} /lesssim /sqrt{s /mu} k^{/star} /frac{/left/|x_{0}-x^{/star}/right/|^{2}}{s^{/frac{3}{2}} /mu^{/frac{1}{2}} k^{3}} /lesssim /frac{/left/|x_{0}-x^{/star}/right/|^{2}}{s^{/frac{3}{2}} /mu^{/frac{1}{2}} k^{3}} .
$$

## Appendix E. Proof of Lemmas in Section 5

First, we prove Lemma 11

Proof To begin with, note that the ODE (3) is equivalent to $/mathrm{d}/left(t^{3} /dot{X}(t)/right) / /mathrm{d} t=-t^{3} /nabla f(X(t))$, which by integration leads to

$$
/begin{equation*}
t^{3} /dot{X}(t)=-/frac{t^{4}}{4} /nabla f/left(x_{0}/right)-/int_{0}^{t} u^{3}/left(/nabla f(X(u))-/nabla f/left(x_{0}/right)/right) /mathrm{d} u=-/frac{t^{4}}{4} /nabla f/left(x_{0}/right)-I(t) /tag{48}
/end{equation*}
$$

Dividing (48) by $t^{4}$ and applying the bound on $I(t)$, we obtain

$$
/frac{/|/dot{X}(t)/|}{t} /leq /frac{/left/|/nabla f/left(x_{0}/right)/right/|}{4}+/frac{/|I(t)/|}{t^{4}} /leq /frac{/left/|/nabla f/left(x_{0}/right)/right/|}{4}+/frac{L M(t) t^{2}}{12}
$$

Note that the right-hand side of the last display is monotonically increasing in $t$. Hence, by taking the supremum of the left-hand side over $(0, t]$, we get

$$
M(t) /leq /frac{/left/|/nabla f/left(x_{0}/right)/right/|}{4}+/frac{L M(t) t^{2}}{12}
$$

which completes the proof by rearrangement.

Next, we prove the lemma used in the proof of Lemma 12.

Lemma 25 The speed restarting time $T$ satisfies

$$
T/left(x_{0}, f/right) /geq /frac{4}{5 /sqrt{L}}
$$

Proof The proof is based on studying $/langle/dot{X}(t), /ddot{X}(t)/rangle$. Dividing (48) by $t^{3}$, we get an expression for $/dot{X}$

$$
/begin{equation*}
/dot{X}(t)=-/frac{t}{4} /nabla f/left(x_{0}/right)-/frac{1}{t^{3}} /int_{0}^{t} u^{3}/left(/nabla f(X(u))-/nabla f/left(x_{0}/right)/right) /mathrm{d} u /tag{49}
/end{equation*}
$$

Differentiating the above, we also obtain an expression for $/ddot{X}$ :

$$
/begin{equation*}
/ddot{X}(t)=-/nabla f(X(t))+/frac{3}{4} /nabla f/left(x_{0}/right)+/frac{3}{t^{4}} /int_{0}^{t} u^{3}/left(/nabla f(X(u))-/nabla f/left(x_{0}/right)/right) /mathrm{d} u /tag{50}
/end{equation*}
$$

Using the two equations we can show that $/mathrm{d}/|/dot{X}/|^{2} / /mathrm{d} t=2/langle/dot{X}(t), /ddot{X}(t)/rangle>0$ for $0<t<$ $4 /(5 /sqrt{L})$. Continue by observing that (49) and (50) yield

$$
/begin{aligned}
& /langle/dot{X}(t), /ddot{X}(t)/rangle=/left/langle-/frac{t}{4} /nabla f/left(x_{0}/right)-/frac{1}{t^{3}} I(t),-/nabla f(X(t))+/frac{3}{4} /nabla f/left(x_{0}/right)+/frac{3}{t^{4}} I(t)/right/rangle //
& /geq /frac{t}{4}/left/langle/nabla f/left(x_{0}/right), /nabla f(X(t))/right/rangle-/frac{3 t}{16}/left/|/nabla f/left(x_{0}/right)/right/|^{2}-/frac{1}{t^{3}}/|I(t)/|/left(/|/nabla f(X(t))/|+/frac{3}{2}/left/|/nabla f/left(x_{0}/right)/right/|/right)-/frac{3}{t^{7}}/|I(t)/|^{2} //
& /geq /frac{t}{4}/left/|/nabla f/left(x_{0}/right)/right/|^{2}-/frac{t}{4}/left/|/nabla f/left(x_{0}/right)/right/|/left/|/nabla f(X(t))-/nabla f/left(x_{0}/right)/right/|-/frac{3 t}{16}/left/|/nabla f/left(x_{0}/right)/right/|^{2} //
& /quad /quad-/frac{L M(t) t^{3}}{12}/left(/left/|/nabla f(X(t))-/nabla f/left(x_{0}/right)/right/|+/frac{5}{2}/left/|/nabla f/left(x_{0}/right)/right/|/right)-/frac{L^{2} M(t)^{2} t^{5}}{48} //
& /geq //
& /geq /frac{t}{16}/left/|/nabla f/left(x_{0}/right)/right/|^{2}-/frac{L M(t) t^{3}/left/|/nabla f/left(x_{0}/right)/right/|}{8}-/frac{L M(t) t^{3}}{12}/left(/frac{L M(t) t^{2}}{2}+/frac{5}{2}/left/|/nabla f/left(x_{0}/right)/right/|/right)-/frac{L^{2} M(t)^{2} t^{5}}{48} //
& =/frac{t}{16}/left/|/nabla f/left(x_{0}/right)/right/|^{2}-/frac{L M(t) t^{3}}{3}/left/|/nabla f/left(x_{0}/right)/right/|-/frac{L^{2} M(t)^{2} t^{5}}{16}
/end{aligned}
$$

To complete the proof, applying Lemma 11, the last inequality yields

$$
/langle/dot{X}(t), /ddot{X}(t)/rangle /geq/left(/frac{1}{16}-/frac{L t^{2}}{12/left(1-L t^{2} / 12/right)}-/frac{L^{2} t^{4}}{256/left(1-L t^{2} / 12/right)^{2}}/right)/left/|/nabla f/left(x_{0}/right)/right/|^{2} t /geq 0
$$

for $t</min /{/sqrt{12 / L}, 4 /(5 /sqrt{L})/}=4 /(5 /sqrt{L})$, where the positivity follows from

$$
/frac{1}{16}-/frac{L t^{2}}{12/left(1-L t^{2} / 12/right)}-/frac{L^{2} t^{4}}{256/left(1-L t^{2} / 12/right)^{2}}>0
$$

which is valid for $0<t /leq 4 /(5 /sqrt{L})$.

## References

A. Beck. Introduction to Nonlinear Optimization: Theory, Algorithms, and Applications with MATLAB. SIAM, 2014.

A. Beck and M. Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM Journal on Imaging Sciences, 2(1):183-202, 2009.

S. Becker, J. Bobin, and E. J. Candès. NESTA: A fast and accurate first-order method for sparse recovery. SIAM Journal on Imaging Sciences, 4(1):1-39, 2011.

M. Bogdan, E. v. d. Berg, C. Sabatti, W. Su, and E. J. Candès. SLOPE-adaptive variable selection via convex optimization. The Annals of Applied Statistics, 9(3):1103-1140, 2015.

S. Boyd and L. Vandenberghe. Convex Optimization. Cambridge University Press, 2004.

S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein. Distributed optimization and statistical learning via the alternating direction method of multipliers. Foundations and Trends in Machine Learning, 3(1):1-122, 2011.

H.-B. Dürr and C. Ebenbauer. On a class of smooth optimization algorithms with applications in control. Nonlinear Model Predictive Control, 4(1):291-298, 2012.

H.-B. Dürr, E. Saka, and C. Ebenbauer. A smooth vector field for quadratic programming. In 51st IEEE Conference on Decision and Control, pages 2515-2520, 2012.

S. Fiori. Quasi-geodesic neural learning algorithms over the orthogonal group: A tutorial. Journal of Machine Learning Research, 6:743-781, 2005.

U. Helmke and J. Moore. Optimization and dynamical systems. Proceedings of the IEEE, 84(6):907, 1996.

D. Hinton. Sturm's 1836 oscillation results evolution of the theory. In Sturm-Liouville theory, pages 1-27. Birkhäuser, Basel, 2005.

J. J. Leader. Numerical Analysis and Scientific Computation. Pearson Addison Wesley, 2004 .

L. Lessard, B. Recht, and A. Packard. Analysis and design of optimization algorithms via integral quadratic constraints. arXiv preprint arXiv:1408.3595, 2014.

R. Monteiro, C. Ortiz, and B. Svaiter. An adaptive accelerated first-order method for convex optimization. Technical report, ISyE, Gatech, 2012.

Y. Nesterov. A method of solving a convex programming problem with convergence rate $O/left(1 / k^{2}/right)$. Soviet Mathematics Doklady, 27(2):372-376, 1983.

Y. Nesterov. Introductory Lectures on Convex Pptimization: A Basic Course, volume 87 of Applied Optimization. Kluwer Academic Publishers, Boston, MA, 2004.

Y. Nesterov. Smooth minimization of non-smooth functions. Mathematical Programming, 103(1):127-152, 2005.

Y. Nesterov. Gradient methods for minimizing composite functions. Mathematical Programming, 140(1):125-161, 2013.

J. Nocedal and S. Wright. Numerical Optimization. Springer Science /& Business Media, 2006 .

B. O'Donoghue and E. J. Candès. Adaptive restart for accelerated gradient schemes. Found. Comput. Math., 2013.

S. Osher, F. Ruan, J. Xiong, Y. Yao, and W. Yin. Sparse recovery via differential inclusions. arXiv preprint arXiv:1406.7728, 2014.

B. T. Polyak. Introduction to optimization. Optimization Software New York, 1987.

Z. Qin and D. Goldfarb. Structured sparsity via alternating direction methods. Journal of Machine Learning Research, 13(1):1435-1468, 2012.

R. T. Rockafellar. Convex Analysis. Princeton Landmarks in Mathematics. Princeton University Press, 1997. Reprint of the 1970 original.

A. P. Ruszczyński. Nonlinear Optimization. Princeton University Press, 2006.

J. Schropp and I. Singer. A dynamical systems approach to constrained minimization. Numerical functional analysis and optimization, 21(3-4):537-551, 2000.

N. Z. Shor. Minimization Methods for Non-Differentiable Functions. Springer Science /& Business Media, 2012.

I. Sutskever, J. Martens, G. Dahl, and G. Hinton. On the importance of initialization and momentum in deep learning. In Proceedings of the 30th International Conference on Machine Learning, pages 1139-1147, 2013.

P. Tseng. On accelerated proximal gradient methods for convex-concave optimization. http://pages.cs.wisc.edu/ brecht/cs726docs/Tseng.APG.pdf, 2008.

P. Tseng. Approximation accuracy, gradient methods, and error bound for structured convex optimization. Mathematical Programming, 125(2):263-295, 2010.

G. N. Watson. A Treatise on the Theory of Bessel Functions. Cambridge Mathematical Library. Cambridge University Press, 1995. Reprint of the second (1944) edition.


[^0]:    2. Up to a constant multiplier, $J_{1}$ is the unique solution to the Bessel's differential equation $u^{2} /ddot{J}_{1}+u /dot{J}_{1}+$ $/left(u^{2}-1/right) J_{1}=0$ that is finite at the origin. In the analytic expansion of $J_{1}, m!$ ! denotes the double factorial defined as $m!!=m /times(m-2) /times /cdots /times 2$ for even $m$, or $m!!=m /times(m-2) /times /cdots /times 1$ for odd $m$.
[^1]:    4. This function does not have a Lipschitz continuous gradient. However, a similar pattern as in Figure 2 can be also observed if we smooth $|x|$ at an arbitrarily small vicinity of 0 .
[^2]:    5. For Figures $4 /mathrm{~d}, 4 /mathrm{e}$ and 4 f , if running generalized Nesterov's schemes with too many iterations (e.g. $10^{5}$ ), the deviations from the ODE will grow. Taking a sufficiently small $s$ can solve this issue.


# CVPR 2020 continual learning in computer vision competition: Approaches, results, current challenges and future directions ${ }^{2 /%}$ 

Vincenzo Lomonaco ${ }^{a, b, *}$, Lorenzo Pellegrini ${ }^{a}$, Pau Rodriguez ${ }^{c}$,<br>Massimo Caccia ${ }^{/mathrm{d}, /mathrm{c}}$, Qi She ${ }^{/mathrm{h}, /mathrm{b}}$, Yu Chen ${ }^{/mathrm{i}}$, Quentin Jodelet ${ }^{/mathrm{j}}{ }^{/mathrm{k}}$, Ruiping Wang ${ }^{1}$,<br>Zheda Mai ${ }^{/mathrm{m}}$, David Vazquez ${ }^{/mathrm{c}}$, German I. Parisi ${ }^{/mathrm{e}, /mathrm{b}}$, Nikhil Churamani ${ }^{/mathrm{f}}$,<br>Marc Pickett ${ }^{g}$, Issam Laradji ${ }^{/mathrm{C}}$, Davide Maltoni ${ }^{/mathrm{a}}$<br>${ }^{a}$ University of Bologna, Italy<br>${ }^{/text {b }}$ ContinualAI Research, Italy<br>${ }^{/text {c Element AI, Canada }}$<br>${ }^{/mathrm{d}}$ MILA, Canada<br>${ }^{e}$ University of Hamburg, Germany<br>${ }^{f}$ University of Cambridge, United Kingdom of Great Britain and Northern Ireland<br>g Google AI, United States of America<br>${ }^{/text {h }}$ ByteDance AI Labs, China<br>${ }^{/mathrm{i}}$ University of Bristol, United Kingdom of Great Britain and Northern Ireland<br>j Tokyo Institute of Technology, Japan<br>${ }^{/mathrm{k}}$ AIST RWBC-OIL, Japan<br>${ }^{1}$ Chinese Academy of Sciences, China<br>${ }^{/mathrm{m}}$ University of Toronto, Canada

## A R T IC L E IN F O

## Article history:

Received 16 September 2020

Received in revised form 10 May 2021

Accepted 21 November 2021

Available online 25 November 2021

## Keywords:

Continual learning

Lifelong learning

Incremental learning

Challenge

Computer vision


#### Abstract

A B S T R A C T In the last few years, we have witnessed a renewed and fast-growing interest in continual learning with deep neural networks with the shared objective of making current AI systems more adaptive, efficient and autonomous. However, despite the significant and undoubted progress of the field in addressing the issue of catastrophic forgetting, benchmarking different continual learning approaches is a difficult task by itself. In fact, given the proliferation of different settings, training and evaluation protocols, metrics and nomenclature, it is often tricky to properly characterize a continual learning algorithm, relate it to other solutions and gauge its real-world applicability. The first Continual Learning in Computer Vision challenge held at CVPR in 2020 has been one of the first opportunities to evaluate different continual learning algorithms on a common hardware with a large set of shared evaluation metrics and 3 different settings based on the realistic CORe50 video benchmark. In this paper, we report the main results of the competition, which counted more than 79 teams registered and 11 finalists. We also summarize the winning approaches, current challenges and future research directions.


(c) 2021 Elsevier B.V. All rights reserved

[^0]
## 1. Introduction

Continual Learning, the new deep learning embodiment of a long-standing paradigm in machine learning research also known as Incremental or Lifelong Learning, has received a renewed attention from the research community over the last few years [26,18,17]. Continual learning, indeed, appears more and more clearly as the only viable option for sustainable AI agents that can scale efficiently in terms of general intelligence capabilities while adapting to ever-changing environments and unpredictable circumstances over time. Even not considering long-term goals of truly intelligent AI agents, from a pure engineering perspective, continual learning is a very desirable option for any AI technology learning on premises or at the edge on embedded devices without the need of moving private data to remote cloud infrastructures [10].

However, gradient-based architectures, such as neural networks trained with Stochastic Gradient Descent (SGD), notably suffer from catastrophic forgetting or interference [24,33,11,29], where the network parameters are rapidly overwritten when learning over non-stationary data distributions to model only the most recent. In the last few years, significant progresses have been made to tame the issue. Nevertheless, comparing continual learning algorithms today constitutes a hard task [8]. This is mainly due to the proliferation of different settings only covering partial aspects of the continual learning paradigm, with diverse training and evaluation protocols, metrics and datasets used [17,4]. Another important question is whether such algorithms, that have mostly been proved on artificial benchmarks such as MNIST [16] or CIFAR [15], can scale and generalize to different settings and real-world applications.

The 1 st Continual Learning in Computer Vision Challenge, organized within the CLVision workshop at CVPR 2020, is one of the first attempts to address these questions. In particular, the main objectives of the competition were:

- Invite the research community to scale up continual learning approaches to natural images and possibly on video benchmarks.
- Invite the community to work on solutions that can generalize over multiple continual learning protocols and settings (e.g. with or without a "task" supervised signal).
- Provide the first opportunity for a comprehensive evaluation on a shared hardware platform for a fair comparison.

Notable competitions previously organized in this area include: the Pascal 2 EU network of excellence challenge on "covariate shift", organized in 2005 [30,31]; the Autonomous Lifelong Machine Learning with Drift challenge organized at NeurIPS 2018 [9] and the IROS 2019 Lifelong Robotic Vision challenge [1]. While the first two competitions can be considered as the first continual learning challenges ever organized, they were based on low-dimensional features benchmarks that made it difficult to understand the scalability of the proposed methods to more complex settings with deep learning based techniques. Moreover, their focus was mostly based on adapting to data drifts rather than being able to learn continually and reduce catastrophic forgetting. The IROS 2019 Lifelong Robotic Vision, instead, has been one of the first challenges organized within robotic vision realistic settings. However, it lacked a general focus on computer vision applications as well as a comprehensive evaluation on 3 different settings and 4 tracks, which we regard as fundamental to understand the general applicability, portability and scalability of the proposed methods.

For transparency and reproducibility, we openly release the finalists' dockerized solutions as well as the initial baselines at the following link: https://github.com/vlomonaco/cvpr_clvision_challenge. All the instructions to reproduce the results and compare original solutions with the strategies presented in this paper are also contained in the same repository.

## 2. Competition

The CLVision competition was planned as a 2-phase event (pre-selection and finals), with 4 tracks and held online from the 15th of February 2020 to the 14th of June 2020. The pre-selection phase, based on the codalab online evaluation framework, ${ }^{1}$ lasted 78 days and was followed by the finals where a dockerized solution had to be submitted for remote evaluation on a shared hardware. In the following section, the dataset, the different tracks, the evaluation metric used and the main rules of the competition are reported in detail. Finally, the main competition statistics, participants and winners are presented.

### 2.1. Dataset

CORe50 [20] was specifically designed as an object recognition video benchmark for continual learning. It consists of 550 egocentric video sessions ( $/sim 300$ frames recorded with a Kinect 2 at 20 fps ) characterized by relevant variations in terms of lighting, background, pose and occlusions and related to 50 domestic objects ( 11 sessions for each object). The 50 objects belong to 10 categories: plug adapters, mobile phones, scissors, light bulbs, cans, glasses, balls, markers, cups and remote controls (see Fig. 1). Classification on CORe50 can hence be performed at object level ( 50 classes) or at category level ( 10 classes). The total number of $128 /times 128$ RGB images composing the videos is 164,866 .

The configuration chosen for the competition was a frame-level classification at object granularity ( 50 classes). Participants were able to exploit temporal coherence during training but not for the test where the frames were randomly shuffled.

/footnotetext{
1 https://codalab.org.

![](https://cdn.mathpix.com/cropped/2024_08_15_07039817fb8d1f0d2cc3g-03.jpg?height=815&width=1616&top_left_y=177&top_left_x=139)

Fig. 1. Example images of the 50 objects in CORe50, the main video dataset used in the challenge. Each column denotes one of the 10 categories [19].

As for the original CORe50 benchmark, three of the eleven sessions (/#3, /#7 and /#10) have been selected for test and the remaining 8 sessions are used for training.

The egocentric vision of hand-held objects allows for the emulation of a scenario where a robot has to incrementally learn to recognize objects while manipulating them. Objects are presented to the robot by a human operator who can also provide the labels, thus enabling a supervised classification (such an applicative scenario is well described in [27,36]).

CORe50 was chosen as the reference dataset for the competition rather than similar datasets such as OpenLORIS [36] or ICubWorld-Transformation [27], given its common adoption in the continual learning community and the multiple scenarios already defined in his benchmark (NI, NC and NIC) that constituted the basis of the challenge tracks.

### 2.2. Tracks

The challenge included four different tracks based on the different settings already proposed in the CORe50 benchmark, namely NI, NC and NIC [20]:

1. New Instances (NI): In this setting 8 training batches (corresponding to the 8 train video sessions of CORe50) with images belonging to the same 50 classes are encountered over time. Each training batch is composed of different images collected in different environmental conditions.
2. Multi-Task New Classes (MT-NC) ${ }^{2}$ : In this setting the 50 different classes are split into 9 different tasks: 10 classes in the first batch and 5 classes in the other 8 following the original CORe50 protocol [20]. However, in this case the task label is provided during training and test.
3. New Instances and Classes (NIC): this protocol is composed of 391 training batches containing 300 images of a single class. No task label will be provided and each batch may contain images of a class seen before as well as a completely new class. For more details about the NIC protocol and classes split and ordering please refer to [20].
4. All together (ALL): All the settings presented above. In this track a single method is expected to be generally applied to the aforementioned three tracks.

Each participant of the challenge could choose in which of the main three tracks (NI, MT-NC, NIC) to compete. Those participants that decided to participate to all the three main tracks were automatically included in the ALL track as well, the most difficult and ambitious track of the competition. For each of the proposed tracks the participants had access to a training, validation and test set. However, the test set did not contain the labels information. The train and validation set corresponded to the original CORe50 train and test set, respectively, while the competition test set was generated with random transformations starting from the original CORe50 test set.

[^1]
### 2.3. Evaluation metric

In the last few years the main evaluation focus in continual learning has always been centered around accuracy-related forgetting metrics [17]. However, as argued by Díaz-Rodríguez et al. [8], this may lead to biased conclusion not accounting for the real scalability of such techniques over an increasing number of tasks/batches and more complex settings. For this reason, in the competition each solution was evaluated across a number of metrics:

1. Final accuracy on the test set $^{3}$ : computed only at the end of the training procedure.
2. Average accuracy over time on the validation set: computed at every batch/task.
3. Total training/test time: total running time from start to end of the main function (in minutes).
4. RAM usage: total memory occupation of the process and its eventual sub-processes. It is computed at every epoch (in MB ).
5. Disk usage: only of additional data produced during training (like replay patterns) and additionally stored parameters. It is computed at every epoch (in MB).

The final aggregation metric ( $C L_{/text {score }}$ ) is the weighted average of the $1-5$ metrics $(0.3,0.1,0.15,0.125,0.125$ respectively). The choice of the weights was carefully defined by the challenge chairs to balance the importance of the accuracy (0.4) with the rest of the metrics (0.6). Additionally increasing the weights assigned to non accuracy-related metrics would have indeed privileged solutions not controlling catastrophic forgetting at all like the Naive baseline introduced in Section 3.1.

### 2.4. Rules and evaluation infrastructure

In order to provide a fair evaluation while not constraining each participants to simplistic solutions due to a limited server-side computational budget, the challenge was based on the following rules:

1. The challenge was based on the Codalab platform. For the pre-selection phase, each team was asked to run the experiments locally on their machines with the help of a Python repository to easily load the data and generate the submission file (with all the necessary data to execute the submission remotely and verify the adherence to the competition rules if needed). The submission file, once uploaded, was used to compute the $C L_{S c o r e}$ which determined the ranking in each scoreboard (one for each track).
2. It was possible to optimize the data loader, but not to change the data order or the protocol itself.
3. The top 11 teams in the scoreboard at the end of the pre-selection phase were selected for the final evaluation. The rest of the teams that did not complete the final submission process with the accompanying reports were excluded from the final evaluation. The maximum number of teams allowed for the finals was capped to 12 based on scoreboard ranking for each track.
4. The final evaluation consisted in a remote evaluation of the final submission for each team. This is to make sure the final ranking was computed in the same computational environment for a fair comparison. In this phase, experiments were run remotely for all the teams over a 32 CPU cores, 1 NVIDIA Titan X GPU, 64 GB RAM Linux system. The max running time was capped at 5 hours for each submission/track.
5. Each team selected for the final evaluation had to submit a single dockerized solution which had to contain the exact same solution submitted for the last codalab evaluation. The initial docker image (provided in the initial challenge repository) could have been customized at will but without exceeding 5 GB .

It is worth noting that only the test accuracy was considered in the ranking of the pre-selection phase of the challenge, since it was the only metric computed server-side and not on participants' local hardware. However, since it was not possible to submit a different solution for the final evaluation, this ensured the competition was not biased on the sole accuracy metric.

The financial budget for the challenge was entirely allocated for the monetary prizes in order to stimulate participation:

- $800 /$$ for the participant with highest average score across the three tracks (e.g. the ALL track).
- $500 /$$ for the participant with highest score on the NI track.
- $500 /$$ for the participant with highest score on the MT-NC track.
- $500 /$$ for the participant with highest score on the NIC track.

These prizes were kindly sponsored by Intel Labs (China), while the remote evaluation was performed thanks to the hardware provided by the University of Bologna.

/footnotetext{
${ }^{3}$ Accuracy in CORe50 is computed on a fixed test set. Rationale behind this choice is explained in [20].

### 2.5. Participants and finalists

The challenge counted the participation of 79 teams worldwide that competed during the pre-selection phase. From those 79 teams only 11 qualified to the finals with a total of 46 people involved and an average team components number of 4. In Table A. 5 of the Appendix, the 11 finalist teams and their members are reported along with their home institutions. The geographically distributed and diverse representation of the 11 finalists teams testifies to the internationality of the competition.

## 3. Continual learning approaches

In this section we discuss the baselines made available as well as the continual learning approaches of the winning teams in more details. On the official competition website an extended report for each of the finalist team detailing their approach is also publicly available. ${ }^{4}$

### 3.1. Baselines

In order to better understand the challenge complexity and the competitiveness of the proposed solutions, three main baselines were included for each of the 4 tracks:

- Naive: This is the basic finetuning strategy, where the standard SGD optimization process is continued on the new batches/tasks without any additional regularization constraint, architectural adjustment or memory replay process.
- Rehearsal: In this baseline the Naive approach is augmented with a basic replay process with a growing external memory, where 20 images for each batch are stored.
- AR1* with Latent Replay: a recently proposed strategy [28] showing competitive results on CORe50 with a shared, non fine-tuned hyper-parametrization across the three main tracks.


### 3.2. Team ICT_VIPL

General techniques for all tracks. To improve their performance the ICT_VIPL team used: (1) Heavy Augmentation with the Python imgaug library ${ }^{5}$; (2) resize the input image to $224 /times 224$ to encourage more knowledge transfer from the ImageNet pretrained model; (3) employ an additional exemplar memory for episodic memory replay to alleviate catastrophic forgetting (randomly select $2 /sim 3 /%$ of the training samples); (4) striking a balance between performance and model capacity by using a moderately deep network ResNet-50. As for efficiency, they leveraged the PyTorch Dataloader module for multi-thread speed-up.

Special techniques for individual tracks. For NI track, there is no special design over the general techniques above and they only tune the best hyper-parameters. For Multi-Task-NC track, they carefully design a pipeline that disentangles representation and classifier learning, which shows very high accuracy and the pipeline is as below ( $D_{i}$ is the set of exemplars for Task $i$ and $/left|D_{i}/right|$ is its size):

For Task 0: (1) Train the feature extractor $f(x)$ and the first head $c_{0}(z)$ with all training samples; (2) Select N samples randomly and store them in the exemplar memory $/left(/left|D_{0}/right|=N/right)$

For Task $i(i=1,2, /ldots, 8)$ : (1) Train head $c_{i}(z)$ with all training samples of Task $i$; (2) Drop some samples randomly from the previous memory, keep $/left|D_{j}/right|=/frac{N}{i+1}$ (for all $j<i$ ); (3) Select $/frac{N}{i+1}$ samples from Task $i$ randomly and store them in the exemplar memory $/left(/left|D_{i}/right|=/frac{N}{i+1}/right.$ ); (4) Fine-tune the feature extractor $f(x)$ with all samples in the memory $/cup_{j} D_{j}(j /leq i)$. (since the feature extractor alone cannot classify images, a temporary head $c(z)$ is used for training); (5) Fine-tune each head $c_{j}(z)$ with the corresponding samples in the memory $D_{j}(j /leq i)$.

For NIC track, based on the assumption that the neural network estimates Bayesian a posteriori probabilities [32], the network outputs are divided by the prior probability for each class inspired by the trick that handles class imbalance [2]. Such a technique can prevent the classifier from biasing minority class (predict to newly added classes) especially in the first few increments.

### 3.3. Team Jodelet

The proposed solution consists in the concatenation of a pre-trained deep convolutional neural network used as a feature extractor and an online trained logistic regression combined with a small reservoir memory [5] used for rehearsal.

Since the guiding principle of the proposed solution is to limit as much as possible the computational complexity, the model is trained in an online continual learning setting: each training example is only used once. In order to further decrease the memory and computational complexity of the solution at the cost of a slight decrease of the accuracy, the

[^2]pre-trained feature extractor is fixed and is not fine-tuned during the training procedure. As a result, it is not necessary to apply the gradient descent algorithm to the large feature extractor and the produced representation is fixed. Therefore, it is possible to store the feature representation in the reservoir memory instead of the whole input raw image. In addition to the memory gain, this implies that the replay patterns do not have to go through the feature extractor again, effectively decreasing the computational complexity of the proposed solution.

Among the different architectures and training procedures considered for the feature extractor, ResNet-50 [14] trained by Facebook AI using the Semi-Weakly Supervised Learning procedure [37] was selected. This training procedure relies on the use of a teacher model and 940 million public images in addition to the ImageNet dataset [34]. Compared with the reference training procedure in which the feature extractor is solely trained on the ImageNet dataset, this novel training procedure allows for a consequent increase of the accuracy without modifying the architecture: while the difference of Top- 1 accuracy between both training procedures for ResNet-50 is about $5.0 /%$ on ImageNet, the difference increases up to $11.1 /%$ on the NIC track of the challenge. Moreover, it should be noted that on the three tracks of the challenge, ResNet- 18 feature extractor trained using this new procedure is able to reach an accuracy comparable with the one of the reference ResNet- 50 feature extractor trained only on ImageNet, while being considerably smaller and faster.

For reasons of consistency, the same hyperparameters have been used for the three tracks of the challenge and have been selected using a grid search.

### 3.4. Team UT_LG

Batch-level experience replay with review In most Experience Replay based methods, the incoming mini-batch is concatenated with another mini-batch of samples retrieved from the memory buffer. Then, they simply take an SGD step with the concatenated samples, followed by an update of the memory [5,3]. Team UT_LG method makes two modifications. Firstly, to reduce the number of retrieval and update steps, they concatenate the memory examples at the batch level instead of at the mini-batch level. Concretely, for every epoch, they draw a batch of data $D_{/mathcal{M}}$ randomly from memory with size replay_sz, concatenate it with the current batch and conduct the gradient descent parameters update. Moreover, they add a review step before the final testing, where they draw a batch of size $D_{R}$ from memory and conduct the gradient update again. To prevent overfitting, the learning rate in the review step is usually lower than the learning rate used when processing incoming batches. The overall training procedure is presented in Algorithm 1.

Data preprocessing (1) Centering-cropping the image with a $(100,100)$ window to make the target object occupy more pixels in the image. (2) Resizing the cropped image to $(224,224)$ to ensure no size discrepancy between the input of the pre-trained model and the training images. (3) Pixel-level and spatial-level data augmentation to improve generalization. The details of their implementation can be found in [22]

```
Algorithm 1 Batch-level experience replay with review.
    procedure $/operatorname{BERR}(/mathcal{D}$, mem_sz, replay_sz, review_sz, lr_replay, lr_review)
        $/mathcal{M} /leftarrow/{/} *$ mem_sz
        for $t /in/{1, /ldots, T/}$ do
            for epochs do
                if $t>1$ then
                    $D_{/mathcal{M}} /stackrel{/text { replay_sz }}{/sim} /mathcal{M}$
                    $D_{/text {train }}=D_{/mathcal{M}} /cup D_{t}$
                else
                    $D_{/text {train }}=D_{t}$
                $/theta /leftarrow /operatorname{SGD}/left(D_{/text {train }}, /theta/right.$, lr_replay $)$
            $/mathcal{M} /leftarrow /operatorname{UpdateMemory}/left(D_{t}, /mathcal{M}/right.$, mem_sz)
        $D_{R} /stackrel{/text { review_sz }}{/sim} /mathcal{M}$
        $/theta /leftarrow /operatorname{SGD}/left(D_{R}, /theta/right.$, lr_review $)$
        return $/theta$
```


### 3.5. Team Yc14600

The use of episodic memories in continual learning is an efficient way to prevent the phenomenon of catastrophic forgetting. In recent studies, several gradient-based approaches have been developed to make more efficient use of compact episodic memories. The essential idea is to use gradients produced by samples from episodic memories to constrain the gradients produced by new samples, e.g. by ensuring the inner product of the pair of gradients is non-negative [21] as follows:

$$
/begin{equation*}
/left/langle g_{t}, g_{k}/right/rangle=/left/langle/frac{/partial /mathcal{L}/left(x_{t}, /theta/right)}{/partial /theta}, /frac{/partial /mathcal{L}/left(x_{k}, /theta/right)}{/partial /theta}/right/rangle /geq 0, /forall k<t /tag{1}
/end{equation*}
$$

where $t$ and $k$ are time indices, $x_{t}$ denotes a new sample from the current task, and $x_{k}$ denotes a sample from the episodic memory. Thus, the updates of parameters are forced to preserve the performance on previous tasks as much as possible.

Equation (1) indicates larger cosine similarities between gradients produced by current and previous tasks result in improved generalization. This in turn indicates that samples that lead to the most diverse gradients provide the most difficulty during learning.

Through empirical studies the team members found that the discrimination ability of representations strongly correlates with the diversity of gradients, and more discriminative representations lead to more consistent gradients. They use this insight to introduce an extra objective Discriminative Representation Loss (DRL) into the optimization objective of classification tasks in continual learning. Instead of explicitly refining gradients during training process, DRL helps with decreasing gradient diversity by optimizing the representations. As defined in Equation (2), DRL consists of two parts: one is for minimizing the similarities of representations between samples from different classes ( $/mathcal{L}_{b t}$ ), the other is for minimizing the similarities of representations between samples from a same class $/left(/mathcal{L}_{w i}/right)$ for preserving information of representations for future tasks.

$$
/begin{align*}
& /min _{/Theta} /mathcal{L}_{D R}=/min /left(/mathcal{L}_{b t}+/mathcal{L}_{w i}/right), //
& /mathcal{L}_{b t}=/frac{1}{B_{b t}} /sum_{l=1}^{L} /sum_{i=1}^{B} /sum_{j=1, y_{j} /neq y_{i}}^{B}/left/langle h_{l, i}, h_{l, j}/right/rangle,  /tag{2}//
& /mathcal{L}_{w i}=/frac{1}{B_{w i}} /sum_{l=1}^{L} /sum_{i=1}^{B} /sum_{j=1, j /neq i, y_{j}=y_{i}}^{B}/left/langle h_{l, i}, h_{l, j}/right/rangle,
/end{align*}
$$

where $/Theta$ denotes the parameters of the model, $L$ is the number of layers of the model, $B$ is training batch size. $B_{b t}$ and $B_{w i}$ denote the number of pairs of samples in the training batch that are from different classes and the same class, respectively, $h_{l, i}$ is the output of layer $l$ by input $x_{i}$ and $y_{i}$ is the label of $x_{i}$. Please refer to [6] for more details.

## 4. Competition results

In this section we detail the main results of the competition for each of the main three tracks (NI, MT-NC /& NIC) as well as the averaged track ALL, which determined the overall winner of the challenge. For each track the teams are ranked as follows: i) each metric is normalized across between 0 and 1 ; ii) the $C L_{/text {score }}$ is computed as a weighted average; ii) results are ordered in descending order. Fig. 2 summarizes the results distributions for the three tracks (NI, MT-NC and NIC) across the 11 finalists solutions and the main evaluation metrics used for the competition.

In the next sections we report the results with their absolute values to better grasp the quality of the solutions proposed and their portability in different applicative contexts.

### 4.1. New Instances (NI) track

In Table 1 the main results for the New Instances (NI) track are reported. In Table A.6, additional details (not taken into account for the evaluation) for each solution are shown. In this track, the $U T_{-} L G$ obtained the best $C L_{S c o r e}$ with a small gap w.r.t. its competitors. The test accuracy tops $91 /%$ for the winning team, showing competitive performance also in real-world non-stationary applications. It is worth noting that the top-4 solutions all employed a rehearsal-based technique, only in one case supported by a regularization counterpart.

### 4.2. Multi-Task NC (MT-NC) track

For the MT-NC track, results are reported in Table 2 and additional details in Table A. 7 of the Appendix. In this scenario, arguably the easiest since it provided an additional supervised signal (the Task label) the AR1 baseline resulted as the best scoring solution. In fact, while achieving lower accuracy results than the other top-7 solutions, it offered a more efficient algorithmic proposal in terms of both memory and computation (even without a careful hyper-parametrization). It is also interesting to note that, in this scenario, it is possible to achieve impressive accuracy performance ( $/sim 99 /%$ ) within reasonable computation and memory constraints as shown by the ICT_VIPL team, the only solution who opted for a diskbased exemplars memorization.

### 4.3. New Instances (NIC) track

The NIC track results are reported in Table 3. Additional details of each solution are also made available in Table A.8. Only 7 over 11 finalist teams submitted a solution for this track. In this case, it is possible to observe generally lower accuracy results and an increase in the running times across the 391 batches.

![](https://cdn.mathpix.com/cropped/2024_08_15_07039817fb8d1f0d2cc3g-08.jpg?height=1277&width=1334&top_left_y=170&top_left_x=275)

Fig. 2. Results distributions for the three tracks (NI, MT-NC and NIC) across the 11 finalists solutions and the main evaluation metrics used for the competition: total test accuracy (/%) at the end of the training, average validation accuracy over time (/%), maximum and average RAM/Disk usage (GB).

Table 1

NI track results for the 11 finalists of the competition and the three baselines.

| TEAM NAME | TEST ACC <br> $(/%)$ | VAL ACC $_{/text {avg }}$ <br> $(/%)$ | RUN $_{/text {time }}$ <br> (M) | RAM $_{/text {avg }}$ <br> (MB) | RAM $_{/text {max }}$ <br> (MB) | DISK $_{/text {avg }}$ <br> (MB) | DISK $_{/text {max }}$ <br> (MB) | $C L_{/text {score }}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| UT_LG | 0.91 | 0.90 | 63.78 | 11429.83 | 11643.63 | 0 | 0 | 0.692 |
| YC14600 | 0.88 | 0.85 | 22.58 | 17336.38 | 18446.90 | 0 | 0.648 |  |
| ICT_VIPL | 0.95 | 0.93 | 113.70 | 2459.42 | 2460.16 | 421.875 | 750 | 0.629 |
| JODELET | 0.84 | 0.85 | 3.11 | 18805.60 | 18829.96 | 0 | 0 | 0.612 |
| SOONY | 0.85 | 0.81 | 25.57 | 16662.73 | 17000.10 | 0 | 0 | 0.602 |
| JIMIB | 0.91 | 0.89 | 248.82 | 19110.84 | 25767.74 | 0 | 0 | 0.573 |
| JUN2TONG | 0.84 | 0.76 | 62.48 | 20968.43 | 23252.39 | 0 | 0 | 0.550 |
| SAHINYU | 0.88 | 0.81 | 156.64 | 26229.77 | 32176.76 | 0 | 0 | 0.538 |
| AR1 | 0.75 | 0.73 | 17.18 | 10550.61 | 10838.79 | 0 | 0 | 0.520 |
| NOOBMASTER | 0.85 | 0.75 | 74.54 | 31750.19 | 39627.31 | 0 | 0 | 0.504 |
| MRGRANDDY | 0.88 | 0.84 | 249.28 | 28384.06 | 33636.52 | 0 | 0 | 0.501 |
| NAÏVE | 0.66 | 0.56 | 2.61 | 18809.50 | 18830.11 | 0 | 0 | 0.349 |
| REHEARSAL | 0.64 | 0.56 | 3.79 | 21685.03 | 21704.76 | 0 | 0 | 0.326 |
| HAORANZHU | 0.70 | 0.67 | 366.22 | 21646.78 | 21688.30 | 0 | 0 | 0.263 |
| AVG |  |  |  |  |  |  | 0 | 0.5 |

### 4.4. All (ALL) track

Finally in Table 4 the results averaged across tracks are reported for the ALL scoreboard. Also in this case the competing teams were 7 over a total of 11 with $U T /_L G$ as the winning team. With an average testing accuracy of $/sim 92 /%$, a average

Table 2

MT-NC track results for the 11 finalists of the competition and the three baselines. Teams not appearing in the table did not compete in this track.

| TEAM NAME | TEST ACC <br> (/%) | VAL ACCavg <br> (/%) | RUNtime <br> (M) | $/mathrm{RAM}_{/text {avg }}$ (MB) <br> (MB) | $/mathrm{RAM}_{/text {max }}$ <br> $(/mathrm{MB}$ ) |  | DISK $_{/text {max }}$ <br> (MB) | $C L_{/text {score }}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AR1 | 0.93 | 0.53 | 16.02 | 10263.19 | 14971.72 | 0 | 0 | 0.693 |
| UT_LG | 0.95 | 0.55 | 19.02 | 13793.31 | 16095.20 | 0 | 0 | 0.691 |
| Yc14600 | 0.97 | 0.54 | 11.81 | 15870.62 | 19403.57 | 0 | 0 | 0.686 |
| Soony | 0.97 | 0.55 | 55.02 | 14005.91 | 16049.12 | 0 | 0 | 0.679 |
| Jodelet | 0.97 | 0.55 | 2.55 | 17893.58 | 23728.84 | 0 | 0 | 0.679 |
| Jun2tong | 0.96 | 0.55 | 28.80 | 18488.68 | 19588.57 | 0 | 0 | 0.671 |
| ICT_VIPL | 0.99 | 0.55 | 25.20 | 2432.56 | 2432.84 | 562.5 | 562.5 | 0.630 |
| Rehearsal | 0.87 | 0.51 | 4.49 | 20446.93 | 28329.14 | 0 | 0 | 0.626 |
| JimiB | 0.95 | 0.78 | 204.56 | 21002.95 | 24528.27 | 0 | 0 | 0.607 |
| MrGrandDy | 0.94 | 0.54 | 46.52 | 27904.55 | 32921.94 | 0 | 0 | 0.604 |
| NoobMASTER | 0.95 | 0.53 | 68.07 | 27899.86 | 32910.23 | 0 | 0 | 0.597 |
| HaoranZhu | 0.57 | 0.32 | 343.50 | 21223.30 | 28366.48 | 0 | 0 | 0.351 |
| NaÏve | 0.02 | 0.13 | 3.41 | 17897.38 | 23726.40 | 0 | 0 | 0.318 |
| AVG | 0.85 | 0.51 | 63.77 | 17624.83 | 21773.26 | 43.27 | 43.27 | 0.60 |

Table 3

NIC track results for the 11 finalists of the competition and the three baselines. Teams not appearing in the table did not compete in this track

| TEAM NAME | TEST ACC <br> (/%) | VAL ACC $_{/text {avg }}$ (/%) | RUN ${ }_{/text {time }}$ <br> (M) | $/mathrm{RAM}_{/text {avg }}$ <br> (MB) | $/mathrm{RAM}_{/text {max }}$ <br> (MB) | DISK $_{/text {avg }}$ <br> (MB) | DISK $_{/text {max }}$ <br> (MB) | ${ }$ score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UT_LG | 0.91 | 0.58 | 123.22 | 6706.61 | 7135.77 | 0 | 0 | 0.706 |
| Jodelet | 0.83 | 0.54 | 14.12 | 10576.67 | 11949.16 | 0 | 0 | 0.694 |
| AR1 | 0.71 | 0.48 | 28.19 | 3307.62 | 4467.64 | 0 | 0 | 0.693 |
| ICT_VIPL | 0.90 | 0.56 | 91.29 | 2485.95 | 2486.03 | 192.187 | 375 | 0.625 |
| Yc14600 | 0.89 | 0.57 | 160.24 | 16069.91 | 21550.97 | 0 | 0 | 0.586 |
| Rehearsal | 0.74 | 0.50 | 60.32 | 15038.34 | 19488.43 | 0 | 0 | 0.585 |
| Soony | 0.82 | 0.52 | 280.39 | 12933.28 | 14241.57 | 0 | 0 | 0.533 |
| JimiB | 0.87 | 0.56 | 272.98 | 13873.04 | 21000.51 | 0 | 0 | 0.533 |
| NOOBMASTER | 0.47 | 0.32 | 300.15 | 14492.13 | 18262.32 | 0 | 0 | 0.346 |
| Naïve | 0.02 | 0.02 | 9.45 | 10583.50 | 11917.55 | 0 | 0 | 0.331 |
| AVG | 0.72 | 0.47 | 134.03 | 10606.70 | 13249.99 | 19.22 | 37.50 | 0.56 |

Table 4

ALL track results for the 11 finalists of the competition and the three baselines. Teams not appearing in the table did not compete in this track

| TEAM NAME | TEST ACC <br> (/%) | VAL ACC avg <br> (/%) | RUN $_{/text {time }}$ <br> (M) | $/mathrm{RAM}_{/text {avg }}$ <br> (MB) | $/mathrm{RAM}_{/text {max }}$ <br> $(/mathrm{MB}$ ) | DISK $_{/text {avg }}$ <br> (MB) | DISK $_{/text {max }}$ <br> (MB) | $C L_{/text {score }}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UT_LG | 0.92 | 0.68 | 68.67 | 10643.25 | 11624.87 | 0 | 0 | 0.694 |
| JoDELET | 0.88 | 0.64 | 6.59 | 15758.62 | 18169.32 | 0 | 0 | 0.680 |
| Ar1 | 0.80 | 0.58 | 20.46 | 8040.47 | 10092.72 | 0 | 0 | 0.663 |
| Yc14600 | 0.91 | 0.65 | 64.88 | 16425.64 | 19800.48 | 0 | 0 | 0.653 |
| ICT_VIPL | 0.95 | 0.68 | 76.73 | 2459.31 | 2459.68 | 392.187 | 562.5 | 0.617 |
| Soony | 0.88 | 0.63 | 120.33 | 14533.97 | 15763.60 | 0 | 0 | 0.612 |
| Rehearsal | 0.75 | 0.52 | 22.87 | 19056.77 | 23174.11 | 0 | 0 | 0.570 |
| JimiB | 0.91 | 0.74 | 242.12 | 17995.61 | 23765.51 | 0 | 0 | 0.542 |
| NOOBMASTER | 0.76 | 0.53 | 147.59 | 24714.06 | 30266.62 | 0 | 0 | 0.464 |
| Naïve | 0.23 | 0.24 | 5.16 | 15763.46 | 18158.02 | 0 | 0 | 0.327 |
| AVG | 0.80 | 0.59 | 77.54 | 14539.12 | 17327.49 | 39.22 | 56.25 | 0.58 |

memory consumption of $/sim 10 /mathrm{~GB}$ and a running time of $/sim 68$ minutes, its relatively simple solution suggests continual learning for practical object recognition applications to be feasible in the real-world, even with a large number of small non-i.i.d. bathes.

### 4.5. Discussion

Given the main competition results and the additional solutions details reported in Appendix A, we can formulate a number of observations to better understand current issues, consolidated approaches and possible future directions for competitive continual learning algorithms tested on real-world computer vision applications.

![](https://cdn.mathpix.com/cropped/2024_08_15_07039817fb8d1f0d2cc3g-10.jpg?height=502&width=754&top_left_y=177&top_left_x=560)

Fig. 3. Percentage (/%) of finalists solutions for each track employing an architectural, regularization or rehearsal strategy. Percentages do not sum to 100/% since many approached used hybrid strategies. Better viewed in colors. (For interpretation of the colors in the figure(s), the reader is referred to the web version of this article.)

In particular, we note:

- Different difficulty for different scenarios: averaging the 11 finalists test accuracy results we can easily deduce that the MT-NC track or scenario was easier than the NI one ( $/sim 85 /%$ vs $/sim 82 /%$ ), while the NIC track was the most difficult with a average accuracy of $/sim 72 /%$. This is not totally surprising, considering that the MT-NC setting allows access to the additional task labels and the NI scenario does not include dramatic distributional shifts, while the NIC one includes a substantially larger number of smaller training batches. Moreover, a number of researchers already pointed out how different training/testing regimes impact forgetting and the continual learning process [25,23,13].
- $100 /%$ of the teams used a pre-trained model: All the solutions, for all the tracks started from a pre-trained model on ImageNet. While starting from a pre-trained model is notably becoming a standard for real-world computer vision applications, we find it interesting to point out such a pervasive use in the challenge. While this does not mean pretrained model should be used for every continual learning algorithm in general, it strongly suggests that for solving real-world computer vision application today, pre-training is mostly needed.
- $/sim 90 /%$ of the teams used a rehearsal strategy: rehearsal constitutes today one of the easiest and effective solution to continual learning where previous works [12,18] have shown that even a very small percentage of previously encountered training data can have huge impacts on the final accuracy performance. Hence, it is not surprising that a large number of teams opted to use it for maximizing the $C L_{/text {score }}$, which only slightly penalized its usage.
- $/sim 45 /%$ of the teams used a regularization approach: regularization strategies have been extensively used in the competition. These approaches are mostly concerned with the idea of regulating the learning process over new data distribution mostly by preserving weights that are important for previous data distributions [17]. It worth noting though, that only 1 team used it alone and not in conjunction with a plain rehearsal or architectural approaches.
- only $/sim 27 /%$ of the teams used an architectural approach: less than one third of the participants did use an architectural approach but only on conjunction with a rehearsal or regularization one. This evidence reinforces the hypothesis that architectural-only approaches are difficult to scale efficiently over a large number of tasks or batches [35].
- Increasing replay usage with track complexity: as shown in Fig. 3, it is worth noting that as the track complexity increased, the proposed solutions tended to include more replay mechanisms. For example, for the NIC track, all the approaches included rehearsal, often used in conjunction with a regularization or architectural approach.
- High memory replay size: it is interesting to note that many CL solutions employing rehearsal have chosen to use a growing memory replay buffer rather than a fixed one with an average maximum memory size (across teams and tracks) of $/sim 26 /mathrm{k}$ patterns. This is a very large number considering that is about $/sim 21 /%$ of the total CORe50 training set images.
- Different hyper-parameters selection: An important note to make is about the hyperparameters selection and its implication to algorithms generalization and robustness. Almost all participants' solutions involved a carefully fine-tuned hyper-parameters selection which was different based on the continual scenario tackled. This somehow highlights the weakness of state-of-the-art algorithms and their inability to truly generalize to novel situations never encountered before. A notable exception is the AR1 baseline, which performed reasonably well in all the tracks with a shared hyperparametrization.


## 5. Conclusions and future improvements

The 1st Continual Learning for Computer Vision Challenge held at CVPR2020 has been one of the first large-scale continual learning competition ever organized with a raised benchmark complexity and targeting real-word applications in computer
vision. This challenge allowed every continual learning algorithm to be fairly evaluated with shared and unifying criteria and pushing the CL community to work on more realistic benchmarks than the more common MNIST or CIFAR.

After a careful investigation and analysis of the competition results we can conclude that continual learning algorithms are mostly ready to face real-world settings involving high-dimensional video streams. This is mostly thanks to hybrid strategies often combined with plain replay mechanisms. The winning strategies designed and implemented for this competition have clearly showed the impact of this simple yet effective idea, proposing original hybrid solutions enhancing the strength points and filling in the gaps of orthogonal approaches. However, it remains unclear if such techniques can scale over longer data sequences and without such an extensive use of replay.

Despite the significant participation and success of the 1st edition of the challenge, a number of possible improvements and suggestions for future continual learning competitions can be formulated:

- Discourage over-engineered solutions: one of the main goals of the competition was to evaluate the applicability of current continual learning algorithms on real-world computer vision problems. However, given the substantial freedom given through the competition rules to achieve this goal, we have noticed a number of over-engineered solutions aimed at improving the $C L_{/text {score }}$ but not really significant in terms of novelty of scientific interest. This in turns forced every other participants to focus on over-engineering rather than on the core continual learning issues. For example, data loading or compression algorithms may be useful to decrease memory and compute overheads but may be applicable to most of the solutions proposed, making them less interesting and out of the scope of competition. For this reason, we believe that finding a good trade-off between realism and scientific interest of the competition will be fundamental for future challenges in this area. We suggest for example to block the possibility to optimize the data loading algorithms and to count the number of replay patterns rather than their bytes overhead. This would also help to modulate benchmark complexity and prevent saturation.
- Automatize evaluation: in the current settings of the challenge the evaluation was client-side (on the participants machines) for the pre-selection phase and on a server-side shared hardware for the finals. To ensure the fairness of the results and the competition rules adherence, the code that generated each submission had to be included as well. However, an always-available remote docker evaluation similar to the one proposed for the AnimalAI Olympics [7], would allow a single phase competition with an always coherent and updated scoreboard, stimulating in turns teams participation and retention over the competition period. This would also alleviate some burdens at the organization levels, reducing the amount of manual interventions.
- Add scalability metrics: An interesting idea to tame the challenge complexity while still providing a good venue for assessing continual learning algorithms advancement, would be to include other than the already proposed metrics, a number of derivative ones taking into account their trend over time rather than their absolute value. This would help to better understand their scalability on more complex problems and longer tasks/batches sequences and incentivize efficient solutions with constant memory/computation overheads.
- Encourage the focus on original learning strategies: Another important possible improvement of the competition would be setting up a number of incentives and disincentives to explore interesting research directions in continual learning. For example, the usage of pre-trained models has been extensively used for the competition by all the participants. However it would have been also interesting to see proposals not taking advantage of it as well. In the next competition edition we plan to discourage the use of pre-trained models, different hyperparameters for each setting track and increase the memory usage weight associated to the $C L_{/text {score }}$.


## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

We would like to thank all the Continual Learning in Computer Vision workshop organizers, challenge chairs and participants for making this competition possible. We also like to acknowledge our sponsors ContinualAI, Element AI, Nvidia and Intel Labs for their support in the organization of the workshop at CVPR 2020.

## Appendix A. Additional details

In this appendix, additional details for each team and track are reported (see Table A.5, Table A.6, Table A. 7 and Table A.8). In particular, for each solution we report: i) the model type; ii) if the model was pre-trained; iii) the type of strategy used; iv) the number of eventual replay examples; v) the number of training epochs per batch; vi) the mini-batch size used.

Table A. 5

11 finalists of the CLVision Competition.

| Team name | Team members | Institution |
| :---: | :---: | :---: |
| HaoranZhu | Haoran Zhu | New York University |
| ICT_VIPL | Chen He, Qiyang Wan, Fengyuan Yang, Ruiping <br> Wang, Shiguang Shan, Xilin Chen | Chinese Academy of Sciences |
| JimiB | Giacomo Bonato, Francesco Lakj, Alex Torci- <br> novich, Alessandro Casella | Independent Researchers |
| Jodelet | Quentin Jodelet, Vincent Gripon, Tsuyoshi Mu- <br> rata | Tokyo Institute of Technology |
| Jun2Tong | Junyong Tong, Amir Nazemi, Mohammad Javad <br> Shafiee, Paul Fieguth | University of Waterloo |
| MrGranddy | Vahit Bugra Yesilkaynak, Firat Oncel, Furkan <br> Ozcelik, Yusuf Huseyin Sahin, Gozde Unal | Istanbul Technical University |
| Noobmaster | Zhaoyang Wu, Yilin Shao, Jiaxuan Zhao, and <br> Bingnan Hu | Xidian University |
| Sahinyu | Yusuf H. Sahin, Furkan Ozcelik, Firat Oncel, <br> Vahit Bugra Yesilkaynak, Gozde Unal | Istanbul Technical University |
| Soony | Soonyong Song, Heechul Bae, Hyonyoung Han, <br> Youngsung Son | Electronics and Telecommunications <br> Research Institute |
| UT_LG | Zheda Mai, Hyunwoo Kim, Jihwan Jeong, Scott <br> Sanner | University of Toronto |
| YC14600 | Yu Chen, Jian Ma, Hanyuan Wang, Yuhang <br> Ming, Jordan Massiah, Tom Diethe | University of Bristol |

Table A. 6

Approaches and baselines details for the NI track.

| Team | Model | Pre-trained | Strategy | Replay Examples ${ }_{/max }$ | Epochs | Mini-batch size |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| UT_LG | DenseNet-161 | yes | rehearsal | 80000 | 2 | 32 |
| Yc14600 | ResNeSt50 | yes | regularization /& rehearsal | 12000 | 1 | 16 |
| ICT_VIPL | WideResNet-50 | yes | rehearsal | 4000 | 2 | 80 |
| Jodelet | ResNet-50 | yes | rehearsal | 6400 | 1 | 32 |
| Soony | ResNext101/50 /& DenseNet161 | yes | architectural /& rehearsal | 119894 | 1 | 900 |
| JimiB | resnext101 | yes | regularization /& rehearsal | 11989 | 8 | 32 |
| Jun2tong | ResNet-50 | yes | regularization /& rehearsal | 12000 | 5 | 32 |
| Sahinyu | EfficientNet-B7 | yes | rehearsal | 8000 | 2 | 27 |
| Ar1 | mobilenetV1 | yes | architectural | 1500 | 4 | 128 |
| Noobmaster | resnet-101 | yes | rehearsal | 24000 | 5 | 32 |
| MrGranddy | EfficientNet-B7 | yes | regularization /& architectural | 0 | 1 | 32 |
| Naïve | mobilenetV1 | yes | n.a. | 0 | 4 | 128 |
| Rehearsal | mobilenetV1 | yes | rehearsal | 160 | 4 | 128 |
| HaoranZhu | ResNet-50 | yes | regularization | 0 | 10 | 32 |

Table A. 7

Approaches and baselines details the MT-NC track.

| Team | Model | Pre-trained | Strategy |  | Epochs | Mini-batch size |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ar1 | mobilenetV1 | yes | architectural /& rehearsal | 1500 | 4 | 128 |
| UT_LG | DenseNet-161 | yes | architectural | 0 | 1 | 32 |
| Yc14600 | ResNeSt50 | yes | regularization /& rehearsal | 4500 | 1 | 16 |
| Soony | ResNext101/50 /& DenseNet161 | yes | architectural /& rehearsal | 119890 | 3 | 100 |
| Jodelet | ResNet-50 | yes | rehearsal | 6400 | 1 | 32 |
| Jun2tong | ResNet-50 | yes | regularization /& rehearsal | 45000 | 1 | 32 |
| ICT_VIPL | ResNeXt-50 | yes | rehearsal | 3000 | 1 | 32 |
| Rehearsal | mobilenetV1 | yes | rehearsal | 180 | 4 | 128 |
| JimiB | resnext101 | yes | regularization /& rehearsal | 11989 | 8 | 32 |
| MrGranddy | EfficientNet-B7 | yes | regularization /& architectural | 0 | 1 | 32 |
| Noobmaster | resnet-101 | yes | rehearsal | 18000 | 5 | 32 |
| HaoranZhu | ResNet-50 | yes | regularization | 0 | 10 | 32 |
| Naïve | mobilenetV1 | yes | n.a. | 0 | 4 | 128 |

Table A. 8

Approaches and baselines details for the NIC track.

| Team | Model | Pre-trained | Strategy | Replay Examples | Epochs | Mini-batch size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| UT_LG | DenseNet-161 | yes | rehearsal | 78200 | 1 | 32 |
| Jodelet | ResNet-50 | yes | rehearsal | 6400 | 1 |  |
| Ar1 | mobilenetV1 | yes | architectural /& rehearsal | 1500 | 4 |  |
| ICT_VIPL | ResNet50 | yes | rehearsal | 2000 | 1 |  |
| Yc14600 | ResNeSt50 | yes | regularization /& rehearsal | 19550 | 128 |  |
| Rehearsal | mobilenetV1 | rehearsal | 7820 | 1 | 34 |  |
| Soony | ResNext101/50 /& DenseNet161 | yes | architectural /& rehearsal | 119890 | 4 |  |
| JimiB | resnext101 | yes | regularization /& rehearsal | 11989 | 1 | 128 |
| Noobmaster | resnet-101 | yes | rehearsal | 23460 | 6 |  |
| Naïve | mobilenetV1 | yes | n.a. | 0 | 500 |  |

## References

[1] H. Bae, E. Brophy, R.H. Chan, B. Chen, F. Feng, G. Graffieti, V. Goel, X. Hao, H. Han, S. Kanagarajah, et al., IROS 2019 lifelong robotic vision: object recognition challenge [competitions], IEEE Robot. Autom. Mag. 27 (2) (2020) 11-16.

[2] M. Buda, A. Maki, M.A. Mazurowski, A systematic study of the class imbalance problem in convolutional neural networks, Neural Netw. 106 (2018) 249-259.

[3] L. Caccia, E. Belilovsky, M. Caccia, J. Pineau, Online learned continual compression with adaptive quantization modules, 2019.

[4] M. Caccia, P. Rodriguez, O. Ostapenko, F. Normandin, M. Lin, L. Caccia, I. Laradji, I. Rish, A. Lacoste, D. Vazquez, et al., Online fast adaptation and knowledge accumulation: a new approach to continual learning, arXiv preprint, arXiv:2003.05856, 2020

[5] A. Chaudhry, M. Rohrbach, M. Elhoseiny, T. Ajanthan, P.K. Dokania, P.H.S. Torr, M. Ranzato, On tiny episodic memories in continual learning, arXiv: 1902.10486, 2019.

[6] Y. Chen, T. Diethe, P. Flach, Bypassing gradients re-projection with episodic memories in online continual learning, arXiv preprint, arXiv:2006.11234, 2020.

[7] M. Crosby, B. Beyret, M. Halina, The Animal-AI Olympics, Nat. Mach. Intell. 1 (5) (2019) 257.

[8] N. Díaz-Rodríguez, V. Lomonaco, D. Filliat, D. Maltoni, Don't forget, there is more than forgetting: new metrics for continual learning, arXiv preprint, arXiv:1810.13166, 2018.

[9] H.J. Escalante, W.W. Tu, I. Guyon, D.L. Silver, E. Viegas, Y. Chen, W. Dai, Q. Yang, AutoML @ NeurIPS 2018 challenge: design and results, in: The NeurIPS'18 Competition, Springer, 2020, pp. 209-229.

[10] S. Farquhar, Y. Gal, Differentially private continual learning, arXiv preprint, arXiv:1902.06497, 2019.

[11] R.M. French, Catastrophic forgetting in connectionist networks, Trends Cogn. Sci. 3 (4) (1999) 128-135.

[12] T.L. Hayes, N.D. Cahill, C. Kanan, Memory efficient experience replay for streaming learning, in: 2019 International Conference on Robotics and Automation (ICRA), IEEE, 2019, pp. 9769-9776.

[13] T.L. Hayes, R. Kemker, N.D. Cahill, C. Kanan, New metrics and experimental paradigms for continual learning, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2018, pp. 2031-2034.

[14] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770-778.

[15] A. Krizhevsky, G. Hinton, et al., Learning multiple layers of features from tiny images, 2009.

[16] Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, Gradient-based learning applied to document recognition, Proc. IEEE 86 (11) (1998) $2278-2324$

[17] T. Lesort, V. Lomonaco, A. Stoian, D. Maltoni, D. Filliat, N. Díaz-Rodríguez, Continual learning for robotics: definition, framework, learning strategies, opportunities and challenges, Inf. Fusion 58 (2020) 52-68, https://doi.org/10.1016/j.inffus.2019.12.004, https://linkinghub.elsevier.com/retrieve/pii/ S1566253519307377.

[18] V. Lomonaco, Continual Learning with Deep Architectures, Ph.D. thesis, alma, 2019, http://amsdottorato.unibo.it/9073/.

[19] V. Lomonaco, D. Maltoni, CORe50: a new dataset and benchmark for continuous object recognition, in: CoRL, 2017, pp. 1-10.

[20] V. Lomonaco, D. Maltoni, CORe50: a new dataset and benchmark for continuous object recognition, in: Proceedings of the 1st Annual Conference on Robot Learning (CoRL), vol. 78, 2017, pp. 17-26, http://proceedings.mlr.press/v78/lomonaco17a.html.

[21] D. Lopez-Paz, M. Ranzato, Gradient episodic memory for continual learning, in: Advances in Neural Information Processing Systems., 2017, pp. 6467-6476.

[22] Z. Mai, H. Kim, J. Jeong, S. Sanner, Batch-level experience replay with review for continual learning, arXiv:2007.05683, 2020.

[23] D. Maltoni, V. Lomonaco, Continuous learning in single-incremental-task scenarios, Neural Netw. 116 (2019) 56-73.

[24] M. McCloskey, N.J. Cohen, Catastrophic interference in connectionist networks: the sequential learning problem, in: Psychology of Learning and Motivation, vol. 24, Elsevier, 1989, pp. 109-165.

[25] S.I. Mirzadeh, M. Farajtabar, R. Pascanu, H. Ghasemzadeh, Understanding the role of training regimes in continual learning, arXiv preprint, arXiv: 2006.06958, 2020.

[26] G.I. Parisi, R. Kemker, J.L. Part, C. Kanan, S. Wermter, Continual lifelong learning with neural networks: a review, Neural Netw. 113 (2019) 54-71, https://doi.org/10.1016/j.neunet.2019.01.012, https://linkinghub.elsevier.com/retrieve/pii/S0893608019300231.

[27] G. Pasquale, C. Ciliberto, F. Odone, L. Rosasco, L. Natale, Are we done with object recognition? The iCub robot's perspective, Robot. Auton. Syst. 112 (2019) 260-281, https://doi.org/10.1016/j.robot.2018.11.001, arXiv:1709.09882v2.

[28] L. Pellegrini, G. Graffieti, V. Lomonaco, D. Maltoni, Latent replay for real-time continual learning, in: International Conference on Intelligent Robots and Systems (IROS), 2020, arXiv:1912.01100

[29] B. Pfülb, A. Gepperth, A comprehensive, application-oriented study of catastrophic forgetting in DNNs, in: International Conference of Learning Representation (ICLR), 2019.

[30] J. Quionero-Candela, M. Sugiyama, A. Schwaighofer, N.D. Lawrence, Pascale 2 challenge: Learning when test and training inputs have different distributions challenge, 2005.

[31] J. Quionero-Candela, M. Sugiyama, A. Schwaighofer, N.D. Lawrence, Dataset Shift in Machine Learning, The MIT Press, 2009.

[32] M.D. Richard, R.P. Lippmann, Neural network classifiers estimate Bayesian a posteriori probabilities, Neural Comput. 3 (4) (1991) 461-483.

[33] A. Robins, Catastrophic forgetting, rehearsal and pseudorehearsal, Connect. Sci. 7 (2) (1995) 123-146.

[34] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, ImageNet large scale visual recognition challenge, Int. J. Comput. Vis. 115 (3) (2015) 211-252, https://doi.org/10.1007/s11263-015-0816-y.

[35] A.A. Rusu, N.C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, R. Hadsell, Progressive neural networks, arXiv preprint, arXiv:1606.04671, 2016

[36] Q. She, F. Feng, X. Hao, Q. Yang, C. Lan, V. Lomonaco, X. Shi, Z. Wang, Y. Guo, Y. Zhang, F Qiao, R.H.M. Chan, OpenLORIS-object: a robotic vision dataset and benchmark for lifelong deep learning, in: 2020 International Conference on Robotics and Automation (ICRA), 2020, pp. 4767-4773.

[37] I.Z. Yalniz, H. Jégou, K. Chen, M. Paluri, D. Mahajan, Billion-scale semi-supervised learning for image classification, arXiv preprint, arXiv:1905.00546, 2019.


[^0]:    at This paper was submitted to the Competition Section of the journal

    * Corresponding author at: Via Cesare Pavese, 50, 47521 Cesena FC, Italy. E-mail address: vincenzo.lomonaco@unibo.it (V. Lomonaco).

[^1]:    2 Multi-Task-NC constitutes a simplified variation of the originally proposed New Classes ( NC ) protocol [20] (where the task label is not provided during train and test).

[^2]:    ${ }^{4}$ https://sites.google.com/view/clvision2020/challenge/challenge-winners.

    5 https://imgaug.readthedocs.io.

# Filtered Channel Features for Pedestrian Detection 

Shanshan Zhang<br>Rodrigo Benenson<br>Bernt Schiele<br>Max Planck Institute for Informatics<br>Saarbrücken, Germany<br>firstname.lastname@mpi-inf.mpg.de


#### Abstract

This paper starts from the observation that multiple top performing pedestrian detectors can be modelled by using an intermediate layer filtering low-level features in combination with a boosted decision forest. Based on this observation we propose a unifying framework and experimentally explore different filter families. We report extensive results enabling a systematic analysis. Using filtered channel features we obtain top performance on the challenging Caltech and KITTI datasets, while using only $H O G+L U V$ as low-level features. When adding optical flow features we further improve detection quality and report the best known results on the Caltech dataset, reaching $93 /%$ recall at 1 FPPI.


## 1. Introduction

Pedestrian detection is an active research area, with 1000+ papers published in the last decade ${ }^{1}$, and well established benchmark datasets [9, 13]. It is considered a canonical case of object detection, and has served as playground to explore ideas that might be effective for generic object detection.

Although many different ideas have been explored, and detection quality has been steadily improving [2], arguably it is still unclear what are the key ingredients for good pedestrian detection; e.g. it remains unclear how effective parts, components, and features learning are for this task.

Current top performing pedestrian detection methods all point to an intermediate layer (such as max-pooling or filtering) between the low-level feature maps and the classification layer [42, 45, 29, 25]. In this paper we explore the simplest of such intermediary: a linear transformation implemented as convolution with a filter bank. We propose a framework for filtered channel features (see figure 1) that unifies multiple top performing methods [8, 1, 45, 25], and that enables a systematic exploration of different filter banks. Our experiments show that, with the proper filter bank, filtered channel features reach top detection quality.

[^0]![](https://cdn.mathpix.com/cropped/2024_08_15_54f9e5398199d0e0fa54g-01.jpg?height=557&width=876&top_left_y=692&top_left_x=1058)

Figure 1: Filtered feature channels illustration, for a single weak classifier reading over a single feature channel. Integral channel features detectors pool features via sums over rectangular regions $[8,1]$. We can equivalently rewrite this operation as convolution with a filter bank followed by single pixel reads (see §2). We aim to answer: What is the effect of selecting different filter banks?

It has been shown that using extra information at test time (such as context, stereo images, optical flow, etc.) can boost detection quality. In this paper we focus on the "core" sliding window algorithm using solely HOG+LUV features (i.e. oriented gradient magnitude and colour features). We consider context information and optical flow as add-ons, included in the experiments section for the sake of completeness and comparison with existing methods. Using only HOG+LUV features we already reach top performance on the challenging Caltech and KITTI datasets, matching results using optical flow and significantly more features (such as LBP and covariance [42, 29]).

### 1.1. Related work

Recent survey papers discuss the diverse set of ideas explored for pedestrian detection [10, 14, 9, 2]. The most recent survey [2] indicates that the classifier choice (e.g. linear/non-linear SVM versus decision forest) is not a clear differentiator regarding quality; rather the features used seem more important.

Creativity regarding different types of features has not
been lacking. HOG) The classic HOG descriptor is based on local image differences (plus pooling and normalization steps), and has been used directly [5], as input for a deformable parts model [11], or as features to be boosted [20, 26]. The integral channel features detector [8, 1] uses a simpler HOG variant with sum pooling and no normalizations. Many extensions of HOG have been proposed (e.g. [17, 11, 6, 34]). LBP) Instead of using the magnitude of local pixel differences, LBP uses the difference sign only as signal [41, 42, 29]. Colour) Although the appearance of pedestrians is diverse, the background and skin areas do exhibit a colour bias. Colour has shown to be an effective feature for pedestrian detection and hence multiple colour spaces have been explored (both hand-crafted and learned) [8, 18, 19, 23]. Local structure) Instead of simple pixel values, some approaches try to encode a larger local structure based on colour similarities (soft-cue) [40, 15], segmentation methods (hard-decision) [27, 32, 36], or by estimating local boundaries [21]. Covariance) Another popular way to encode richer information is to compute the covariance amongst features (commonly colour, gradient, and oriented gradient) [38, 29]. Etc.) Other features include bag-of-words over colour, HOG, or LBP features [4]; learning sparse dictionary encoders [33]; and training features via a convolutional neural network [35] ( $[37,16]$ appeared while preparing this manuscript). Additional features specific for stereo depth or optical flow have been proposed, however we consider these beyond the focus of this paper. For our flow experiments we will use difference of frames from weakly stabilized videos (SDt) [30].

All the feature types listed above can be used in the integral channel features detector framework [8]. This family of detectors is an extension of the old ideas from Viola/&Jones [39]. Sums of rectangular regions are used as input to decision trees trained via Adaboost. Both the regions to pool from and the thresholds in the decision trees are selected during training. The crucial difference from the pioneer work [39] is that the sums are done over feature channels other than simple image luminance.

Current top performing pedestrian detection methods (dominating INRIA [5], Caltech [9] and KITTI datasets [13]) are all extensions of the basic integral channel features detector (named ChnFtrs in [8], which uses only HOG+LUV features). SquaresChnFtrs [2], InformedHaar [45], and LDCF [25], are discussed in detail in section 2.2. Katamari exploits context and optical flow for improved performance. SpatialPooling (+) [29] adds max-pooling on top of sum-pooling, and uses additional features such as covariance, LBP, and optical flow. Similarly, Regionlets [42] also uses extended features and max-pooling, together with stronger weak classifiers and training a cascade of classifiers. Out of these, Regionlets is the only method that has also shown good performance on general classes datasets such as Pascal VOC and ImageNet.

In this paper we will show that vanilla HOG+LUV features have not yet saturated, and that, when properly used, they can reach top performance for pedestrian detection.

### 1.2. Contributions

- We point out the link between ACF [7], (Squares) ChnFtrs [8, 1, 2], InformedHaar [45], and LDCF [25]. See section 2.
- We provide extensive experiments to enable a systematic analysis of the filtered integral channels, covering aspects not explored by related work. We report the summary of $65+$ trained models ( $/sim 10$ days of single machine computation). See sections 4,5 and 7 .
- We show that top detection performance can be reached on Caltech and KITTI using HOG+LUV features only. We additionally report the best known results on Caltech. See section 7.


## 2. Filtered channel features

Before entering the experimental section, let us describe our general architecture. Methods such as ChnFtrs [8], SquaresChnFtrs [1, 2] and ACF [7] all use the basic architecture depicted in figure 1 (top part, best viewed in colours). The input image is transformed into a set of feature channels (also called feature maps), the feature vector is constructed by sum-pooling over a (large) set of rectangular regions. This feature vector is fed into a decision forest learned via Adaboost. The split nodes in the trees are a simple comparison between a feature value and a learned threshold. Commonly only a subset of the feature vector is used by the learned decision forest. Adaboost serves both for feature selection and for learning the thresholds in the split nodes. For more details on this basic architecture, please consult $[8,1]$.

A key observation, illustrated in figure 1 (bottom), is that such sum-pooling can be re-written as convolution with a filter bank (one filter per rectangular shape) followed by reading a single value of the convolution's response map. This "filter + pick" view generalizes the integral channel features [8] detectors by allowing to use any filter bank (instead of only rectangular shapes). We name this generalization "filtered channel features detectors".

In our framework, ACF [7] has a single filter in its bank, corresponding to a uniform $4 /times 4$ pixels pooling region. ChnFtrs [8] was a very large (tens of thousands) filter bank comprised of random rectangular shapes. SquaresChnFtrs [1, 2], on the other hand, has only 16 filters, each with a square-shaped uniform pooling region of different sizes. See figure 2 a for an illustration of the SquaresChnFtrs filters, the upper-left filter corresponds to ACF's one.

The InformedHaar [45] method can also be seen as a filtered channel features detector, where the filter bank (and read locations) are based on a human shape template (thus the "informed" naming). LDCF [25] is also a particular instance of this framework, where the filter bank consists of PCA bases of patches from the training dataset. In sections 4 and 5 we provide experiments revisiting some of the design decisions of these methods.

Note that all the methods mentioned above (and in the majority of experiments below) use only HOG+LUV feature channels ${ }^{2}$ ( 10 channels total). Using linear filters and decision trees on top of these does not allow to reconstruct the decision functions obtained when using LBP or covariance features (used by [42, 29]). Compared to SpatialPooling [29] and Regionlets [42] the main differences are that we use simpler features, do pooling only via filtering (instead of mixed mean and maxpooling), use simpler weak classifiers (short decision trees), and vanilla discrete Adaboost. We consider the approach considered here mainly orthogonal to the ideas in [42, 29].

### 2.1. Evaluation protocol

For our experiments we use the Caltech [9, 2] and KITTI datasets [13]. The popular INRIA dataset is considered too small and too close to saturation to provide interesting results. All Caltech results are evaluated using the provided toolbox, and summarised by log-average miss-rate (MR, lower is better) in the $/left[10^{-2}, 10^{0}/right]$ FPPI range for the "reasonable" setup. KITTI results are evaluated via the online evaluation portal, and summarised as average precision (AP, higher is better) for the "moderate" setup.

Caltech10x The raw Caltech dataset consists of videos (acquired at 30 Hz ) with every frame annotated. The standard training and evaluation considers one out of each 30 frames ( 1631 pedestrians over 4250 frames in training, 1014 pedestrians over 4024 frames in testing).

In our experiments of section 5 we will also consider a $10 /times$ increased training set where every 3rd frame is used (linear growth in pedestrians and images). We name this extended training set "Caltech10x". LDCF [25] uses a similar extended set for training its model (every 4th frame).

Flow Methods using optical flow do not only use additional neighbour frames during training ( $1 /leftrightarrow 4$ depending on the method), but they also do so at test time. Because they have access to additional information at test time, we consider them as a separate group in our results section.

Validation set In order to explore the design space of our pedestrian detector we setup a Caltech validation set by splitting the six training videos into five for training and one for testing (one of the splits suggested in [9]). Most of

[^1]our experiments use this validation setup. We also report (a posteriori) our key results on the standard test set for comparison to the state of the art.

For the KITTI experiments we also validate some design choices (such as search range and number of scales) before submission on the evaluation server. There we use a $2 / 3+1 / 3$ validation setup.

### 2.2. Baselines

ACF Our experiments are based on the open source release of ACF [7]. Our first baseline is vanilla $A C F$ re-trained on the standard Caltech set (not Caltech10x), all parameter details are described in section 2.3, and kept identical across experiments unless explicitly stated. On the Caltech test set it obtains $32.6 /%$ MR ( $50.2 /%$ MR on validation set). Note that this baseline already improves over more than 50 previously published methods [2] on this dataset. There is also a large gap between ACF-Ours ( $32.6 /% /mathrm{MR}$ ) and the original number from ACF-Caltech (44.2/% MR [7]). This improvement is mainly due to the change towards a larger model size (from $32 /times 64$ pixels in [7] to $60 /times 120$ here).

InformedHaar Our second baseline is a reimplementation of InformedHaar [45]. Here again we observe an important gain from using a larger model size (same change as for ACF). While the original InformedHaar paper reports $34.6 /% /mathrm{MR}$, InformedHaar-Ours reaches $27.0 /% /mathrm{MR}$ on the Caltech test set ( $39.3 /%$ MR on validation set).

For both our baselines we use exactly the same training set as the original papers. Note that the InformedHaar-Ours baseline ( $27.0 /% /mathrm{MR}$ ) is right away the best known result for a method trained on the standard Caltech training set. In section 3 we will discuss our reimplementation of LDCF [25].

### 2.3. Model parameters

Unless otherwise specified we train all our models using the following parameters. Feature channels are HOG+LUV only. The final classifier includes 4096 level-2 decision trees (L2, 3 stumps per tree), trained via vanilla discrete Adaboost. Each tree is built by doing exhaustive greedy search for each node (no randomization). The model has size $60 /times 120$ pixels, and is built via four rounds of hard negative mining (starting from a model with 32 trees, and then $512,1024,2048,4096$ trees). Each round adds 10000 additional negatives to the training set. The sliding window stride is 6 pixels (both during hard negative mining and at test time).

Compared to the default ACF parameters, we use a bigger model, more trees, more negative samples, and more boosting rounds. But we do use the same code-base and the same training set.

![](https://cdn.mathpix.com/cropped/2024_08_15_54f9e5398199d0e0fa54g-04.jpg?height=351&width=356&top_left_y=280&top_left_x=212)

(a) SquaresChntrs filters

![](https://cdn.mathpix.com/cropped/2024_08_15_54f9e5398199d0e0fa54g-04.jpg?height=364&width=359&top_left_y=696&top_left_x=211)

(c) RandomFilters

![](https://cdn.mathpix.com/cropped/2024_08_15_54f9e5398199d0e0fa54g-04.jpg?height=367&width=356&top_left_y=1123&top_left_x=212)

(e) LDCF 8 filters

![](https://cdn.mathpix.com/cropped/2024_08_15_54f9e5398199d0e0fa54g-04.jpg?height=353&width=351&top_left_y=276&top_left_x=600)

(b) Checkerboards filters

![](https://cdn.mathpix.com/cropped/2024_08_15_54f9e5398199d0e0fa54g-04.jpg?height=364&width=351&top_left_y=696&top_left_x=600)

(d) InformedFilters

![](https://cdn.mathpix.com/cropped/2024_08_15_54f9e5398199d0e0fa54g-04.jpg?height=366&width=356&top_left_y=1118&top_left_x=597)

(f) PcaForeground filters
Figure 2: Illustration of the different filter banks considered. Except for SquaresChntrs filters, only a random subset of the full filter bank is shown. /{ $/square$ Red, $/square$ White, $/square$ Green /} indicate $/{-1,0,+1/}$.

Starting from section 5 we will consider results using Caltech10x. There, better performance is reached when using level-4 decision trees (L4), and Realboost [12] instead of discrete Adaboost. All other parameters are left unchanged.

## 3. Filter bank families

Given the general architecture and the baselines described in section 2, we now proceed to explore different types of filter banks. Some of them are designed using prior knowledge and they do not change when applied across datasets, others exploit data-driven techniques for learning their filters. Sections 4 and 5 will compare their detection quality.

InformedFilters Starting from the InformedHaar [45] baseline we use the same "informed" filters but let free the positions where they are applied (instead of fixed in InformedHaar); these are selected during the boosting learning. Our initial experiments show that removing the position constraint has a small (positive) effect. Additionally we observe that the original InformedHaar filters do not include simple square pooling regions (à la SquaresChnFtrs), we thus add these too. We end up with 212 filters in total, to be applied over each of the 10 feature channels. This is equivalent to training decision trees over 2120 (non filtered) channel features.

As illustrated in figure 2d the InformedFilters have different sizes, from $1 /times 1$ to $4 /times 3$ cells ( 1 cell $=6 /times$ 6 pixels), and each cell takes a value in $/{-1,0,+1/}$. These filters are applied with a step size of 6 pixels. For a model of $60 /times 120$ pixels this results in 200 features per channel, $2120 /cdot 200=424000$ features in total ${ }^{3}$. In practice considering border effects (large filters are not applied on the border of the model to avoid reading outside it) we end up with $/sim 300000$ features. When training 4096 level- 2 decision trees, at most $4096 /cdot 3=12288$ features will be used, that is $/sim 3 /%$ of the total. In this scenario (and all others considered in this paper) Adaboost has a strong role of feature selection.

Checkerboards As seen in section 2.2 InformedHaar is a strong baseline. It is however unclear how much the "informed" design of the filters is effective compared to other possible choices. Checkerboards is a naïve set of filters that covers the same sizes (in number of cells) as InformedHaar/InformedFilters and for each size defines (see figure 2b): a uniform square, all horizontal and vertical gradient detectors ( $/pm 1$ values), and all possible checkerboard patterns. These configurations are comparable to InformedFilters but do not use the human shape as prior.

The total number of filters is a direct function of the maximum size selected. For up to $4 /times 4$ cells we end up with 61 filters, up to $4 /times 3$ cells 39 filters, up to $3 /times 3$ cells 25 filters, and up to $2 /times 2$ cells 7 filters.

RandomFilters Our next step towards removing a hand-crafted design is simply using random filters (see figure 2c). Given a desired number of filters and a maximum filter size (in cells), we sample the filter size with uniform distribution, and set its cell values to $/pm 1$ with uniform probability. We also experimented with values $/{-1,0,+1/}$ and observed a (small) quality decrease compared to the binary option).

The design of the filters considered above completely ignores the available training data. In the following, we consider additional filters learned from data.

[^2]![](https://cdn.mathpix.com/cropped/2024_08_15_54f9e5398199d0e0fa54g-05.jpg?height=581&width=836&top_left_y=235&top_left_x=165)

Figure 3: Detection quality (log-average miss-rate MR, lower is better) versus number of filters used. All models trained and tested on the Caltech validation set (see §4).

LDCF [25] The work on PCANet [3] showed that applying arbitrary non-linearities on top of PCA projections of image patches can be surprisingly effective for image classification. Following this intuition LDCF [25] uses learned PCA eigenvectors as filters (see figure 2e).

We present a re-implementation of [25] based on ACF's [7] source code. We follow the original description as closely as possible. We use the same top 4 filters of $10 /times 10$ pixels, selected per feature channel based on their eigenvalues ( 40 filters total). We do change some parameters to be consistent amongst all experiments, see sections 2.3 and 5 . The main changes are the training set (we use Caltech10x, sampled every 3 frames, instead of every 4 frames in [25]), and the model size $(60 /times 120$ pixels instead of $32 /times 64)$. As seen in section 7, our implementation (LDCF-Ours) clearly improves over the previously published numbers [25], showing the potential of the method.

For comparison with PcaForeground we also consider training LDCF 8 where the top 8 filters are selected per channel ( 80 filters total).

PcaForeground In LDCF the filters are learned using all of the training data available. In practice this means that the learned filters will be dominated by background information, and will have minimal information about the pedestrians. Put differently, learning filters from all the data assumes that the decision boundary is defined by a single distribution (like in Linear Discriminant Analysis [24]), while we might want to define it based on the relation between the background distribution and the foreground distribution (like Fisher's Discriminant Analysis [24]). In PcaForeground we train 8 filters per feature channel, 4 learned from background image patches, and 4 learned from patches extracted over pedestrians (see figure 2 f ). Compared to LDCF 8 the obtained filters are similar but not identical, all other parameters are kept identical.
Other than via PcaForeground/LDCF8, it is not clear how to further increase the number of filters used in LDCF. Past 8 filters per channel, the eigenvalues decrease to negligible values and the eigenvectors become essentially random (similar to RandomFilters).

To keep the filtered channel features setup close to InformedHaar, the filters are applied with a step of 6 pixels. However, to stay close to the original LDCF, the LDCF/PcaForeground filters are evaluated every 2 pixels. Although (for example) LDCF 8 uses only $/sim 10 /%$ of the number of filters per channel compared to Checkerboards $4 /times 4$, due to the step size increase, the obtained feature vector size is $/sim 40 /%$.

## 4. How many filters?

Given a fixed set of channel features, a larger filter bank provides a richer view over the data compared to a smaller one. With enough training data one would expect larger filter banks to perform best. We want thus to analyze the trade-off between number of filters and detection quality, as well as which filter bank family performs best.

Figure 3 presents the results of our initial experiments on the Caltech validation set. It shows detection quality versus number of filters per channel. This figure densely summarizes $/sim 30$ trained models.

InformedFilters The first aspect to notice is that there is a meaningful gap between InformedHaar-Ours and InformedFilters despite having a similar number of filters ( 209 versus 212). This validates the importance of letting Adaboost choose the pooling locations instead of hand-crafting them. Keep in mind that InformedHaar-Ours is a top performing baseline (see §2.2).

Secondly, we observe that (for the fixed training data available) $/sim 50$ filters is better than $/sim 200$. Below 50 filters the performance degrades for all methods (as expected).

To change the number of filters in InformedFilters we train a full model ( 212 filters), pick the $N$ most frequently used filters (selected from node splitting in the decision forest), and use these to train the desired reduced model. We can select the most frequent filters across channels or per channel (marked as Inf.FiltersPerChannel). We observe that per channel selection is slightly worse than across channels, thus we stick to the latter.

Using the most frequently used filters for selection is clearly a crude strategy since frequent usage does not guarantee discriminative power, and it ignores relation amongst filters. We find this strategy good enough to convey the main points of this work.

Checkerboards also reaches best results in the $/sim 50$ filters region. Here the number of filters is varied by chan-

| Training | Method | L2 | L3 | L4 | L5 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Caltech | ACF | 50.2 | 42.1 | 48.8 | 48.7 |
| Caltech10x |  | 52.6 | 49.9 | 44.9 | 41.3 |
| Caltech | Checker- | 32.9 | 30.4 | 28.0 | 31.5 |
| Caltech10x | boards | 37.0 | 31.6 | 24.7 | 24.7 |

Table 1: Effect of the training volume and decision tree depth ( $/mathrm{L} n$ ) over the detection quality (average miss-rate on validation set, lower is better), for ACF-Ours and Checkerboards variant with (61) filters of $4 /times 4$ cells. We observe a similar trend for other filter banks.

ging the maximum filter size (in number of cells). Regarding the lowest miss-rate there is no large gap between the "informed" filters and this naïve baseline.

RandomFilters The hexagonal dots and their deviation bars indicate the mean, maximum and minimum missrate obtained out of five random runs. When using a larger number of filters (50) we observe a lower (better) mean but a larger variance compared to when using fewer filters (15). Here again the gap between the best random run and the best result of other methods is not large.

Given a set of five models, we select the $N$ most frequently used filters and train new reduced models; these are shown in the RandomFilters line. Overall the random filters are surprisingly close to the other filter families. This indicates that expanding the feature channels via filtering is the key step for improving detection quality, while selecting the "perfect" filters is a secondary concern.

LDCF/PcaForeground In contrast to the other filter bank families, LDCF under-performs when increasing the number of filters (from 4 to 8 ) while using the standard Caltech training set (consistent with the observations in [25]). PcaForeground improves marginally over LDCF8.

Takeaways From figure 3 we observe two overall trends. First, the more filters the merrier, with $/sim 50$ filters as sweet spot for Caltech training data. Second, there is no flagrant difference between the different filter types.

## 5. Additional training data

One caveat of the previous experiments is that as we increase the number of filters used, so does the number of features Adaboost must pick from. Since we increased the model capacity (compared to ACF which uses a single filter), we consider using the Caltech10x dataset (§2.1) to verify that our models are not starving for data. Similar to the experiments in [25], we also reconsider the decision tree depth, since additional training data enables bigger models.

Results for two representative methods are collected in table 1. First we observe that already with the original training data, deeper trees do provide significant improvement over level-2 (which was selected when tuning over INRIA

| Aspect | MR | $/Delta$ MR |
| :--- | :---: | :---: |
| ACF-Ours | 50.8 | - |
| + filters | 32.9 | +17.9 |
| + L4 | 28.0 | +4.9 |
| + Caltech10x | 24.7 | +3.3 |
| + Realboost | 24.4 | +0.3 |
| Checkerboards4x4 | 24.4 | +26.4 |

Table 2: Ingredients to build our strong detectors (using Checkerboards $4 /times 4$ in this example, 61 filters). Validation set log-average miss-rate (MR).

data $[8,1])$. Second, we notice that increasing the training data volume does provide the expected improvement only when the decision trees are deep enough. For our following experiments we choose to use level-4 decision trees (L4) as a good balance between increased detection quality and reasonable training times.

Realboost Although previous papers on ChnFtrs detectors reported that different boosting variants all obtain equal results on this task [8, 1], the recent [25] indicated that Realboost has an edge over discrete Adaboost when additional training data is used. We observe the same behaviour in our Caltech 10x setup.

As summarized in table 2 using filtered channels, deeper trees, additional training data, and Realboost does provide a significant detection quality boost. For the rest of the paper our models trained on Caltech10x all use level-4 trees and RealBoost, instead of level-2 and discrete Adaboost for the Caltech1x models.

Timing When using Caltech data ACF takes about one hour for training and one for testing. Checkerboards$4 /times 4$ takes about 4 and 2 hours respectively. When using Caltech10x the training times for these methods augment to 2 and 29 hours, respectively. The training time does not increase proportionally with the training data volume because the hard negative mining reads a variable amount of images to attain the desired quota of negative samples. This amount increases when a detector has less false positive mistakes.

### 5.1. Validation set experiments

Based on the results in table 2 we proceed to evaluate on Caltech10x the most promising configurations (filter type and number) from section 4. The results over the Caltech validation set are collected in table 3. We observe a clear overall gain from increasing the training data.

Interestingly with enough RandomFilters we can outperform the strong performance of LDCF-Ours. We also notice that the naïve Checkerboards outperforms the manual design of InformedFilters.

| Filters type | $/#$ <br> filters | Caltech <br> MR | Caltech10x <br> MR | $/Delta$ MR |
| :---: | :---: | :---: | :---: | :---: |
| ACF-Ours | 1 | 50.2 | 39.8 | 10.4 |
| LDCF-Ours | 4 | 37.3 | 34.1 | 3.2 |
| LDCF8 | 8 | 42.6 | 30.7 | 11.9 |
| PcaForeground | 8 | 41.6 | 28.6 | 13.0 |
| RandomFilters | 50 | 36.5 | 28.2 | 8.3 |
| InformedFilters | 50 | 30.3 | 26.6 | 3.7 |
| Checkerboards | 39 | 30.9 | 25.9 | 5.0 |
| Checkerboards | 61 | 32.9 | 24.4 | 8.5 |

Table 3: Effect of increasing the training set for different methods, quality measured on Caltech validation set (MR: log-average miss-rate).

## 6. Add-ons

Before presenting the final test set results of our "core" method (section 7), we review some "add-ons" based on the suggestions from [2]. For the sake of evaluating complementarity, comparison with existing methods, and reporting the best possible detection quality, we consider extending our detector with context and optical flow information.

Context Context is modelled via the 2 Ped re-scoring method of [28]. It is a post-processing step that merges our detection scores with the results of a two person DPM [11] trained on the INRIA dataset (with extended annotations). In [28] the authors reported an improvement of $/sim 5 /mathrm{pp}$ (percent points) on the Caltech set, across different methods. In [2] an improvement of 2.8 pp is reported over their strong detector (SquaresChnFtrs+DCT+SDt $25.2 /% /mathrm{MR}$ ). In our experiments however we obtain a gain inferior to 0.5 pp . We have also investigated fusing the 2 Ped detection results via a different, more principled, fusion method [43]. We observe consistent results: as the strength of the starting point increases, the gain from 2 Ped decreases. When reaching our Checkerboards results, all gains have evaporated. We believe that the 2 Ped approach is a promising one, but our experiments indicate that the used DPM template is simply too weak in comparison to our filtered channels.

Optical flow Optical flow is fed to our detector as an additional set of 2 channels (not filtered). We use the implementation from SDt [30] which uses differences of weakly stabilized video frames. On Caltech, the authors of [30] reported a $/sim 7 /mathrm{pp}$ gain over ACF ( $44.2 /% /mathrm{MR}$ ), while [2] reported a $/sim 5 /mathrm{pp}$ percent points improvement over their strong baseline (SquaresChnFtrs+DCT+2Ped $27.4 /% /mathrm{MR}$ ) . When using + SDt our results are directly comparable to Katamari [2] and SpatialPooling+ [29] which both use optical flow too.

Using our stronger Checkerboards results SDt provides a 1.4 pp gain. Here again we observe an erosion as the starting point improves (for confirmation, reproduced

![](https://cdn.mathpix.com/cropped/2024_08_15_54f9e5398199d0e0fa54g-07.jpg?height=630&width=763&top_left_y=238&top_left_x=1104)

Figure 4: Some of the top quality detection methods on the Caltech test set.

the $/mathrm{ACF}+/mathrm{SDt}$ results [30], $43.9 /% /rightarrow 33.9 /% /mathrm{MR}$ ). We name our Checkerboards+SDt detector All-in-one.

Our filtered channel features results are strong enough to erode existing context and flow features. Although these remain complementary cues, more sophisticated ways of extracting this information will be required to further progress in detection quality.

It should be noted that despite our best efforts we could not reproduce the results from neither 2Ped [28] nor SDt [30] on the KITTI dataset (in spite of its apparent similarity to Caltech). Effective methods for context and optical flow across datasets have yet to be shown. Our main contribution remains on the core detector (only HOG+LUV features over local sliding window pixels in a single frame).

## 7. Test set results

Having done our exploration of the parameters space on the validation set, we now evaluate the most promising methods on the Caltech and KITTI test sets.

Caltech test set Figures 5 and 4 present our key results on the Caltech test set. For proper comparison, only methods using the same training set should be compared (see [2, figure 3] for a similar table comparing 50+ previous methods). We include for comparison the baselines mentioned in section 2.2, Roerei [1] the best known method trained without any Caltech images, MT-DPM [44] the best known method based on DPM, and SDN [22] the best known method using convolutional neural networks. We also include the top performers Katamari [2] and SpatialPooling+ [29]. We mark as "Caltech $N /times$ " both the Caltech10x training set and the one used in LDCF [25] (see section 5).

KITTI test set Figure 6 presents the results on the KITTI test set ("moderate" setup), together with all other reported

![](https://cdn.mathpix.com/cropped/2024_08_15_54f9e5398199d0e0fa54g-08.jpg?height=779&width=836&top_left_y=234&top_left_x=165)

Figure 5: Some of the top quality detection methods for Caltech test set (see text), and our results (highlighted with white hatch). Methods using optical flow are trained on original Caltech except our All-in-one which uses Caltech10x. Caltech $N /times$ indicates Caltech10x for all methods but the original LDCF (see section 2.1).

methods using only monocular image content (no stereo or LIDAR data). The KITTI evaluation server only recently has started receiving submissions ( 14 for this task, 11 in the last year), and thus is less prone to dataset over-fitting.

We train our model on the KITTI training set using almost identical parameters as for Caltech. The only change is a subtle pre-processing step in the HOG+LUV computation. On KITTI the input image is smoothed (radius 1 pixel) before the feature channels are computed, while on Caltech we do not. This subtle change provided a $/sim 4 /mathrm{pp}$ (percent points) improvement on the KITTI validation set.

### 7.1. Analysis

With a $/sim 10 /mathrm{pp}$ (percent points) gap between $/mathrm{ACF} / /mathrm{In}-$ formedHaar and ACF/InformedHaar-Ours (see figure 5), the results of our baselines show the importance of proper validation of training parameters (large enough model size and negative samples). InformedHaar-Ours is the best reported result trained with Caltech1x.

When considering methods trained on Caltech10x, we obtain a clear gap with the previous best results (LDCF $24.8 /%$ MR $/rightarrow$ Checkerboards 18.5/% MR). Using our architecture and an adequate number of filters one can obtain strong results using only HOG+LUV features. The amongst the options we considered the filter type seems not critical, in our experiments Checkerboards $4 /times 3$ reaches the best performance given the available training data. RandomFilters reaches the same result, but requires

![](https://cdn.mathpix.com/cropped/2024_08_15_54f9e5398199d0e0fa54g-08.jpg?height=500&width=743&top_left_y=235&top_left_x=1103)

Figure 6: Pedestrian detection on the KITTI dataset (using images only).

training and merging multiple models.

Our results cut by half miss-rate of the best known convnet for pedestrian detection (SDN [22]), which in principle could learn similar low-level features and their filtering.

When adding optical flow we further push the state of the art and reach $17.1 /% /mathrm{MR}$, a comfortable $/sim 5 /mathrm{pp}$ improvement over the previous best optical flow method (SpatialPooling+). This is the best reported result on this challenging dataset.

The results on the KITTI dataset confirm the strength of our approach, reaching $54.0 /% /mathrm{AP}$, just 1 pp below the best known result on this dataset. Competing methods (Regionlets [42] and SpatialPooling [29]) both use HOG and additional LBP and covariance features, as well as an intermediate max-pooling step. Adding these remains a possibility for our system. Our results also improve over methods using LIDAR+Image, such as Fusion-DPM [31] ( $46.7 /% /mathrm{AP}$, not included in figure 6 for clarity).

## 8. Conclusion

Through this paper we have shown that the seemingly disconnected methods ACF, (Squares) ChnFtrs, InformedHaar, and LDCF can be all put under the filtered channel features detectors umbrella. We have systematically explored different filter banks for such architecture and shown that they provide means for important improvements for pedestrian detection. Our results indicate that HOG+LUV features have not yet saturated, and that competitive results (over Caltech and KITTI datasets) can be obtained using only them. When optical flow information is added we set the new state of art for the Caltech dataset, reaching $17.1 /% /mathrm{MR}(93 /%$ recall at 1 false positive per image).

In future work we plan to explore how the insights of this work can be exploited into a more general detection architecture such as convolutional neural networks.

## References

[1] R. Benenson, M. Mathias, T. Tuytelaars, and L. Van Gool. Seeking the strongest rigid detector. In $C V P R$, 2013. 1, 2, 6, 7

[2] R. Benenson, M. Omran, J. Hosang, , and B. Schiele. Ten years of pedestrian detection, what have we learned? In ECCV, CVRSUAD workshop, 2014. 1, 2, 3, 7

[3] T.-H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma. Pcanet: A simple deep learning baseline for image classification? In arXiv, 2014. 5

[4] A. D. Costea and S. Nedevschi. Word channel based multiscale pedestrian detection without image resizing and using only one classifier. In $C V P R$, June 2014. 2

[5] N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In $C V P R, 2005.2$

[6] L. Dang, B. Bui, P. D. Vo, T. N. Tran, and B. H. Le. Improved hog descriptors. In KSE, 2011. 2

[7] P. Dollár, R. Appel, S. Belongie, and P. Perona. Fast feature pyramids for object detection. PAMI, 2014. 2, 3,5

[8] P. Dollár, Z. Tu, P. Perona, and S. Belongie. Integral channel features. In $B M V C, 2009.1,2,6$

[9] P. Dollár, C. Wojek, B. Schiele, and P. Perona. Pedestrian detection: An evaluation of the state of the art. TPAMI, 2011. 1, 2, 3

[10] M. Enzweiler and D. M. Gavrila. Monocular pedestrian detection: Survey and experiments. PAMI, 2009.

[11] P. Felzenszwalb, R. Girshick, D. McAllester, and D. Ramanan. Object detection with discriminatively trained part-based models. PAMI, 2010. 2, 7

[12] J. Friedman, T. Hastie, R. Tibshirani, et al. Additive logistic regression: a statistical view of boosting. The annals of statistics, 2000. 4

[13] A. Geiger, P. Lenz, and R. Urtasun. Are we ready for autonomous driving? the kitti vision benchmark suite. In $C V P R$, 2012. 1, 2, 3

[14] D. Geronimo, A. M. Lopez, A. D. Sappa, and T. Graf. Survey of pedestrian detection for advanced driver assistance systems. PAMI, 2010. 1

[15] Y. Goto, Y. Yamauchi, and H. Fujiyoshi. Cs-hog: Color similarity-based hog. In Korea-Japan Joint Workshop on Frontiers of Computer Vision, 2013. 2

[16] J. Hosang, M. Omran, R. Benenson, and B. Schiele. Taking a deeper look at pedestrians. In $C V P R, 2015$. 2

[17] C. Hou, H. Ai, and S. Lao. Multiview pedestrian detection based on vector boosting. In ACCV. 2007. 2
[18] F. Khan, R. Anwer, J. van de Weijer, A. Bagdanov, M. Vanrell, and A. Lopez. Color attributes for object detection. In CVPR, 2012. 2

[19] R. Khan, J. Van de Weijer, F. S. Khan, D. Muselet, C. Ducottet, and C. Barat. Discriminative color descriptors. In CVPR, 2013. 2

[20] I. Laptev. Improving object detection with boosted histograms. Image and Vision Computing, 2009. 2

[21] J. Lim, C. L. Zitnick, and P. Dollár. Sketch tokens: A learned mid-level representation for contour and object detection. In $C V P R, 2013.2$

[22] P. Luo, Y. Tian, X. Wang, and X. Tang. Switchable deep network for pedestrian detection. In $C V P R$, 2014. 7, 8

[23] M. Mathias, R. Benenson, M. Pedersoli, and L. Van Gool. Face detection without bells and whistles. In ECCV, 2014. 2

[24] K. Murphy. Machine learning: a probabilistic perspective. MIT press, 2012. 5

[25] W. Nam, P. Dollár, and J. H. Han. Local decorrelation for improved detection. In NIPS, 2014. 1, 2, 3, 5, 6, 7

[26] W. Nam, B. Han, and J. Han. Improving object localization using macrofeature layout selection. In ICCV, Visual Surveillance Workshop, 2011. 2

[27] P. Ott and M. Everingham. Implicit color segmentation features for pedestrian and object detection. In CVPR, 2009. 2

[28] W. Ouyang and X. Wang. Single-pedestrian detection aided by multi-pedestrian detection. In $C V P R, 2013$. 7

[29] S. Paisitkriangkrai, C. Shen, and A. van den Hengel. Strengthening the effectiveness of pedestrian detection with spatially pooled features. In $E C C V, 2014.1,2$, 3, 7, 8

[30] D. Park, C. L. Zitnick, D. Ramanan, and P. Dollár. Exploring weak stabilization for motion feature extraction. In $C V P R$, 2013. 2, 7

[31] C. Premebida, J. Carreira, J. Batista, and U. Nunes. Pedestrian detection combining RGB and dense LIDAR data. In IROS, 2014. 8

[32] D. Ramanan. Using segmentation to verify object hypotheses. In CVPR, 2007. 2

[33] X. Ren and D. Ramanan. Histograms of sparse codes for object detection. In CVPR, 2013. 2

[34] A. Satpathy, X. Jiang, and H.-L. Eng. Human detection by quadratic classification on subspace of extended histogram of gradients. IEEE Transactions on Image Processing, 2014. 2

[35] P. Sermanet, K. Kavukcuoglu, S. Chintala, and Y. LeCun. Pedestrian detection with unsupervised multi-stage feature learning. In CVPR, 2013. 2

[36] Y. Socarras, D. Vazquez, A. Lopez, D. Geronimo, and T. Gevers. Improving hog with image segmentation: Application to human detection. In Advanced Concepts for Intelligent Vision Systems. 2012. 2

[37] Y. Tian, P. Luo, X. Wang, and X. Tang. Pedestrian detection aided by deep learning semantic tasks. In CVPR, 2015. 2

[38] O. Tuzel, F. Porikli, and P. Meer. Pedestrian detection via classification on riemannian manifolds. PAMI, 2008. 2

[39] P. Viola, M. Jones, and D. Snow. Detecting pedestrians using patterns of motion and appearance. IJCV, 2005. 2

[40] S. Walk, N. Majer, K. Schindler, and B. Schiele. New features and insights for pedestrian detection. In CVPR, 2010. 2

[41] X. Wang, X. Han, and S. Yan. An hog-lbp human detector with partial occlusion handling. In ICCV, 2009. 2

[42] X. Wang, M. Yang, S. Zhu, and Y. Lin. Regionlets for generic object detection. In ICCV. IEEE, 2013. 1, 2, 3, 8

[43] P. Xu, F. Davoine, and T. Denoeux. Evidential combination of pedestrian detectors. In $B M V C, 2014.7$

[44] J. Yan, X. Zhang, Z. Lei, S. Liao, and S. Z. Li. Robust multi-resolution pedestrian detection in traffic scenes. In $C V P R$, 2013. 7

[45] S. Zhang, C. Bauckhage, and A. B. Cremers. Informed haar-like features improve pedestrian detection. In CVPR, 2014. 1, 2, 3, 4


[^0]:    ${ }^{1}$ Papers from 2004 to 2014 with "pedestrian detection" in the title, according to Google Scholar.

[^1]:    ${ }^{2}$ We use "raw" HOG, without any clamping, cell normalization, block normalization, or dimensionality reduction.

[^2]:    3"Feature channel" refers to the output of the first transformation in figure 1 bottom. "Filters" are the convolutional operators applied to the feature channels. And "features" are entries in the response maps of al filters applied over all channels. A subset of these features are the input to the learned decision forest.

the riddle house the villagers of little hangleton still called it “the riddle house,” even though it had been many years since the riddle family had lived there. it stood on a hill overlooking the village some of its windows boarded tiles missing from its roof and ivy spreading unchecked over its face. once a fine-looking manor and easily the largest and grandest building for miles around the riddle house was now damp derelict and unoccupied. the little hangletons all agreed that the old house was “creepy.” half a century ago something strange and horrible had happened there something that the older inhabitants of the village still liked to discuss when topics for gossip were scarce. the story had been picked over so many times and had been embroidered in so many places that nobody was quite sure what the truth was anymore. every version of the tale however started in the same place fifty years before at daybreak on a fine summer’s morning when the riddle house had still been well kept and impressive a maid had entered the drawing room to find all three riddles dead. the maid had run screaming down the hill into the village and roused as many people as she could. “lying there with their eyes wide open! cold as ice! still in their dinner things!” the police were summoned and the whole of little hangleton had seethed with shocked curiosity and ill disguised excitement. nobody wasted their breath pretending to feel very sad about the riddles for they had been most unpopular. elderly mr. and mrs. riddle had been rich snobbish and rude and their grown-up son tom had been if anything worse. all the villagers cared about was the identity of their murderer — for plainly three apparently healthy people did not all drop dead of natural causes on the same night. the hanged man the village pub did a roaring trade that night the whole village seemed to have turned out to discuss the murders. they were rewarded for leaving their firesides when the riddles’ cook arrived dramatically in their midst and announced to the suddenly silent pub that a man called frank bryce had just been arrested. “frank!” cried several people. “never!” frank bryce was the riddles’ gardener. he lived alone in a rundown cottage on the grounds of the riddle house. frank had come back from the war with a very stiff leg and a great dislike of crowds and loud noises and had been working for the riddles ever since. there was a rush to buy the cook drinks and hear more details. “always thought he was odd,” she told the eagerly listening villagers after her fourth sherry. “unfriendly like. i’m sure if i’ve offered him a cuppa once i’ve offered it a hundred times. never wanted to mix he didn’t.” “ah now,” said a woman at the bar “he had a hard war frank. he likes the quiet life. that’s no reason to “who else had a key to the back door then?” barked the cook. “there’s been a spare key hanging in the gardener’s cottage far back as i can remember! nobody forced the door last night! no broken windows! all frank had to do was creep up to the big house while we was all sleeping. ...” the villagers exchanged dark looks. “i always thought he had a nasty look about him right enough,” grunted a man at the bar. “war turned him funny if you ask me,” said the landlord. “told you i wouldn’t like to get on the wrong side of frank didn’t i dot?” said an excited woman in the corner. “horrible temper,” said dot nodding fervently. “i remember when he was a kid ...” by the following morning hardly anyone in little hangleton doubted that frank bryce had killed the riddles. but over in the neighboring town of great hangleton in the dark and dingy police station frank was stubbornly repeating again and again that he was innocent and that the only person he had seen near the house on the day of the riddles’ deaths had been a teenage boy a stranger dark-haired and pale. nobody else in the village had seen any such boy and the police were quite sure that frank had invented him. then just when things were looking very serious for frank the report on the riddles’ bodies came back and changed everything. the police had never read an odder report. a team of doctors had examined the bodies and had concluded that none of the riddles had been poisoned stabbed shot strangled suffocated or as far as they could tell harmed at all. in fact the report continued in a tone of unmistakable bewilderment the riddles all appeared to be in perfect health — apart from the fact that they were all dead. the doctors did note as though determined to find something wrong with the bodies that each of the riddles had a look of terror upon his or her face — but as the frustrated police said whoever heard of three people being frightened to death as there was no proof that the riddles had been murdered at all the police were forced to let frank go. the riddles were buried in the little hangleton churchyard and their graves remained objects of curiosity for a while. to everyone’s surprise and amid a cloud of suspicion frank bryce returned to his cottage on the grounds of the riddle house. “ ’s far as i’m concerned he killed them and i don’t care what the police say,” said dot in the hanged man. “and if he had any decency he’d leave here knowing as how we knows he did it.” but frank did not leave. he stayed to tend the garden for the next family who lived in the riddle house and then the next — for neither family stayed long. perhaps it was partly because of frank that the new owners said there was a nasty feeling about the place which in the absence of inhabitants started to fall into disrepair. the wealthy man who owned the riddle house these days neither lived there nor put it to any use they said in the village that he kept it for “tax reasons,” though nobody was very clear what these might be. the wealthy owner continued to pay frank to do the gardening however. frank was nearing his seventy seventh birthday now very deaf his bad leg stiffer than ever but could be seen pottering around the flower beds in fine weather even though the weeds were starting to creep up on him try as he might to suppress them. weeds were not the only things frank had to contend with either. boys from the village made a habit of throwing stones through the windows of the riddle house. they rode their bicycles over the lawns frank worked so hard to keep smooth. once or twice they broke into the old house for a dare. they knew that old frank’s devotion to the house and grounds amounted almost to an obsession and it amused them to see him limping across the garden brandishing his stick and yelling croakily at them. frank for his part believed the boys tormented him because they like their parents and grandparents thought him a murderer. so when frank awoke one night in august and saw something very odd up at the old house he merely assumed that the boys had gone one step further in their attempts to punish him. it was frank’s bad leg that woke him it was paining him worse than ever in his old age. he got up and limped downstairs into the kitchen with the idea of refilling his hot-water bottle to ease the stiffness in his knee. standing at the sink filling the kettle he looked up at the riddle house and saw lights glimmering in its upper windows. frank knew at once what was going on. the boys had broken into the house again and judging by the flickering quality of the light they had started a fire. frank had no telephone and in any case he had deeply mistrusted the police ever since they had taken him in for questioning about the riddles’ deaths. he put down the kettle at once hurried back upstairs as fast as his bad leg would allow and was soon back in his kitchen fully dressed and removing a rusty old key from its hook by the door. he picked up his walking stick which was propped against the wall and set off into the night. the front door of the riddle house bore no sign of being forced nor did any of the windows. frank limped around to the back of the house until he reached a door almost completely hidden by ivy took out the old key put it into the lock and opened the door noiselessly. he let himself into the cavernous kitchen. frank had not entered it for many years nevertheless although it was very dark he remembered where the door into the hall was and he groped his way toward it his nostrils full of the smell of decay ears pricked for any sound of footsteps or voices from overhead. he reached the hall which was a little lighter owing to the large mullioned windows on either side of the front door and started to climb the stairs blessing the dust that lay thick upon the stone because it muffled the sound of his feet and stick. on the landing frank turned right and saw at once where the intruders were at the very end of the passage a door stood ajar and a flickering light shone through the gap casting a long sliver of gold across the black floor. frank edged closer and closer grasping his walking stick firmly. several feet from the entrance he was able to see a narrow slice of the room beyond. the fire he now saw had been lit in the grate. this surprised him. then he stopped moving and listened intently for a man’s voice spoke within the room it sounded timid and fearful. “there is a little more in the bottle my lord if you are still hungry.” “later,” said a second voice. this too belonged to a man — but it was strangely high-pitched and cold as a sudden blast of icy wind. something about that voice made the sparse hairs on the back of frank’s neck stand up. “move me closer to the fire wormtail.” frank turned his right ear toward the door the better to hear. there came the clink of a bottle being put down upon some hard surface and then the dull scraping noise of a heavy chair being dragged across the floor. frank caught a glimpse of a small man his back to the door pushing the chair into place. he was wearing a long black cloak and there was a bald patch at the back of his head. then he went out of sight again. “where is nagini?” said the cold voice. “i — i don’t know my lord,” said the first voice nervously. “she set out to explore the house i think. “you will milk her before we retire wormtail,” said the second voice. “i will need feeding in the night. the journey has tired me greatly.” brow furrowed frank inclined his good ear still closer to the door listening very hard. there was a pause and then the man called wormtail spoke again. “my lord may i ask how long we are going to stay here?” “a week,” said the cold voice. “perhaps longer. the place is moderately comfortable and the plan cannot proceed yet. it would be foolish to act before the quidditch world cup is over.” frank inserted a gnarled finger into his ear and rotated it. owing no doubt to a buildup of earwax he had heard the word “quidditch,” which was not a word at all. “the — the quidditch world cup my lord?” said wormtail. frank dug his finger still more vigorously into his ear. “forgive me but — i do not understand — why should we wait until the world cup is over?” “because fool at this very moment wizards are pouring into the country from all over the world and every meddler from the ministry of magic will be on duty on the watch for signs of unusual activity checking and double-checking identities. they will be obsessed with security lest the muggles notice anything. so we wait.” frank stopped trying to clear out his ear. he had distinctly heard the words “ministry of magic,” “wizards,” and “muggles.” plainly each of these expressions meant something secret and frank could think of only two sorts of people who would speak in code spies and criminals. frank tightened his hold on his walking stick once more and listened more closely still. “your lordship is still determined then?” wormtail said quietly. “certainly i am determined wormtail.” there was a note of menace in the cold voice now. a slight pause followed — and then wormtail spoke the words tumbling from him in a rush as though he was forcing himself to say this before he lost his nerve. “it could be done without harry potter my lord.” another pause more protracted and then — “without harry potter?” breathed the second voice softly. “i see ...” “my lord i do not say this out of concern for the boy!” said wormtail his voice rising squeakily. “the boy is nothing to me nothing at all! it is merely that if we were to use another witch or wizard — any wizard — the thing could be done so much more quickly! if you allowed me to leave you for a short while — you know that i can disguise myself most effectively — i could be back here in as little as two days with a suitable person — ” “i could use another wizard,” said the cold voice softly “that is true. ...” “my lord it makes sense,” said wormtail sounding thoroughly relieved now. “laying hands on harry potter would be so difficult he is so well protected — ” “and so you volunteer to go and fetch me a substitute i wonder . . . perhaps the task of nursing me has become wearisome for you “wormtail could this suggestion of abandoning the plan be nothing more than an attempt to desert me?” “my lord! i — i have no wish to leave you none at all “do not lie to me!” hissed the second voice. “i can always tell wormtail! you are regretting that you ever returned to me. i revolt you. i see you flinch when you look at me feel you shudder when you touch me. ...” “no! my devotion to your lordship — ” “your devotion is nothing more than cowardice. you would not be here if you had anywhere else to go. how am i to survive without you when i need feeding every few hours who is to milk nagini?” “but you seem so much stronger my lord — ” “liar,” breathed the second voice. “i am no stronger and a few days alone would be enough to rob me of the little health i have regained under your clumsy care. silencel” wormtail who had been sputtering incoherently fell silent at once. for a few seconds frank could hear nothing but the fire crackling. then the second man spoke once more in a whisper that was almost a hiss. “i have my reasons for using the boy as i have already explained to you and i will use no other. i have waited thirteen years. a few more months will make no difference. as for the protection surrounding the boy i believe my plan will be effective. all that is needed is a little courage from you wormtail — courage you will find unless you wish to feel the full extent of lord voldemort’s wrath — ” “my lord i must speak!” said wormtail panic in his voice now. “all through our journey i have gone over the plan in my head — my lord bertha jorkins’s disappearance will not go unnoticed for long and if we proceed if i murder — ” “ip” whispered the second voice. “if if you follow the plan wormtail the ministry need never know that anyone else has died. you will do it quietly and without fuss i only wish that i could do it myself but in my present condition ... come wormtail one more death and our path to harry potter is clear. i am not asking you to do it alone. by that time my faithful servant will have rejoined us — ” “i am a faithful servant,” said wormtail the merest trace of sullenness in his voice. “wormtail i need somebody with brains somebody whose loyalty has never wavered and you unfortunately fulfill neither requirement.” “i found you,” said wormtail and there was definitely a sulky edge to his voice now. “i was the one who found you. i brought you bertha jorkins.” “that is true,” said the second man sounding amused. “a stroke of brilliance i would not have thought possible from you wormtail — though if truth be told you were not aware how useful she would be when you caught her were you?” “i — i thought she might be useful my lord — ” “liar,” said the second voice again the cruel amusement more pronounced than ever. “however i do not deny that her information was invaluable. without it i could never have formed our plan and for that you will have your reward wormtail. i will allow you to perform an essential task for me one that many of my followers would give their right hands to perform. ...” “r-really my lord what — ” wormtail sounded terrified again. “ah wormtail you don’t want me to spoil the surprise your part will come at the very end ... but i promise you you will have the honor of being just as useful as bertha jorkins.” “you ... you ...” wormtail’s voice suddenly sounded hoarse as though his mouth had gone very dry. “you . . . are going ... to kill me too?” “wormtail wormtail,” said the cold voice silkily “why would i kill you i killed bertha because i had to. she was fit for nothing after my questioning quite useless. in any case awkward questions would have been asked if she had gone back to the ministry with the news that she had met you on her holidays. wizards who are supposed to be dead would do well not to run into ministry of magic witches at wayside inns. ...” wormtail muttered something so quietly that frank could not hear it but it made the second man laugh — an entirely mirthless laugh cold as his speech. “we could have modified her memory but memory charms can be broken by a powerful wizard as i proved when i questioned her. it would be an insult to her memory not to use the information i extracted from her wormtail.” out in the corridor frank suddenly became aware that the hand gripping his walking stick was slippery with sweat. the man with the cold voice had killed a woman. he was talking about it without any kind of remorse — with amusement. he was dangerous — a madman. and he was planning more murders — this boy harry potter whoever he was — was in danger — frank knew what he must do. now if ever was the time to go to the police. he would creep out of the house and head straight for the telephone box in the village . . . but the cold voice was speaking again and frank remained where he was frozen to the spot listening with all his might. “one more murder . . . my faithful servant at hogwarts ... harry potter is as good as mine wormtail. it is decided. there will be no more argument. but quiet ... i think i hear nagini. ...” and the second man’s voice changed. he started making noises such as frank had never heard before he was hissing and spitting without drawing breath. frank thought he must be having some sort of fit or seizure. and then frank heard movement behind him in the dark passageway. he turned to look and found himself paralyzed with fright. something was slithering toward him along the dark corridor floor and as it drew nearer to the sliver of firelight he realized with a thrill of terror that it was a gigantic snake at least twelve feet long. horrified transfixed frank stared as its undulating body cut a wide curving track through the thick dust on the floor coming closer and closer — what was he to do the only means of escape was into the room where two men sat plotting murder yet if he stayed where he was the snake would surely kill him — but before he had made his decision the snake was level with him and then incredibly miraculously it was passing it was following the spitting hissing noises made by the cold voice beyond the door and in seconds the tip of its diamond-patterned tail had vanished through the gap. there was sweat on frank’s forehead now and the hand on the walking stick was trembling. inside the room the cold voice was continuing to hiss and frank was visited by a strange idea an impossible idea. . . . this man could talk to snakes. frank didn’t understand what was going on. he wanted more than anything to be back in his bed with his hot-water bottle. the problem was that his legs didn’t seem to want to move. as he stood there shaking and trying to master himself the cold voice switched abruptly to english again. “nagini has interesting news wormtail,” it said. “in-indeed my lord?” said wormtail. “indeed yes,” said the voice. “according to nagini there is an old muggle standing right outside this room listening to every word we say.” frank didn’t have a chance to hide himself. there were footsteps and then the door of the room was flung wide open. a short balding man with graying hair a pointed nose and small watery eyes stood before frank a mixture of fear and alarm in his face. “invite him inside wormtail. where are your manners?” the cold voice was coming from the ancient armchair before the fire but frank couldn’t see the speaker. the snake on the other hand was curled up on the rotting hearth rug like some horrible travesty of a pet dog. wormtail beckoned frank into the room. though still deeply shaken frank took a firmer grip upon his walking stick and limped over the threshold. the fire was the only source of light in the room it cast long spidery shadows upon the walls. frank stared at the back of the armchair the man inside it seemed to be even smaller than his servant for frank couldn’t even see the back of his head. “you heard everything muggle?” said the cold voice. “what’s that you’re calling me?” said frank defiantly for now that he was inside the room now that the time had come for some sort of action he felt braver it had always been so in the war. “i am calling you a muggle,” said the voice coolly. “it means that you are not a wizard.” “i don’t know what you mean by wizard,” said frank his voice growing steadier. “all i know is i’ve heard enough to interest the police tonight i have. you’ve done murder and you’re planning more! and i’ll tell you this too,” he added on a sudden inspiration “my wife knows i’m up here and if i don’t come back — ” “you have no wife,” said the cold voice very quietly. “nobody knows you are here. you told nobody that you were coming. do not lie to lord voldemort muggle for he knows ... he always knows. ...” “is that right?” said frank roughly. “lord is it well i don’t think much of your manners my lord. turn ’round and face me like a man why don’t you?” “but i am not a man muggle,” said the cold voice barely audible now over the crackling of the flames. “i am much much more than a man. however ... why not i will face you. ... wormtail come turn my chair around.” the servant gave a whimper. “you heard me wormtail.” slowly with his face screwed up as though he would rather have done anything than approach his master and the hearth rug where the snake lay the small man walked forward and began to turn the chair. the snake lifted its ugly triangular head and hissed slightly as the legs of the chair snagged on its rug. and then the chair was facing frank and he saw what was sitting in it. his walking stick fell to the floor with a clatter. he opened his mouth and let out a scream. he was screaming so loudly that he never heard the words the thing in the chair spoke as it raised a wand. there was a flash of green light a rushing sound and frank bryce crumpled. he was dead before he hit the floor. two hundred miles away the boy called harry potter woke with a start. the scar harry lay flat on his back breathing hard as though he had been running. he had awoken from a vivid dream with his hands pressed over his face. the old scar on his forehead which was shaped like a bolt of lightning was burning beneath his fingers as though someone had just pressed a white-hot wire to his skin. he sat up one hand still on his scar the other reaching out in the darkness for his glasses which were on the bedside table. he put them on and his bedroom came into clearer focus lit by a faint misty orange light that was filtering through the curtains from the street lamp outside the window. harry ran his fingers over the scar again. it was still painful. he turned on the lamp beside him scrambled out of bed crossed the room opened his wardrobe and peered into the mirror on the inside of the door. a skinny boy of fourteen looked back at him his bright green eyes puzzled under his untidy black hair. he examined the lightning-bolt scar of his reflection more closely. it looked normal but it was still stinging. harry tried to recall what he had been dreaming about before he had awoken. it had seemed so real. ... there had been two people he knew and one he didn’t. ... he concentrated hard frowning trying to remember. ... the dim picture of a darkened room came to him. ... there had been a snake on a hearth rug ... a small man called peter nicknamed wormtail ... and a cold high voice ... the voice of lord voldemort. harry felt as though an ice cube had slipped down into his stomach at the very thought. ... he closed his eyes tightly and tried to remember what voldemort had looked like but it was impossible. ... all harry knew was that at the moment when voldemort’s chair had swung around and he harry had seen what was sitting in it he had felt a spasm of horror which had awoken him ... or had that been the pain in his scar and who had the old man been for there had definitely been an old man harry had watched him fall to the ground. it was all becoming confused. harry put his face into his hands blocking out his bedroom trying to hold on to the picture of that dimly lit room but it was like trying to keep water in his cupped hands the details were now trickling away as fast as he tried to hold on to them. ... voldemort and wormtail had been talking about someone they had killed though harry could not remember the name . . . and they had been plotting to kill someone else . . . him). harry took his face out of his hands opened his eyes and stared around his bedroom as though expecting to see something unusual there. as it happened there were an extraordinary number of unusual things in this room. a large wooden trunk stood open at the foot of his bed revealing a cauldron broomstick black robes and assorted spellbooks. rolls of parchment littered that part of his desk that was not taken up by the large empty cage in which his snowy owl hedwig usually perched. on the floor beside his bed a book lay open harry had been reading it before he fell asleep last night. the pictures in this book were all moving. men in bright orange robes were zooming in and out of sight on broomsticks throwing a red ball to one another. harry walked over to the book picked it up and watched one of the wizards score a spectacular goal by putting the ball through a fifty-foot-high hoop. then he snapped the book shut. even quidditch — in harry’s opinion the best sport in the world — couldn’t distract him at the moment. he placed flying with the cannons on his bedside table crossed to the window and drew back the curtains to survey the street below. privet drive looked exactly as a respectable suburban street would be expected to look in the early hours of saturday morning. all the curtains were closed. as far as harry could see through the darkness there wasn’t a living creature in sight not even a cat. and yet . . . and yet . . . harry went restlessly back to the bed and sat down on it running a finger over his scar again. it wasn’t the pain that bothered him harry was no stranger to pain and injury. he had lost all the bones from his right arm once and had them painfully regrown in a night. the same arm had been pierced by a venomous foot-long fang not long afterward. only last year harry had fallen fifty feet from an airborne broomstick. he was used to bizarre accidents and injuries they were unavoidable if you attended hogwarts school of witchcraft and wizardry and had a knack for attracting a lot of trouble. no the thing that was bothering harry was that the last time his scar had hurt him it had been because voldemort had been close by. ... but voldemort couldn’t be here now. ... the idea of voldemort lurking in privet drive was absurd impossible. ... harry listened closely to the silence around him. was he half-expecting to hear the creak of a stair or the swish of a cloak and then he jumped slightly as he heard his cousin dudley give a tremendous grunting snore from the next room. harry shook himself mentally he was being stupid. there was no one in the house with him except uncle vernon aunt petunia and dudley and they were plainly still asleep their dreams untroubled and painless. asleep was the way harry liked the dursleys best it wasn’t as though they were ever any help to him awake. uncle vernon aunt petunia and dudley were harry’s only living relatives. they were muggles who hated and despised magic in any form which meant that harry was about as welcome in their house as dry rot. they had explained away harry’s long absences at hogwarts over the last three years by telling everyone that he went to st. brutus’s secure center for incurably criminal boys. they knew perfectly well that as an underage wizard harry wasn’t allowed to use magic outside hogwarts but they were still apt to blame him for anything that went wrong about the house. harry had never been able to confide in them or tell them anything about his life in the wizarding world. the very idea of going to them when they awoke and telling them about his scar hurting him and about his worries about voldemort was laughable. and yet it was because of voldemort that harry had come to live with the dursleys in the first place. if it hadn’t been for voldemort harry would not have had the lightning scar on his forehead. if it hadn’t been for voldemort harry would still have had parents. ... harry had been a year old the night that voldemort — the most powerful dark wizard for a century a wizard who had been gaining power steadily for eleven years — arrived at his house and killed his father and mother. voldemort had then turned his wand on harry he had performed the curse that had disposed of many full-grown witches and wizards in his steady rise to power — and incredibly it had not worked. instead of killing the small boy the curse had rebounded upon voldemort. harry had survived with nothing but a lightning-shaped cut on his forehead and voldemort had been reduced to something barely alive. his powers gone his life almost extinguished voldemort had fled the terror in which the secret community of witches and wizards had lived for so long had lifted voldemort’s followers had disbanded and harry potter had become famous. it had been enough of a shock for harry to discover on his eleventh birthday that he was a wizard it had been even more disconcerting to find out that everyone in the hidden wizarding world knew his name. harry had arrived at hogwarts to find that heads turned and whispers followed him wherever he went. but he was used to it now at the end of this summer he would be starting his fourth year at hogwarts and harry was already counting the days until he would be back at the castle again. but there was still a fortnight to go before he went back to school. he looked hopelessly around his room again and his eye paused on the birthday cards his two best friends had sent him at the end of july. what would they say if harry wrote to them and told them about his scar hurting at once hermione granger’s voice seemed to fill his head shrill and panicky. “ your scar hurt harry that’s really serious. . . . write to professor dumbledorel and i’ll go and check common magical ailments and afflictions. ... maybe there’s something in there about curse scars. ...” yes that would be hermione’s advice go straight to the headmaster of hogwarts and in the meantime consult a book. harry stared out of the window at the inky blue-black sky. he doubted very much whether a book could help him now. as far as he knew he was the only living person to have survived a curse like voldemort’s it was highly unlikely therefore that he would find his symptoms listed in common magical ailments and afflictions. as for informing the headmaster harry had no idea where dumbledore went during the summer holidays. he amused himself for a moment picturing dumbledore with his long silver beard full-length wizard’s robes and pointed hat stretched out on a beach somewhere rubbing suntan lotion onto his long crooked nose. wherever dumbledore was though harry was sure that hedwig would be able to find him harry’s owl had never yet failed to deliver a letter to anyone even without an address. but what would he write dear professor dumbledore sorry to bother you but my scar hurt this morning. yours sincerely harry potter. even inside his head the words sounded stupid. and so he tried to imagine his other best friend ron weasley’s reaction and in a moment ron’s red hair and long-nosed freckled face seemed to swim before harry wearing a bemused expression. “ your scar hurt but ... but you-knotu-who can’t be near you now can he i mean ... you’d know wouldn’t you he’d be trying to do you in again wouldn’t he i dunno harry maybe curse scars always twinge a bit ... i’ll ask dad. ...” mr. weasley was a fully qualified wizard who worked in the misuse of muggle artifacts office at the ministry of magic but he didn’t have any particular expertise in the matter of curses as far as harry knew. in any case harry didn’t like the idea of the whole weasley family knowing that he harry was getting jumpy about a few moments’ pain. mrs. weasley would fuss worse than hermione and fred and george ron’s sixteen-year-old twin brothers might think harry was losing his nerve. the weasleys were harry’s favorite family in the world he was hoping that they might invite him to stay any time now ron had mentioned something about the quidditch world cup and he somehow didn’t want his visit punctuated with anxious inquiries about his scar. harry kneaded his forehead with his knuckles. what he really wanted and it felt almost shameful to admit it to himself was someone like — someone like a parent an adult wizard whose advice he could ask without feeling stupid someone who cared about him who had had experience with dark magic. ... and then the solution came to him. it was so simple and so obvious that he couldn’t believe it had taken so long — sirius. harry leapt up from the bed hurried across the room and sat down at his desk he pulled a piece of parchment toward him loaded his eagle-feather quill with ink wrote dear sirius then paused wondering how best to phrase his problem still marveling at the fact that he hadn’t thought of sirius straight away. but then perhaps it wasn’t so surprising — after all he had only found out that sirius was his godfather two months ago. there was a simple reason for sirius’s complete absence from harry’s life until then — sirius had been in azkaban the terrifying wizard jail guarded by creatures called dementors sightless soul-sucking fiends who had come to search for sirius at hogwarts when he had escaped. yet sirius had been innocent — the murders for which he had been convicted had been committed by wormtail voldemort’s supporter whom nearly everybody now believed dead. harry ron and hermione knew otherwise however they had come face-to-face with wormtail only the previous year though only professor dumbledore had believed their story. for one glorious hour harry had believed that he was leaving the dursleys at last because sirius had offered him a home once his name had been cleared. but the chance had been snatched away from him — wormtail had escaped before they could take him to the ministry of magic and sirius had had to flee for his life. harry had helped him escape on the back of a hippogriff called buckbeak and since then sirius had been on the run. the home harry might have had if wormtail had not escaped had been haunting him all summer. it had been doubly hard to return to the dursleys knowing that he had so nearly escaped them forever. nevertheless sirius had been of some help to harry even if he couldn’t be with him. it was due to sirius that harry now had all his school things in his bedroom with him. the dursleys had never allowed this before their general wish of keeping harry as miserable as possible coupled with their fear of his powers had led them to lock his school trunk in the cupboard under the stairs every summer prior to this. but their attitude had changed since they had found out that harry had a dangerous murderer for a godfather — for harry had conveniently forgotten to tell them that sirius was innocent. harry had received two letters from sirius since he had been back at privet drive. both had been delivered not by owls as was usual with wizards but by large brightly colored tropical birds. hedwig had not approved of these flashy intruders she had been most reluctant to allow them to drink from her water tray before flying off again. harry on the other hand had liked them they put him in mind of palm trees and white sand and he hoped that wherever sirius was sirius never said in case the letters were intercepted he was enjoying himself. somehow harry found it hard to imagine dementors surviving for long in bright sunlight perhaps that was why sirius had gone south. sirius’s letters which were now hidden beneath the highly useful loose floorboard under harry’s bed sounded cheerful and in both of them he had reminded harry to call on him if ever harry needed to. well he needed to now all right. ... harry’s lamp seemed to grow dimmer as the cold gray light that precedes sunrise slowly crept into the room. finally when the sun had risen when his bedroom walls had turned gold and when sounds of movement could be heard from uncle vernon and aunt petunia’s room harry cleared his desk of crumpled pieces of parchment and reread his finished letter. dear sirius thanks for your last letter. that bird was enormous it could hardly get through my window. things are the same as usual here. dudley’s diet isn’t going too well. my aunt found him smuggling doughnuts into his room yesterday. they told him they ’d have to cut his pocket money if he keeps doing it so he got really angry and chucked his playstation out of the window. that’s a sort of computer thing you can play games on. bit stupid really now he hasn’t even got mega-mutilation part three to take his mind off things. i’m okay mainly because the dursleys are terrified you might turn up and turn them all into bats if i ask you to. a weird thing happened this morning though. my scar hurt again. last time that happened it was because voldemort was at hogwarts. but i don’t reckon he can be anywhere near me now can he do you know if curse scars sometimes hurt years afterward i’ll send this with hedwig when she gets back she’s offhunt-\ing at the moment. say hello to buckbeak for me. harry yes thought harry that looked all right. there was no point putting in the dream he didn’t want it to look as though he was too worried. he folded up the parchment and laid it aside on his desk ready for when hedwig returned. then he got to his feet stretched and opened his wardrobe once more. without glancing at his reflection he started to get dressed before going down to breakfast. 3 the invitation by the time harry arrived in the kitchen the three dursleys were already seated around the table. none of them looked up as he entered or sat down. uncle vernon’s large red face was hidden behind the morning’s daily mail and aunt petunia was cutting a grapefruit into quarters her lips pursed over her horselike teeth. dudley looked furious and sulky and somehow seemed to be taking up even more space than usual. this was saying something as he always took up an entire side of the square table by himself. when aunt petunia put a quarter of unsweetened grapefruit onto dudley’s plate with a tremulous “there you are diddy darling,” dudley glowered at her. his life had taken a most unpleasant turn since he had come home for the summer with his end-of-year report. uncle vernon and aunt petunia had managed to find excuses for his bad marks as usual aunt petunia always insisted that dudley was a very gifted boy whose teachers didn’t understand him while uncle vernon maintained that “he didn’t want some swotty little nancy boy for a son anyway.” they also skated over the accusations of bullying in the report — “he’s a boisterous little boy but he wouldn’t hurt a fly!” aunt petunia had said tearfully. however at the bottom of the report there were a few well-chosen comments from the school nurse that not even uncle vernon and aunt petunia could explain away. no matter how much aunt petunia wailed that dudley was big-boned and that his poundage was really puppy fat and that he was a growing boy who needed plenty of food the fact remained that the school outfitters didn’t stock knickerbockers big enough for him anymore. the school nurse had seen what aunt petunia’s eyes — so sharp when it came to spotting fingerprints on her gleaming walls and in observing the comings and goings of the neighbors — simply refused to see that far from needing extra nourishment dudley had reached roughly the size and weight of a young killer whale. so — after many tantrums after arguments that shook harry’s bedroom floor and many tears from aunt petunia — the new regime had begun. the diet sheet that had been sent by the smeltings school nurse had been taped to the fridge which had been emptied of all dudley’s favorite things — fizzy drinks and cakes chocolate bars and burgers — and filled instead with fruit and vegetables and the sorts of things that uncle vernon called “rabbit food.” to make dudley feel better about it all aunt petunia had insisted that the whole family follow the diet too. she now passed a grapefruit quarter to harry. he noticed that it was a lot smaller than dudley’s. aunt petunia seemed to feel that the best way to keep up dudley’s morale was to make sure that he did at least get more to eat than harry. but aunt petunia didn’t know what was hidden under the loose floorboard upstairs. she had no idea that harry was not following the diet at all. the moment he had got wind of the fact that he was expected to survive the summer on carrot sticks harry had sent hedwig to his friends with pleas for help and they had risen to the occasion magnificently. hedwig had returned from hermione’s house with a large box stuffed full of sugar-free snacks. hermione’s parents were dentists. hagrid the hogwarts gamekeeper had obliged with a sack full of his own homemade rock cakes. harry hadn’t touched these he had had too much experience of hagrid ’s cooking. mrs. weasley however had sent the family owl errol with an enormous fruitcake and assorted meat pies. poor errol who was elderly and feeble had needed a full five days to recover from the journey. and then on harry’s birthday which the dursleys had completely ignored he had received four superb birthday cakes one each from ron hermione hagrid and sirius. harry still had two of them left and so looking forward to a real breakfast when he got back upstairs he ate his grapefruit without complaint. uncle vernon laid aside his paper with a deep sniff of disapproval and looked down at his own grapefruit quarter. “is this it?” he said grumpily to aunt petunia. aunt petunia gave him a severe look and then nodded pointedly at dudley who had already finished his own grapefruit quarter and was eyeing harry’s with a very sour look in his piggy little eyes. uncle vernon gave a great sigh which ruffled his large bushy mustache and picked up his spoon. the doorbell rang. uncle vernon heaved himself out of his chair and set off down the hall. quick as a flash while his mother was occupied with the kettle dudley stole the rest of uncle vernon’s grapefruit. harry heard talking at the door and someone laughing and uncle vernon answering curtly. then the front door closed and the sound of ripping paper came from the hall. aunt petunia set the teapot down on the table and looked curiously around to see where uncle vernon had got to. she didn’t have to wait long to find out after about a minute he was back. he looked livid. “you,” he barked at harry. “in the living room. now.” bewildered wondering what on earth he was supposed to have done this time harry got up and followed uncle vernon out of the kitchen and into the next room. uncle vernon closed the door sharply behind both of them. “so,” he said marching over to the fireplace and turning to face harry as though he were about to pronounce him under arrest. “so.” harry would have dearly loved to have said “so what?” but he didn’t feel that uncle vernon’s temper should be tested this early in the morning especially when it was already under severe strain from lack of food. he therefore settled for looking politely puzzled. “this just arrived,” said uncle vernon. he brandished a piece of purple writing paper at harry. “a letter. about you.” harry’s confusion increased. who would be writing to uncle vernon about him who did he know who sent letters by the postman uncle vernon glared at harry then looked down at the letter and began to read aloud dear mr. and mrs. dursley we have never been introduced but i am sure you have heard a great deal from harry about my son ron. as harry might have told you the final of the quidditch world cup takes place this monday night and my husband arthur has just managed to get prime tickets through his connections at the department of magical games and sports. i do hope you will allow us to take harry to the match as this really is a once-in-a-lifetime opportunity britain hasn’t hosted the cup for thirty years and tickets are extremely hard to come by. we would of course be glad to have harry stay for the remainder of the summer holidays and to see him safely onto the train back to school. it would be best for harry to send us your answer as quickly as possible in the normal way because the muggle postman has never delivered to our house and i am not sure he even knows where it is. hoping to see harry soon yours sincerely molly weasley p.s. i do hope we’ve put enough stamps on. uncle vernon finished reading put his hand back into his breast pocket and drew out something else. “look at this,” he growled. he held up the envelope in which mrs. weasley’s letter had come and harry had to fight down a laugh. every bit of it was covered in stamps except for a square inch on the front into which mrs. weasley had squeezed the dursleys’ address in minute writing. “she did put enough stamps on then,” said harry trying to sound as though mrs. weasley’s was a mistake anyone could make. his uncle’s eyes flashed. “the postman noticed,” he said through gritted teeth. “very interested to know where this letter came from he was. that’s why he rang the doorbell. seemed to think it was funny.” harry didn’t say anything. other people might not understand why uncle vernon was making a fuss about too many stamps but harry had lived with the dursleys too long not to know how touchy they were about anything even slightly out of the ordinary. their worst fear was that someone would find out that they were connected however distantly with people like mrs. weasley. uncle vernon was still glaring at harry who tried to keep his expression neutral. if he didn’t do or say anything stupid he might just be in for the treat of a lifetime. he waited for uncle vernon to say something but he merely continued to glare. harry decided to break the silence. “so — can i go then?” he asked. a slight spasm crossed uncle vernon’s large purple face. the mustache bristled. harry thought he knew what was going on behind the mustache a furious battle as two of uncle vernon’s most fundamental instincts came into conflict. allowing harry to go would make harry happy something uncle vernon had struggled against for thirteen years. on the other hand allowing harry to disappear to the weasleys’ for the rest of the summer would get rid of him two weeks earlier than anyone could have hoped and uncle vernon hated having harry in the house. to give himself thinking time it seemed he looked down at mrs. weasley’s letter again. “who is this woman?” he said staring at the signature with distaste. “you’ve seen her,” said harry. “she’s my friend ron’s mother she was meeting him off the hog — off the school train at the end of last term.” he had almost said “hogwarts express,” and that was a sure way to get his uncle’s temper up. nobody ever mentioned the name of harry’s school aloud in the dursley household. uncle vernon screwed up his enormous face as though trying to remember something very unpleasant. “dumpy sort of woman?” he growled finally. “load of children with red hair?” harry frowned. he thought it was a bit rich of uncle vernon to call anyone “dumpy,” when his own son dudley had finally achieved what he’d been threatening to do since the age of three and become wider than he was tall. uncle vernon was perusing the letter again. “quidditch,” he muttered under his breath. “quidditch — what is this rubbish?” harry felt a second stab of annoyance. “it’s a sport,” he said shortly. “played on broom — ” “all right all right!” said uncle vernon loudly. harry saw with some satisfaction that his uncle looked vaguely panicky. apparently his nerves couldn’t stand the sound of the word “broomsticks” in his living room. he took refuge in perusing the letter again. harry saw his lips form the words “send us your answer ... in the normal way.” he scowled. “what does she mean ‘the normal way’?” he spat. “normal for us,” said harry and before his uncle could stop him he added “you know owl post. that’s what’s normal for wizards.” uncle vernon looked as outraged as if harry had just uttered a disgusting swear word. shaking with anger he shot a nervous look through the window as though expecting to see some of the neighbors with their ears pressed against the glass. “how many times do i have to tell you not to mention that unnaturalness under my roof?” he hissed his face now a rich plum color. “you stand there in the clothes petunia and i have put on your ungrateful back — ” “only after dudley finished with them,” said harry coldly and indeed he was dressed in a sweatshirt so large for him that he had had to roll back the sleeves five times so as to be able to use his hands and which fell past the knees of his extremely baggy jeans. “i will not be spoken to like that!” said uncle vernon trembling with rage. but harry wasn’t going to stand for this. gone were the days when he had been forced to take every single one of the dursleys’ stupid rules. he wasn’t following dudley’s diet and he wasn’t going to let uncle vernon stop him from going to the quidditch world cup not if he could help it. harry took a deep steadying breath and then said “okay i can’t see the world cup. can i go now then only i’ve got a letter to sirius i want to finish. you know — my godfather.” he had done it. he had said the magic words. now he watched the purple recede blotchily from uncle vernon’s face making it look like badly mixed black currant ice cream. “you’re — you’re writing to him are you?” said uncle vernon in a would-be calm voice — but harry had seen the pupils of his tiny eyes contract with sudden fear. “well — yeah,” said harry casually. “it’s been a while since he heard from me and you know if he doesn’t he might start thinking something’s wrong.” he stopped there to enjoy the effect of these words. he could almost see the cogs working under uncle vernon’s thick dark neatly parted hair. if he tried to stop harry writing to sirius sirius would think harry was being mistreated. if he told harry he couldn’t go to the quidditch world cup harry would write and tell sirius who would know harry was being mistreated. there was only one thing for uncle vernon to do. harry could see the conclusion forming in his uncle’s mind as though the great mustached face were transparent. harry tried not to smile to keep his own face as blank as possible. and then — “well all right then. you can go to this ruddy ... this stupid ... this world cup thing. you write and tell these — these weasleys they’re to pick you up mind. i haven’t got time to go dropping you off all over the country. and you can spend the rest of the summer there. and you can tell your — your godfather ... tell him ... tell him you’re going.” “okay then,” said harry brightly. he turned and walked toward the living room door fighting the urge to jump into the air and whoop. he was going ... he was going to the weasleys’ he was going to watch the quidditch world cup! outside in the hall he nearly ran into dudley who had been lurking behind the door clearly hoping to overhear harry being told off. he looked shocked to see the broad grin on harry’s face. “that was an excellent breakfast wasn’t it?” said harry. “i feel really full don’t you?” laughing at the astonished look on dudley’s face harry took the stairs three at a time and hurled himself back into his bedroom. the first thing he saw was that hedwig was back. she was sitting in her cage staring at harry with her enormous amber eyes and clicking her beak in the way that meant she was annoyed about something. exactly what was annoying her became apparent almost at once. “ouch!” said harry as what appeared to be a small gray feathery tennis ball collided with the side of his head. harry massaged the spot furiously looking up to see what had hit him and saw a minute owl small enough to fit into the palm of his hand whizzing excitedly around the room like a loose firework. harry then realized that the owl had dropped a letter at his feet. harry bent down recognized ron’s handwriting then tore open the envelope. inside was a hastily scribbled note. harry — dad got the tickets — ireland versus bulgaria monday night. mum’s writing to the muggles to ask you to stay. they might already have the letter i don’t know how fast muggle post is. thought i’d send this with pig anyway. harry stared at the word “pig,” then looked up at the tiny owl now zooming around the light fixture on the ceiling. he had never seen anything that looked less like a pig. maybe he couldn’t read ron’s writing. he went back to the letter we’re coming for you whether the muggles like it or not you can’t miss the world cup only mum and dad reckon it’s better if we pretend to ask their permission first. if they say yes send pig back with your answer pronto and we’ll come and get you at five o’clock on sunday. if they say no send pig back pronto and we’ll come and get you at five o’clock on sunday anyway. hermione’s arriving this afternoon. percy’s started work — the department of international magical cooperation. don’t mention anything about abroad while you’re here unless you want the pants bored off you. see you soon — ron “calm down!” harry said as the small owl flew low over his head twittering madly with what harry could only assume was pride at having delivered the letter to the right person. “come here i need you to take my answer back!” the owl fluttered down on top of hedwig’s cage. hedwig looked coldly up at it as though daring it to try and come any closer. harry seized his eagle-feather quill once more grabbed a fresh piece of parchment and wrote ron it’s all okay the muggles say i can come. see you five o’clock tomorrow. can’t wait. harry he folded this note up very small and with immense difficulty tied it to the tiny owl’s leg as it hopped on the spot with excitement. the moment the note was secure the owl was off again it zoomed out of the window and out of sight. harry turned to hedwig. “feeling up to a long journey?” he asked her. hedwig hooted in a dignified sort of a way. “can you take this to sirius for me?” he said picking up his letter. “hang on ... i just want to finish it.” he unfolded the parchment and hastily added a postscript. if you want to contact me i’ll be at my friend ron weasley’s for the rest of the summer. his dad’s got us tickets for the quidditch world cup the letter finished he tied it to hedwig’s leg she kept unusually still as though determined to show him how a real post owl should behave. “i’ll be at ron’s when you get back all right?” harry told her. she nipped his finger affectionately then with a soft swooshing noise spread her enormous wings and soared out of the open window. harry watched her out of sight then crawled under his bed wrenched up the loose floorboard and pulled out a large chunk of birthday cake. he sat there on the floor eating it savoring the happiness that was flooding through him. he had cake and dudley had nothing but grapefruit it was a bright summer’s day he would be leaving privet drive tomorrow his scar felt perfectly normal again and he was going to watch the quidditch world cup. it was hard just now to feel worried about anything — even lord voldemort. back to the burrow by twelve o’clock the next day harry’s school trunk was packed with his school things and all his most prized possessions — the invisibility cloak he had inherited from his father the broomstick he had gotten from sirius the enchanted map of hogwarts he had been given by fred and george weasley last year. he had emptied his hiding place under the loose floorboard of all food double-checked every nook and cranny of his bedroom for forgotten spellbooks or quills and taken down the chart on the wall counting down the days to september the first on which he liked to cross off the days remaining until his return to hogwarts. the atmosphere inside number four privet drive was extremely tense. the imminent arrival at their house of an assortment of wizards was making the dursleys uptight and irritable. uncle vernon had looked downright alarmed when harry informed him that the weasley s would be arriving at five o’clock the very next day. “i hope you told them to dress properly these people,” he snarled at once. “i’ve seen the sort of stuff your lot wear. they’d better have the decency to put on normal clothes that’s all.” harry felt a slight sense of foreboding. he had rarely seen mr. or mrs. weasley wearing anything that the dursleys would call “normal.” their children might don muggle clothing during the holidays but mr. and mrs. weasley usually wore long robes in varying states of shabbiness. harry wasn’t bothered about what the neighbors would think but he was anxious about how rude the dursleys might be to the weasleys if they turned up looking like their worst idea of wizards. uncle vernon had put on his best suit. to some people this might have looked like a gesture of welcome but harry knew it was because uncle vernon wanted to look impressive and intimidating. dudley on the other hand looked somehow diminished. this was not because the diet was at last taking effect but due to fright. dudley had emerged from his last encounter with a fully-grown wizard with a curly pig’s tail poking out of the seat of his trousers and aunt petunia and uncle vernon had had to pay for its removal at a private hospital in london. it wasn’t altogether surprising therefore that dudley kept running his hand nervously over his backside and walking sideways from room to room so as not to present the same target to the enemy. lunch was an almost silent meal. dudley didn’t even protest at the food cottage cheese and grated celery). aunt petunia wasn’t eating anything at all. her arms were folded her lips were pursed and she seemed to be chewing her tongue as though biting back the furious diatribe she longed to throw at harry. “they’ll be driving of course?” uncle vernon barked across the table. “er,” said harry. he hadn’t thought of that. how were the weasleys going to pick him up they didn’t have a car anymore the old ford anglia they had once owned was currently running wild in the forbidden forest at hogwarts. but mr. weasley had borrowed a ministry of magic car last year possibly he would do the same today “i think so,” said harry. uncle vernon snorted into his mustache. normally uncle vernon would have asked what car mr. weasley drove he tended to judge other men by how big and expensive their cars were. but harry doubted whether uncle vernon would have taken to mr. weasley even if he drove a ferrari. harry spent most of the afternoon in his bedroom he couldn’t stand watching aunt petunia peer out through the net curtains every few seconds as though there had been a warning about an escaped rhinoceros. finally at a quarter to five harry went back downstairs and into the living room. aunt petunia was compulsively straightening cushions. uncle vernon was pretending to read the paper but his tiny eyes were not moving and harry was sure he was really listening with all his might for the sound of an approaching car. dudley was crammed into an armchair his porky hands beneath him clamped firmly around his bottom. harry couldn’t take the tension he left the room and went and sat on the stairs in the hall his eyes on his watch and his heart pumping fast from excitement and nerves. but five o’clock came and then went. uncle vernon perspiring slightly in his suit opened the front door peered up and down the street then withdrew his head quickly. “they’re late!” he snarled at harry. “i know,” said harry. “maybe — er — the traffic’s bad or something.” ten past five ... then a quarter past five ... harry was starting to feel anxious himself now. at half past he heard uncle vernon and aunt petunia conversing in terse mutters in the living room. “no consideration at all.” “we might’ve had an engagement.” “maybe they think theyll get invited to dinner if they’re late.” “well they most certainly won’t be,” said uncle vernon and harry heard him stand up and start pacing the living room. “they’ll take the boy and go there 11 be no hanging around. that’s if they’re coming at all. probably mistaken the day. i daresay their kind don’t set much store by punctuality. either that or they drive some tin-pot car that’s broken d — aaaaaaaarrrrrgh ! ” harry jumped up. from the other side of the living room door came the sounds of the three dursleys scrambling panic-stricken across the room. next moment dudley came flying into the hall looking terrified. “what happened?” said harry. “what’s the matter?” but dudley didn’t seem able to speak. hands still clamped over his buttocks he waddled as fast as he could into the kitchen. harry hurried into the living room. loud hangings and scrapings were coming from behind the dursleys’ boarded-up fireplace which had a fake coal fire plugged in front of it. “what is it?” gasped aunt petunia who had backed into the wall and was staring terrified toward the fire. “what is it vernon?” but they were left in doubt barely a second longer. voices could be heard from inside the blocked fireplace. “ouch! fred no — go back go back there’s been some kind of mistake — tell george not to — ouch! george no there’s no room go back quickly and tell ron — ” “maybe harry can hear us dad — maybe he’ll be able to let us out — ” there was a loud hammering of fists on the boards behind the electric fire. “harry harry can you hear us?” the dursleys rounded on harry like a pair of angry wolverines. “what is this?” growled uncle vernon. “what’s going on?” “they — they’ve tried to get here by floo powder,” said harry fighting a mad desire to laugh. “they can travel by fire — only you’ve blocked the fireplace — hang on — ” he approached the fireplace and called through the boards. “mr. weasley can you hear me?” the hammering stopped. somebody inside the chimney piece said “shh!” “mr. weasley it’s harry ... the fireplace has been blocked up. you won’t be able to get through there.” “damn!” said mr. weasley’s voice. “what on earth did they want to block up the fireplace for?” “they’ve got an electric fire,” harry explained. “really?” said mr. weasley’s voice excitedly. “eclectic you say with a plug gracious i must see that. ... let’s think ... ouch ron!” ron’s voice now joined the others’. “what are we doing here has something gone wrong?” “oh no ron,” came fred’s voice very sarcastically. “no this is exactly where we wanted to end up.” “yeah we’re having the time of our lives here,” said george whose voice sounded muffled as though he was squashed against the wall. “boys boys ...” said mr. weasley vaguely. “i’m trying to think what to do. ... yes ... only way ... stand back harry.” harry retreated to the sofa. uncle vernon however moved forward. “wait a moment!” he bellowed at the fire. “what exactly are you going to — ” bang. the electric fire shot across the room as the boarded up fireplace burst outward expelling mr. weasley fred george and ron in a cloud of rubble and loose chippings. aunt petunia shrieked and fell backward over the coffee table uncle vernon caught her before she hit the floor and gaped speechless at the weasleys all of whom had bright red hair including fred and george who were identical to the last freckle. “that’s better,” panted mr. weasley brushing dust from his long green robes and straightening his glasses. “ah — you must be harry’s aunt and uncle!” tall thin and balding he moved toward uncle vernon his hand outstretched but uncle vernon backed away several paces dragging aunt petunia. words utterly failed uncle vernon. his best suit was covered in white dust which had settled in his hair and mustache and made him look as though he had just aged thirty years. “er — yes — sorry about that,” said mr. weasley lowering his hand and looking over his shoulder at the blasted fireplace. “it’s all my fault. it just didn’t occur to me that we wouldn’t be able to get out at the other end. i had your fireplace connected to the floo network you see — just for an afternoon you know so we could get harry. muggle fireplaces aren’t supposed to be connected strictly speaking — but i’ve got a useful contact at the floo regulation panel and he fixed it for me. i can put it right in a jiffy though don’t worry. i’ll light a fire to send the boys back and then i can repair your fireplace before i disapparate.” harry was ready to bet that the dursleys hadn’t understood a single word of this. they were still gaping at mr. weasley thunderstruck. aunt petunia staggered upright again and hid behind uncle vernon. “hello harry!” said mr. weasley brightly. “got your trunk ready?” “it’s upstairs,” said harry grinning back. “well get it,” said fred at once. winking at harry he and george left the room. they knew where harry’s bedroom was having once rescued him from it in the dead of night. harry suspected that fred and george were hoping for a glimpse of dudley they had heard a lot about him from harry. “well,” said mr. weasley swinging his arms slightly while he tried to find words to break the very nasty silence. “very — erm — very nice place you’ve got here.” as the usually spotless living room was now covered in dust and bits of brick this remark didn’t go down too well with the dursleys. uncle vernon’s face purpled once more and aunt petunia started chewing her tongue again. however they seemed too scared to actually say anything. mr. weasley was looking around. he loved everything to do with muggles. harry could see him itching to go and examine the television and the video recorder. “they run off eckeltricity do they?” he said knowledgeably. “ah yes i can see the plugs. i collect plugs,” he added to uncle vernon. “and batteries. got a very large collection of batteries. my wife thinks i’m mad but there you are.” uncle vernon clearly thought mr. weasley was mad too. he moved ever so slightly to the right screening aunt petunia from view as though he thought mr. weasley might suddenly run at them and attack. dudley suddenly reappeared in the room. harry could hear the clunk of his trunk on the stairs and knew that the sounds had scared dudley out of the kitchen. dudley edged along the wall gazing at mr. weasley with terrified eyes and attempted to conceal himself behind his mother and father. unfortunately uncle vernon’s bulk while sufficient to hide bony aunt petunia was nowhere near enough to conceal dudley. “ah this is your cousin is it harry?” said mr. weasley taking another brave stab at making conversation. “yep,” said harry “that’s dudley.” he and ron exchanged glances and then quickly looked away from each other the temptation to burst out laughing was almost overwhelming. dudley was still clutching his bottom as though afraid it might fall off. mr. weasley however seemed genuinely concerned at dudley’s peculiar behavior. indeed from the tone of his voice when he next spoke harry was quite sure that mr. weasley thought dudley was quite as mad as the dursleys thought he was except that mr. weasley felt sympathy rather than fear. “having a good holiday dudley?” he said kindly. dudley whimpered. harry saw his hands tighten still harder over his massive backside. fred and george came back into the room carrying harry’s school trunk. they glanced around as they entered and spotted dudley. their faces cracked into identical evil grins. “ah right,” said mr. weasley. “better get cracking then.” he pushed up the sleeves of his robes and took out his wand. harry saw the dursleys draw back against the wall as one. “incendio\” said mr. weasley pointing his wand at the hole in the wall behind him. flames rose at once in the fireplace crackling merrily as though they had been burning for hours. mr. weasley took a small drawstring bag from his pocket untied it took a pinch of the powder inside and threw it onto the flames which turned emerald green and roared higher than ever. “off you go then fred,” said mr. weasley. “coming,” said fred. “oh no — hang on — ” a bag of sweets had spilled out of fred’s pocket and the contents were now rolling in every direction — big fat toffees in brightly colored wrappers. fred scrambled around cramming them back into his pocket then gave the dursleys a cheery wave stepped forward and walked right into the fire saying “the burrow!” aunt petunia gave a little shuddering gasp. there was a whooshing sound and fred vanished. “right then george,” said mr. weasley “you and the trunk.” harry helped george carry the trunk forward into the flames and turn it onto its end so that he could hold it better. then with a second whoosh george had cried “the burrow!” and vanished too. “ron you next,” said mr. weasley. “see you,” said ron brightly to the dursleys. he grinned broadly at harry then stepped into the fire shouted “the burrow!” and disappeared. now harry and mr. weasley alone remained. “well ... t>ye then,” harry said to the dursleys. they didn’t say anything at all. harry moved toward the fire but just as he reached the edge of the hearth mr. weasley put out a hand and held him back. he was looking at the dursleys in amazement. “harry said good-bye to you,” he said. “didn’t you hear him?” “it doesn’t matter,” harry muttered to mr. weasley. “honestly i don’t care.” mr. weasley did not remove his hand from harry’s shoulder. “you aren’t going to see your nephew till next summer,” he said to uncle vernon in mild indignation. “surely you’re going to say good-bye?” uncle vernon’s face worked furiously. the idea of being taught consideration by a man who had just blasted away half his living room wall seemed to be causing him intense suffering. but mr. weasley’s wand was still in his hand and uncle vernon’s tiny eyes darted to it once before he said very resentfully “good-bye then.” “see you,” said harry putting one foot forward into the green flames which felt pleasantly like warm breath. at that moment however a horrible gagging sound erupted behind him and aunt petunia started to scream. harry wheeled around. dudley was no longer standing behind his parents. he was kneeling beside the coffee table and he was gagging and sputtering on a foot-long purple slimy thing that was protruding from his mouth. one bewildered second later harry realized that the foot-long thing was dudley’s tongue — and that a brightly colored toffee wrapper lay on the floor before him. aunt petunia hurled herself onto the ground beside dudley seized the end of his swollen tongue and attempted to wrench it out of his mouth unsurprisingly dudley yelled and sputtered worse than ever trying to fight her off. uncle vernon was bellowing and waving his arms around and mr. weasley had to shout to make himself heard. “not to worry i can sort him out!” he yelled advancing on dudley with his wand outstretched but aunt petunia screamed worse than ever and threw herself on top of dudley shielding him from mr. weasley. “no really!” said mr. weasley desperately. “it’s a simple process — it was the toffee — my son fred — real practical joker — but it’s only an engorgement charm — at least i think it is — please i can correct it — ” but far from being reassured the dursleys became more panic-stricken aunt petunia was sobbing hysterically tugging dudley’s tongue as though determined to rip it out dudley appeared to be suffocating under the combined pressure of his mother and his tongue and uncle vernon who had lost control completely seized a china figure from on top of the sideboard and threw it very hard at mr. weasley who ducked causing the ornament to shatter in the blasted fireplace. “now really!” said mr. weasley angrily brandishing his wand. “i’m trying to help\” bellowing like a wounded hippo uncle vernon snatched up another ornament. “harry go! just go!” mr. weasley shouted his wand on uncle vernon. “i’ll sort this out!” harry didn’t want to miss the fun but uncle vernon’s second ornament narrowly missed his left ear and on balance he thought it best to leave the situation to mr. weasley. he stepped into the fire looking over his shoulder as he said “the burrow!” his last fleeting glimpse of the living room was of mr. weasley blasting a third ornament out of uncle vernon’s hand with his wand aunt petunia screaming and lying on top of dudley and dudley’s tongue lolling around like a great slimy python. but next moment harry had begun to spin very fast and the dursleys’ living room was whipped out of sight in a rush of emerald-green flames. 5 weasleys’ wizard wheezes harry spun faster and faster elbows tucked tightly to his sides blurred fireplaces flashing past him until he started to feel sick and closed his eyes. then when at last he felt himself slowing down he threw out his hands and came to a halt in time to prevent himself from falling face forward out of the weasleys’ kitchen fire. “did he eat it?” said fred excitedly holding out a hand to pull harry to his feet. “yeah,” said harry straightening up. “what was it?” “ton-tongue toffee,” said fred brightly. “george and i invented them and we’ve been looking for someone to test them on all summer. ...” the tiny kitchen exploded with laughter harry looked around and saw that ron and george were sitting at the scrubbed wooden table with two red-haired people harry had never seen before though he knew immediately who they must be bill and charlie the two eldest weasley brothers. “how’re you doing harry?” said the nearer of the two grinning at him and holding out a large hand which harry shook feeling calluses and blisters under his fingers. this had to be charlie who worked with dragons in romania. charlie was built like the twins shorter and stockier than percy and ron who were both long and lanky. he had a broad good-natured face which was weather-beaten and so freckly that he looked almost tanned his arms were muscular and one of them had a large shiny burn on it. bill got to his feet smiling and also shook harry’s hand. bill came as something of a surprise. harry knew that he worked for the wizarding bank gringotts and that bill had been head boy at hogwarts harry had always imagined bill to be an older version of percy fussy about rule-breaking and fond of bossing everyone around. however bill was — there was no other word for it — cool. he was tall with long hair that he had tied back in a ponytail. he was wearing an earring with what looked like a fang dangling from it. bill’s clothes would not have looked out of place at a rock concert except that harry recognized his boots to be made not of leather but of dragon hide. before any of them could say anything else there was a faint popping noise and mr. weasley appeared out of thin air at george’s shoulder. he was looking angrier than harry had ever seen him. “that wasn’t funny fred!” he shouted. “what on earth did you give that muggle boy?” “i didn’t give him anything,” said fred with another evil grin. “i just dropped it. ... it was his fault he went and ate it i never told him to.” “you dropped it on purpose!” roared mr. weasley. “you knew he’d eat it you knew he was on a diet — ” “how big did his tongue get?” george asked eagerly. “it was four feet long before his parents would let me shrink it!” harry and the weasleys roared with laughter again. “it isn’t funny\” mr. weasley shouted. “that sort of behavior seriously undermines wizard-muggle relations! i spend half my life campaigning against the mistreatment of muggles and my own sons — ” “we didn’t give it to him because he’s a muggle!” said fred indignantly. “no we gave it to him because he’s a great bullying git,” said george. “isn’t he harry?” “yeah he is mr. weasley,” said harry earnestly. “that’s not the point!” raged mr. weasley. “you wait until i tell your mother — ” “tell me what?” said a voice behind them. mrs. weasley had just entered the kitchen. she was a short plump woman with a very kind face though her eyes were presently narrowed with suspicion. “oh hello harry dear,” she said spotting him and smiling. then her eyes snapped back to her husband. “tell me what  arthur?” mr. weasley hesitated. harry could tell that however angry he was with fred and george he hadn’t really intended to tell mrs. weasley what had happened. there was a silence while mr. weasley eyed his wife nervously. then two girls appeared in the kitchen doorway behind mrs. weasley. one with very bushy brown hair and rather large front teeth was harry’s and ron’s friend hermione granger. the other who was small and red-haired was ron’s younger sister ginny. both of them smiled at harry who grinned back which made ginny go scarlet — she had been very taken with harry ever since his first visit to the burrow. “tell me what arthur?” mrs. weasley repeated in a dangerous sort of voice. “it’s nothing molly,” mumbled mr. weasley “fred and george just — but i’ve had words with them — ” “what have they done this time?” said mrs. weasley. “if it’s got anything to do with weasleys’ wizard wheezes — ” “why don’t you show harry where he’s sleeping ron?” said hermione from the doorway. “he knows where he’s sleeping,” said ron “in my room he slept there last — ” “we can all go,” said hermione pointedly. “oh,” said ron cottoning on. “right.” “yeah we’ll come too,” said george. “you stay where you are!” snarled mrs. weasley. harry and ron edged out of the kitchen and they hermione and ginny set off along the narrow hallway and up the rickety staircase that zigzagged through the house to the upper stories. “what are weasleys’ wizard wheezes?” harry asked as they climbed. ron and ginny both laughed although hermione didn’t. “mum found this stack of order forms when she was cleaning fred and george’s room,” said ron quietly. “great long price lists for stuff they’ve invented. joke stuff you know. fake wands and trick sweets loads of stuff. it was brilliant i never knew they’d been inventing all that ...” “we’ve been hearing explosions out of their room for ages but we never thought they were actually making things,” said ginny. “we thought they just liked the noise.” “only most of the stuff — well all of it really — was a bit dangerous,” said ron “and you know they were planning to sell it at hogwarts to make some money and mum went mad at them. told them they weren’t allowed to make any more of it and burned all the order forms. ... she’s furious at them anyway. they didn’t get as many o.w.l.s as she expected.” o.w.l.s were ordinary wizarding levels the examinations hogwarts students took at the age of fifteen. “and then there was this big row,” ginny said “because mum wants them to go into the ministry of magic like dad and they told her all they want to do is open a joke shop.” just then a door on the second landing opened and a face poked out wearing horn-rimmed glasses and a very annoyed expression. “hi percy,” said harry. “oh hello harry,” said percy. “i was wondering who was making all the noise. i’m trying to work in here you know — i’ve got a report to finish for the office — and it’s rather difficult to concentrate when people keep thundering up and down the stairs.” “we’re not thundering,” said ron irritably. “we’re walking. sorry if we’ve disturbed the top-secret workings of the ministry of magic.” “what are you working on?” said harry. “a report for the department of international magical cooperation,” said percy smugly. “we’re trying to standardize cauldron thickness. some of these foreign imports are just a shade too thin — leakages have been increasing at a rate of almost three percent a year — ” “that’ll change the world that report will,” said ron. “front page of the daily prophet i expect cauldron leaks.” percy went slightly pink. “you might sneer ron,” he said heatedly “but unless some sort of international law is imposed we might well find the market flooded with flimsy shallow bottomed products that seriously endanger — ” “yeah yeah all right,” said ron and he started off upstairs again. percy slammed his bedroom door shut. as harry hermione and ginny followed ron up three more flights of stairs shouts from the kitchen below echoed up to them. it sounded as though mr. weasley had told mrs. weasley about the toffees. the room at the top of the house where ron slept looked much as it had the last time that harry had come to stay the same posters of ron’s favorite quidditch team the chudley cannons were whirling and waving on the walls and sloping ceiling and the fish tank on the windowsill which had previously held frog spawn now contained one extremely large frog. ron’s old rat scabbers was here no more but instead there was the tiny gray owl that had delivered ron’s letter to harry in privet drive. it was hopping up and down in a small cage and twittering madly. “shut up pig,” said ron edging his way between two of the four beds that had been squeezed into the room. “fred and george are in here with us because bill and charlie are in their room,” he told harry. “percy gets to keep his room all to himself because he’s got to work.” “er — why are you calling that owl pig?” harry asked ron. “because he’s being stupid,” said ginny. “its proper name is pigwidgeon.” “yeah and that’s not a stupid name at all,” said ron sarcastically. “ginny named him,” he explained to harry. “she reckons it’s sweet. and i tried to change it but it was too late he won’t answer to anything else. so now he’s pig. i’ve got to keep him up here because he annoys errol and hermes. he annoys me too come to that.” pigwidgeon zoomed happily around his cage hooting shrilly. harry knew ron too well to take him seriously. he had moaned continually about his old rat scabbers but had been most upset when hermione’s cat crookshanks appeared to have eaten him. “where’s crookshanks?” harry asked hermione now. “out in the garden i expect,” she said. “he likes chasing gnomes. he’s never seen any before.” “percy’s enjoying work then?” said harry sitting down on one of the beds and watching the chudley cannons zooming in and out of the posters on the ceiling. “enjoying it?” said ron darkly. “i don’t reckon he’d come home if dad didn’t make him. he’s obsessed. just don’t get him onto the subject of his boss. according to mr. crouch ... as i was saying to mr. crouch . . . mr. crouch is of the opinion . . . mr. crouch was telling me ... they 11 be announcing their engagement any day now.” “have you had a good summer harry?” said hermione. “did you get our food parcels and everything?” “yeah thanks a lot,” said harry. “they saved my life those cakes.” “and have you heard from — ” ron began but at a look from hermione he fell silent. harry knew ron had been about to ask about sirius. ron and hermione had been so deeply involved in helping sirius escape from the ministry of magic that they were almost as concerned about harry’s godfather as he was. however discussing him in front of ginny was a bad idea. nobody but themselves and professor dumbledore knew about how sirius had escaped or believed in his innocence. “i think they’ve stopped arguing,” said hermione to cover the awkward moment because ginny was looking curiously from ron to harry. “shall we go down and help your mum with dinner?” “yeah all right,” said ron. the four of them left ron’s room and went back downstairs to find mrs. weasley alone in the kitchen looking extremely bad-tempered. “we’re eating out in the garden,” she said when they came in. “there’s just not room for eleven people in here. could you take the plates outside girls bill and charlie are setting up the tables. knives and forks please you two,” she said to ron and harry pointing her wand a little more vigorously than she had intended at a pile of potatoes in the sink which shot out of their skins so fast that they ricocheted off the walls and ceiling. “oh for heaven’s sake,” she snapped now directing her wand at a dustpan which hopped off the sideboard and started skating across the floor scooping up the potatoes. “those two!” she burst out savagely now pulling pots and pans out of a cupboard and harry knew she meant fred and george. “i don’t know what’s going to happen to them i really don’t. no ambition unless you count making as much trouble as they possibly can. ...” mrs. weasley slammed a large copper saucepan down on the kitchen table and began to wave her wand around inside it. a creamy sauce poured from the wand tip as she stirred. “it’s not as though they haven’t got brains,” she continued irritably taking the saucepan over to the stove and lighting it with a further poke of her wand “but they’re wasting them and unless they pull themselves together soon they’ll be in real trouble. i’ve had more owls from hogwarts about them than the rest put together. if they carry on the way they’re going they’ll end up in front of the improper use of magic office.” mrs. weasley jabbed her wand at the cutlery drawer which shot open. harry and ron both jumped out of the way as several knives soared out of it flew across the kitchen and began chopping the potatoes which had just been tipped back into the sink by the dustpan. “i don’t know where we went wrong with them,” said mrs. weasley putting down her wand and starting to pull out still more saucepans. “it’s been the same for years one thing after another and they won’t listen to — oh not again.” she had picked up her wand from the table and it had emitted a loud squeak and turned into a giant rubber mouse. “one of their fake wands again!” she shouted. “how many times have i told them not to leave them lying around?” she grabbed her real wand and turned around to find that the sauce on the stove was smoking. “c’mon,” ron said hurriedly to harry seizing a handful of cutlery from the open drawer “let’s go and help bill and charlie.” they left mrs. weasley and headed out the back door into the yard. they had only gone a few paces when hermione’s bandy-legged ginger cat crookshanks came pelting out of the garden bottle-brush tail held high in the air chasing what looked like a muddy potato on legs. harry recognized it instantly as a gnome. barely ten inches high its horny little feet pattered very fast as it sprinted across the yard and dived headlong into one of the wellington boots that lay scattered around the door. harry could hear the gnome giggling madly as crookshanks inserted a paw into the boot trying to reach it. meanwhile a very loud crashing noise was coming from the other side of the house. the source of the commotion was revealed as they entered the garden and saw that bill and charlie both had their wands out and were making two battered old tables fly high above the lawn smashing into each other each attempting to knock the other’s out of the air. fred and george were cheering ginny was laughing and hermione was hovering near the hedge apparently torn between amusement and anxiety. bill’s table caught charlie’s with a huge bang and knocked one of its legs off. there was a clatter from overhead and they all looked up to see percy’s head poking out of a window on the second floor. “will you keep it down?!” he bellowed. “sorry perce,” said bill grinning. “how’re the cauldron bottoms coming on?” “very badly,” said percy peevishly and he slammed the window shut. chuckling bill and charlie directed the tables safely onto the grass end to end and then with a flick of his wand bill reattached the table leg and conjured tablecloths from nowhere. by seven o’clock the two tables were groaning under dishes and dishes of mrs. weasley’s excellent cooking and the nine weasleys harry and hermione were settling themselves down to eat beneath a clear deep blue sky. to somebody who had been living on meals of increasingly stale cake all summer this was paradise and at first harry listened rather than talked as he helped himself to chicken and ham pie boiled potatoes and salad. at the far end of the table percy was telling his father all about his report on cauldron bottoms. “i’ve told mr. crouch that i’ll have it ready by tuesday,” percy was saying pompously. “that’s a bit sooner than he expected it but i like to keep on top of things. i think he’ll be grateful i’ve done it in good time i mean it’s extremely busy in our department just now what with all the arrangements for the world cup. we’re just not getting the support we need from the department of magical games and sports. ludo bagman — ” “i like ludo,” said mr. weasley mildly. “he was the one who got us such good tickets for the cup. i did him a bit of a favor his brother otto got into a spot of trouble — a lawnmower with unnatural powers — i smoothed the whole thing over.” “oh bagman’s likable enough of course,” said percy dismissively “but how he ever got to be head of department ... when i compare him to mr. crouch! i can’t see mr. crouch losing a member of our department and not trying to find out what’s happened to them. you realize bertha jorkins has been missing for over a month now went on holiday to albania and never came back?” “yes i was asking ludo about that,” said mr. weasley frowning. “he says bertha’s gotten lost plenty of times before now — though i must say if it was someone in my department i’d be worried. ...” “oh bertha’s hopeless all right,” said percy. “i hear she’s been shunted from department to department for years much more trouble than she’s worth ... but all the same bagman ought to be trying to find her. mr. crouch has been taking a personal interest she worked in our department at one time you know and i think mr. crouch was quite fond of her — but bagman just keeps laughing and saying she probably misread the map and ended up in australia instead of albania. however” — percy heaved an impressive sigh and took a deep swig of elderflower wine — “we’ve got quite enough on our plates at the department of international magical cooperation without trying to find members of other departments too. as you know we’ve got another big event to organize right after the world cup.” percy cleared his throat significantly and looked down toward the end of the table where harry ron and hermione were sitting. “ you know the one i’m talking about father.” he raised his voice slightly. “the top secret one.” ron rolled his eyes and muttered to harry and hermione “he’s been trying to get us to ask what that event is ever since he started work. probably an exhibition of thick-bottomed cauldrons.” in the middle of the table mrs. weasley was arguing with bill about his earring which seemed to be a recent acquisition. “... with a horrible great fang on it. really bill what do they say at the bank?” “mum no one at the bank gives a damn how i dress as long as i bring home plenty of treasure,” said bill patiently. “and your hair’s getting silly dear,” said mrs. weasley fingering her wand lovingly. “i wish you’d let me give it a trim. ...” “i like it,” said ginny who was sitting beside bill. “you’re so old-fashioned mum. anyway it’s nowhere near as long as professor dumbledore’s. ...” next to mrs. weasley fred george and charlie were all talking spiritedly about the world cup. “it’s got to be ireland,” said charlie thickly through a mouthful of potato. “they flattened peru in the semifinals.” “bulgaria has got viktor krum though,” said fred. “krum’s one decent player ireland has got seven,” said charlie shortly. “i wish england had got through. that was embarrassing that was.” “what happened?” said harry eagerly regretting more than ever his isolation from the wizarding world when he was stuck on privet drive. “went down to transylvania three hundred and ninety to ten,” said charlie gloomily. “shocking performance. and wales lost to uganda and scotland was slaughtered by luxembourg.” harry had been on the gryffindor house quidditch team ever since his first year at hogwarts and owned one of the best racing brooms in the world a firebolt. flying came more naturally to harry than anything else in the magical world and he played in the position of seeker on the gryffindor house team. mr. weasley conjured up candles to light the darkening garden before they had their homemade strawberry ice cream and by the time they had finished moths were fluttering low over the table and the warm air was perfumed with the smells of grass and honeysuckle. harry was feeling extremely well fed and at peace with the world as he watched several gnomes sprinting through the rosebushes laughing madly and closely pursued by crookshanks. ron looked carefully up the table to check that the rest of the family were all busy talking then he said very quietly to harry “so — have you heard from sirius lately?” hermione looked around listening closely. “yeah,” said harry softly “twice. he sounds okay. i wrote to him yesterday. he might write back while i’m here.” he suddenly remembered the reason he had written to sirius and for a moment was on the verge of telling ron and hermione about his scar hurting again and about the dream that had awoken him . . . but he really didn’t want to worry them just now not when he himself was feeling so happy and peaceful. “look at the time,” mrs. weasley said suddenly checking her wristwatch. “you really should be in bed the whole lot of you — you’ll be up at the crack of dawn to get to the cup. harry if you leave your school list out i’ll get your things for you tomorrow in diagon alley. i’m getting everyone else’s. there might not be time after the world cup the match went on for five days last time.” “wow — hope it does this time!” said harry enthusiastically. “well i certainly don’t,” said percy sanctimoniously. “i shudder to think what the state of my in-tray would be if i was away from work for five days.” “yeah someone might slip dragon dung in it again eh perce?” said fred. “that was a sample of fertilizer from norway!” said percy going very red in the face. “it was nothing personall” “it was,” fred whispered to harry as they got up from the table. “we sent it.” the portkey harry felt as though he had barely lain down to sleep in ron’s room when he was being shaken awake by mrs. weasley. “time to go harry dear,” she whispered moving away to wake ron. harry felt around for his glasses put them on and sat up. it was still dark outside. ron muttered indistinctly as his mother roused him. at the foot of harry’s mattress he saw two large disheveled shapes emerging from tangles of blankets. “ ’s’ time already?” said fred groggily. they dressed in silence too sleepy to talk then yawning and stretching the four of them headed downstairs into the kitchen. mrs. weasley was stirring the contents of a large pot on the stove while mr. weasley was sitting at the table checking a sheaf of large parchment tickets. he looked up as the boys entered and spread his arms so that they could see his clothes more clearly. he was wearing what appeared to be a golfing sweater and a very old pair of jeans slightly too big for him and held up with a thick leather belt. “what d’you think?” he asked anxiously. “we’re supposed to go incognito — do i look like a muggle harry?” “yeah,” said harry smiling “very good.” “where ’re bill and charlie and per-per-percy?” said george failing to stifle a huge yawn. “well they’re apparating aren’t they?” said mrs. weasley heaving the large pot over to the table and starting to ladle porridge into bowls. “so they can have a bit of a lie-in.” harry knew that apparating meant disappearing from one place and reappearing almost instantly in another but had never known any hogwarts student to do it and understood that it was very difficult. “so they’re still in bed?” said fred grumpily pulling his bowl of porridge toward him. “why can’t we apparate too?” “because you’re not of age and you haven’t passed your test,” snapped mrs. weasley. “and where have those girls got to?” she bustled out of the kitchen and they heard her climbing the stairs. “you have to pass a test to apparate?” harry asked. “oh yes,” said mr. weasley tucking the tickets safely into the back pocket of his jeans. “the department of magical transportation had to fine a couple of people the other day for apparating without a license. it’s not easy apparition and when it’s not done properly it can lead to nasty complications. this pair i’m talking about went and splinched themselves.” everyone around the table except harry winced. “er — splinched?” said harry. “they left half of themselves behind,” said mr. weasley now spooning large amounts of treacle onto his porridge. “so of course they were stuck. couldn’t move either way. had to wait for the accidental magic reversal squad to sort them out. meant a fair old bit of paperwork i can tell you what with the muggles who spotted the body parts they’d left behind. ...” harry had a sudden vision of a pair of legs and an eyeball lying abandoned on the pavement of privet drive. “were they okay?” he asked startled. “oh yes,” said mr. weasley matter-of-factly. “but they got a heavy fine and i don’t think they’ll be trying it again in a hurry. you don’t mess around with apparition. there are plenty of adult wizards who don’t bother with it. prefer brooms — slower but safer.” “but bill and charlie and percy can all do it?” “charlie had to take the test twice,” said fred grinning. “he failed the first time apparated five miles south of where he meant to right on top of some poor old dear doing her shopping remember?” “yes well he passed the second time,” said mrs. weasley marching back into the kitchen amid hearty sniggers. “percy only passed two weeks ago,” said george. “he’s been apparating downstairs every morning since just to prove he can.” there were footsteps down the passageway and hermione and ginny came into the kitchen both looking pale and drowsy. “why do we have to be up so early?” ginny said rubbing her eyes and sitting down at the table. “we’ve got a bit of a walk,” said mr. weasley. “walk?” said harry. “what are we walking to the world cup?” “no no that’s miles away,” said mr. weasley smiling. “we only need to walk a short way. it’s just that it’s very difficult for a large number of wizards to congregate without attracting muggle attention. we have to be very careful about how we travel at the best of times and on a huge occasion like the quidditch world cup — ” “george!” said mrs. weasley sharply and they all jumped. “what?” said george in an innocent tone that deceived nobody. “what is that in your pocket?” “nothing!” “don’t you lie to me!” mrs. weasley pointed her wand at george’s pocket and said “acciol” several small brightly colored objects zoomed out of george’s pocket he made a grab for them but missed and they sped right into mrs. weasley ’s outstretched hand. “we told you to destroy them!” said mrs. weasley furiously holding up what were unmistakably more ton-tongue toffees. “we told you to get rid of the lot! empty your pockets go on both of you!” it was an unpleasant scene the twins had evidently been trying to smuggle as many toffees out of the house as possible and it was only by using her summoning charm that mrs. weasley managed to find them all. “acciol acciol acciol” she shouted and toffees zoomed from all sorts of unlikely places including the lining of george’s jacket and the turn-ups of fred’s jeans. “we spent six months developing those!” fred shouted at his mother as she threw the toffees away. “oh a fine way to spend six months!” she shrieked. “no wonder you didn’t get more o.w.l.s!” all in all the atmosphere was not very friendly as they took their departure. mrs. weasley was still glowering as she kissed mr. weasley on the cheek though not nearly as much as the twins who had each hoisted their rucksacks onto their backs and walked out without a word to her. “well have a lovely time,” said mrs. weasley “and behave yourselves,” she called after the twins’ retreating backs but they did not look back or answer. “i’ll send bill charlie and percy along around midday,” mrs. weasley said to mr. weasley as he harry ron hermione and ginny set off across the dark yard after fred and george. it was chilly and the moon was still out. only a dull greenish tinge along the horizon to their right showed that daybreak was drawing closer. harry having been thinking about thousands of wizards speeding toward the quidditch world cup sped up to walk with mr. weasley. “so how does everyone get there without all the muggles noticing?” he asked. “it’s been a massive organizational problem,” sighed mr. weasley. “the trouble is about a hundred thousand wizards turn up at the world cup and of course we just haven’t got a magical site big enough to accommodate them all. there are places muggles can’t penetrate but imagine trying to pack a hundred thousand wizards into diagon alley or platform nine and three-quarters. so we had to find a nice deserted moor and set up as many anti-muggle precautions as possible. the whole ministry’s been working on it for months. first of course we have to stagger the arrivals. people with cheaper tickets have to arrive two weeks beforehand. a limited number use muggle transport but we can’t have too many clogging up their buses and trains — remember wizards are coming from all over the world. some apparate of course but we have to set up safe points for them to appear well away from muggles. i believe there’s a handy wood they’re using as the apparition point. for those who don’t want to apparate or can’t we use portkeys. they’re objects that are used to transport wizards from one spot to another at a prearranged time. you can do large groups at a time if you need to. there have been two hundred portkeys placed at strategic points around britain and the nearest one to us is up at the top of stoatshead hill so that’s where we’re headed.” mr. weasley pointed ahead of them where a large black mass rose beyond the village of ottery st. catchpole. “what sort of objects are portkeys?” said harry curiously. “well they can be anything,” said mr. weasley. “unobtrusive things obviously so muggles don’t go picking them up and playing with them . . . stuff they’ll just think is litter. ...” they trudged down the dark dank lane toward the village the silence broken only by their footsteps. the sky lightened very slowly as they made their way through the village its inky blackness diluting to deepest blue. harry’s hands and feet were freezing. mr. weasley kept checking his watch. they didn’t have breath to spare for talking as they began to climb stoatshead hill stumbling occasionally in hidden rabbit holes slipping on thick black tuffets of grass. each breath harry took was sharp in his chest and his legs were starting to seize up when at last his feet found level ground. “whew,” panted mr. weasley taking off his glasses and wiping them on his sweater. “well we’ve made good time — we’ve got ten minutes. ...” hermione came over the crest of the hill last clutching a stitch in her side. “now we just need the portkey,” said mr. weasley replacing his glasses and squinting around at the ground. “it won’t be big. ... come on ...” they spread out searching. they had only been at it for a couple of minutes however when a shout rent the still air. “over here arthur! over here son we’ve got it!” two tall figures were silhouetted against the starry sky on the other side of the hilltop. “amos!” said mr. weasley smiling as he strode over to the man who had shouted. the rest of them followed. mr. weasley was shaking hands with a ruddy-faced wizard with a scrubby brown beard who was holding a moldy-looking old boot in his other hand. “this is amos diggory everyone,” said mr. weasley. “he works for the department for the regulation and control of magical creatures. and i think you know his son cedric?” cedric diggory was an extremely handsome boy of around seventeen. he was captain and seeker of the hufflepuff house quidditch team at hogwarts. “hi,” said cedric looking around at them all. everybody said hi back except fred and george who merely nodded. they had never quite forgiven cedric for beating their team gryffindor in the first quidditch match of the previous year. “long walk arthur?” cedric’s father asked. “not too bad,” said mr. weasley. “we live just on the other side of the village there. you?” “had to get up at two didn’t we ced i tell you i’ll be glad when he’s got his apparition test. still ... not complaining ... quidditch world cup wouldn’t miss it for a sackful of galleons — and the tickets cost about that. mind you looks like i got off easy. ...” amos diggory peered good-naturedly around at the three weasley boys harry hermione and ginny. “all these yours arthur?” “oh no only the redheads,” said mr. weasley pointing out his children. “this is hermione friend of ron’s — and harry another friend — ” “merlin’s beard,” said amos diggory his eyes widening. “harry harry potter?” “er — yeah,” said harry. harry was used to people looking curiously at him when they met him used to the way their eyes moved at once to the lightning scar on his forehead but it always made him feel uncomfortable. “ced’s talked about you of course,” said amos diggory. “told us all about playing against you last year. ... i said to him i said — ced that’ll be something to tell your grandchildren that will. ... you beat harry potterl” harry couldn’t think of any reply to this so he remained silent. fred and george were both scowling again. cedric looked slightly embarrassed. “harry fell off his broom dad,” he muttered. “i told you ... it was an accident. ...” “yes but you didn’t fall off did you?” roared amos genially slapping his son on his back. “always modest our ced always the gentleman ... but the best man won i’m sure harry’d say the same wouldn’t you eh one falls off his broom one stays on you don’t need to be a genius to tell which one’s the better flier!” “must be nearly time,” said mr. weasley quickly pulling out his watch again. “do you know whether we’re waiting for any more amos?” “no the lovegoods have been there for a week already and the fawcetts couldn’t get tickets,” said mr. diggory. “there aren’t any more of us in this area are there?” “not that i know of,” said mr. weasley. “yes it’s a minute off. ... we’d better get ready. ...” he looked around at harry and hermione. “you just need to touch the portkey that’s all a finger will do — ” with difficulty owing to their bulky backpacks the nine of them crowded around the old boot held out by amos diggory. they all stood there in a tight circle as a chill breeze swept over the hilltop. nobody spoke. it suddenly occurred to harry how odd this would look if a muggle were to walk up here now ... nine people two of them grown men clutching this manky old boot in the semidarkness waiting. ... “three ...” muttered mr. weasley one eye still on his watch “two ... one ...” it happened immediately harry felt as though a hook just behind his navel had been suddenly jerked irresistibly forward. his feet left the ground he could feel ron and hermione on either side of him their shoulders banging into his they were all speeding forward in a howl of wind and swirling color his forefinger was stuck to the boot as though it was pulling him magnetically onward and then — his feet slammed into the ground ron staggered into him and he fell over the portkey hit the ground near his head with a heavy thud. harry looked up. mr. weasley mr. diggory and cedric were still standing though looking very windswept everybody else was on the ground. “seven past five from stoatshead hill,” said a voice. 7 bagman and crouch harry disentangled himself from ron and got to his feet. they had arrived on what appeared to be a deserted stretch of misty moor. in front of them was a pair of tired and grumpy-looking wizards one of whom was holding a large gold watch the other a thick roll of parchment and a quill. both were dressed as muggles though very inexpertly the man with the watch wore a tweed suit with thigh-length galoshes his colleague a kilt and a poncho. “morning basil,” said mr. weasley picking up the boot and handing it to the kilted wizard who threw it into a large box of used portkeys beside him harry could see an old newspaper an empty drinks can and a punctured football. “hello there arthur,” said basil wearily. “not on duty eh it’s all right for some. ... we’ve been here all night. ... you’d better get out of the way we’ve got a big party coming in from the black forest at five fifteen. hang on i’ll find your campsite. ... weasley ... weasley ...” he consulted his parchment list. “about a quarter of a mile’s walk over there first field you come to. site manager’s called mr. roberts. diggory ... second field ... ask for mr. payne.” “thanks basil,” said mr. weasley and he beckoned everyone to follow him. they set off across the deserted moor unable to make out much through the mist. after about twenty minutes a small stone cottage next to a gate swam into view. beyond it harry could just make out the ghostly shapes of hundreds and hundreds of tents rising up the gentle slope of a large field toward a dark wood on the horizon. they said good-bye to the diggorys and approached the cottage door. a man was standing in the doorway looking out at the tents. harry knew at a glance that this was the only real muggle for several acres. when he heard their footsteps he turned his head to look at them. “morning!” said mr. weasley brightly. “morning,” said the muggle. “would you be mr. roberts?” “aye i would,” said mr. roberts. “and who’re you?” “weasley — two tents booked a couple of days ago?” “aye,” said mr. roberts consulting a list tacked to the door. “you’ve got a space up by the wood there. just the one night?” “that’s it,” said mr. weasley. “you’ll be paying now then?” said mr. roberts. “ah — right — certainly — ” said mr. weasley. he retreated a short distance from the cottage and beckoned harry toward him. “help me harry,” he muttered pulling a roll of muggle money from his pocket and starting to peel the notes apart. “this one’s a — a — a ten ah yes i see the little number on it now. ... so this is a five?” “a twenty,” harry corrected him in an undertone uncomfortably aware of mr. roberts trying to catch every word. “ah yes so it is. ... i don’t know these little bits of paper ...” “you foreign?” said mr. roberts as mr. weasley returned with the correct notes. “foreign?” repeated mr. weasley puzzled. “you’re not the first one who’s had trouble with money,” said mr. roberts scrutinizing mr. weasley closely. “i had two try and pay me with great gold coins the size of hubcaps ten minutes ago.” “did you really?” said mr. weasley nervously. mr. roberts rummaged around in a tin for some change. “never been this crowded,” he said suddenly looking out over the misty field again. “hundreds of pre bookings. people usually just turn up. ...” “is that right?” said mr. weasley his hand held out for his change but mr. roberts didn’t give it to him. “aye,” he said thoughtfully. “people from all over. loads of foreigners. and not just foreigners. weirdos you know there’s a bloke walking ’round in a kilt and a poncho.” “shouldn’t he?” said mr. weasley anxiously. “it’s like some sort of ... i dunno ... like some sort of rally,” said mr. roberts. “they all seem to know each other. like a big party.” at that moment a wizard in plus-fours appeared out of thin air next to mr. roberts’s front door. “obliviate\” he said sharply pointing his wand at mr. roberts. instantly mr. roberts’s eyes slid out of focus his brows unknitted and a look of dreamy unconcern fell over his face. harry recognized the symptoms of one who had just had his memory modified. “a map of the campsite for you,” mr. roberts said placidly to mr. weasley. “and your change.” “thanks very much,” said mr. weasley. the wizard in plus-fours accompanied them toward the gate to the campsite. he looked exhausted his chin was blue with stubble and there were deep purple shadows under his eyes. once out of earshot of mr. roberts he muttered to mr. weasley “been having a lot of trouble with him. needs a memory charm ten times a day to keep him happy. and ludo bagman’s not helping. trotting around talking about bludgers and quaffles at the top of his voice not a worry about anti-muggle security. blimey i’ll be glad when this is over. see you later arthur.” he disapparated. “i thought mr. bagman was head of magical games and sports,” said ginny looking surprised. “he should know better than to talk about bludgers near muggles shouldn’t he?” “he should,” said mr. weasley smiling and leading them through the gates into the campsite “but ludo’s always been a bit ... well ... lax about security. you couldn’t wish for a more enthusiastic head of the sports department though. he played quidditch for england himself you know. and he was the best beater the wimbourne wasps ever had.” they trudged up the misty field between long rows of tents. most looked almost ordinary their owners had clearly tried to make them as muggle-like as possible but had slipped up by adding chimneys or bellpulls or weather vanes. however here and there was a tent so obviously magical that harry could hardly be surprised that mr. roberts was getting suspicious. halfway up the field stood an extravagant confection of striped silk like a miniature palace with several live peacocks tethered at the entrance. a little farther on they passed a tent that had three floors and several turrets and a short way beyond that was a tent that had a front garden attached complete with birdbath sundial and fountain. “always the same,” said mr. weasley smiling. “we can’t resist showing off when we get together. ah here we are look this is us.” they had reached the very edge of the wood at the top of the field and here was an empty space with a small sign hammered into the ground that read weezly. “couldn’t have a better spot!” said mr. weasley happily. “the field is just on the other side of the wood there we’re as close as we could be.” he hoisted his backpack from his shoulders. “right,” he said excitedly “no magic allowed strictly speaking not when we’re out in these numbers on muggle land. we’ll be putting these tents up by hand! shouldn’t be too difficult. ... muggles do it all the time. ... here harry where do you reckon we should start?” harry had never been camping in his life the dursleys had never taken him on any kind of holiday preferring to leave him with mrs. figg an old neighbor. however he and hermione worked out where most of the poles and pegs should go and though mr. weasley was more of a hindrance than a help because he got thoroughly overexcited when it came to using the mallet they finally managed to erect a pair of shabby two-man tents. all of them stood back to admire their handiwork. nobody looking at these tents would guess they belonged to wizards harry thought but the trouble was that once bill charlie and percy arrived they would be a party of ten. hermione seemed to have spotted this problem too she gave harry a quizzical look as mr. weasley dropped to his hands and knees and entered the first tent. “well be a bit cramped,” he called “but i think we’ll all squeeze in. come and have a look.” harry bent down ducked under the tent flap and felt his jaw drop. he had walked into what looked like an old-fashioned three-room flat complete with bathroom and kitchen. oddly enough it was furnished in exactly the same sort of style as mrs. figg’s house there were crocheted covers on the mismatched chairs and a strong smell of cats. “well it’s not for long,” said mr. weasley mopping his bald patch with a handkerchief and peering in at the four bunk beds that stood in the bedroom. “i borrowed this from perkins at the office. doesn’t camp much anymore poor fellow he’s got lumbago.” he picked up the dusty kettle and peered inside it. “well need water. ...” “there’s a tap marked on this map the muggle gave us,” said ron who had followed harry inside the tent and seemed completely unimpressed by its extraordinary inner proportions. “it’s on the other side of the field.” “well why don’t you harry and hermione go and get us some water then” — mr. weasley handed over the kettle and a couple of saucepans — “and the rest of us will get some wood for a fire?” “but we’ve got an oven,” said ron. “why can’t we just “ron anti-muggle security!” said mr. weasley his face shining with anticipation. “when real muggles camp they cook on fires outdoors. i’ve seen them at it! after a quick tour of the girls’ tent which was slightly smaller than the boys’ though without the smell of cats harry ron and hermione set off across the campsite with the kettle and saucepans. now with the sun newly risen and the mist lifting they could see the city of tents that stretched in every direction. they made their way slowly through the rows staring eagerly around. it was only just dawning on harry how many witches and wizards there must be in the world he had never really thought much about those in other countries. their fellow campers were starting to wake up. first to stir were the families with small children harry had never seen witches and wizards this young before. a tiny boy no older than two was crouched outside a large pyramid-shaped tent holding a wand and poking happily at a slug in the grass which was swelling slowly to the size of a salami. as they drew level with him his mother came hurrying out of the tent. “ how ma.ny times kevin you don’t — touch — daddy’s — wand — yecchh!” she had trodden on the giant slug which burst. her scolding carried after them on the still air mingling with the little boy’s yells — “you bust slug! you bust slug!” a short way farther on they saw two little witches barely older than kevin who were riding toy broomsticks that rose only high enough for the girls’ toes to skim the dewy grass. a ministry wizard had already spotted them as he hurried past harry ron and hermione he muttered distractedly “in broad daylight! parents having a lie-in i suppose — ” here and there adult wizards and witches were emerging from their tents and starting to cook breakfast. some with furtive looks around them conjured fires with their wands others were striking matches with dubious looks on their faces as though sure this couldn’t work. three african wizards sat in serious conversation all of them wearing long white robes and roasting what looked like a rabbit on a bright purple fire while a group of middle-aged american witches sat gossiping happily beneath a spangled banner stretched between their tents that read the salem witches’ institute. harry caught snatches of conversation in strange languages from the inside of tents they passed and though he couldn’t understand a word the tone of every single voice was excited. “er — is it my eyes or has everything gone green?” said ron. it wasn’t just ron’s eyes. they had walked into a patch of tents that were all covered with a thick growth of shamrocks so that it looked as though small oddly shaped hillocks had sprouted out of the earth. grinning faces could be seen under those that had their flaps open. then from behind them they heard their names. “harry! ron! hermione!” it was seamus finnigan their fellow gryffindor fourth year. he was sitting in front of his own shamrock covered tent with a sandy-haired woman who had to be his mother and his best friend dean thomas also of gryffindor. “like the decorations?” said seamus grinning. “the ministry’s not too happy.” “ah why shouldn’t we show our colors?” said mrs. finnigan. “you should see what the bulgarians have got dangling all over their tents. you’ll be supporting ireland of course?” she added eyeing harry ron and hermione beadily. when they had assured her that they were indeed supporting ireland they set off again though as ron said “like we’d say anything else surrounded by that lot.” “i wonder what the bulgarians have got dangling all over their tents?” said hermione. “let’s go and have a look,” said harry pointing to a large patch of tents upheld where the bulgarian flag — white green and red — was fluttering in the breeze. the tents here had not been bedecked with plant life but each and every one of them had the same poster attached to it a poster of a very surly face with heavy black eyebrows. the picture was of course moving but all it did was blink and scowl. “krum,” said ron quietly. “what?” said hermione. “krum!” said ron. “viktor krum the bulgarian seeker!” “he looks really grumpy,” said hermione looking around at the many krums blinking and scowling at them. “ ‘really grumpy’?” ron raised his eyes to the heavens. “who cares what he looks like he’s unbelievable. he’s really young too. only just eighteen or something. he’s a genius you wait until tonight you’ll see.” there was already a small queue for the tap in the corner of the field. harry ron and hermione joined it right behind a pair of men who were having a heated argument. one of them was a very old wizard who was wearing a long flowery nightgown. the other was clearly a ministry wizard he was holding out a pair of pinstriped trousers and almost crying with exasperation. “just put them on archie there’s a good chap. you can’t walk around like that the muggle at the gate’s already getting suspicious — ” “i bought this in a muggle shop,” said the old wizard stubbornly. “muggles wear them.” “muggle women wear them archie not the men they wear these,” said the ministry wizard and he brandished the pinstriped trousers. “i’m not putting them on,” said old archie in indignation. “i like a healthy breeze ’round my privates thanks.” hermione was overcome with such a strong fit of the giggles at this point that she had to duck out of the queue and only returned when archie had collected his water and moved away. walking more slowly now because of the weight of the water they made their way back through the campsite. here and there they saw more familiar faces other hogwarts students with their families. oliver wood the old captain of harry’s house quidditch team who had just left hogwarts dragged harry over to his parents’ tent to introduce him and told him excitedly that he had just been signed to the puddlemere united reserve team. next they were hailed by ernie macmillan a hufflepuff fourth year and a little farther on they saw cho chang a very pretty girl who played seeker on the ravenclaw team. she waved and smiled at harry who slopped quite a lot of water down his front as he waved back. more to stop ron from smirking than anything harry hurriedly pointed out a large group of teenagers whom he had never seen before. “who d’you reckon they are?” he said. “they don’t go to hogwarts do they?” “ ’spect they go to some foreign school,” said ron. “i know there are others. never met anyone who went to one though. bill had a penfriend at a school in brazil . . . this was years and years ago . . . and he wanted to go on an exchange trip but mum and dad couldn’t afford it. his penfriend got all offended when he said he wasn’t going and sent him a cursed hat. it made his ears shrivel up.” harry laughed but didn’t voice the amazement he felt at hearing about other wizarding schools. he supposed now that he saw representatives of so many nationalities in the campsite that he had been stupid never to realize that hogwarts couldn’t be the only one. he glanced at hermione who looked utterly unsurprised by the information. no doubt she had run across the news about other wizarding schools in some book or other. “you’ve been ages,” said george when they finally got back to the weasleys’ tents. “met a few people,” said ron setting the water down. “you not got that fire started yet?” “dad’s having fun with the matches,” said fred. mr. weasley was having no success at all in lighting the fire but it wasn’t for lack of trying. splintered matches littered the ground around him but he looked as though he was having the time of his life. “oops!” he said as he managed to light a match and promptly dropped it in surprise. “come here mr. weasley,” said hermione kindly taking the box from him and showing him how to do it properly. at last they got the fire lit though it was at least another hour before it was hot enough to cook anything. there was plenty to watch while they waited however. their tent seemed to be pitched right alongside a kind of thoroughfare to the field and ministry members kept hurrying up and down it greeting mr. weasley cordially as they passed. mr. weasley kept up a running commentary mainly for harry’s and hermione’s benefit his own children knew too much about the ministry to be greatly interested. “that was cuthbert mockridge head of the goblin liaison office. ... here comes gilbert wimple he’s with the committee on experimental charms he’s had those horns for awhile now. ... hello arnie ... arnold peasegood he’s an obliviator — member of the accidental magic reversal squad you know. ... and that’s bode and croaker ... they’re unspeakables. ...” “they’re what?” “from the department of mysteries top secret no idea what they get up to. ...” at last the fire was ready and they had just started cooking eggs and sausages when bill charlie and percy came strolling out of the woods toward them. “just apparated dad,” said percy loudly. “ah excellent lunch!” they were halfway through their plates of eggs and sausages when mr. weasley jumped to his feet waving and grinning at a man who was striding toward them. “aha!” he said. “the man of the moment! ludo!” ludo bagman was easily the most noticeable person harry had seen so far even including old archie in his flowered nightdress. he was wearing long quidditch robes in thick horizontal stripes of bright yellow and black. an enormous picture of a wasp was splashed across his chest. he had the look of a powerfully built man gone slightly to seed the robes were stretched tightly across a large belly he surely had not had in the days when he had played quidditch for england. his nose was squashed probably broken by a stray bludger harry thought but his round blue eyes short blond hair and rosy complexion made him look like a very overgrown schoolboy. “ahoy there!” bagman called happily. he was walking as though he had springs attached to the balls of his feet and was plainly in a state of wild excitement. “arthur old man,” he puffed as he reached the campfire “what a day eh what a day! could we have asked for more perfect weather a cloudless night coming . . . and hardly a hiccough in the arrangements. ... not much for me to do!” behind him a group of haggard-looking ministry wizards rushed past pointing at the distant evidence of some sort of a magical fire that was sending violet sparks twenty feet into the air. percy hurried forward with his hand outstretched. apparently his disapproval of the way ludo bagman ran his department did not prevent him from wanting to make a good impression. “ah — yes,” said mr. weasley grinning “this is my son percy. he’s just started at the ministry — and this is fred — no george sorry — that’s fred — bill charlie ron — my daughter ginny — and ron’s friends hermione granger and harry potter.” bagman did the smallest of double takes when he heard harry’s name and his eyes performed the familiar flick upward to the scar on harry’s forehead. “everyone,” mr. weasley continued “this is ludo bagman you know who he is it’s thanks to him we’ve got such good tickets — ” bagman beamed and waved his hand as if to say it had been nothing. “fancy a flutter on the match arthur?” he said eagerly jingling what seemed to be a large amount of gold in the pockets of his yellow-and-black robes. “i’ve already got roddy pontner betting me bulgaria will score first — i offered him nice odds considering ireland’s front three are the strongest i’ve seen in years — and little agatha timms has put up half shares in her eel farm on a week-long match.” “oh ... go on then,” said mr. weasley. “let’s see ... a galleon on ireland to win?” “a galleon?” ludo bagman looked slightly disappointed but recovered himself. “very well very well . . . any other takers?” “they’re a bit young to be gambling,” said mr. weasley. “molly wouldn’t like — ” “well bet thirty-seven galleons fifteen sickles three knuts,” said fred as he and george quickly pooled all their money “that ireland wins — but viktor krum gets the snitch. oh and we’ll throw in a fake wand.” “you don’t want to go showing mr. bagman rubbish like that — ” percy hissed but bagman didn’t seem to think the wand was rubbish at all on the contrary his boyish face shone with excitement as he took it from fred and when the wand gave a loud squawk and turned into a rubber chicken bagman roared with laughter. “excellent! i haven’t seen one that convincing in years! i’d pay five galleons for that!” percy froze in an attitude of stunned disapproval. “boys,” said mr. weasley under his breath “i don’t want you betting. ... that’s all your savings. ... your mother — ” “don’t be a spoilsport arthur!” boomed ludo bagman rattling his pockets excitedly. “they’re old enough to know what they want! you reckon ireland will win but krum ’ll get the snitch not a chance boys not a chance. ... i’ll give you excellent odds on that one. ... we’ll add five galleons for the funny wand then shall we. ...” mr. weasley looked on helplessly as ludo bagman whipped out a notebook and quill and began jotting down the twins’ names. “cheers,” said george taking the slip of parchment bagman handed him and tucking it away carefully. bagman turned most cheerfully back to mr. weasley. “couldn’t do me a brew i suppose i’m keeping an eye out for barty crouch. my bulgarian opposite number’s making difficulties and i can’t understand a word he’s saying. barty’ll be able to sort it out. he speaks about a hundred and fifty languages.” “mr. crouch?” said percy suddenly abandoning his look of poker-stiff disapproval and positively writhing with excitement. “he speaks over two hundred! mermish and gobbledegook and troll ...” “anyone can speak troll,” said fred dismissively. “all you have to do is point and grunt.” percy threw fred an extremely nasty look and stoked the fire vigorously to bring the kettle back to the boil. “any news of bertha jorkins yet ludo?” mr. weasley asked as bagman settled himself down on the grass beside them all. “not a dicky bird,” said bagman comfortably. “but she’ll turn up. poor old bertha ... memory like a leaky cauldron and no sense of direction. lost you take my word for it. she’ll wander back into the office sometime in october thinking it’s still july.” “you don’t think it might be time to send someone to look for her?” mr. weasley suggested tentatively as percy handed bagman his tea. “barty crouch keeps saying that,” said bagman his round eyes widening innocently “but we really can’t spare anyone at the moment. oh — talk of the devil! barty!” a wizard had just apparated at their fireside and he could not have made more of a contrast with ludo bagman sprawled on the grass in his old wasp robes. barty crouch was a stiff upright elderly man dressed in an impeccably crisp suit and tie. the parting in his short gray hair was almost unnaturally straight and his narrow toothbrush mustache looked as though he trimmed it using a slide rule. his shoes were very highly polished. harry could see at once why percy idolized him. percy was a great believer in rigidly following rules and mr. crouch had complied with the rule about muggle dressing so thoroughly that he could have passed for a bank manager harry doubted even uncle vernon would have spotted him for what he really was. “pull up a bit of grass barty,” said ludo brightly patting the ground beside him. “no thank you ludo,” said crouch and there was a bite of impatience in his voice. “i’ve been looking for you everywhere. the bulgarians are insisting we add another twelve seats to the top box.” “oh is that what they’re after?” said bagman. “i thought the chap was asking to borrow a pair of tweezers. bit of a strong accent.” “mr. crouch!” said percy breathlessly sunk into a kind of half-bow that made him look like a hunchback. “would you like a cup of tea?” “oh,” said mr. crouch looking over at percy in mild surprise. “yes — thank you weatherby.” fred and george choked into their own cups. percy very pink around the ears busied himself with the kettle. “oh and i’ve been wanting a word with you too arthur,” said mr. crouch his sharp eyes falling upon mr. weasley. “ali bashir’s on the warpath. he wants a word with you about your embargo on flying carpets.” mr. weasley heaved a deep sigh. “i sent him an owl about that just last week. if i’ve told him once i’ve told him a hundred times carpets are defined as a muggle artifact by the registry of proscribed charmable objects but will he listen?” “i doubt it,” said mr. crouch accepting a cup from percy. “he’s desperate to export here.” “well they’ll never replace brooms in britain will they?” said bagman. “ali thinks there’s a niche in the market for a family vehicle,” said mr. crouch. “i remember my grandfather had an axminster that could seat twelve — but that was before carpets were banned of course.” he spoke as though he wanted to leave nobody in any doubt that all his ancestors had abided strictly by the law. “so been keeping busy barty?” said bagman breezily. “fairly,” said mr. crouch dryly. “organizing portkeys across five continents is no mean feat ludo.” “i expect you’ll both be glad when this is over?” said mr. weasley. ludo bagman looked shocked. “glad! don’t know when i’ve had more fun. ... still it’s not as though we haven’t got anything to look forward to eh barty eh plenty left to organize eh?” mr. crouch raised his eyebrows at bagman. “we agreed not to make the announcement until all the details — ” “oh details!” said bagman waving the word away like a cloud of midges. “they’ve signed haven’t they they’ve agreed haven’t they i bet you anything these kids’ll know soon enough anyway. i mean it’s happening at hogwarts — ” “ludo we need to meet the bulgarians you know,” said mr. crouch sharply cutting bagman’s remarks short. “thank you for the tea weatherby.” he pushed his undrunk tea back at percy and waited for ludo to rise bagman struggled to his feet swigging down the last of his tea the gold in his pockets chinking merrily. “see you all later!” he said. “you’ll be up in the top box with me — i’m commentating!” he waved barty crouch nodded curtly and both of them disapparated. “what’s happening at hogwarts dad?” said fred at once. “what were they talking about?” “you’ll find out soon enough,” said mr.weasley smiling. “it’s classified information until such time as the ministry decides to release it,” said percy stiffly. “mr. crouch was quite right not to disclose it.” “oh shut up weatherby,” said fred. a sense of excitement rose like a palpable cloud over the campsite as the afternoon wore on. by dusk the still summer air itself seemed to be quivering with anticipation and as darkness spread like a curtain over the thousands of waiting wizards the last vestiges of pretence disappeared the ministry seemed to have bowed to the inevitable and stopped fighting the signs of blatant magic now breaking out everywhere. salesmen were apparating every few feet carrying trays and pushing carts full of extraordinary merchandise. there were luminous rosettes — green for ireland red for bulgaria — which were squealing the names of the players pointed green hats bedecked with dancing shamrocks bulgarian scarves adorned with lions that really roared flags from both countries that played their national anthems as they were waved there were tiny models of firebolts that really flew and collectible figures of famous players which strolled across the palm of your hand preening themselves. “been saving my pocket money all summer for this,” ron told harry as they and hermione strolled through the salesmen buying souvenirs. though ron purchased a dancing shamrock hat and a large green rosette he also bought a small figure of viktor krum the bulgarian seeker. the miniature krum walked backward and forward over ron’s hand scowling up at the green rosette above him. “wow look at these!” said harry hurrying over to a cart piled high with what looked like brass binoculars except that they were covered with all sorts of weird knobs and dials. “omnioculars,” said the saleswizard eagerly. “you can replay action . . . slow everything down . . . and they flash up a play-by-play breakdown if you need it. bargain — ten galleons each.” “wish i hadn’t bought this now,” said ron gesturing at his dancing shamrock hat and gazing longingly at the omnioculars. “three pairs,” said harry firmly to the wizard. “no — don’t bother,” said ron going red. he was always touchy about the fact that harry who had inherited a small fortune from his parents had much more money than he did. “you won’t be getting anything for christmas,” harry told him thrusting omnioculars into his and hermione’s hands. “for about ten years mind.” “fair enough,” said ron grinning. “oooh thanks harry,” said hermione. “and i’ll get us some programs look — ” their money bags considerably lighter they went back to the tents. bill charlie and ginny were all sporting green rosettes too and mr. weasley was carrying an irish flag. fred and george had no souvenirs as they had given bagman all their gold. and then a deep booming gong sounded somewhere beyond the woods and at once green and red lanterns blazed into life in the trees lighting a path to the field. “it’s time!” said mr. weasley looking as excited as any of them. “come on let’s go!” 4 the quidditch world cup clutching their purchases mr. weasley in the lead they all hurried into the wood following the lantern lit trail. they could hear the sounds of thousands of people moving around them shouts and laughter snatches of singing. the atmosphere of feverish excitement was highly infectious harry couldn’t stop grinning. they walked through the wood for twenty minutes talking and joking loudly until at last they emerged on the other side and found themselves in the shadow of a gigantic stadium. though harry could see only a fraction of the immense gold walls surrounding the field he could tell that ten cathedrals would fit comfortably inside it. “seats a hundred thousand,” said mr. weasley spotting the awestruck look on harry’s face. “ministry task force of five hundred have been working on it all year. muggle repelling charms on every inch of it. every time muggles have got anywhere near here all year they’ve suddenly remembered urgent appointments and had to dash away again . . . bless them,” he added fondly leading the way toward the nearest entrance which was already surrounded by a swarm of shouting witches and wizards. “prime seats!” said the ministry witch at the entrance when she checked their tickets. “top box! straight upstairs arthur and as high as you can go.” the stairs into the stadium were carpeted in rich purple. they clambered upward with the rest of the crowd which slowly filtered away through doors into the stands to their left and right. mr. weasley’s party kept climbing and at last they reached the top of the staircase and found themselves in a small box set at the highest point of the stadium and situated exactly halfway between the golden goal posts. about twenty purple-and-gilt chairs stood in two rows here and harry filing into the front seats with the weasleys looked down upon a scene the likes of which he could never have imagined. a hundred thousand witches and wizards were taking their places in the seats which rose in levels around the long oval field. everything was suffused with a mysterious golden light which seemed to come from the stadium itself. the field looked smooth as velvet from their lofty position. at either end of the field stood three goal hoops fifty feet high right opposite them almost at harry’s eye level was a gigantic blackboard. gold writing kept dashing across it as though an invisible giant’s hand were scrawling upon the blackboard and then wiping it off again watching it harry saw that it was flashing advertisements across the field. the bluebottle a broom for all the family — safe reliable and with built-in anti-burglar buzzer ... mrs. skower’s all-purpose magical mess remover no pain no stain! ... gladrags wizardwear — london paris hogsmeade ... harry tore his eyes away from the sign and looked over his shoulder to see who else was sharing the box with them. so far it was empty except for a tiny creature sitting in the second from last seat at the end of the row behind them. the creature whose legs were so short they stuck out in front of it on the chair was wearing a tea towel draped like a toga and it had its face hidden in its hands. yet those long batlike ears were oddly familiar. . . . “ dobby ” said harry incredulously. the tiny creature looked up and stretched its fingers revealing enormous brown eyes and a nose the exact size and shape of a large tomato. it wasn’t dobby — it was however unmistakably a house-elf as harry’s friend dobby had been. harry had set dobby free from his old owners the malfoy family. “did sir just call me dobby?” squeaked the elf curiously from between its fingers. its voice was higher even than dobby’s had been a teeny quivering squeak of a voice and harry suspected — though it was very hard to tell with a house-elf — that this one might just be female. ron and hermione spun around in their seats to look. though they had heard a lot about dobby from harry they had never actually met him. even mr. weasley looked around in interest. “sorry,” harry told the elf “i just thought you were someone i knew.” “but i knows dobby too sir!” squeaked the elf. she was shielding her face as though blinded by light though the top box was not brightly lit. “my name is winky sir — and you sir — ” her dark brown eyes widened to the size of side plates as they rested upon harry’s scar. “you is surely harry potter!” “yeah i am,” said harry. “but dobby talks of you all the time sir!” she said lowering her hands very slightly and looking awestruck. “how is he?” said harry. “how’s freedom suiting him?” “ah sir,” said winky shaking her head “ah sir meaning no disrespect sir but i is not sure you did dobby a favor sir when you is setting him free.” “why?” said harry taken aback. “what’s wrong with him?” “freedom is going to dobby’s head sir,” said winky sadly. “ideas above his station sir. can’t get another position sir.” “why not?” said harry. winky lowered her voice by a half-octave and whispered “he is wanting paying for his work sir.” “paying?” said harry blankly. “well — why shouldn’t he be paid?” winky looked quite horrified at the idea and closed her fingers slightly so that her face was half-hidden again. “house-elves is not paid sir!” she said in a muffled squeak. “no no no. i says to dobby i says go find yourself a nice family and settle down dobby. he is getting up to all sorts of high jinks sir what is unbecoming to a house-elf. you goes racketing around like this dobby i says and next thing i hear you’s up in front of the department for the regulation and control of magical creatures like some common goblin.” “well it’s about time he had a bit of fun,” said harry. “house-elves is not supposed to have fun harry potter,” said winky firmly from behind her hands. “house-elves does what they is told. i is not liking heights at all harry potter” — she glanced toward the edge of the box and gulped — “but my master sends me to the top box and i comes sir.” “why’s he sent you up here if he knows you don’t like heights?” said harry frowning. “master — master wants me to save him a seat harry potter. he is very busy,” said winky tilting her head toward the empty space beside her. “winky is wishing she is back in master’s tent harry potter but winky does what she is told. winky is a good house-elf.” she gave the edge of the box another frightened look and hid her eyes completely again. harry turned back to the others. “so that’s a house-elf?” ron muttered. “weird things aren’t they?” “dobby was weirder,” said harry fervently. ron pulled out his omnioculars and started testing them staring down into the crowd on the other side of the stadium. “wild!” he said twiddling the replay knob on the side. “i can make that old bloke down there pick his nose again ... and again ... and again ...” hermione meanwhile was skimming eagerly through her velvet-covered tasseled program. “ ‘a display from the team mascots will precede the match,’ ” she read aloud. “oh that’s always worth watching,” said mr. weasley. “national teams bring creatures from their native land you know to put on a bit of a show.” the box filled gradually around them over the next half hour. mr. weasley kept shaking hands with people who were obviously very important wizards. percy jumped to his feet so often that he looked as though he were trying to sit on a hedgehog. when cornelius fudge the minister of magic himself arrived percy bowed so low that his glasses fell off and shattered. highly embarrassed he repaired them with his wand and thereafter remained in his seat throwing jealous looks at harry whom cornelius fudge had greeted like an old friend. they had met before and fudge shook harry’s hand in a fatherly fashion asked how he was and introduced him to the wizards on either side of him. “harry potter you know,” he told the bulgarian minister loudly who was wearing splendid robes of black velvet trimmed with gold and didn’t seem to understand a word of english. “harry potter ... oh come on now you know who he is . . . the boy who survived you-know-who ... you do know who he is — ” the bulgarian wizard suddenly spotted harry’s scar and started gabbling loudly and excitedly pointing at it. “knew we’d get there in the end,” said fudge wearily to harry. “i’m no great shakes at languages i need barty crouch for this sort of thing. ah i see his house-elf’s saving him a seat. ... good job too these bulgarian blighters have been trying to cadge all the best places ... ah and here’s lucius!” harry ron and hermione turned quickly. edging along the second row to three still-empty seats right behind mr. weasley were none other than dobby the house-elf’s former owners lucius malfoy his son draco and a woman harry supposed must be draco’s mother. harry and draco malfoy had been enemies ever since their very first journey to hogwarts. a pale boy with a pointed face and white-blond hair draco greatly resembled his father. his mother was blonde too tall and slim she would have been nice-looking if she hadn’t been wearing a look that suggested there was a nasty smell under her nose. “ah fudge,” said mr. malfoy holding out his hand as he reached the minister of magic. “how are you i don’t think you’ve met my wife narcissa or our son draco?” “how do you do how do you do?” said fudge smiling and bowing to mrs. malfoy. “and allow me to introduce you to mr. oblansk — obalonsk — mr. — well he’s the bulgarian minister of magic and he can’t understand a word i’m saying anyway so never mind. and let’s see who else — you know arthur weasley i daresay?” it was a tense moment. mr. weasley and mr. malfoy looked at each other and harry vividly recalled the last time they had come face-to-face it had been in flourish and blotts’ bookshop and they had had a fight. mr. malfoy’s cold gray eyes swept over mr. weasley and then up and down the row. “good lord arthur,” he said softly. “what did you have to sell to get seats in the top box surely your house wouldn’t have fetched this much?” fudge who wasn’t listening said “lucius has just given a very generous contribution to st. mungo’s hospital for magical maladies and injuries arthur. he’s here as my guest.” “how — how nice,” said mr. weasley with a very strained smile. mr. malfoy’s eyes had returned to hermione who went slightly pink but stared determinedly back at him. harry knew exactly what was making mr. malfoy’s lip curl like that. the malfoys prided themselves on being purebloods in other words they considered anyone of muggle descent like hermione second-class. however under the gaze of the minister of magic mr. malfoy didn’t dare say anything. he nodded sneeringly to mr. weasley and continued down the line to his seats. draco shot harry ron and hermione one contemptuous look then settled himself between his mother and father. “slimy gits,” ron muttered as he harry and hermione turned to face the field again. next moment ludo bagman charged into the box. “everyone ready?” he said his round face gleaming like a great excited edam. “minister — ready to go?” “ready when you are ludo,” said fudge comfortably. ludo whipped out his wand directed it at his own throat and said “sonorusl” and then spoke over the roar of sound that was now filling the packed stadium his voice echoed over them booming into every corner of the stands. “ladies and gentlemen ... welcome! welcome to the final of the four hundred and twenty-second quidditch world cup!” the spectators screamed and clapped. thousands of flags waved adding their discordant national anthems to the racket. the huge blackboard opposite them was wiped clear of its last message bertie bott’s every flavor beans — a risk with every mouthfull and now showed bulgaria 0 ireland 0. “and now without further ado allow me to introduce ... the bulgarian national team mascots!” the right-hand side of the stands which was a solid block of scarlet roared its approval. “i wonder what they’ve brought,” said mr. weasley leaning forward in his seat. “aaah!” he suddenly whipped off his glasses and polished them hurriedly on his robes. “veelal” “what are veel — ” but a hundred veela were now gliding out onto the field and harry’s question was answered for him. veela were women . . . the most beautiful women harry had ever seen ... except that they weren’t — they couldn’t be — human. this puzzled harry for a moment while he tried to guess what exactly they could be what could make their skin shine moon bright like that or their white-gold hair fan out behind them without wind . . . but then the music started and harry stopped worrying about them not being human — in fact he stopped worrying about anything at all. the veela had started to dance and harry’s mind had gone completely and blissfully blank. all that mattered in the world was that he kept watching the veela because if they stopped dancing terrible things would happen. ... and as the veela danced faster and faster wild half formed thoughts started chasing through harry’s dazed mind. he wanted to do something very impressive right now. jumping from the box into the stadium seemed a good idea . . . but would it be good enough “harry what are you doing?” said hermione’s voice from a long way off. the music stopped. harry blinked. he was standing up and one of his legs was resting on the wall of the box. next to him ron was frozen in an attitude that looked as though he were about to dive from a springboard. angry yells were filling the stadium. the crowd didn’t want the veela to go. harry was with them he would of course be supporting bulgaria and he wondered vaguely why he had a large green shamrock pinned to his chest. ron meanwhile was absent-mindedly shredding the shamrocks on his hat. mr. weasley smiling slightly leaned over to ron and tugged the hat out of his hands. “you’ll be wanting that,” he said “once ireland have had their say.” “huh?” said ron staring openmouthed at the veela who had now lined up along one side of the field. hermione made a loud tutting noise. she reached up and pulled harry back into his seat. “honestlyl” she said. “and now,” roared ludo bagman’s voice “kindly put your wands in the air ... for the irish national team mascots!” next moment what seemed to be a great green-and gold comet came zooming into the stadium. it did one circuit of the stadium then split into two smaller comets each hurtling toward the goal posts. a rainbow arced suddenly across the field connecting the two balls of light. the crowd oooohed and aaaaahed as though at a fireworks display. now the rainbow faded and the balls of light reunited and merged they had formed a great shimmering shamrock which rose up into the sky and began to soar over the stands. something like golden rain seemed to be falling from it — “excellent!” yelled ron as the shamrock soared over them and heavy gold coins rained from it bouncing off their heads and seats. squinting up at the shamrock harry realized that it was actually comprised of thousands of tiny little bearded men with red vests each carrying a minute lamp of gold or green. “leprechauns!” said mr. weasley over the tumultuous applause of the crowd many of whom were still fighting and rummaging around under their chairs to retrieve the gold. “there you go,” ron yelled happily stuffing a fistful of gold coins into harry’s hand “for the omnioculars! now you’ve got to buy me a christmas present ha!” the great shamrock dissolved the leprechauns drifted down onto the field on the opposite side from the veela and settled themselves cross-legged to watch the match. “and now ladies and gentlemen kindly welcome — the bulgarian national quidditch team! i give you — dimitrov!” a scarlet-clad figure on a broomstick moving so fast it was blurred shot out onto the field from an entrance far below to wild applause from the bulgarian supporters. “ivanova!” a second scarlet-robed player zoomed out. “zograf! levski! vulchanov! volkov! aaaaaaand — krum\” “that’s him that’s him!” yelled ron following krum with his omnioculars. harry quickly focused his own. viktor krum was thin dark and sallow-skinned with a large curved nose and thick black eyebrows. he looked like an overgrown bird of prey. it was hard to believe he was only eighteen. “and now please greet — the irish national quidditch team!” yelled bagman. “presenting — connolly! ryan! troy! mullet! moran! quigley! aaaaaand — lynch).” seven green blurs swept onto the field harry spun a small dial on the side of his omnioculars and slowed the players down enough to read the word “firebolt” on each of their brooms and see their names embroidered in silver upon their backs. “and here all the way from egypt our referee acclaimed chairwizard of the international association of quidditch hassan mostafa!” a small and skinny wizard completely bald but with a mustache to rival uncle vernon’s wearing robes of pure gold to match the stadium strode out onto the field. a silver whistle was protruding from under the mustache and he was carrying a large wooden crate under one arm his broomstick under the other. harry spun the speed dial on his omnioculars back to normal watching closely as mostafa mounted his broomstick and kicked the crate open — four balls burst into the air the scarlet quaffle the two black bludgers and harry saw it for the briefest moment before it sped out of sight the minuscule winged golden snitch. with a sharp blast on his whistle mostafa shot into the air after the balls. “theeeeeeeey’re off!” screamed bagman. “and it’s mullet! troy! moran! dimitrov! back to mullet! troy! levski! moran!” it was quidditch as harry had never seen it played before. he was pressing his omnioculars so hard to his glasses that they were cutting into the bridge of his nose. the speed of the players was incredible — the chasers were throwing the quaffle to one another so fast that bagman only had time to say their names. harry spun the slow dial on the right of his omnioculars again pressed the play-by-play button on the top and he was immediately watching in slow motion while glittering purple lettering flashed across the lenses and the noise of the crowd pounded against his eardrums. hawkshead attacking formation he read as he watched the three irish chasers zoom closely together troy in the center slightly ahead of mullet and moran bearing down upon the bulgarians. porskoff ploy flashed up next as troy made as though to dart upward with the quaffle drawing away the bulgarian chaser ivanova and dropping the quaffle to moran. one of the bulgarian beaters volkov swung hard at a passing bludger with his small club knocking it into moran’s path moran ducked to avoid the bludger and dropped the quaffle and levski soaring beneath caught it — “troy scores!” roared bagman and the stadium shuddered with a roar of applause and cheers. “ten zero to ireland!” “what?” harry yelled looking wildly around through his omnioculars. “but levski’s got the quaffle!” “harry if you’re not going to watch at normal speed you’re going to miss things!” shouted hermione who was dancing up and down waving her arms in the air while troy did a lap of honor around the field. harry looked quickly over the top of his omni-oculars and saw that the leprechauns watching from the sidelines had all risen into the air again and formed the great glittering shamrock. across the field the veela were watching them sulkily. furious with himself harry spun his speed dial back to normal as play resumed. harry knew enough about quidditch to see that the irish chasers were superb. they worked as a seamless team their movements so well coordinated that they appeared to be reading one another’s minds as they positioned themselves and the rosette on harry’s chest kept squeaking their names “troy — mullet — moran).” and within ten minutes ireland had scored twice more bringing their lead to thirty-zero and causing a thunderous tide of roars and applause from the green-clad supporters. the match became still faster but more brutal. volkov and vulchanov the bulgarian beaters were whacking the bludgers as fiercely as possible at the irish chasers and were starting to prevent them from using some of their best moves twice they were forced to scatter and then finally ivanova managed to break through their ranks dodge the keeper ryan and score bulgaria’s first goal. “fingers in your ears!” bellowed mr. weasley as the veela started to dance in celebration. harry screwed up his eyes too he wanted to keep his mind on the game. after a few seconds he chanced a glance at the field. the veela had stopped dancing and bulgaria was again in possession of the quaffle. “dimitrov! levski! dimitrov! ivanova — oh i say!” roared bagman. one hundred thousand wizards gasped as the two seekers krum and lynch plummeted through the center of the chasers so fast that it looked as though they had just jumped from airplanes without parachutes. harry followed their descent through his omnioculars squinting to see where the snitch was “they’re going to crash!” screamed hermione next to harry. she was half right — at the very last second viktor krum pulled out of the dive and spiraled off. lynch however hit the ground with a dull thud that could be heard throughout the stadium. a huge groan rose from the irish seats. “fool!” moaned mr. weasley. “krum was feinting!” “it’s time-out!” yelled bagman’s voice “as trained mediwizards hurry onto the field to examine aidan lynch!” “he’ll be okay he only got ploughed!” charlie said reassuringly to ginny who was hanging over the side of the box looking horror-struck. “which is what krum was after of course. ...” harry hastily pressed the replay and play-by-play buttons on his omnioculars twiddled the speed dial and put them back up to his eyes. he watched as krum and lynch dived again in slow motion. wronski defensive feint — dangerous seeker diversion read the shining purple lettering across his lenses. he saw krum’s face contorted with concentration as he pulled out of the dive just in time while lynch was flattened and he understood — krum hadn’t seen the snitch at all he was just making lynch copy him. harry had never seen anyone fly like that krum hardly looked as though he was using a broomstick at all he moved so easily through the air that he looked unsupported and weightless. harry turned his omnioculars back to normal and focused them on krum. he was now circling high above lynch who was being revived by mediwizards with cups of potion. harry focusing still more closely upon krum’s face saw his dark eyes darting all over the ground a hundred feet below. he was using the time while lynch was revived to look for the snitch without interference. lynch got to his feet at last to loud cheers from the green-clad supporters mounted his firebolt and kicked back off into the air. his revival seemed to give ireland new heart. when mostafa blew his whistle again the chasers moved into action with a skill unrivaled by anything harry had seen so far. after fifteen more fast and furious minutes ireland had pulled ahead by ten more goals. they were now leading by one hundred and thirty points to ten and the game was starting to get dirtier. as mullet shot toward the goal posts yet again clutching the quaffle tightly under her arm the bulgarian keeper zograf flew out to meet her. whatever happened was over so quickly harry didn’t catch it but a scream of rage from the irish crowd and mostafa’s long shrill whistle blast told him it had been a foul. “and mostafa takes the bulgarian keeper to task for cobbing — excessive use of elbows!” bagman informed the roaring spectators. “and — yes it’s a penalty to ireland!” the leprechauns who had risen angrily into the air like a swarm of glittering hornets when mullet had been fouled now darted together to form the words “ha ha ha!” the veela on the other side of the field leapt to their feet tossed their hair angrily and started to dance again. as one the weasley boys and harry stuffed their fingers into their ears but hermione who hadn’t bothered was soon tugging on harry’s arm. he turned to look at her and she pulled his fingers impatiently out of his ears. “look at the referee!” she said giggling. harry looked down at the field. hassan mostafa had landed right in front of the dancing veela and was acting very oddly indeed. he was flexing his muscles and smoothing his mustache excitedly. “now we can’t have that!” said ludo bagman though he sounded highly amused. “somebody slap the referee!” a mediwizard came tearing across the field his fingers stuffed into his own ears and kicked mostafa hard in the shins. mostafa seemed to come to himself harry watching through the omnioculars again saw that he looked exceptionally embarrassed and had started shouting at the veela who had stopped dancing and were looking mutinous. “and unless i’m much mistaken mostafa is actually attempting to send off the bulgarian team mascots!” said bagman’s voice. “now there’s something we haven’t seen before. ... oh this could turn nasty. ...” it did the bulgarian beaters volkov and vulchanov landed on either side of mostafa and began arguing furiously with him gesticulating toward the leprechauns who had now gleefully formed the words “hee hee hee.” mostafa was not impressed by the bulgarians’ arguments however he was jabbing his finger into the air clearly telling them to get flying again and when they refused he gave two short blasts on his whistle. “two penalties for ireland!” shouted bagman and the bulgarian crowd howled with anger. “and volkov and vulchanov had better get back on those brooms . . . yes ... there they go ... and troy takes the quaffle ...” play now reached a level of ferocity beyond anything they had yet seen. the beaters on both sides were acting without mercy volkov and vulchanov in particular seemed not to care whether their clubs made contact with bludger or human as they swung them violently through the air. dimitrov shot straight at moran who had the quaffle nearly knocking her off her broom. “foul\” roared the irish supporters as one all standing up in a great wave of green. “foul!” echoed ludo bagman’s magically magnified voice. “dimitrov skins moran — deliberately flying to collide there — and it’s got to be another penalty — yes there’s the whistle!” the leprechauns had risen into the air again and this time they formed a giant hand which was making a very rude sign indeed at the veela across the field. at this the veela lost control. instead of dancing they launched themselves across the field and began throwing what seemed to be handfuls of fire at the leprechauns. watching through his omnioculars harry saw that they didn’t look remotely beautiful now. on the contrary their faces were elongating into sharp cruel-beaked bird heads and long scaly wings were bursting from their shoulders — “and that boys,” yelled mr. weasley over the tumult of the crowd below “is why you should never go for looks alone!” ministry wizards were flooding onto the field to separate the veela and the leprechauns but with little success meanwhile the pitched battle below was nothing to the one taking place above. harry turned this way and that staring through his omnioculars as the quaffle changed hands with the speed of a bullet. “levski — dimitrov — moran — troy — mullet — ivanova — moran again — moran — moran scores!” but the cheers of the irish supporters were barely heard over the shrieks of the veela the blasts now issuing from the ministry members’ wands and the furious roars of the bulgarians. the game recommenced immediately now levski had the quaffle now dimitrov — the irish beater quigley swung heavily at a passing bludger and hit it as hard as possible toward krum who did not duck quickly enough. it hit him full in the face. there was a deafening groan from the crowd krum’s nose looked broken there was blood everywhere but hassan mostafa didn’t blow his whistle. he had become distracted and harry couldn’t blame him one of the veela had thrown a handful of fire and set his broom tail alight. harry wanted someone to realize that krum was injured even though he was supporting ireland krum was the most exciting player on the field. ron obviously felt the same. “time-out! ah come on he can’t play like that look at him — ” “look at lynch 1” harry yelled. for the irish seeker had suddenly gone into a dive and harry was quite sure that this was no wronski feint this was the real thing. ... “he’s seen the snitch!” harry shouted. “he’s seen it! look at him go!” half the crowd seemed to have realized what was happening the irish supporters rose in another great wave of green screaming their seeker on . . . but krum was on his tail. how he could see where he was going harry had no idea there were flecks of blood flying through the air behind him but he was drawing level with lynch now as the pair of them hurtled toward the ground again — “they’re going to crash!” shrieked hermione. “they’re not!” roared ron. “lynch is!” yelled harry. and he was right — for the second time lynch hit the ground with tremendous force and was immediately stampeded by a horde of angry veela. “the snitch where’s the snitch?” bellowed charlie along the row. “he’s got it — krum’s got it — it’s all over!” shouted harry. krum his red robes shining with blood from his nose was rising gently into the air his fist held high a glint of gold in his hand. the scoreboard was flashing bulgaria 160 ireland 170 across the crowd who didn’t seem to have realized what had happened. then slowly as though a great jumbo jet were revving up the rumbling from the ireland supporters grew louder and louder and erupted into screams of delight. “ireland wins!” bagman shouted who like the irish seemed to be taken aback by the sudden end of the match. “krum gets the snitch — but ireland wins — good lord i don’t think any of us were expecting that!” “what did he catch the snitch for?” ron bellowed even as he jumped up and down applauding with his hands over his head. “he ended it when ireland were a hundred and sixty points ahead the idiot!” “he knew they were never going to catch up!” harry shouted back over all the noise also applauding loudly. “the irish chasers were too good. ... he wanted to end it on his terms that’s all. ...” “he was very brave wasn’t he?” hermione said leaning forward to watch krum land as a swarm of mediwizards blasted a path through the battling leprechauns and veela to get to him. “he looks a terrible mess. ...” harry put his omnioculars to his eyes again. it was hard to see what was happening below because leprechauns were zooming delightedly all over the field but he could just make out krum surrounded by mediwizards. he looked surlier than ever and refused to let them mop him up. his team members were around him shaking their heads and looking dejected a short way away the irish players were dancing gleefully in a shower of gold descending from their mascots. flags were waving all over the stadium the irish national anthem blared from all sides the veela were shrinking back into their usual beautiful selves now though looking dispirited and forlorn. “veil ve fought bravely,” said a gloomy voice behind harry. he looked around it was the bulgarian minister of magic. “you can speak english!” said fudge sounding outraged. “and you’ve been letting me mime everything all day!” “veil it vos very funny,” said the bulgarian minister shrugging. “and as the irish team performs a lap of honor flanked by their mascots the quidditch world cup itself is brought into the top box!” roared bagman. harry’s eyes were suddenly dazzled by a blinding white light as the top box was magically illuminated so that everyone in the stands could see the inside. squinting toward the entrance he saw two panting wizards carrying a vast golden cup into the box which they handed to cornelius fudge who was still looking very disgruntled that he’d been using sign language all day for nothing. “let’s have a really loud hand for the gallant losers — bulgaria!” bagman shouted. and up the stairs into the box came the seven defeated bulgarian players. the crowd below was applauding appreciatively harry could see thousands and thousands of omniocular lenses flashing and winking in their direction. one by one the bulgarians filed between the rows of seats in the box and bagman called out the name of each as they shook hands with their own minister and then with fudge. krum who was last in line looked a real mess. two black eyes were blooming spectacularly on his bloody face. he was still holding the snitch. harry noticed that he seemed much less coordinated on the ground. he was slightly duck footed and distinctly round-shouldered. but when krum’s name was announced the whole stadium gave him a resounding earsplitting roar. and then came the irish team. aidan lynch was being supported by moran and connolly the second crash seemed to have dazed him and his eyes looked strangely unfocused. but he grinned happily as troy and quigley lifted the cup into the air and the crowd below thundered its approval. harry’s hands were numb with clapping. at last when the irish team had left the box to perform another lap of honor on their brooms aidan lynch on the back of connolly’s clutching hard around his waist and still grinning in a bemused sort of way bagman pointed his wand at his throat and muttered “quietus.” “they’ll be talking about this one for years,” he said hoarsely “a really unexpected twist that. ... shame it couldn’t have lasted longer. ... ah yes. ... yes i owe you . . . how much?” for fred and george had just scrambled over the backs of their seats and were standing in front of ludo bagman with broad grins on their faces their hands outstretched. 9 the dark mark “ don’t tell your mother you’ve been gambling,” mr. weasley implored fred and george as they all made their way slowly down the purple-carpeted stairs. “don’t worry dad,” said fred gleefully “we’ve got big plans for this money. we don’t want it confiscated.” mr. weasley looked for a moment as though he was going to ask what these big plans were but seemed to decide upon reflection that he didn’t want to know. they were soon caught up in the crowds now flooding out of the stadium and back to their campsites. raucous singing was borne toward them on the night air as they retraced their steps along the lantern-lit path and leprechauns kept shooting over their heads cackling and waving their lanterns. when they finally reached the tents nobody felt like sleeping at all and given the level of noise around them mr. weasley agreed that they could all have one last cup of cocoa together before turning in. they were soon arguing enjoyably about the match mr. weasley got drawn into a disagreement about cobbing with charlie and it was only when ginny fell asleep right at the tiny table and spilled hot chocolate all over the floor that mr. weasley called a halt to the verbal replays and insisted that everyone go to bed. hermione and ginny went into the next tent and harry and the rest of the weasleys changed into pajamas and clambered into their bunks. from the other side of the campsite they could still hear much singing and the odd echoing bang. “oh i am glad i’m not on duty,” muttered mr. weasley sleepily. “i wouldn’t fancy having to go and tell the irish they’ve got to stop celebrating.” harry who was on a top bunk above ron lay staring up at the canvas ceiling of the tent watching the glow of an occasional leprechaun lantern flying overhead and picturing again some of krum’s more spectacular moves. he was itching to get back on his own firebolt and try out the wronski feint. ... somehow oliver wood had never managed to convey with all his wriggling diagrams what that move was supposed to look like. ... harry saw himself in robes that had his name on the back and imagined the sensation of hearing a hundred-thousand-strong crowd roar as ludo bagman’s voice echoed throughout the stadium “i give you ... pottea.” harry never knew whether or not he had actually dropped off to sleep — his fantasies of flying like krum might well have slipped into actual dreams — all he knew was that quite suddenly mr. weasley was shouting. “get up! ron — harry — come on now get up this is urgent!” harry sat up quickly and the top of his head hit canvas. “ ’s’ matter?” he said. dimly he could tell that something was wrong. the noises in the campsite had changed. the singing had stopped. he could hear screams and the sound of people running. he slipped down from the bunk and reached for his clothes but mr. weasley who had pulled on his jeans over his own pajamas said “no time harry — just grab a jacket and get outside — quickly!” harry did as he was told and hurried out of the tent ron at his heels. by the light of the few fires that were still burning he could see people running away into the woods fleeing something that was moving across the field toward them something that was emitting odd flashes of light and noises like gunfire. loud jeering roars of laughter and drunken yells were drifting toward them then came a burst of strong green light which illuminated the scene. a crowd of wizards tightly packed and moving together with wands pointing straight upward was marching slowly across the field. harry squinted at them. ... they didn’t seem to have faces. ... then he realized that their heads were hooded and their faces masked. high above them floating along in midair four struggling figures were being contorted into grotesque shapes. it was as though the masked wizards on the ground were puppeteers and the people above them were marionettes operated by invisible strings that rose from the wands into the air. two of the figures were very small. more wizards were joining the marching group laughing and pointing up at the floating bodies. tents crumpled and fell as the marching crowd swelled. once or twice harry saw one of the marchers blast a tent out of his way with his wand. several caught fire. the screaming grew louder. the floating people were suddenly illuminated as they passed over a burning tent and harry recognized one of them mr. roberts the campsite manager. the other three looked as though they might be his wife and children. one of the marchers below flipped mrs. roberts upside down with his wand her nightdress fell down to reveal voluminous drawers and she struggled to cover herself up as the crowd below her screeched and hooted with glee. “that’s sick,” ron muttered watching the smallest muggle child who had begun to spin like a top sixty feet above the ground his head flopping limply from side to side. “that is really sick. ...” hermione and ginny came hurrying toward them pulling coats over their nightdresses with mr. weasley right behind them. at the same moment bill charlie and percy emerged from the boys’ tent fully dressed with their sleeves rolled up and their wands out. “we’re going to help the ministry!” mr. weasley shouted over all the noise rolling up his own sleeves. “you lot — get into the woods and stick together. i’ll come and fetch you when we’ve sorted this out!” bill charlie and percy were already sprinting away toward the oncoming marchers mr. weasley tore after them. ministry wizards were dashing from every direction toward the source of the trouble. the crowd beneath the roberts family was coming ever closer. “c’mon,” said fred grabbing ginny’s hand and starting to pull her toward the wood. harry ron hermione and george followed. they all looked back as they reached the trees. the crowd beneath the roberts family was larger than ever they could see the ministry wizards trying to get through it to the hooded wizards in the center but they were having great difficulty. it looked as though they were scared to perform any spell that might make the roberts family fall. the colored lanterns that had lit the path to the stadium had been extinguished. dark figures were blundering through the trees children were crying anxious shouts and panicked voices were reverberating around them in the cold night air. harry felt himself being pushed hither and thither by people whose faces he could not see. then he heard ron yell with pain. “what happened?” said hermione anxiously stopping so abruptly that harry walked into her. “ron where are you oh this is stupid — lumos\” she illuminated her wand and directed its narrow beam across the path. ron was lying sprawled on the ground. “tripped over a tree root,” he said angrily getting to his feet again. “well with feet that size hard not to,” said a drawling voice from behind them. harry ron and hermione turned sharply. draco malfoy was standing alone nearby leaning against a tree looking utterly relaxed. his arms folded he seemed to have been watching the scene at the campsite through a gap in the trees. ron told malfoy to do something that harry knew he would never have dared say in front of mrs. weasley “language weasley,” said malfoy his pale eyes glittering. “hadn’t you better be hurrying along now you wouldn’t like her spotted would you?” he nodded at hermione and at the same moment a blast like a bomb sounded from the campsite and a flash of green light momentarily lit the trees around them. “what’s that supposed to mean?” said hermione defiantly. “granger they’re after muggles,” said malfoy. “d’you want to be showing off your knickers in midair because if you do hang around ... they’re moving this way and it would give us all a laugh.” “hermione’s a witch,” harry snarled. “have it your own way potter,” said malfoy grinning maliciously. “if you think they can’t spot a mudblood stay where you are.” “you watch your mouth!” shouted ron. everybody present knew that “mudblood” was a very offensive term for a witch or wizard of muggle parentage. “never mind ron,” said hermione quickly seizing ron’s arm to restrain him as he took a step toward malfoy. there came a bang from the other side of the trees that was louder than anything they had heard. several people nearby screamed. malfoy chuckled softly. “scare easily don’t they?” he said lazily. “i suppose your daddy told you all to hide what’s he up to — trying to rescue the muggles?” “where’re your parents?” said harry his temper rising. “out there wearing masks are they?” malfoy turned his face to harry still smiling. “well ... if they were i wouldn’t be likely to tell you would i potter?” “oh come on,” said hermione with a disgusted look at malfoy “let’s go and find the others.” “keep that big bushy head down granger,” sneered malfoy. “come on,” hermione repeated and she pulled harry and ron up the path again. “i’ll bet you anything his dad is one of that masked lot!” said ron hotly. “well with any luck the ministry will catch him!” said hermione fervently. “oh i can’t believe this. where have the others got to?” fred george and ginny were nowhere to be seen though the path was packed with plenty of other people all looking nervously over their shoulders toward the commotion back at the campsite. a huddle of teenagers in pajamas was arguing vociferously a little way along the path. when they saw harry ron and hermione a girl with thick curly hair turned and said quickly “ ou est madame maxime nous vavons perdue — ” “er — what?” said ron. “oh ...” the girl who had spoken turned her back on him and as they walked on they distinctly heard her say “ ’ogwarts.” “beauxbatons,” muttered hermione. “sorry?” said harry. “they must go to beauxbatons,” said hermione. “you know . . . beauxbatons academy of magic ... i read about it in an appraisal of magical education in europe.” “oh ... yeah ... right,” said harry. “fred and george can’t have gone that far,” said ron pulling out his wand lighting it like hermione’s and squinting up the path. harry dug in the pockets of his jacket for his own wand — but it wasn’t there. the only thing he could find was his omnioculars. “ah no i don’t believe it ... i’ve lost my wand!” “you’re kidding!” ron and hermione raised their wands high enough to spread the narrow beams of light farther on the ground harry looked all around him but his wand was nowhere to be seen. “maybe it’s back in the tent,” said ron. “maybe it fell out of your pocket when we were running?” hermione suggested anxiously. “yeah,” said harry “maybe ...” he usually kept his wand with him at all times in the wizarding world and finding himself without it in the midst of a scene like this made him feel very vulnerable. a rustling noise nearby made all three of them jump. winky the house-elf was fighting her way out of a clump of bushes nearby. she was moving in a most peculiar fashion apparently with great difficulty it was as though someone invisible were trying to hold her back. “there is bad wizards about!” she squeaked distractedly as she leaned forward and labored to keep running. “people high — high in the air! winky is getting out of the way!” and she disappeared into the trees on the other side of the path panting and squeaking as she fought the force that was restraining her. “what’s up with her?” said ron looking curiously after winky. “why can’t she run properly?” “bet she didn’t ask permission to hide,” said harry. he was thinking of dobby every time he had tried to do something the malfoys wouldn’t like the house-elf had been forced to start beating himself up. “you know house-elves get a very raw deal!” said hermione indignantly. “it’s slavery that’s what it is! that mr. crouch made her go up to the top of the stadium and she was terrified and he’s got her bewitched so she can’t even run when they start trampling tents! why doesn’t anyone do something about it?” “well the elves are happy aren’t they?” ron said. “you heard old winky back at the match ... ‘house elves is not supposed to have fun’ ... that’s what she likes being bossed around. ...” “it’s people like you ron,” hermione began hotly “who prop up rotten and unjust systems just because they’re too lazy to — ” another loud bang echoed from the edge of the wood. “let’s just keep moving shall we?” said ron and harry saw him glance edgily at hermione. perhaps there was truth in what malfoy had said perhaps hermione was in more danger than they were. they set off again harry still searching his pockets even though he knew his wand wasn’t there. they followed the dark path deeper into the wood still keeping an eye out for fred george and ginny. they passed a group of goblins who were cackling over a sack of gold that they had undoubtedly won betting on the match and who seemed quite unperturbed by the trouble at the campsite. farther still along the path they walked into a patch of silvery light and when they looked through the trees they saw three tall and beautiful veela standing in a clearing surrounded by a gaggle of young wizards all of whom were talking very loudly. “i pull down about a hundred sacks of galleons a year!” one of them shouted. “i’m a dragon killer for the committee for the disposal of dangerous creatures.” “no you’re not!” yelled his friend. “you’re a dishwasher at the leaky cauldron. ... but i’m a vampire hunter i’ve killed about ninety so far — ” a third young wizard whose pimples were visible even by the dim silvery light of the veela now cut in “i’m about to become the youngest ever minister of magic i am.” harry snorted with laughter. he recognized the pimply wizard his name was stan shunpike and he was in fact a conductor on the triple-decker knight bus. he turned to tell ron this but ron’s face had gone oddly slack and next second ron was yelling “did i tell you i’ve invented a broomstick that’ll reach jupiter?” “honestly\” said hermione and she and harry grabbed ron firmly by the arms wheeled him around and marched him away. by the time the sounds of the veela and their admirers had faded completely they were in the very heart of the wood. they seemed to be alone now everything was much quieter. harry looked around. “i reckon we can just wait here you know. well hear anyone coming a mile off.” the words were hardly out of his mouth when ludo bagman emerged from behind a tree right ahead of them. even by the feeble light of the two wands harry could see that a great change had come over bagman. he no longer looked buoyant and rosy-faced there was no more spring in his step. he looked very white and strained. “who’s that?” he said blinking down at them trying to make out their faces. “what are you doing in here all alone?” they looked at one another surprised. “well — there’s a sort of riot going on,” said ron. bagman stared at him. “what?” “at the campsite . . . some people have got hold of a family of muggles. ...” bagman swore loudly. “damn them!” he said looking quite distracted and without another word he disapparated with a small pop “not exactly on top of things mr. bagman is he?” said hermione frowning. “he was a great beater though,” said ron leading the way off the path into a small clearing and sitting down on a patch of dry grass at the foot of a tree. “the wimbourne wasps won the league three times in a row while he was with them.” he took his small figure of krum out of his pocket set it down on the ground and watched it walk around. like the real krum the model was slightly duck footed and round-shouldered much less impressive on his splayed feet than on his broomstick. harry was listening for noise from the campsite. everything seemed much quieter perhaps the riot was over. “i hope the others are okay,” said hermione after a while. “they’ll be fine,” said ron. “imagine if your dad catches lucius malfoy,” said harry sitting down next to ron and watching the small figure of krum slouching over the fallen leaves. “he’s always said he’d like to get something on him.” “that’d wipe the smirk off old draco’s face all right,” said ron. “those poor muggles though,” said hermione nervously. “what if they can’t get them down?” “they will,” said ron reassuringly. “they’ll find a way.” “mad though to do something like that when the whole ministry of magic’s out here tonight!” said hermione. “i mean how do they expect to get away with it do you think they’ve been drinking or are they just — ” but she broke off abruptly and looked over her shoulder. harry and ron looked quickly around too. it sounded as though someone was staggering toward their clearing. they waited listening to the sounds of the uneven steps behind the dark trees. but the footsteps came to a sudden halt. “hello?” called harry. there was silence. harry got to his feet and peered around the tree. it was too dark to see very far but he could sense somebody standing just beyond the range of his vision. “who’s there?” he said. and then without warning the silence was rent by a voice unlike any they had heard in the wood and it uttered not a panicked shout but what sounded like a spell. “morsmordrel ” and something vast green and glittering erupted from the patch of darkness harry’s eyes had been struggling to penetrate it flew up over the treetops and into the sky. “what the — ” gasped ron as he sprang to his feet again staring up at the thing that had appeared. for a split second harry thought it was another leprechaun formation. then he realized that it was a colossal skull comprised of what looked like emerald stars with a serpent protruding from its mouth like a tongue. as they watched it rose higher and higher blazing in a haze of greenish smoke etched against the black sky like a new constellation. suddenly the wood all around them erupted with screams. harry didn’t understand why but the only possible cause was the sudden appearance of the skull which had now risen high enough to illuminate the entire wood like some grisly neon sign. he scanned the darkness for the person who had conjured the skull but he couldn’t see anyone. “who’s there?” he called again. “harry come on move\” hermione had seized the collar of his jacket and was tugging him backward. “what’s the matter?” harry said startled to see her face so white and terrified. “it’s the dark mark harry!” hermione moaned pulling him as hard as she could. “you-know-who’s sign!” “voldemort’s — ” “harry come on!” harry turned — ron was hurriedly scooping up his miniature krum — the three of them started across the clearing — but before they had taken a few hurried steps a series of popping noises announced the arrival of twenty wizards appearing from thin air surrounding them. harry whirled around and in an instant he registered one fact each of these wizards had his wand out and every wand was pointing right at himself ron and hermione. without pausing to think he yelled “duck!” he seized the other two and pulled them down onto the ground. “stupefyl” roared twenty voices — there was a blinding series of flashes and harry felt the hair on his head ripple as though a powerful wind had swept the clearing. raising his head a fraction of an inch he saw jets of fiery red light flying over them from the wizards’ wands crossing one another bouncing off tree trunks rebounding into the darkness — “stop!” yelled a voice he recognized. “stop! that’s my son\” harry’s hair stopped blowing about. he raised his head a little higher. the wizard in front of him had lowered his wand. he rolled over and saw mr. weasley striding toward them looking terrified. “ron — harry” — his voice sounded shaky — “hermione — are you all right?” “out of the way arthur,” said a cold curt voice. it was mr. crouch. he and the other ministry wizards were closing in on them. harry got to his feet to face them. mr. crouch’s face was taut with rage. “which of you did it?” he snapped his sharp eyes darting between them. “which of you conjured the dark mark?” “we didn’t do that!” said harry gesturing up at the skull. “we didn’t do anything!” said ron who was rubbing his elbow and looking indignantly at his father. “what did you want to attack us for?” “do not lie sir!” shouted mr. crouch. his wand was still pointing directly at ron and his eyes were popping — he looked slightly mad. “you have been discovered at the scene of the crime!” “barty,” whispered a witch in a long woolen dressing gown “they’re kids barty they’d never have been able to — ” “where did the mark come from you three?” said mr. weasley quickly. “over there,” said hermione shakily pointing at the place where they had heard the voice. “there was someone behind the trees . . . they shouted words — an incantation — ” “oh stood over there did they?” said mr. crouch turning his popping eyes on hermione now disbelief etched all over his face. “said an incantation did they you seem very well informed about how that mark is summoned missy — ” but none of the ministry wizards apart from mr. crouch seemed to think it remotely likely that harry ron or hermione had conjured the skull on the contrary at hermione’s words they had all raised their wands again and were pointing in the direction she had indicated squinting through the dark trees. “we’re too late,” said the witch in the woolen dressing gown shaking her head. “they’ll have disapparated.” “i don’t think so,” said a wizard with a scrubby brown beard. it was amos diggory cedric’s father. “our stunners went right through those trees. ... there’s a good chance we got them. ...” “amos be careful!” said a few of the wizards warningly as mr. diggory squared his shoulders raised his wand marched across the clearing and disappeared into the darkness. hermione watched him vanish with her hands over her mouth. a few seconds later they heard mr. diggory shout. “yes! we got them! there’s someone here! unconscious! it’s — but — blimey ...” “you’ve got someone?” shouted mr. crouch sounding highly disbelieving. “who who is it?” they heard snapping twigs the rustling of leaves and then crunching footsteps as mr. diggory reemerged from behind the trees. he was carrying a tiny limp figure in his arms. harry recognized the tea towel at once. it was winky. mr. crouch did not move or speak as mr. diggory deposited his elf on the ground at his feet. the other ministry wizards were all staring at mr. crouch. for a few seconds crouch remained transfixed his eyes blazing in his white face as he stared down at winky. then he appeared to come to life again. “this — cannot — be,” he said jerkily. “no — ” he moved quickly around mr. diggory and strode off toward the place where he had found winky. “no point mr. crouch,” mr. diggory called after him. “there’s no one else there.” but mr. crouch did not seem prepared to take his word for it. they could hear him moving around and the rustling of leaves as he pushed the bushes aside searching. “bit embarrassing,” mr. diggory said grimly looking down at winky’s unconscious form. “barty crouch’s house-elf ... i mean to say ...” “come off it amos,” said mr. weasley quietly “you don’t seriously think it was the elf the dark mark’s a wizard’s sign. it requires a wand.” “yeah,” said mr. diggory “and she had a wand.” “what?” said mr. weasley. “here look.” mr. diggory held up a wand and showed it to mr. weasley. “had it in her hand. so that’s clause three of the code of wand use broken for a start. no non-human creature is permitted to carry or use a wand.” just then there was another pop and ludo bagman apparated right next to mr. weasley. looking breathless and disorientated he spun on the spot goggling upward at the emerald-green skull. “the dark mark!” he panted almost trampling winky as he turned inquiringly to his colleagues. “who did it did you get them barty! what’s going on?” mr. crouch had returned empty-handed. his face was still ghostly white and his hands and his toothbrush mustache were both twitching. “where have you been barty?” said bagman. “why weren’t you at the match your elf was saving you a seat too — gulping gargoyles!” bagman had just noticed winky lying at his feet. “what happened to her?” “i have been busy ludo,” said mr. crouch still talking in the same jerky fashion barely moving his lips. “and my elf has been stunned.” “stunned by you lot you mean but why — ” comprehension dawned suddenly on bagman’s round shiny face he looked up at the skull down at winky and then at mr. crouch. “ivo!” he said. “winky conjure the dark mark she wouldn’t know how! she’d need a wand for a start!” “and she had one,” said mr. diggory. “i found her holding one ludo. if it’s all right with you mr. crouch i think we should hear what she’s got to say for herself.” crouch gave no sign that he had heard mr. diggory but mr. diggory seemed to take his silence for assent. he raised his own wand pointed it at winky and said “rennervatel” winky stirred feebly. her great brown eyes opened and she blinked several times in a bemused sort of way. watched by the silent wizards she raised herself shakily into a sitting position. she caught sight of mr. diggory’s feet and slowly tremulously raised her eyes to stare up into his face then more slowly still she looked up into the sky. harry could see the floating skull reflected twice in her enormous glassy eyes. she gave a gasp looked wildly around the crowded clearing and burst into terrified sobs. “elf!” said mr. diggory sternly. “do you know who i am i’m a member of the department for the regulation and control of magical creatures!” winky began to rock backward and forward on the ground her breath coming in sharp bursts. harry was reminded forcibly of dobby in his moments of terrified disobedience. “as you see elf the dark mark was conjured here a short while ago,” said mr. diggory. “and you were discovered moments later right beneath it! an explanation if you please!” “i — i — i is not doing it sir!” winky gasped. “i is not knowing how sir!” “you were found with a wand in your hand!” barked mr. diggory brandishing it in front of her. and as the wand caught the green light that was filling the clearing from the skull above harry recognized it. “hey — that’s mine!” he said. everyone in the clearing looked at him. “excuse me?” said mr. diggory incredulously. “that’s my wand!” said harry. “i dropped it!” “you dropped it?” repeated mr. diggory in disbelief. “is this a confession you threw it aside after you conjured the mark?” “amos think who you’re talking to!” said mr. weasley very angrily. “is harry potter likely to conjure the dark mark?” “er — of course not,” mumbled mr. diggory. “sorry ... carried away ...” “i didn’t drop it there anyway,” said harry jerking his thumb toward the trees beneath the skull. “i missed it right after we got into the wood.” “so,” said mr. diggory his eyes hardening as he turned to look at winky again cowering at his feet. “you found this wand eh elf and you picked it up and thought you’d have some fun with it did you?” “i is not doing magic with it sir!” squealed winky tears streaming down the sides of her squashed and bulbous nose. “i is ... i is ... i is just picking it up sir! i is not making the dark mark sir i is not knowing how!” “it wasn’t her!” said hermione. she looked very nervous speaking up in front of all these ministry wizards yet determined all the same. “winky’s got a squeaky little voice and the voice we heard doing the incantation was much deeper!” she looked around at harry and ron appealing for their support. “it didn’t sound anything like winky did it?” “no,” said harry shaking his head. “it definitely didn’t sound like an elf.” “yeah it was a human voice,” said ron. “well we’ll soon see,” growled mr. diggory looking unimpressed. “there’s a simple way of discovering the last spell a wand performed elf did you know that?” winky trembled and shook her head frantically her ears flapping as mr. diggory raised his own wand again and placed it tip to tip with harry’s. “ prior incantatol” roared mr. diggory. harry heard hermione gasp horrified as a gigantic serpent-tongued skull erupted from the point where the two wands met but it was a mere shadow of the green skull high above them it looked as though it were made of thick gray smoke the ghost of a spell. “deletrius\” mr. diggory shouted and the smoky skull vanished in a wisp of smoke. “so,” said mr. diggory with a kind of savage triumph looking down upon winky who was still shaking convulsively. “i is not doing it!” she squealed her eyes rolling in terror. “i is not i is not i is not knowing how! i is a good elf i isn’t using wands i isn’t knowing how!” “ you’ve been caught red-handed elf.” mr. diggory roared. “caught with the guilty wand in your hand\” “amos,” said mr. weasley loudly “think about it ... precious few wizards know how to do that spell. ... where would she have learned it?” “perhaps amos is suggesting,” said mr. crouch cold anger in every syllable “that i routinely teach my servants to conjure the dark mark?” there was a deeply unpleasant silence. amos diggory looked horrified. “mr. crouch ... not ... not at all ...” “you have now come very close to accusing the two people in this clearing who are least likely to conjure that mark!” barked mr. crouch. “harry potter — and myself! i suppose you are familiar with the boy’s story amos?” “of course — everyone knows — ” muttered mr. diggory looking highly discomforted. “and i trust you remember the many proofs i have given over a long career that i despise and detest the dark arts and those who practice them?” mr. crouch shouted his eyes bulging again. “mr. crouch i — i never suggested you had anything to do with it!” amos diggory muttered again now reddening behind his scrubby brown beard. “if you accuse my elf you accuse me diggory!” shouted mr. crouch. “where else would she have learned to conjure it?” “she — she might’ve picked it up anywhere — ” “precisely amos,” said mr. weasley. “ she might have picked it up anywhere. ... winky?” he said kindly turning to the elf but she flinched as though he too was shouting at her. “where exactly did you find harry’s wand?” winky was twisting the hem of her tea towel so violently that it was fraying beneath her fingers. “i — i is finding it ... finding it there sir. ...” she whispered “there ... in the trees sir. ...” “you see amos?” said mr. weasley. “whoever conjured the mark could have disapparated right after they’d done it leaving harry’s wand behind. a clever thing to do not using their own wand which could have betrayed them. and winky here had the misfortune to come across the wand moments later and pick it up.” “but then she’d have been only a few feet away from the real culprit!” said mr. diggory impatiently. “elf did you see anyone?” winky began to tremble worse than ever. her giant eyes flickered from mr. diggory to ludo bagman and onto mr. crouch. then she gulped and said “i is seeing no one sir ... no one ...” “amos,” said mr. crouch curtly “i am fully aware that in the ordinary course of events you would want to take winky into your department for questioning. i ask you however to allow me to deal with her.” mr. diggory looked as though he didn’t think much of this suggestion at all but it was clear to harry that mr. crouch was such an important member of the ministry that he did not dare refuse him. “you may rest assured that she will be punished,” mr. crouch added coldly. “m-m-master ...” winky stammered looking up at mr. crouch her eyes brimming with tears. “m-m-master p-p-please ...” mr. crouch stared back his face somehow sharpened each line upon it more deeply etched. there was no pity in his gaze. “winky has behaved tonight in a manner i would not have believed possible,” he said slowly. “i told her to remain in the tent. i told her to stay there while i went to sort out the trouble. and i find that she disobeyed me. this means clothes.” “no!” shrieked winky prostrating herself at mr. crouch’s feet. “no master! not clothes not clothes!” harry knew that the only way to turn a house-elf free was to present it with proper garments. it was pitiful to see the way winky clutched at her tea towel as she sobbed over mr. crouch’s feet. “but she was frightened!” hermione burst out angrily glaring at mr. crouch. “your elf’s scared of heights and those wizards in masks were levitating people! you can’t blame her for wanting to get out of their way!” mr. crouch took a step backward freeing himself from contact with the elf whom he was surveying as though she were something filthy and rotten that was contaminating his over-shined shoes. “i have no use for a house-elf who disobeys me,” he said coldly looking over at hermione. “i have no use for a servant who forgets what is due to her master and to her master’s reputation.” winky was crying so hard that her sobs echoed around the clearing. there was a very nasty silence which was ended by mr. weasley who said quietly “well i think i’ll take my lot back to the tent if nobody’s got any objections. amos that wand’s told us all it can — if harry could have it back please — ” mr. diggory handed harry his wand and harry pocketed it. “come on you three,” mr. weasley said quietly. but hermione didn’t seem to want to move her eyes were still upon the sobbing elf. “hermione!” mr. weasley said more urgently. she turned and followed harry and ron out of the clearing and off through the trees. “what’s going to happen to winky?” said hermione the moment they had left the clearing. “i don’t know,” said mr. weasley. “the way they were treating her!” said hermione furiously. “mr. diggory calling her ‘elf’ all the time ... and mr. crouch! he knows she didn’t do it and he’s still going to sack her! he didn’t care how frightened she’d been or how upset she was — it was like she wasn’t even human!” “well she’s not,” said ron. hermione rounded on him. “that doesn’t mean she hasn’t got feelings ron. it’s disgusting the way — ” “hermione i agree with you,” said mr. weasley quickly beckoning her on “but now is not the time to discuss elf rights. i want to get back to the tent as fast as we can. what happened to the others?” “we lost them in the dark,” said ron. “dad why was everyone so uptight about that skull thing?” “i’ll explain everything back at the tent,” said mr. weasley tensely. but when they reached the edge of the wood their progress was impeded. a large crowd of frightened looking witches and wizards was congregated there and when they saw mr. weasley coming toward them many of them surged forward. “what’s going on in there?” “who conjured it?” “arthur — it’s not — him?” “of course it’s not him,” said mr. weasley impatiently. “we don’t know who it was it looks like they disapparated. now excuse me please i want to get to bed.” he led harry ron and hermione through the crowd and back into the campsite. all was quiet now there was no sign of the masked wizards though several ruined tents were still smoking. charlie’s head was poking out of the boys’ tent. “dad what’s going on?” he called through the dark. “fred george and ginny got back okay but the others — ” “i’ve got them here,” said mr. weasley bending down and entering the tent. harry ron and hermione entered after him. bill was sitting at the small kitchen table holding a bedsheet to his arm which was bleeding profusely. charlie had a large rip in his shirt and percy was sporting a bloody nose. fred george and ginny looked unhurt though shaken. “did you get them dad?” said bill sharply. “the person who conjured the mark?” “no,” said mr. weasley. “we found barty crouch’s elf holding harry’s wand but we’re none the wiser about who actually conjured the mark.” “what?” said bill charlie and percy together. “harry’s wand?” said fred. “mr. crouch’s elf?” said percy sounding thunderstruck. with some assistance from harry ron and hermione mr. weasley explained what had happened in the woods. when they had finished their story percy swelled indignantly. “well mr. crouch is quite right to get rid of an elf like that!” he said. “running away when he’d expressly told her not to ... embarrassing him in front of the whole ministry . . . how would that have looked if she’d been brought up in front of the department for the regulation and control — ” “she didn’t do anything — she was just in the wrong place at the wrong time!” hermione snapped at percy who looked very taken aback. hermione had always got on fairly well with percy — better indeed than any of the others. “hermione a wizard in mr. crouch’s position can’t afford a house-elf who’s going to run amok with a wand!” said percy pompously recovering himself. “she didn’t run amok!” shouted hermione. “she just picked it up off the ground!” “look can someone just explain what that skull thing was?” said ron impatiently. “it wasn’t hurting anyone. ... why’s it such a big deal?” “i told you it’s you-know-who’s symbol ron,” said hermione before anyone else could answer. “i read about it in the rise and fall of the dark arts.” “and it hasn’t been seen for thirteen years,” said mr. weasley quietly. “of course people panicked ... it was almost like seeing you-know-who back again.” “i don’t get it,” said ron frowning. “i mean ... it’s still only a shape in the sky. ...” “ron you-know-who and his followers sent the dark mark into the air whenever they killed,” said mr. weasley. “the terror it inspired ... you have no idea you’re too young. just picture coming home and finding the dark mark hovering over your house and knowing what you’re about to find inside. ...” mr. weasley winced. “everyone’s worst fear ... the very worst ...” there was silence for a moment. then bill removing the sheet from his arm to check on his cut said “well it didn’t help us tonight whoever conjured it. it scared the death eaters away the moment they saw it. they all disapparated before we’d got near enough to unmask any of them. we caught the robertses before they hit the ground though. they’re having their memories modified right now.” “death eaters?” said harry. “what are death eaters?” “it’s what you-know who’s supporters called themselves,” said bill. “i think we saw what’s left of them tonight — the ones who managed to keep themselves out of azkaban anyway.” “we can’t prove it was them bill,” said mr. weasley. “though it probably was,” he added hopelessly. “yeah i bet it was!” said ron suddenly. “dad we met draco malfoy in the woods and he as good as told us his dad was one of those nutters in masks! and we all know the malfoys were right in with you-know-who!” “but what were voldemort’s supporters — ” harry began. everybody flinched — like most of the wizarding world the weasleys always avoided saying voldemort’s name. “sorry,” said harry quickly. “what were you know who’s supporters up to levitating muggles i mean what was the point?” “the point?” said mr. weasley with a hollow laugh. “harry that’s their idea of fun. half the muggle killings back when you-know-who was in power were done for fun. i suppose 

          """]