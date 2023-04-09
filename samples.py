sample_abstract1 = "We focus on federated learning in practical recommender \
                    systems and natural language processing scenarios. The \
                    global model for federated optimization typically \
                    contains a large and sparse embedding layer, while \
                    each client’s local data tend to interact with part \
                    of features, updating only a small submodel with the \
                    feature-related embedding vectors. We identify a new \
                    and important issue that distinct data features normally \
                    involve different numbers of clients, generating the \
                    differentiation of hot and cold features. We further \
                    reveal that the classical federated averaging algorithm \
                    (FedAvg) or its variants, which randomly selects clients \
                    to participate and uniformly averages their submodel updates\
                    , will be severely slowed down, because different \
                    parameters of the global model are optimized at different \
                    speeds. More specifically, the model parameters related to \
                    hot (resp., cold) features will be updated quickly \
                    (resp., slowly). We thus propose federated submodel \
                    averaging (FedSubAvg), which introduces the number of \
                    feature-related clients as the metric of feature heat \
                    to correct the aggregation of submodel updates. We prove \
                    that due to the dispersion of feature heat, the global \
                    objective is ill-conditioned, and FedSubAvg works as a \
                    suitable diagonal preconditioner. We also rigorously \
                    analyze FedSubAvg’s convergence rate to stationary points. \
                    We finally evaluate FedSubAvg over several public and \
                    industrial datasets. The evaluation results demonstrate \
                    that FedSubAvg significantly outperforms FedAvg and its variants."
sample_abstract2 = "We study a stochastic bandit problem with a general unknown reward function and a general unknown constraint function. Both functions can be non-linear (even non-convex) and are assumed to lie in a reproducing kernel Hilbert space (RKHS) with a bounded norm. In contrast to safety-type hard constraints studied in prior works, we consider soft constraints that may be violated in any round as long as the cumulative violations are small. Our ultimate goal is to study how to utilize the nature of soft constraints to attain a finer complexity-regret-constraint trade-off in the kernelized bandit setting. To this end, leveraging primal-dual optimization, we propose a general framework for both algorithm design and performance analysis. This framework builds upon a novel sufficient condition, which not only is satisfied under general exploration strategies, including upper confidence bound (UCB), Thompson sampling (TS), and new ones based on random exploration, but also enables a unified analysis for showing both sublinear regret and sublinear or even zero constraint violation. We demonstrate the superior performance of our proposed algorithms via numerical experiments based on both synthetic and real-world datasets. Along the way, we also make the first detailed comparison between two popular methods for analyzing constrained bandits and Markov decision processes (MDPs) by discussing the key difference and some subtleties in the analysis, which could be of independent interest to the communities."

sample_abstract3 = "A novel approach to rank estimation, called geometric order learning (GOL), is proposed in this paper. First, we construct an embedding space, in which the direction and distance between objects represent order and metric relations between their ranks, by enforcing two geometric constraints: the order constraint compels objects to be sorted according to their ranks, while the metric constraint makes the distance between objects reflect their rank difference. Then, we perform the simple k nearest neighbor (k-NN) search in the embedding space to estimate the rank of a test object. Moreover, to assess the quality of embedding spaces for rank estimation, we propose a metric called discriminative ratio for ranking (DRR). Extensive experiments on facial age estimation, historical color image (HCI) classification, and aesthetic score regression demonstrate that GOL constructs effective embedding spaces and thus yields excellent rank estimation performances. The source codes are available at https://github.com/seon92/GOL"

sample_abstract4 = "A key goal of unsupervised learning is to go beyond density estimation and sample generation to reveal the structure inherent within observed data. Such structure can be expressed in the pattern of interactions between explanatory latent variables captured through a probabilistic graphical model. Although the learning of structured graphical models has a long history, much recent work in unsupervised modelling has instead emphasised flexible deep-network-based generation, either transforming independent latent generators to model complex data or assuming that distinct observed variables are derived from different latent nodes. Here, we extend amortised variational inference to incorporate structured factors over multiple variables, able to capture the observation-induced posterior dependence between latents that results from “explaining away” and thus allow complex observations to depend on multiple nodes of a structured graph. We show that appropriately parametrised factors can be combined efficiently with variational message passing in rich graphical structures. We instantiate the framework in nonlinear Gaussian Process Factor Analysis, evaluating the structured recognition framework using synthetic data from known generative processes. We fit the GPFA model to high-dimensional neural spike data from the hippocampus of freely moving rodents, where the model successfully identifies latent signals that correlate with behavioural covariates."