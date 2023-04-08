import bert
sample_abstract = "We focus on federated learning in practical recommender \
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

bert_model = bert.BertForEmbedding('bert-base-uncased')
print(bert_model.bertify_single_abstract(sample_abstract))
