import bert
import samples
import sklearn.mixture
import numpy as np

bert_model = bert.BertForEmbedding('bert-base-uncased')
embedding1 = bert_model.bertify_single_abstract(samples.sample_abstract1)
embedding2 = bert_model.bertify_single_abstract(samples.sample_abstract2)
embedding3 = bert_model.bertify_single_abstract(samples.sample_abstract3)
embedding4 = bert_model.bertify_single_abstract(samples.sample_abstract4)

# print(embedding1)
# print(embedding2)
# print(embedding3)
# print(embedding4)

gmm = sklearn.mixture.GaussianMixture(2)
X = np.array([embedding1.numpy().flatten(),
                  embedding2.numpy().flatten(),
                  embedding3.numpy().flatten(),
                  embedding4.numpy().flatten()])
gmm.fit(X)



