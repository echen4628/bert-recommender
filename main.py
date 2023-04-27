import bert
import samples
import sklearn.mixture
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as  plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import *
import pdb

from numpy import dot
from numpy.linalg import norm

def cos_sim(query_embedding,cluster_embedding):
    q_norm = norm(query_embedding)
    c_norm = norm(cluster_embedding, axis=1)
    q_c_dot = dot(cluster_embedding, query_embedding)
    return q_c_dot/(q_norm*c_norm)
# dot(x, y)/(norm(x)*norm(y))

if __name__ == "__main__":
    # bert_model = bert.BertForEmbedding('bert-base-uncased')
    # nips_2022_df = pd.read_csv("nips_2022.csv")

    nips_2018 = pd.read_csv("nips_2018.csv")
    embeddings = create_embeddings(nips_2018, save_path="outputs/nips_2018_embeddings")
    embeddings = np.array(load_embeddings())


    # t_sne = TSNE(n_components=2)
    # embeddings_2d = t_sne.fit_transform(embeddings)
    # pca = PCA(n_components=2)
    # pca.fit(embeddings)
    # embeddings_2d = pca.transform(embeddings)

    x = embeddings[0]
    gmm20 = load_gmm("gmm_20.pkl")
    y = gmm20.predict(embeddings)

    x_class = gmm20.predict(x.reshape(1,-1))
    mask = y == x_class
    relevant_nips = nips_2022[mask]
    relevant_embeddings = embeddings[mask]

    scores = cos_sim(x, relevant_embeddings)
    best_matches = np.argsort(scores)[-20:][::-1]
    for x in nips_2022.iloc[best_matches]["title"]:
        print(x)



    # embeddings[mask]


    # for num_cluster in range(1, 21):
    #     gmm = load_gmm(f"gmm_{num_cluster}.pkl")
    #     y = gmm.predict(embeddings)
    #     fig, ax = plt.subplots()
    #     ax.scatter(embeddings[:,0], embeddings[:,1], c=y)
    #     plt.show()
    #     # fig.savefig(f"gmm{num_cluster}_pca2.png")
    #     fig.savefig(f"gmm{num_cluster}_tsne2.png")

    
    # gmm = sklearn.mixture.GaussianMixture(5)
    # gmm.fit(embeddings)
    # file = open("gmm_5", 'wb')
    # pickle.dump(gmm, file)
    # file.close()
    



# embedding1 = bert_model.bertify_single_abstract(nips_2022_df["abstract"][0])
# embedding2 = bert_model.bertify_single_abstract([nips_2022_df["abstract"][0], nips_2022_df["abstract"][0]])
# embedding3 = bert_model.bertify_single_abstract(samples.sample_abstract3)
# embedding4 = bert_model.bertify_single_abstract(samples.sample_abstract4)

# print(embedding1)
# print(embedding2)
# print(embedding3)
# print(embedding4)

# Running GMM
# gmm = sklearn.mixture.GaussianMixture(2)
# X = np.array([embedding1.numpy().flatten(),
#                   embedding2.numpy().flatten(),
#                   embedding3.numpy().flatten(),
#                   embedding4.numpy().flatten()])
# gmm.fit(X)



