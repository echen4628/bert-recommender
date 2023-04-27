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
import umap
import hdbscan


from numpy import dot
from numpy.linalg import norm

if __name__ == "__main__":
    nips_df = pd.read_csv("nips_2018.csv")
    embeddings = load_embeddings("outputs/nips_2018_embeddings.pkl")
    embeddings = [x.numpy() for x in embeddings]
    embeddings = np.array(embeddings)
    # reduce dimensions to 5 using umap
    reduced_embeddings = reduce_dim(embeddings, "umap", 200)
    # reduced_embeddings = embeddings
    # cluster
    clusters, model = cluster_and_predict(reduced_embeddings, "gmm", num_cluster=10)

    # import matplotlib.pyplot as plt

    # Prepare data
    umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

    fig, ax = plt.subplots()
    ax.scatter(umap_data[:,0], umap_data[:,1], c=clusters)
    plt.show()


    # for dbscan visuals
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = clusters.labels_

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()
    plt.show()

#     # t_sne = TSNE(n_components=2)
#     # embeddings_2d = t_sne.fit_transform(embeddings)
#     # pca = PCA(n_components=2)
#     # pca.fit(embeddings)
#     # embeddings_2d = pca.transform(embeddings)

#     x = embeddings[0]
#     gmm20 = load_gmm("gmm_20.pkl")
#     y = gmm20.predict(embeddings)

#     x_class = gmm20.predict(x.reshape(1,-1))
#     mask = y == x_class
#     relevant_nips = nips_2022[mask]
#     relevant_embeddings = embeddings[mask]

#     scores = cos_sim(x, relevant_embeddings)
#     best_matches = np.argsort(scores)[-20:][::-1]
#     for x in nips_2022.iloc[best_matches]["title"]:
#         print(x)



#     # embeddings[mask]


#     # for num_cluster in range(1, 21):
#     #     gmm = load_gmm(f"gmm_{num_cluster}.pkl")
#     #     y = gmm.predict(embeddings)
#     #     fig, ax = plt.subplots()
#     #     ax.scatter(embeddings[:,0], embeddings[:,1], c=y)
#     #     plt.show()
#     #     # fig.savefig(f"gmm{num_cluster}_pca2.png")
#     #     fig.savefig(f"gmm{num_cluster}_tsne2.png")

    
#     # gmm = sklearn.mixture.GaussianMixture(5)
#     # gmm.fit(embeddings)
#     # file = open("gmm_5", 'wb')
#     # pickle.dump(gmm, file)
#     # file.close()
    



# # embedding1 = bert_model.bertify_single_abstract(nips_2022_df["abstract"][0])
# # embedding2 = bert_model.bertify_single_abstract([nips_2022_df["abstract"][0], nips_2022_df["abstract"][0]])
# # embedding3 = bert_model.bertify_single_abstract(samples.sample_abstract3)
# # embedding4 = bert_model.bertify_single_abstract(samples.sample_abstract4)

# # print(embedding1)
# # print(embedding2)
# # print(embedding3)
# # print(embedding4)

# # Running GMM
# # gmm = sklearn.mixture.GaussianMixture(2)
# # X = np.array([embedding1.numpy().flatten(),
# #                   embedding2.numpy().flatten(),
# #                   embedding3.numpy().flatten(),
# #                   embedding4.numpy().flatten()])
# # gmm.fit(X)



