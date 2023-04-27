import bert
import samples
import sklearn.mixture
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as  plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from bert import BertForEmbedding
import umap
import hdbscan

from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer


def create_embeddings(file_path, save_path="nips_2022_embeddings"):
    bert_model = BertForEmbedding()
    nips_2022_df = pd.read_csv(file_path)
    embeddings = bert_model.bertify_abstracts(nips_2022_df["abstract"])
    file = open(save_path, 'wb')
    pickle.dump(embeddings, file)
    file.close()
    return embeddings

def load_embeddings(file_path):
    file = open(file_path, 'rb')
    embeddings = pickle.load(file)
    file.close()
    return embeddings

def load_gmm(gmm_path):
    file=open(gmm_path, 'rb')
    gmm=pickle.load(file)
    file.close()
    return gmm

def cos_sim(query_embedding,cluster_embedding):
    q_norm = norm(query_embedding)
    c_norm = norm(cluster_embedding, axis=1)
    q_c_dot = dot(cluster_embedding, query_embedding)
    return q_c_dot/(q_norm*c_norm)
# dot(x, y)/(norm(x)*norm(y))

def select_cluster(cluster_num, embeddings, df, labels):
    relevant_idx = labels == cluster_num
    relevant_embeddings = embeddings[relevant_idx]
    relevant_df = df.loc[relevant_idx]
    return relevant_embeddings, relevant_df

def reduce_dim(embeddings, method, dim, custom=None):
    if custom is not None:
        return custom.fit_transform(embeddings)
    elif method == "umap":
        return umap.UMAP(n_neighbors=15, n_components=dim, metric='cosine').fit_transform(embeddings)
    elif method == "pca":
        return PCA(n_components=dim).transform(embeddings)
    elif method == "tsne":
        return TSNE(n_components=dim).fit_transform(embeddings)
    else:
        raise Exception("Please use one of {umap, pca, tsne}")
    
def cluster_and_predict(embeddings, method, num_cluster=None):
    if method == "gmm":
        gmm = sklearn.mixture.GaussianMixture(num_cluster)
        gmm.fit(embeddings)
        clusters = gmm.predict(embeddings)
        return clusters, gmm
    elif method == "dbscan":
        dbscan = hdbscan.HDBSCAN(min_cluster_size=15,
                          metric='euclidean',                      
                          cluster_selection_method='eom', prediction_data=True)
        clusters = dbscan.fit(embeddings)
        # use hdbscan.approximate_predict(clusters, test_points)
        return clusters, None
    else:
        raise Exception("Please use one of {gmm, dbscan}")

# can use word to vec 30,000 vector instead of bert output and see which one is better
from functools import reduce
def detect_keywords(x, keywords):
    return reduce(lambda x, y: x or y , map(lambda w: w in x, keywords))

def visualize(embeddings, labels):
    umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

    result = pd.DataFrame(umap_data, columns=['x', 'y'])

    result['labels'] = labels

    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()
    plt.show()

def metrics(metrics_l, embeddings, labels, dic = {}):
    for metric in metrics_l:
        if metric == "sil":
            dic["sil"] = silhouette_score(embeddings, labels)
        else:
            print("don't have those implemented yet.")
    return dic

# topic modeling
def create_docs_per_topic(df, labels):
    df["Topic"] = labels
    df["Doc_ID"] = range(len(df))
    docs_per_topic = df.groupby(['Topic'], as_index = False).agg({'abstract': ' '.join})
    return df, docs_per_topic

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .abstract
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "abstract": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

########
