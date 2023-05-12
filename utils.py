from functools import reduce
import bert
import samples
import sklearn.mixture
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from bert import BertForEmbedding
import umap
import hdbscan

from numpy import dot
from numpy.linalg import norm
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import os
import pdb


def create_embeddings(file_path, save_path="nips_2022_embeddings"):
    """
    Used for creating a set of abstract embeddings using bert.

    Args:
        file_path (string): the path to a csv containing scrapped NIPS data
        save_path (string): the path to save a pickle file

    Returns:
        embeddings: A 2d numpy array containing all the abstract embeddings
    """
    bert_model = BertForEmbedding()
    nips_2022_df = pd.read_csv(file_path)
    embeddings = bert_model.bertify_abstracts(nips_2022_df["abstract"])
    file = open(save_path, 'wb')
    pickle.dump(embeddings, file)
    file.close()
    return embeddings


def load_embeddings(file_path):
    """
    load abstract embeddings

    Args:
        file_path (stiring): The path where the pickle file is stored

    Returns:
         embeddings: A 2d numpy array containing all the abstract embeddings
    """
    file = open(file_path, 'rb')
    embeddings = pickle.load(file)
    file.close()
    return embeddings


def load_gmm(gmm_path):
    """
    loads a mixture of gaussian model

    Args:
        file_path (stiring): The path where the pickle file is stored

    Returns:
         embeddings: a mixture of gaussian model from sklearn

    """
    file = open(gmm_path, 'rb')
    gmm = pickle.load(file)
    file.close()
    return gmm


def cos_sim(query_embedding, cluster_embedding):
    """
    calculates the cosine similarity.

    Args:
        query_embedding: a single embedding representing the user input
        cluster_embedding: the 2D numpy array representing all abstract embeddings.

    Returns:
        numpy of doubles: the cosine similarity between query_embedding and each cluster_embedding.
    """
    q_norm = norm(query_embedding)
    c_norm = norm(cluster_embedding, axis=1)
    q_c_dot = dot(cluster_embedding, query_embedding.flatten())
    return q_c_dot/(q_norm*c_norm)


def select_cluster(cluster_num, embeddings, df, labels):
    """
    Finds all embedding and paper entries corresponding to a particular cluster

    Args:
        cluster_num: the cluster label
        embeddings: all of the embeddings in a 2d numpy array
        df: the nips data as a dataframe
        labels: a list of cluster membership for each of the embeddings

    Returns:
        relevant_embeddings: 2d numpy array filled with the relevant embeddings
        relevant_df: dataframe whose rows are papers corresponding to the 
                    relevant embeddings.
    """
    relevant_idx = labels == cluster_num
    relevant_embeddings = embeddings[relevant_idx]
    relevant_df = df.loc[relevant_idx]
    return relevant_embeddings, relevant_df


def reduce_dim(embeddings, method, dim, custom=None):
    """
    reduce the vector dimensions

    Args:
        embeddings: 2d numpy array of the abstract embeddings
        method (string): "umap" | "pca"
        dim: the new dimension
        custom: another method of dimensional reduction

    Returns:
        2D Numpy array: the embeddings in the reduced dimension
        dim_reduce_model: the model that was used for data reduction

    """
    if custom is not None:
        return custom.fit_transform(embeddings)
    elif method == "umap":
        # return umap.UMAP(n_neighbors=15, n_components=dim, metric='cosine').fit_transform(embeddings)
        dim_reduce_model = umap.UMAP(
            n_neighbors=15, n_components=dim, metric='cosine').fit(embeddings)
        return dim_reduce_model.transform(embeddings), dim_reduce_model
    elif method == "pca":
        # return PCA(n_components=dim).transform(embeddings)
        dim_reduce_model = PCA(n_components=dim).fit(embeddings)
        return dim_reduce_model.transform(embeddings), dim_reduce_model
    # elif method == "tsne":
    #     return TSNE(n_components=dim).fit_transform(embeddings)
    else:
        raise Exception("Please use one of {umap, pca}")


def cluster_and_predict(embeddings, method, num_cluster=None):
    """
    clusters the data and assign a class to each paper

    Args:
        embeddings: a 2d numpy array containing all the abstract embeddings:
        method (string): "gmm" | "kmeans" | "dbscan"
        num_cluster: the desired number of clusters

    Returns:
        clusters: labels for each embedding.
        model: a fitted sklearn model
    """
    if method == "gmm":
        gmm = sklearn.mixture.GaussianMixture(num_cluster)
        gmm.fit(embeddings)
        clusters = gmm.predict(embeddings)
        return clusters, gmm
    elif method == "kmeans":
        kmeans = KMeans(num_cluster)
        kmeans.fit(embeddings)
        clusters = kmeans.predict(embeddings)
        return clusters, kmeans
    elif method == "dbscan":
        dbscan = hdbscan.HDBSCAN(min_cluster_size=15,
                                 metric='euclidean',
                                 cluster_selection_method='eom', prediction_data=True)
        clusters = dbscan.fit(embeddings)
        # use hdbscan.approximate_predict(clusters, test_points)
        return clusters, None
    else:
        raise Exception("Please use one of {gmm, dbscan}")


def train_cluster(embeddings, method, metric, num_cluster_range=None, saving_path=None):
    """
    [DEPRECATED; USE dim_reduce_train_cluster]
    trains a series of mixture of gaussian model and give their losses

    Args:
        embeddings: 2D numpy array of abstract embeddings.
        method (string): "gmm"
        metric (string): "sil" for silhoutte loss
        num_clsuter_range: a Range or List of a number of cluster parameter
        saving_path (string): where to store the models

    Returns:
        models: a dictionary. the keys are the number of clusters and the values are trained models
        scores: a dictionary, the keys are the number of clusters and the values are the sil score.
    """
    scores = {}
    models = {}
    if method == "gmm":
        for num_cluster in tqdm(num_cluster_range):
            clusters, gmm = cluster_and_predict(
                embeddings, method, num_cluster)
            current_metric = metrics([metric], embeddings, clusters)
            scores[num_cluster] = current_metric[metric]
            models[num_cluster] = gmm
            if saving_path:
                file = open(os.path.join(
                    saving_path, f"gmm_{num_cluster}.pkl"), 'wb')
                pickle.dump(gmm, file)
                file.close()
    return models, scores


def dim_reduce_train_cluster(embeddings,
                             cluster_method,
                             dim_reduc_method,
                             metric,
                             dim_range,
                             num_cluster_range=None,
                             saving_path=None):
    """
    reduces dimension first and trains many models

    Args:
        embeddings: 2D numpy array of abstract embeddings
        cluster_method (string): "gmm" | "kmeans" | "dbscan"
        dim_reduc_method: "umap"
        metric: "sil"
        dim_range: a Range or List of dimensions to try reducing to
        num_cluster_range: a Range or List of number of clusters to try
        saving_path (string): the base path to save to

    Returns:
        cluster_models: a dictionary. The key has format (dim, num_cluster). The values are models.
        dim_reduc_models: a dictionary. The key is the dim. The values are the dimensional reduction models.
        scores: a dictionary. The key has format (dim, num_cluster). The values are the silhoutte scores.
    """
    scores = {}
    cluster_models = {}
    dim_reduc_models = {}
    for dim in tqdm(dim_range):
        if dim == -1:
            reduced_embeddings, dim_reduce_model = embeddings, None
        else:
            reduced_embeddings, dim_reduce_model = reduce_dim(
                embeddings, dim_reduc_method, dim)
        dim_reduc_models[dim] = dim_reduce_model
        if saving_path:
            os.makedirs(saving_path, exist_ok=True)
            file = open(os.path.join(saving_path, f"dim_{dim}.pkl"), 'wb')
            pickle.dump(dim_reduce_model, file)
            file.close()
        for num_cluster in tqdm(num_cluster_range):
            clusters, cluster_model = cluster_and_predict(
                reduced_embeddings, cluster_method, num_cluster)
            if cluster_method == "dbscan":
                not_outliers = clusters.labels_ != -1
                try:
                    current_metric = metrics(
                        [metric], reduced_embeddings[not_outliers], clusters.labels_[not_outliers])
                    scores[(dim, num_cluster)] = current_metric[metric]
                    print(current_metric[metric])

                except:
                    scores[((dim, num_cluster))] = -1
                    print(-1)
            else:
                current_metric = metrics(
                    [metric], reduced_embeddings, clusters)
                scores[(dim, num_cluster)] = current_metric[metric]
                print(current_metric[metric])
            cluster_models[(dim, num_cluster)] = cluster_model
            if saving_path:
                os.makedirs(saving_path, exist_ok=True)
                file = open(os.path.join(
                    saving_path, f"dim_{dim}_{cluster_method}_{num_cluster}.pkl"), 'wb')
                pickle.dump(cluster_model, file)
                file.close()
    if saving_path:
        os.makedirs(saving_path, exist_ok=True)
        file = open(os.path.join(saving_path, f"scores.pkl"), 'wb')
        pickle.dump(cluster_model, file)
        file.close()
    return cluster_models, dim_reduc_models, scores

def simplify_scores(scores, dim_range, cluster_range, use_dim=True):
    """
    separates a subset of the scores dictionary into dimensions and scores

    Args:
        scores: a dictionary. The key has format (dim, num_cluster). The values are the silhoutte scores.
        dim_range: a Range or List of desireable dimensions
        cluster_range: a Range or List of desireable number of clusters
        use_dim: a boolean that whether to return the dimensions.
    
    Returns:
        categories: a list of dimensions if use_dim is True
        select_scores: a list of silhoutte scores corresponding to the categories
    """
    categories = []
    select_scores = []
    for dim in dim_range:
        for cluster in cluster_range:
            if use_dim:
                if dim == -1:
                    categories.append(768)
                else:
                    categories.append(dim)
            select_scores.append(scores[(dim, cluster)])
    return categories, select_scores

def detect_keywords(x, keywords):
    """
    generates a boolean mask that's True for input with the keywords.

    Args:
        x: a series of text
        keywords: a list of keywords we are interested in finding

    Returns:
        numpy boolean array: True for text containing keywords. False otherwise
    """
    return reduce(lambda x, y: x or y, map(lambda w: w in x, keywords))


def visualize(embeddings, labels):
    """
    plots the clusters in two dimensions and color coded by cluster membership

    Args:
        embeddings: a 2D numpy array of abstract embeddings.
        labels: a list of cluster label corresponding to the embeddings.
    """
    umap_data = umap.UMAP(n_neighbors=15, n_components=2,
                          min_dist=0.0, metric='cosine').fit_transform(embeddings)

    result = pd.DataFrame(umap_data, columns=['x', 'y'])

    result['labels'] = labels

    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD')
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, cmap='hsv_r')
    plt.colorbar()
    plt.show()


def metrics(metrics_l, embeddings, labels, dic={}):
    """
    calculates metrics for clustering.

    Args:
        metrics_l: a list of metrics, right now only ["sil"] will work
        embeddings: 2D numpy array of abstract embeddings
        labels: a list of cluster membership corresponding to embeddings
        dic: a dictory to store the metrics

    Return:
        dic: a dictory storing the metrics
    """
    for metric in metrics_l:
        if metric == "sil":
            dic["sil"] = silhouette_score(embeddings, labels)
        else:
            print("don't have those implemented yet.")
    return dic

# topic modeling


def create_docs_per_topic(df, labels):
    """
    combine abstracts that have the same label.
    Taken from: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

    Args:
        df: a dataframe containing the scrapped nips data
        labels: cluster membership for each abstract embedding

    Returns:
        df: the original dataframe with two new columns "Topic" and "Doc_ID"
        docs_per_topic: a new dataframe that contains the cluster-combined abstracts
    """
    df["Topic"] = labels
    df["Doc_ID"] = range(len(df))
    docs_per_topic = df.groupby(
        ['Topic'], as_index=False).agg({'abstract': ' '.join})
    return df, docs_per_topic


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    """
    calculate the c-tf-idf scores for the cluster-combined abstracts.
    Taken from: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

    Args:
        documents: list of the cluster-combined abstracts from create_docs_per_topic
        m: the number of clusters
        ngram_range: the number of sequential words considered as one unit

    Returns:
        tf_idf: a 2D array containing the tf_idf score for each word fo rall the clusters
        count: a CountVectorizer object containing the counts of words.
    """
    count = CountVectorizer(ngram_range=ngram_range,
                            stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    """
    returns the top words from each cluster.
    Taken from https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6

    Args:
        tf_idf: a 2D array containing the tf_idf score for each word fo rall the clusters
        count: a CountVectorizer object containing the counts of words.
        docs_per_topic: a new dataframe that contains the cluster-combined abstracts
        n: the number of top words to show

    Return:
        top_n_words: a list of the most important words and their tf-idf scores.
    """
    words = count.get_feature_names_out()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j])
                           for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words


def extract_topic_sizes(df):
    """
    finds the size of each cluster

    Args:
        df: a dataframe of scrapped nips data with cluster labels

    Returns:
        topic_sizes: a dataframe containing the size for each topic.
    """
    topic_sizes = (df.groupby(['Topic'])
                     .abstract
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "abstract": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

########
