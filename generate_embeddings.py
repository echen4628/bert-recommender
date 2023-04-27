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

embeddings = create_embeddings("nips_2019_clean.csv", "outputs/clean_nips_2019_embeddings.pkl")
