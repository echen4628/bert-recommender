from utils import *
from tqdm import tqdm

def train_cluster()



# load embeddings
embeddings = np.array(load_embeddings(file_path="nips_2022_embeddings_n.pkl"))
# create loop
for num_cluster in tqdm(range(1, 21)):
    gmm = sklearn.mixture.GaussianMixture(num_cluster)
    gmm.fit(embeddings)
    file = open(f"gmm_{num_cluster}.pkl", 'wb')
    pickle.dump(gmm, file)
    file.close()
# X = np.array([embedding1.numpy().flatten(),
#                   embedding2.numpy().flatten(),
#                   embedding3.numpy().flatten(),
#                   embedding4.numpy().flatten()])
# gmm.fit(X)
# make gmm

# save and delete old gmm
