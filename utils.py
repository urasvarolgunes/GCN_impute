from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from graphs.graphs import *

def build_affinity_matrix(embed_matrix):
    
    aff_mat = None
    
    return aff_mat


def KNN(X, y, n):
    l = len(y)
    y_hat = []
    
    for i in range(l): 
        X_train = np.delete(X, i, axis = 0) 
        y_train = np.delete(y, i, axis = 0) 
    
        neigh = KNeighborsClassifier(n_neighbors = n)
        neigh.fit(X_train, y_train)
        y_hat.extend(neigh.predict(X[i].reshape(1,-1)))
        
    print("%d-NN" %n)
    print(sum(np.array(y_hat) == y) / l)
    
def prepare_adj(df, method = 'gaussian', sig = 1, alpha = 0.5, delta = 20):

    """
    Input: Adjacency matrix or feature matrix with the last column including the labels
    Output: Row normalized gaussian kernel similarity matrix
    """
    X = df.values[:,:-1] #consider X a graph or a feature matrix, both fine
    np.fill_diagonal(X,0) #set diagonal to zero / remove self loops
    Q_index = range(X.shape[0]) # for now always use this

    dis = distanceEuclidean(X, Q_index, n_jobs=-1)
    similarity = kerGauss(dis, sigma = sig) #try different sigma

    # origianl similarity matrix, using gaussian kernel, row normalize
    if method == 'gaussian':
        graph = RandomWalkNormalize(similarity)
        
    elif method == 'MSTKNN':
        A_KNN = MSTKNN(dis,Q_index,delta,n_jobs=-1,spanning=True)
        A_KNN_ker = A_KNN*similarity
        graph = RandomWalkNormalize(A_KNN_ker)
        
    elif method == 'nnlsw':
        A_KNN = MSTKNN(dis,Q_index,delta,n_jobs=-1,spanning=True)
        A_KNN_nnls = multicoreNNLS(X,A_KNN,Q_index,n_jobs=-1)
        graph = lazy(A_KNN_nnls, alpha= alpha) # convert to lazy
        
    return graph

def get_train_and_val_mask(train_mask, val_size):
    
    """
    Input: Indices of the p rows in the Domain matrix
    
    split the training set into training and validation sets
    train_ind and val_ind are used to filter the base_embeddings during training
    """
    p = len(train_mask)
    indices = np.arange(p)
    np.random.shuffle(indices)
    
    train_lim = int((1-val_size)* p) # 90% training, 10% validation
    
    train_ind = [indices[i] for i in range(train_lim)]
    val_ind = [indices[i] for i in range(train_lim, p)]
    
    val_mask = [train_mask[i] for i in val_ind]
    train_mask = [train_mask[i] for i in train_ind]
    
    return train_mask, val_mask, train_ind, val_ind