from sklearn.neighbors import KNeighborsClassifier
import numpy as np

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
    
    
def normalize_adj(adj):

    rowsum = np.sum(adj, 1)
    D_inv_sqrt = np.power(rowsum, -0.5).flatten()
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
    D_mat_inv_sqrt = np.diag(D_inv_sqrt)
    return D_mat_inv_sqrt.dot(adj).dot(D_mat_inv_sqrt)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx