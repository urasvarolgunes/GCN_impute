'''
Accelerate code running via CPU multiprocessing. 
Shibo Yao, Oct. 25 2020
In here we assume there is no modification on the graph like what I did in
Latent Semantic Imputation, as the neural units and backpropogation in GCN
can handle the deviation of those known guys. 
A comparison between w/without modification on graph can be carried out later.
'''
import numpy as np
import pandas as pd #can remove when not testing
from scipy.optimize import nnls
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import multiprocessing as mp



def dis_base(pid, index, return_dic, x):
    '''
    the base for Euclidean distance matrix calculation
    pid: process ID, index: matrix row index, return_dic: result container
    x: feature matrix being shared by all processes
    '''
    p = len(index)
    n = x.shape[0]
    small_dis = np.zeros([p,n])
    for i in range(p):
        vec = x[index[i]]
        small_dis[i] = [np.linalg.norm(vec-x[j]) for j in range(n)]

    return_dic[pid] = small_dis


def distanceEuclidean(x, Q_index, n_jobs, func=dis_base):
    '''
    multiprocessing Euclidean distance matrix calculation
    x: feature matrix, Q_index: index of unknown guys(to impute)
    for now always use the full index for Q_index
    n_jobs: number of jobs, default number of processes on CPU
    '''
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    index_list = np.array_split(Q_index, n_jobs)
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(i,index_list[i],return_dic,x))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()

    dis_mat = [return_dic[i] for i in range(n_jobs)]
    dis_mat = np.concatenate(dis_mat, axis=0)
    
    return dis_mat


def kerGauss(dis, sigma):
    '''
    similarity matrix given by Gaussian kernel
    dis: Euclidean matrix, sigma: same as in Gaussian kernel
    '''
    s = sigma**2
    return np.exp(-dis ** 2 / s)


def MST(dis, Q_index):
    '''
    minimum spanning tree based on Euclidean matrix
    return a zero-one graph
    '''
    dis = dis[:, Q_index]
    d = csr_matrix(dis.astype('float'))
    Tcsr = minimum_spanning_tree(d)
    del d
    mpn = Tcsr.toarray().astype(float)
    del Tcsr
    mpn[mpn!=0] = 1
    n = dis.shape[0]

    for i in range(n):
        for j in range(n):
            if mpn[i,j] != 0:
                mpn[j,i] = 1

    return mpn


def mst_knn_base(pid, index, dis, mpn, delta, return_dic):
    '''
    base for minimum-spanning-tree-k-nearest-neighbor graph
    pid: process ID, index: --, mpn: minimum spanning tree
    dis: Euclidean distance matrix, delta: node in-degree
    return_dic: result container
    '''
    n = len(index)
    small_graph = mpn[index]
    
    for i in range(n):
        ind = index[i]
        nn_index = np.argsort(dis[ind])[1:(delta+1)]
        degree = small_graph[i].sum()
        j = 0
        while degree < delta:
            if small_graph[i, nn_index[j]] == 0:
                small_graph[i, nn_index[j]] = 1
                degree += 1
            j += 1

    return_dic[pid] = small_graph


def MSTKNN(dis, Q_index, delta, n_jobs, spanning=True, func=mst_knn_base):
    '''
    minimum spanning tree k nearest neighbor graph
    when spanning is false, it degenrates to KNN graph, hence delta is k
    zero - one graph
    '''
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu
    
    mpn = np.zeros(dis.shape)
    if spanning:
        mst = MST(dis, Q_index)
        mpn[:,Q_index[0]:] = mst
    
    index_list = np.array_split(range(len(Q_index)), n_jobs)
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(i,index_list[i],dis,mpn,delta,return_dic))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()

    graph = [return_dic[i] for i in range(n_jobs)]
    graph = np.concatenate(graph, axis=0)
    
    return graph

    
def nnlsw(X, graph, pid, sub_list, return_dic, epsilon):
    '''
    X: feature matrix, graph: what ever zero-one graph
    epsilon: small nonnegative that cures singularity
    '''
    nrows = len(sub_list)
    ncols = graph.shape[1]
    W = np.zeros((nrows, ncols))
    for i in range(nrows):
        ind_i = sub_list[i]
        vec = X[ind_i]#b vector in scipy documentation
        gvec = graph[ind_i]
        indK = [j for j in range(ncols) if gvec[j] == 1]
        delta = len(indK) 
        mat = X[indK]#A matrix in scipy documentation
        w = nnls(mat.T, vec)[0]#return both weights and residual
        if epsilon is not None:
            tmp = w[w!=0].copy()
            w = w + epsilon*min(tmp)#all neighbors nonzero
        if sum(w)==0: #in case funny things happen
            w = np.ones(len(w))
        w = w/sum(w) #need to normalize, w bounded between 0 and 1
    
        for ii in range(delta):
            W[i, indK[ii]] = w[ii]
    
    return_dic[pid] = W


def multicoreNNLS(X, graph, Q_index, n_jobs, epsilon=1e-1, func=nnlsw):
    '''
    nonnegative least square 
    return a weighted (unnormalized) nonnegative graph
    '''
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify correct job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu

    graph_list = np.array_split(range(len(Q_index)), n_jobs)#default axis=0
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(X,graph,i,graph_list[i],return_dic,epsilon))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()

    W_p = np.eye(X.shape[0]-graph.shape[0], graph.shape[1])
    W_q = [return_dic[i] for i in range(n_jobs)]
    W_q = np.concatenate(W_q, axis=0)

    return np.concatenate([W_p,W_q], axis=0)


def RandomWalkNormalize(A):
    D = A.sum(1) # row sum, degree vector
    return A / D.reshape(-1,1)


def lazy(W, alpha=.5):
    '''
    lazy walk, add self loop to nodes / add constant to diagonal
    alpha: the weight of self-loops
    '''
    return W*(1-alpha) + np.eye(W.shape[0])*alpha




if __name__ == '__main__':
    df = pd.read_csv("../data/sp500/affMat.csv", index_col=0)
    X = df.values[:,:-1] #consider X a graph or a feature matrix, both fine
    np.fill_diagonal(X,0) #set diagonal to zero / remove self loops
    Q_index = range(X.shape[0]) # for now always use this

    dis = distanceEuclidean(X, Q_index, n_jobs=-1)
    similarity = kerGauss(dis, sigma=1.) #try different sigma
    
    ## some examples
    # origianl similarity matrix, using gaussian kernel, row normalize
    A_kernel_norm = RandomWalkNormalize(similarity)
    # MSTKNN graph, using gaussian kernel, row normalize
    A_MSTKNN = MSTKNN(dis,Q_index,delta=20,n_jobs=-1,spanning=True)
    A_MSTKNN_ker = A_MSTKNN*similarity
    A_MSTKNN_norm = RandomWalkNormalize(A_MSTKNN_ker)
    # KNN graph, using gaussian kernel, row normalize
    A_KNN = MSTKNN(dis,Q_index,delta=20,n_jobs=-1,spanning=False)
    A_KNN_ker = A_KNN*similarity
    A_KNN_norm = RandomWalkNormalize(A_KNN_ker)
    # MSTKNN graph, NNLS, row normalize
    A_MSTKNN_nnls = multicoreNNLS(X,A_MSTKNN,Q_index,n_jobs=-1)
    # KNN graph, NNLS, row normalize
    A_KNN_nnls = multicoreNNLS(X,A_KNN,Q_index,n_jobs=-1)
    
    print("kernel similarity row normalize")
    print(abs(A_kernel_norm[0].sum()-1) < 1e-7, (A_kernel_norm<0).sum()==0)
    print("MSTKNN kernel row normalize")
    print(abs(A_MSTKNN_norm[0].sum()-1) < 1e-7, (A_MSTKNN_norm<0).sum()==0)
    print("KNN kernel row normalize")
    print(abs(A_KNN_norm[0].sum()-1) < 1e-7, (A_KNN_norm<0).sum()==0)
    print("MSTKNN NNLS row normalize")
    print(abs(A_MSTKNN_nnls[0].sum()-1) < 1e-7, (A_MSTKNN_nnls<0).sum()==0)
    print("KNN NNLS row normalize")
    print(abs(A_KNN_nnls[0].sum()-1) < 1e-7, (A_KNN_nnls<0).sum()==0)
    
    print(A_KNN_nnls.sum(1)) # can have some float error
    
    A_lazy = lazy(A_KNN_nnls, alpha=.5) # can switch to any A_* above
    print("lazy walk matrix")
    print(abs(A_lazy[0].sum()-1) < 1e-7, (A_lazy<0).sum()==0)
