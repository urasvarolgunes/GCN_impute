import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    
    def __init__(self, adj_matrix, d, s, layer1_out_dim):
        super(GCN, self).__init__()
        
        self.n = adj_matrix.shape[0]
        
        self.d = d # domain space embedding dim.
        self.s = s # semantic space embedding dim.
        self.layer1_dim = layer1_out_dim
        
        self.linear1 = nn.Linear(self.d, self.layer1_dim)
        self.linear2 = nn.Linear(self.layer1_dim, self.s) 
        self.A = adj_matrix
        
    def forward(self, X):
        
        #Layer 1, output_shape = n x layer1_out
        X = torch.matmul(self.A,X)
        X = self.linear1(X)
        X = F.leaky_relu_(X)
        
        #Layer 2, output_shape = n x s 
        
        X = torch.matmul(self.A, X)
        X = self.linear2(X)
        X = F.leaky_relu_(X)
        
        return X