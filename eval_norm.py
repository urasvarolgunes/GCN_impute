import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from model import GCN
from utils import KNN, normalize_adj
import numpy as np


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--layer1_out", help = "output dimension of layer 1", default= 200, type=int)
parser.add_argument("--lr", help = "learning rate", default=0.00001, type=float)
parser.add_argument("--wd", help = "weight decay, (L2 regularization)", default=0.00001, type=float)
parser.add_argument("--epochs", help = "how many epochs to train", default=10000, type=int)
parser.add_argument("--es", help = "early stop tolerance", default=20, type=int)
parser.add_argument("--base_embed", help = "glove, fast or google", default='glove', type= str)
args = parser.parse_args()


adj_and_labels = pd.read_csv("./LatentSemanticImputation/word_classification/matrices/affMat.csv", index_col=0) 
y = adj_and_labels['y']

adj_mat = adj_and_labels.drop('y', axis = 1)
adj_mat_index = adj_mat.index.to_list()

glove_mat = pd.read_csv("./LatentSemanticImputation/word_classification/matrices/%sMat.csv" % args.base_embed, index_col=0)
Y_p_labels = glove_mat['300']
glove_mat_index = glove_mat.index.to_list()
glove_mat = glove_mat.drop(['300','301'], axis = 1)

train_mask = []
Y_q_index = []

for i in range(len(adj_mat)):
    
    if adj_mat_index[i] in glove_mat_index:
        train_mask.append(i)
    else:
        Y_q_index.append(i)

train_ind = int(0.9*len(train_mask))

val_mask = train_mask[train_ind:]
train_mask = train_mask[:train_ind]

print('training: %i points' % len(train_mask))
print('validation: %i points' % len(val_mask))

                       
layer1_out_dim = args.layer1_out
s = glove_mat.shape[1]

glove_mat = torch.FloatTensor(glove_mat.to_numpy()).cuda()

corr_mat = torch.FloatTensor(adj_mat.to_numpy()).cuda()
adj_mat = torch.FloatTensor(normalize_adj(adj_mat.to_numpy())).cuda()


model = GCN(adj_mat, s, layer1_out_dim).cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay= args.wd)


losses = []
prev_val_loss = 10000000
es_count = 0

for i in range(args.epochs):

    print('EPOCH', i)
    optimizer.zero_grad()

    outputs = model(corr_mat)

    loss = criterion(outputs[train_mask], glove_mat[:train_ind])
    val_loss = criterion(outputs[val_mask], glove_mat[train_ind:])

    print('train loss:', loss.item())
    print('val loss:', val_loss.item())

    loss.backward()
    optimizer.step()

    if val_loss > prev_val_loss:
        es_count += 1
        print("early stop count:", es_count)
        # torch.save(model.state_dict(), PATH)
        
        if es_count == args.es:
            print('early stopping...')
            break

    prev_val_loss = val_loss

    losses.append(loss.item())
    

Y_q_embeds = outputs[Y_q_index].detach().cpu().numpy()
Y_q_labels = y[Y_q_index]
y = np.r_[Y_p_labels, Y_q_labels]
X = np.r_[glove_mat.detach().cpu().numpy(), Y_q_embeds]


for n in [2,5,8,10,15]:
    KNN(X,y,n)

#plt.plot(losses)