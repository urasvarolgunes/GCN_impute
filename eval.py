import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from model import GCN
from utils import KNN, get_train_and_val_mask, prepare_adj
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--layer1_out", help = "output dimension of layer 1", default= 200, type=int)
parser.add_argument("--lr", help = "learning rate", default=0.00001, type=float)
parser.add_argument("--wd", help = "weight decay, (L2 regularization)", default=0.00001, type=float)
parser.add_argument("--epochs", help = "how many epochs to train", default=20000, type=int)
parser.add_argument("--es", help = "early stop tolerance", default=20, type=int)
parser.add_argument("--base_embed", help = "glove, fast or google", default='glove', type= str)
parser.add_argument("--sigma", help = "value of sigma for the gaussian kernel", default= 1., type= float)
args = parser.parse_args()


corr_and_labels = pd.read_csv("./data/sp500/affMat.csv", index_col=0) # corr. matrix, labels in the last column

A_kernel_norm = prepare_adj(corr_and_labels) #prepare the adjacency matrix

y = corr_and_labels['y'] #save all the labels both (p and q)

corr_mat = corr_and_labels.drop('y', axis = 1) #drop the label column

company_names_all = corr_mat.index.to_list() #save the company names to filter the known p rows


glove_mat = pd.read_csv("./data/sp500/%sMat.csv" % args.base_embed, index_col=0) #get base embeddings (Y_p)
Y_p_labels = glove_mat['300'] # save labels for Y_p
company_names_p = glove_mat.index.to_list() # save the company names with known embeddings
glove_mat = glove_mat.drop(['300','301'], axis = 1) #drop the labels and word frequencies

train_mask = [] # to keep the indices of p (known embeddings)
Y_q_index = [] # to keep the indices of q (unknown embeddings)

for i in range(len(corr_mat)):
    
    if company_names_all[i] in company_names_p:
        train_mask.append(i)
    else:
        Y_q_index.append(i)

train_mask, val_mask, train_ind, val_ind = get_train_and_val_mask(train_mask)

print('training: %i points' % len(train_mask))
print('validation: %i points' % len(val_mask))

                       
layer1_out_dim = args.layer1_out # Number of output neurons in the first layer
s = glove_mat.shape[1] #save the shape of base embeddings
d = corr_mat.shape[1]

glove_mat = torch.FloatTensor(glove_mat.to_numpy()).cuda() 

adj_mat = torch.FloatTensor(A_kernel_norm).cuda() # save the similarity matrix in GPU

corr_mat = torch.FloatTensor(corr_mat.to_numpy()).cuda()

model = GCN(adj_mat, d, s, layer1_out_dim).cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay= args.wd)


losses = []
prev_val_loss = 10000000
es_count = 0

for i in range(args.epochs):
    
    optimizer.zero_grad()

    outputs = model(corr_mat) #input is the correlation matrix

    loss = criterion(outputs[train_mask], glove_mat[train_ind])
    val_loss = criterion(outputs[val_mask], glove_mat[val_ind])

    if i % 50 == 0:
        print('EPOCH', i + 1)
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
    

Y_q_embeds = outputs[Y_q_index].detach().cpu().numpy() #Get imputed embedding Y_q
Y_q_labels = y[Y_q_index] # get Y_q labels
y = np.r_[Y_p_labels, Y_q_labels] # permute the labels accordingly
Y_p_embeds = glove_mat.detach().cpu().numpy()
X = np.r_[Y_p_embeds, Y_q_embeds] # Concatenate Y_p and Y_q


for n in [2,5,8,10,15]:
    KNN(X,y,n)

#plt.plot(losses)