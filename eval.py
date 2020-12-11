import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from model import GCN, GCN3
from utils import *
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--layer1_out", help = "output dimension of layer 1", default= 200, type=int)
parser.add_argument("--layer2_out", help = "output dimension of layer 2", default= 1000, type=int)
parser.add_argument("--lr", help = "learning rate", default = 0.001, type=float)
parser.add_argument("--val_size", help = "validation set percentage", default = 0.1, type=float)
parser.add_argument("--wd", help = "weight decay, (L2 regularization)", default=0.00005, type=float)
parser.add_argument("--es", help = "early stop tolerance", default=50, type=int)
parser.add_argument("--base_embed", help = "glove, fast or google", default='glove', type= str)
parser.add_argument("--sigma", help = "value of sigma for the gaussian kernel", default= 0.1, type= float)
parser.add_argument("--alpha", help = "value of alpha for lazy walk", default= 0.7, type= float)
parser.add_argument("--delta", help = "value of delta for MSTKNN", default= 5, type=int)
parser.add_argument("--lazy", help = "use lazy walk or not", default= True, type=bool)
parser.add_argument("--inv_lap", help = "use lazy walk or not", default= False, type=bool)
parser.add_argument("--sim", help = "how to build similarity graph: gaussian, MSTKNN, nnlsw", default='nnlsw', type= str)
parser.add_argument("--data", help = "small or large", default='large', type= str)
parser.add_argument("--model", help = " '1' for 2 layer model, '2' for 3 layer model", default= 1, type=int)
args = parser.parse_args()

if args.data == "small":
    input_and_labels = pd.read_csv("./data/sp500/affMat.csv", index_col=0) # corr. matrix, labels in the last column
    glove_mat = pd.read_csv("./data/sp500/%sMat.csv" % args.base_embed, index_col=0) #get base embeddings (Y_p)

else:
    input_and_labels = pd.read_csv("./data/finance/priceMat_4000.csv", index_col=0)
    glove_mat = pd.read_csv("./data/finance/gloveMat_4000.csv", index_col=0)

A_kernel_norm = prepare_adj(input_and_labels, method = args.sim, sig = args.sigma, alpha = args.alpha, lazy_flag = args.lazy)

if args.inv_lap:
    print('USING INVERSE LAPLACIAN...\n')
    A_kernel_norm = apply_laplacian(A_kernel_norm)

print(A_kernel_norm)

y = input_and_labels.iloc[:,-1] #save all the labels both (p and q)

X = input_and_labels.iloc[:,:-1] #drop the label column

company_names_all = X.index.to_list() #save the company names to filter the known p rows

Y_p_labels = glove_mat.iloc[:,-2] # save labels for Y_p
company_names_p = glove_mat.index.to_list() # save the company names with known embeddings
glove_mat = glove_mat.iloc[:,:-2] #drop the labels and word frequencies

train_mask = [] # to keep the indices of p (known embeddings)
Y_q_index = [] # to keep the indices of q (unknown embeddings)

for i in range(len(X)):
    
    if company_names_all[i] in company_names_p:
        train_mask.append(i)
    else:
        Y_q_index.append(i)

train_mask, val_mask, train_ind, val_ind = get_train_and_val_mask(train_mask, val_size = args.val_size)

print('training: %i points' % len(train_mask))
print('validation: %i points' % len(val_mask))

                       
layer1_out_dim = args.layer1_out # Number of output neurons in the first layer
layer2_out_dim = args.layer2_out # Number of output neurons in the first layer
s = glove_mat.shape[1] #save the shape of base embeddings
d = X.shape[1]

glove_mat = torch.FloatTensor(glove_mat.values).cuda() 

adj_mat = torch.FloatTensor(A_kernel_norm).cuda() # put the similarity matrix on GPU

X = torch.FloatTensor(X.values).cuda()

if args.model == 1: #2 layer GCN
    model = GCN(adj_mat, d, s, layer1_out_dim).cuda()
elif args.model == 2: #3 layer GCN
    model = GCN3(adj_mat, d, s, layer1_out_dim, layer2_out_dim).cuda()
    
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay= args.wd)


losses = []
prev_val_loss = 10000000  # set initial validation loss
es_count = 0  # initialize early stopping counter to 0

for i in range(10000): # default max epoch 2000
    
    optimizer.zero_grad()

    outputs = model(X) #input is the correlation matrix

    loss = criterion(outputs[train_mask], glove_mat[train_ind])
    val_loss = criterion(outputs[val_mask], glove_mat[val_ind])

    if i % 100 == 0:
        print('EPOCH', i + 1)
        print('train loss:', loss.item())
        print('val loss:', val_loss.item())

    loss.backward()
    optimizer.step()

    if val_loss > prev_val_loss:
        es_count += 1
        if es_count > 5:
            print("early stop count:", es_count)
        # torch.save(model.state_dict(), PATH)
        
        if es_count == args.es:
            print('early stopping...')
            break
    else:
        es_count = 0

    prev_val_loss = val_loss

    losses.append(loss.item())
    

Y_q_embeds = outputs[Y_q_index].detach().cpu().numpy() #Get imputed embedding Y_q
Y_q_labels = y[Y_q_index] # get Y_q labels
y = np.r_[Y_p_labels, Y_q_labels] # permute the labels accordingly
Y_p_embeds = glove_mat.detach().cpu().numpy()
X = np.r_[Y_p_embeds, Y_q_embeds] # Concatenate Y_p and Y_q

np.save("./data/GCN_embeds/GCN_" + args.base_embed + "_4000_" + str(args.delta) + '_' + str(args.sigma), np.c_[X, y]) # save for TSNE, with labels

if args.data == "large":
    total = len(company_names_all)
    X_uncommon = pd.read_csv("./data/finance/" + args.base_embed + "Mat_4000_uncommon.csv", index_col = 0)
    word_names = company_names_p + [company_names_all[i] for i in range(total) if i in Y_q_index] + X_uncommon.index.to_list()
    print('length of word_names:', len(word_names))
    
    to_save = pd.DataFrame(np.r_[X, X_uncommon.values])
    to_save.index = word_names
    print(to_save.shape)
    to_save.to_csv("./data/GCN_embeds/lm_GCN_4000_" + args.base_embed + '_' + str(args.delta) + '_' + str(args.sigma) + ".csv") # for language model task

for n in [2,5,8,10,15]:
    
    if args.data == "small":
        KNN(X,y,n)
    else:
        KNN_large(X,y,n)