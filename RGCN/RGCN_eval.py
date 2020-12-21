import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from model import GCN, GCN3
from utils import *
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--base_embed", help = "glove, fast or google", default='glove', type= str)
parser.add_argument("--sigma", help = "value of sigma for the gaussian kernel", default= 0.1, type= float)
parser.add_argument("--model", help = " '1' for 2 layer model, '2' for 3 layer model", default= 1, type=int)
parser.add_argument("--data", help = "small or large", default='large', type= str)
parser.add_argument("--sim", help = "how to build similarity graph: gaussian, MSTKNN, nnlsw", default='nnlsw', type= str)
parser.add_argument("--epochs", help = "maximum epochs", default= 200, type=int)
parser.add_argument("--layer1_out", help = "output dimension of layer 1", default= 2000, type=int)
parser.add_argument("--layer2_out", help = "output dimension of layer 2", default= 200, type=int)
parser.add_argument("--lr", help = "learning rate", default = 1e-3, type=float)
parser.add_argument("--val_size", help = "validation set percentage", default = 0.1, type=float)
parser.add_argument("--wd", help = "weight decay, (L2 regularization)", default= 1e-7, type=float)
parser.add_argument("--es", help = "early stop tolerance", default=10, type=int)
parser.add_argument("--alpha", help = "value of alpha for lazy walk", default= 0.2, type= float)
parser.add_argument("--delta", help = "value of delta for MSTKNN", default= 20, type=int)
parser.add_argument("--lazy", help = "use lazy walk or not", default= True, type=bool)
parser.add_argument("--inv_lap", help = "use inverse laplacian or not", default= 0, type=int)
#parser.add_argument("--leaky", help = "nonlinearity in the output layer", default= True, type=bool)

args = parser.parse_args()

input_and_labels = pd.read_csv("../data/finance/priceMat_4000.csv", index_col=0)
glove_mat = pd.read_csv("../data/finance/%sMat_4000.csv" % args.base_embed, index_col=0)

A_kernel_norm = prepare_adj(input_and_labels, method = args.sim, sig = args.sigma, alpha = args.alpha, lazy_flag = args.lazy)

full_list = input_and_labels.index.to_list()
glove_index = glove_mat.index.to_list()
glove_full = pd.DataFrame(np.zeros((4092, 300)))
glove_full.index = input_and_labels.index

train_list = []
test_list = []
for i in range(len(full_list)): #initialize unknown embeddings with zero. 
    if full_list[i] in glove_index:
        glove_full.iloc[i,:] = glove_mat.filter(like = full_list[i], axis = 0).iloc[0,:300].to_list()
        train_list.append(i)
    else:
        test_list.append(i)

print('length of train_list:', len(train_list))
print('length of train_list:', len(test_list))

y = input_and_labels.iloc[:,-1] #save all the labels both (p and q)
X = glove_full

train_mask = np.array(train_list)
np.random.shuffle(train_mask)
val_mask = train_mask[int(len(train_mask)*0.9):]
train_mask = train_mask[:int(len(train_mask)*0.9)]


print('training: %i points' % len(train_mask))
print('validation: %i points' % len(val_mask))

                       
layer1_out_dim = args.layer1_out # Number of output neurons in the first layer
layer2_out_dim = args.layer2_out # Number of output neurons in the first layer

d = 300

glove_full = torch.FloatTensor(glove_full.values).cuda() 

adj_mat = torch.FloatTensor(A_kernel_norm).cuda() # put the similarity matrix on GPU

X = torch.FloatTensor(X.values).cuda()

if args.model == 1: #2 layer GCN
    model = GCN(adj_mat, d, layer1_out_dim).cuda()
elif args.model == 2: #3 layer GCN
    model = GCN3(adj_mat, d, s, layer1_out_dim, layer2_out_dim).cuda()
    
criterion = nn.MSELoss()
#criterion = nn.L1Loss(reduction = 'sum')
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay= args.wd)


losses = []
prev_val_loss = 10000000  # set initial validation loss
es_count = 0  # initialize early stopping counter to 0


for i in range(args.epochs): # default max epoch 2000
    
    optimizer.zero_grad()

    outputs = model(X)

    loss = criterion(outputs[train_mask], glove_full[train_mask])
    val_loss = criterion(outputs[val_mask], glove_full[val_mask])
    #loss = -(outputs[train_mask]*glove_mat[train_ind]).sum()
    #val_loss = -(outputs[val_mask]*glove_mat[val_ind]).sum()
    
    if i % (args.epochs // 10) == 0:
        print('EPOCH', i + 1)
        print('train loss:', loss.item())
        print('val loss:', val_loss.item())

    loss.backward()
    optimizer.step()

    if val_loss > prev_val_loss:
        es_count += 1
        #if es_count > 5:
            #print("early stop count:", es_count)
        
        if es_count == args.es:
            print('early stopping...')
            break
    else:
        es_count = 0

    prev_val_loss = val_loss

    losses.append(loss.item())
    

Y_q_embeds = outputs[test_list].detach().cpu().numpy() #Get imputed embedding Y_q
Y_p_embeds = X[train_list].detach().cpu().numpy()

print('Y_p_embeds shape:', Y_p_embeds.shape)
print('Y_q_embeds shape:', Y_q_embeds.shape)


X = np.r_[Y_p_embeds, Y_q_embeds] # Concatenate Y_p and Y_q

print('X shape:', X.shape)

X_uncommon = pd.read_csv("../data/finance/" + args.base_embed + "Mat_4000_uncommon.csv", index_col = 0)
word_names = [full_list[i] for i in train_list] + [full_list[i] for i in test_list] + X_uncommon.index.to_list()

print('X_uncommon shape:', X_uncommon.shape)
print('wordnames length:', len(word_names))

to_save = pd.DataFrame(np.r_[X, X_uncommon.values])
to_save.index = word_names
to_save.to_csv("../data/RGCN_embeds/GCN_fin_" + args.base_embed + '_' + str(args.delta) + ".txt", sep = ' ', header = False) # for language model task

y = y.values[np.r_[np.array(train_list), np.array(test_list)]]

for n in [2,5,8,10,15]:
    
    if args.data == "small":
        KNN(X,y,n)
    else:
        KNN_large(X,y,n)