import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from model import GCN
from utils import KNN
import numpy as np
import sys

glove_mat = pd.read_csv("./LatentSemanticImputation/word_classification/matrices/%sMat.csv" % str(sys.argv[1]), index_col=0)

print(sys.argv[1])

y = glove_mat['300'].values
X = glove_mat.drop(['300','301'], axis = 1).values

for n in [2,5,8,10, 15]:
    KNN(X,y,n)

#plt.plot(losses)