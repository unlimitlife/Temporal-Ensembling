# ZCA and MeanOnlyBNLayer implementations copied from
#   https://github.com/TimSalimans/weight_norm/blob/master/nn.py
#
# Modifications made to MeanOnlyBNLayer:
# - Added configurable momentum.
# - Added 'modify_incoming' flag for weight matrix sharing (not used in this project).
# - Sums and means use float32 datatype.

import os
import numpy as np
import torch
from scipy import linalg
import matplotlib.pyplot as plt

def show(i):
    i = i.reshape((32,32,3))
    m,M = i.min(), i.max()
    plt.imshow((i - m) / (M - m))
    plt.show()

class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if x is not None:
            self.fit(x)

    def fit(self, x):
        s = x.shape
        x = x.copy().reshape((s[0],np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x -= m
        #do Global Contrast Normalization, which is quite often applied to image data. 
        #I'll use the L2 norm, which makes every image have vector magnitude 1
        #x = x / np.sqrt((x ** 2).sum(axis=1))[:,None]
        sigma = np.dot(x.T,x) / x.shape[0]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1./np.sqrt(S+self.regularization)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S+self.regularization)))
        self.ZCA_mat = torch.from_numpy(np.dot(tmp, U.T).astype(np.float64)).to(self.device)
        self.inv_ZCA_mat = torch.from_numpy(np.dot(tmp2, U.T).astype(np.float64)).to(self.device)
        self.mean = torch.from_numpy(m.astype(np.float64)).to(self.device)
        self.ZCA_mat = torch.from_numpy(np.load('data/ZCA/zca_matrix.npy')).to(self.device)
        self.mean = torch.from_numpy(np.load('data/ZCA/zca_mean.npy').astype(np.float64)).to(self.device)
        #if not os.path.isdir('data/ZCA'):
            #os.mkdir('data/ZCA')
        #np.save('data/ZCA/zca_matrix', self.ZCA_mat.cpu())
        #np.save('data/ZCA/zca_mean', self.mean.cpu())
        
    def apply(self, x):
        s = x.shape
        x = torch.Tensor(x).to(self.device)
        if isinstance(x, torch.Tensor):
            return (torch.mm(x.flatten(1) - torch.unsqueeze(self.mean,0), self.ZCA_mat).reshape(s)).cpu().numpy()
        else:
            raise NotImplementedError("Whitening only implemented for Torch TensorVariables")
            
    def invert(self, x):
        s = x.shape
        x = torch.Tensor(x).to(self.device)
        if isinstance(x, torch.Tensor):
            return (torch.mm(x.flatten(1), self.inv_ZCA_mat) + torch.unsqueeze(self.mean,0)).reshape(s).cpu().numpy()
        else:
            raise NotImplementedError("Whitening only implemented for Torch TensorVariables")