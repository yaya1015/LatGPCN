import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_sparse import spmm
from torch_scatter import scatter_add

class LatGPCN(nn.Module):
    def __init__(self, nfeat, args, nclass, norm_H=False):
        super(LatGPCN, self).__init__()

        self.gc1 = LatGPCLayer(nfeat, args.hidden, args, args.lamda1, norm_H)
        self.gc2 = LatGPCLayer(args.hidden, nclass, args, args.lamda2, norm_H)
        self.dropout = args.dropout

    def forward(self, x, edge, A_value):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, edge, A_value))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge, A_value)
        return F.log_softmax(x, dim=1)


class LatGPCLayer(Module):

    def __init__(self, in_features, out_features, args, lamba, norm_H=False, bias=True):
        super(LatGPCLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lamba = 0.5*lamba
        self.gamma = args.gamma
        self.iter = args.iter
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.norm_H = norm_H
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def NormalizeS(self, S_value, edge, X):
        deg = scatter_add(S_value, edge[0], dim=0, dim_size=X.shape[0])
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        v = deg_inv_sqrt[edge[0]] * S_value
        return v

    def NormalizeH(self, mx):
        norm_inv = torch.norm(mx,dim=1).pow(-1)
        norm_inv[norm_inv == float('inf')] = 0
        mx = mx*norm_inv.unsqueeze(1)
        return mx

    def computeH(self, S_value, edge, X):
        S_value = self.NormalizeS(S_value, edge, X)
        return spmm(edge, S_value, X.shape[0], X.shape[0], X)

    def computeS(self, feat, X, edge, A_value, XX_value):
        HH = torch.sum(torch.pow(feat,2),1).expand(X.shape[0],-1).t()
        HH_value = HH[edge[0],edge[1]]
        HX_value = torch.matmul(feat[edge[0],:].unsqueeze(1), X[edge[1],:].unsqueeze(2)).squeeze()
        B = HH_value-2*HX_value+XX_value
        return F.relu(A_value-self.lamba*B)

    def forward(self, X, edge, A_value):

        X = torch.mm(X, self.weight)
        X = X + self.bias if self.bias is not None else X

        H,S_value = X.clone(),A_value.clone()
        XX = torch.sum(torch.pow(X,2),1).expand(X.shape[0],-1)
        XX_value = XX[edge[0],edge[1]]
        for i in range(self.iter):
            S_value = self.computeS(H, X, edge, A_value, XX_value)
            H = self.computeH(S_value, edge, X)

        H = self.NormalizeH(H) if self.norm_H else H
        output = (H+self.gamma*X)/(1+self.gamma)

        return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'