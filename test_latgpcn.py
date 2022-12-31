import time
import argparse
import numpy as np
import torch
import os.path as osp
import scipy.sparse as sp
from tqdm import tqdm
import scipy.io as sio
import os.path as osp
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.global_attack import Random
from latgpcn import LatGPCNTrain
from latgpcn_model import LatGPCN
from data_utils import load_wiki, preprocess


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=15, help='Random seed of network.')
parser.add_argument('--aseed', type=int, default=15, help='Random seed of Random attack.')
parser.add_argument('--record', type=bool, default="True", help='Record result into txt.')
parser.add_argument('--stop-type', type=str, default="Type1", help='The type of early stop.')

parser.add_argument('--dataset', type=str, default='cora', choices=['cora', 'citeseer', 'pubmed','cora_ml','wiki'], help='dataset')
parser.add_argument('--attack', type=str, default='meta', choices=['meta', 'random'])
parser.add_argument('--ptb_rate', type=float, default=0.25, help="noise ptb_rate")
parser.add_argument('--iter_all', type=int, default=1, help='Number of executions.')

parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=5000, help='Number of epochs to train.')
parser.add_argument('--iter', type=int, default=3, help='The number of recurrent time R of each block.')
parser.add_argument('--lamda1', type=float, default=1., help='Parameter lamda of first layer.')
parser.add_argument('--lamda2', type=float, default=0.1, help='Parameter lamda of second layer.')
parser.add_argument('--gamma', type=float, default=1.5, help='Parameter gamma.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print("=================================================")
print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# load data
data = load_wiki() if args.dataset=="wiki" else Dataset(root='./data', name=args.dataset, setting='prognn')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

def test(perturbed_adj):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    numv = np.zeros(args.iter_all)
    norm_H = True if args.dataset=="wiki" else False

    for it in range(args.iter_all):
        model = LatGPCN(nfeat=features.shape[1],
                        args=args,
                        nclass=labels.max().item() + 1,
                        norm_H=norm_H)

        edge, value, feat, label = preprocess(perturbed_adj, features, labels, device=device)
        
        latgpcn = LatGPCNTrain(model, args, device)
        latgpcn.fit(feat, edge, value, label, idx_train, idx_val)
        numv[it] = latgpcn.test(label, idx_test)
        print("iter",it,": ",numv[it])
    print("-------------------------------------------------")
    print("Final average result:",round(numv.mean(),4),"std:",round(numv.std(),4))

    if args.record:
        f = open('result.txt','r+')
        f.read()
        f.write(str(round(numv.mean(),4))+'+'+str(round(numv.std(),4))+',')
        f.close()
    

if args.attack == 'random' or args.ptb_rate == 0.:
    attacker = Random()
    n_perturbations = int(args.ptb_rate * (adj.sum()//2))
    attacker.attack(adj, n_perturbations, type='add')
    perturbed_adj = attacker.modified_adj
    test(perturbed_adj)

elif args.attack == 'meta':
    data_filename = osp.join('./data/meta',
                '{}_meta_adj_{}.npz'.format(args.dataset, args.ptb_rate, args.aseed))
    perturbed_adj = sp.load_npz(data_filename)
    test(perturbed_adj)

print("=================================================")