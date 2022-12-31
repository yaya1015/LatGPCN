import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from deeprobust.graph.utils import accuracy

class EarlyStop:
    def __init__(self, patience=100, etype="Type1"):
        self.patience = patience
        self.etype = etype
        self.best_val_acc = 0
        self.best_val_loss = 1e+20
        self.counter = 0
        self.best_epoch = 0
        self.output = None

    def judge_stop(self, output, loss_val, acc, epoch):
        if self.etype=="Type1":
            return self.judge_stop_type1(output, loss_val, acc, epoch)
        else:
            return self.judge_stop_type2(output, loss_val, epoch)

    def judge_stop_type1(self, output, loss_val, acc_val, epoch):
        if self.best_val_loss > loss_val:
            self.best_val_loss = loss_val
            self.output = output
            self.best_epoch = epoch

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.output = output
        return False

    def judge_stop_type2(self, output, loss_val, epoch):
        if loss_val <= self.best_val_loss :
            self.output, self.best_val_loss, self.counter, self.best_epoch = output, loss_val, 0, epoch
        else:
            self.counter += 1
        return False if self.counter < self.patience else True


class LatGPCNTrain:

    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
        self.earlystop = EarlyStop(etype=args.stop_type)

    def fit(self, features, edge, value, labels, idx_train, idx_val):
        """Training the network on training set
        """
        for epoch in range(self.args.epochs):
            t = time.time()

            #train
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(features, edge, value)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            self.optimizer.step()

            # val
            self.model.eval()
            output = self.model(features, edge, value)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val]).item()
            acc_val = accuracy(output[idx_val], labels[idx_val]).item()

            #print
            if self.args.debug:
                print('Epoch: {:04d}'.format(epoch+1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'loss_val: {:.4f}'.format(loss_val),
                      'acc_val: {:.4f}'.format(acc_val),
                      'time: {:.4f}s'.format(time.time() - t))

            # Determine whether to stop early
            Flag_stop = self.earlystop.judge_stop(output, loss_val, acc_val, epoch)
            if Flag_stop:
                break

    def test(self, labels, idx_test):
        """Evaluate the peformance on test set
        """
        output = self.earlystop.output
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        if self.args.debug:
            print("Test set results:",
                  "best_epoch: {:d}".format(self.earlystop.best_epoch),
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()