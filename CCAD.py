import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from evaluate import get_highest_scores, get_scores,plot_roc_curve_inout, plot_roc_curve
from load_data import KyotoMonth, get_loaders
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score,roc_curve
import logging
import time
import os
import argparse
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from PIL import Image
from ipdb import set_trace as debug
import torch.nn.functional as F
import pickle

class CCAD_Net(nn.Module):
    def __init__(self,in_dim, rep_dim,temp=1):
        super(CCAD_Net,self).__init__()
        self.rep_dim = rep_dim
        self.temp = temp
        prototype = torch.randn(1,self.rep_dim)
        self.prototype = nn.Parameter(prototype,requires_grad=True)
        self.encoder = nn.Sequential(
            torch.nn.Linear(in_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, rep_dim))

    def phi(self,z,prototype):
        prototype = F.normalize(prototype,dim=1)
        p = torch.exp(-1/self.temp*torch.sum((z-prototype)**2,dim=-1))
        # debug()
        return p

    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z,dim=1)
        return z

class CCADSolver():
    def __init__(self,model, device,replay, k=5):
        self.net = model.to(device)
        self.device = device
        self.replay = replay
        self.k = k
        self.replay_buffer = []
        self.criterion = nn.BCELoss()

    def rehearsal_selection(self,trainloader):
        self.net.eval()
        with torch.no_grad():
            x = trainloader.dataset.data
            x = x.to(self.device)
            z = self.net(x)
            p = self.net.phi(z,self.net.prototype)
            indices = torch.topk(p,self.k)[1]
            rehearsal = x[indices]
            self.replay_buffer.append(rehearsal)

    def compute_homo_loss(self,z,prototype,y):
        p = self.net.phi(z,prototype)
        pred = 1 - p
        loss = self.criterion(pred.view(-1, ), y)
        return loss

    def compute_hetero_loss(self, z, replay_buffer,y):
        t = len(replay_buffer)
        rehearsal = torch.stack(replay_buffer,dim=0) # [t,k,D] z[B,D] y[B,]
        rehearsal_z = self.net(rehearsal)
        p = self.net.phi(z.unsqueeze(1).unsqueeze(1), rehearsal_z.unsqueeze(0))
        y = y.unsqueeze(-1).unsqueeze(-1).repeat([1,t,self.k])
        loss = self.criterion(1-p.view(-1,), y.view(-1,))
        # debug()
        return loss

    def train(self,trainloader, testloader, past_testloaders, n_epochs, lr,lmbda):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        for epoch in tqdm(range(n_epochs)):
            self.net.train()
            loss_batch = 0
            n_batches = 0
            epoch_start_time = time.time()
            for x,y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                z = self.net(x)
                if len(self.replay_buffer)==0:
                    loss = self.compute_homo_loss(z,self.net.prototype,y)
                else:
                    homo_loss = self.compute_homo_loss(z,self.net.prototype,y)
                    hetero_loss = self.compute_hetero_loss(z,self.replay_buffer,y)
                    loss = homo_loss+lmbda*hetero_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_batch += loss.item()
                n_batches += 1
            epoch_train_time = time.time() - epoch_start_time

        auroc = self.test(testloader)
        if self.replay:
            self.rehearsal_selection(trainloader)

        bwts = []
        bwt_mean = None
        if len(past_testloaders)>0:
            bwts = []
            for testloader in past_testloaders:
                bwt = self.test(testloader)
                bwts.append(bwt)
            bwt_mean = round(sum(bwts)/len(bwts),4)

        return auroc, bwt_mean, bwts

    def test(self,testloader):
        self.net.eval()
        all_labels = []
        all_scores = []
        with torch.no_grad():
            for x, y in testloader:
                x = x.to(self.device)
                z = self.net(x)
                p = self.net.phi(z, self.net.prototype)
                pred = (1 - p).view(-1, )
                all_scores.append(pred.detach().cpu().data.numpy())
                all_labels.append(y.detach().cpu().data.numpy())
        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        auroc = roc_auc_score(all_labels, all_scores)
        return round(auroc,4)

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # Training options
    parser.add_argument('--bsize', default=64, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lmbda', default=1., type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--replay', action='store_true')

    # Exp options
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()


    input_dim = 447
    net = CCAD_Net(in_dim=input_dim,rep_dim=64)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    solver = CCADSolver(net,device,args.replay)

    src_path = '../datasets/Kyoto/kyoto_processed/monthly/onehot'
    files = os.listdir(src_path)
    months = [file[0:7] for file in files]
    months = list(set(months))
    months.sort()
    past_testloaders = []

    BWTs = {}
    AUROCs = {}

    for i, month in enumerate(months):
        trainloader = get_loaders(months=[month],unsup=False,train=True)
        testloader = get_loaders(months=[month], unsup=False, train=False)
        if month[-2:]=='12':
            auroc, bwt_mean, bwts = solver.train(trainloader,testloader,past_testloaders,args.epochs, lr=args.lr, lmbda=args.lmbda)
            AUROCs[month]=auroc
            BWTs[month+'_'+str(bwt_mean)]=bwts
            print('Time: {}\t  AUROC: {:.4f} \t BWT: {:.4f}'.format(month, auroc, bwt_mean))
        else:
            auroc, bwt, bwts = solver.train(trainloader,testloader,[],args.epochs, lr=args.lr, lmbda=args.lmbda)
            AUROCs[month]=auroc
            print('Time: {}\t  AUROC: {:.4f}'.format(month, auroc))

        past_testloaders.append(testloader)

    if not os.path.exists('./eval_results'):
        os.mkdir('./eval_results')

    flag = 'replay' if args.replay else 'wo'
    with open('./eval_results/ccad_{}_auroc.pkl'.format(flag), 'wb') as f:
        pickle.dump(AUROCs, f)
    with open('./eval_results/ccad_{}_bwt.pkl'.format(flag), 'wb') as f:
        pickle.dump(BWTs, f)



