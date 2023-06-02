from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import numpy as np
import os
from ipdb import set_trace as debug


class KyotoMonth(Dataset):
    def __init__(self,months,unsup=False,train=True,src='../datasets/Kyoto/kyoto_processed/monthly/onehot'):
        self.months = months
        self.unsup = unsup
        self.train = train
        self.src = src
        self.data, self.targets = self.load_data()

    def load_data(self):
        flag = 'train' if self.train else 'test'
        data_months = []
        target_months = []
        for month in self.months:
            month_data= pd.read_parquet(os.path.join(self.src,'{}_{}_subset_onehot.parquet'.format(month,flag))).values
            data_months.append(month_data[:,:-1])
            target_months.append(month_data[:,-1])
        data_months = np.concatenate(data_months,axis=0).astype(np.float32)
        target_months = np.concatenate(target_months,axis=0).astype(np.float32)
        if self.train and self.unsup:
            normal_indexs = np.where(target_months==0.)[0]
            data_months =data_months[normal_indexs]
            target_months = target_months[normal_indexs]
        return torch.FloatTensor(data_months),torch.FloatTensor(target_months)

    def __getitem__(self,index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]

def get_loaders(months,unsup=False,train=True,batch_size=32):
    dataset = KyotoMonth(months,unsup,train)
    if train:
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    else:
        dataloader = DataLoader(dataset,batch_size=dataset.data.shape[0],shuffle=False)
    return dataloader


if __name__=='__main__':
    dataset = KyotoMonth(months=['2009_08', '2008_09'], train=True,unsup=False)
    debug()