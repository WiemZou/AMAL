import torch
from torch.autograd import Function
import numpy as np
import pandas as pd
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self,length,batch,dim,latent,dimout):
        super(RNN, self).__init__()
        self.length=length
        self.batch=batch
        self.dim=dim
        self.dimout=dimout
        self.latent=latent
        self.linear = torch.nn.Linear(self.dim,self.latent)
        self.hidden = torch.nn.Linear(self.latent,self.latent)
        self.tanh=torch.nn.Tanh()
#pytorch traite un bacth tout seul on met les dim que d'un seul element

    def forward(self,x,h=None):
        if h is None :
            h=torch.zeros(self.batch,self.latent)
            h=h.float()
        H=[]
        H.append(h)
        for time in range(self.length-1):
            elem = x[:,time]
            new_h=self.one_step(elem,h)
            h=new_h.clone()
            self.H.append(h)
        y_x=self.linear.forward(x[:,self.length-1])
        y_h=self.hidden.forward(h)
        return y_h
                
    def one_step(self,x,h):
        y_x=self.linear.forward(x)
        y_h=self.hidden.forward(h)
        y=self.tanh(y_x+y_h)

        return y


if __name__ == '__main__':

    seqSize=50

    df=pd.read_csv('tempAMAL_train.csv')
    col_names=df.columns.values[1:]
    train_x=[]
    train_y=[]
    for city in col_names:
        d = np.array(df[city])
        d = np.where(np.isfinite(d), np.nanmean(d,axis=0), d)
        train_x.append(d[:seqSize])
        train_y.append(city)
    train_x=np.array(train_x)
    train_y=np.array(train_y)
    train_x= torch.from_numpy(train_x.T)
    train_x=train_x.float()

    #train_y=torch.from_numpy(train_y)

    batch=train_x[0].shape[0]
    dim=1

    rnn=RNN(seqSize,batch,dim,10,train_x.shape[0])        
    rnn.forward(train_x)


