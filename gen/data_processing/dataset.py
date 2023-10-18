import torch
from torch.utils.data import Dataset
import numpy as np
import MDAnalysis as MDA
import os


class TrajData(Dataset):   
    def __init__(self, data_path=None, encoder=None, device="cpu",flattening=False,**kwargs):
        self.device = device
        self.data_path = data_path
        self.encoder = encoder
        self.flattening = flattening
        if data_path is not None:
            self.traj = self.load_traj(data_path)
    def __len__(self):
        return len(self.traj)
    
    def __getitem__(self, idx): 
        data = self.traj[idx]
        if self.encoder is not None:
            data = self.encoder(torch.tensor(data))
        return data

    def update_data(self,file,append=False):
        traj = self.load_traj(file)
        if append:
            self.traj = torch.cat((self.traj,traj),axis=0)
        else:
            self.traj = traj

    def load_traj(self,data_path):
        ext = os.path.splitext(data_path)[-1].lower()
        if ext == ".xyz":
            traj = MDA.coordinates.XYZ.XYZReader(data_path)
            traj = torch.tensor(np.array([np.array(traj[i]) for i in range(len(traj))])).to(self.device)
        elif ext == ".pt":
            traj = torch.tensor(torch.load(data_path)).float().to(self.device)
        elif ext == ".npy":
            traj = np.load(data_path,mmap_mode='c').astype('float32')
        else:
            raise NotImplementedError
        if self.flattening:
            traj = traj.reshape(len(traj),-1)
        return traj
    
class LiveSimulation(Dataset):
    def __init__(self, sampler, encoder=None, batch_size=100,
                  nbatches_per_epoch=1000, device="cuda",flattening=False,**kwargs):
        self.device = device
        self.encoder = encoder
        self.flattening = flattening
        self.sampler = sampler
        self.nbatches_per_epoch = nbatches_per_epoch
        self.batch_size = batch_size
        
    def __len__(self):
        return self.nbatches_per_epoch
    
    def __getitem__(self, idx):    
        data = self.sampler.sample(self.batch_size)
        data = torch.tensor(data).to(self.device)
        if self.encoder is not None:
            data = self.encoder(data)
        if self.flattening:
            data = data.reshape(self.batch_size,-1)
        return data

