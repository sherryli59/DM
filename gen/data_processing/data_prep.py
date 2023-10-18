import numpy as np
import mdtraj as md
import torch
import pickle
from math import pi

def prep_dataset(data,dir,name):
    size = len(data)
    indices= np.arange(size)
    np.random.shuffle(indices)
    test_indices = indices[:1000]
    train_indices = indices[1000:]
    np.save(dir+name+"_train.npy",data[train_indices])
    np.save(dir+name+"_test.npy",data[test_indices])

def rescale(data,saving_dir='./',inverse=False):
    #assume data is in the following format:
    distances = data[:,0]
    angles = data[:,1]
    dihedrals = data[:,2]
  
    if inverse:
        with open(saving_dir+'stats.pkl', 'rb+') as f:
            dict = pickle.load(f)
        max_distance = torch.tensor(dict['max_distance']).to(data.device)
        distances = (distances +1)/2*max_distance
        angles = (angles+1)/2*pi
        dihedrals = (dihedrals+1)/2*2*pi
    else:
        max_distance = torch.max(distances)
        distances  = distances /max_distance*2-1
        mean= torch.mean(distances,axis=0)
        std = torch.std(distances,axis=0)

        angles = angles/pi*2 - 1
        mean = torch.cat((mean,torch.mean(angles,axis=0)))
        std= torch.cat((std,torch.std(angles,axis=0)))

        dihedrals = dihedrals/(2*pi)*2 - 1
        mean = torch.cat((mean,torch.mean(dihedrals,axis=0)))
        std = torch.cat((std, torch.std(dihedrals,axis=0)))
        dict = {'mean':mean.cpu().numpy(),'std':std.cpu().numpy()
                ,'max_distance':max_distance.cpu().numpy()}

        with open(saving_dir+'stats.pkl', 'wb+') as f:
            pickle.dump(dict, f)
    return torch.stack((distances,angles,dihedrals),dim=1)

def pca(x):
    x = x.reshape(len(x),-1)
    mu = torch.mean(x,axis=0)
    cov = torch.cov(x.T)
    u,s,v = torch.linalg.svd(cov)
    x_whitened = (torch.diag(1/torch.sqrt(s))@u.T@(x-mu).T).T
    return x_whitened,u,torch.diag(torch.sqrt(s)),mu

if __name__=="__main__":

    dir = "run/"
    filename = dir+"adp"
    traj = md.load_hdf5(filename+".h5")
    data = traj.xyz
    data = torch.tensor(data)
    prep_dataset(data,dir,"pos")
    data = data - torch.mean(data,axis=1).unsqueeze(1) #center the coords
    prep_dataset(data,dir,"pos_centered")
    atomic_number = np.array([atom.element.number for atom in traj.topology.atoms])
    np.save(dir+"atomic_numbers.npy",atomic_number)
    #traj.save_pdb(filename+".pdb")
