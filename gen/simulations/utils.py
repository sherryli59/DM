import MDAnalysis as MDA
import numpy as np
import os
import torch

def load_position(file,flatten=False):
    ext = os.path.splitext(file)[-1].lower()
    if ext == ".xyz":
        traj = MDA.coordinates.XYZ.XYZReader(file)
        pos = torch.from_numpy(np.array([np.array(traj[i]) for i in range(len(traj))])).flatten(start_dim=1)
    elif ext == ".pt":
        pos = torch.tensor(torch.load(file)).float()
    elif ext == ".npy":
        pos = torch.tensor(np.load(file)).float()
    else:
        raise NotImplementedError
    if flatten:
        pos = pos.flatten(start_dim=1)
    return pos

def read_coord(dir,format="torch"):
    with open(dir, 'rb') as coord:
        n_atoms=int(coord.readline())
        counter=0
        coord.seek(0)
        pos=[]
        while True:
            line = coord.readline()
            if not line:
                break
            if (counter%(n_atoms+2)==0):
                pos.append(np.zeros((n_atoms,3))) 
            if (counter%(n_atoms+2)>1): 
                pos[-1][counter%(n_atoms+2)-2]=line.split()[1:4]
            counter+=1
        if format=="torch":
            pos=torch.from_numpy(np.array(pos))
        else:
            pos=np.array(pos)
    return pos
    
def write_lammps_coord(file_dir,traj,nparticles,boxlength=None):
    traj=traj.reshape((-1,nparticles,3))
    with open(file_dir, 'a') as pos:
        for j in range(len(traj)):
                atom_index=np.arange(nparticles)
                type_index=np.ones(nparticles)
                config = np.column_stack((atom_index, type_index, traj[j].reshape((-1, 3)).cpu()))
                np.savetxt(pos, config, fmt=['%u','%u', '%.5f', '%.5f', '%.5f'])

def write_coord(file_dir,traj,nparticles=None,boxlength=None,append=False):
    if nparticles is not None:
        traj=traj.reshape((-1,nparticles,3))
    if not append:
        with open(file_dir, 'w') as pos:
            pass
    with open(file_dir, 'a') as pos:
        for j in range(len(traj)):
                #U=LJ_potential(traj[j],boxlength,cutoff=2.7)
                pos.write('%d\n'%nparticles)
                pos.write(' Atoms\n')
                #pos.write('U: %d\n' % U)
                atom_index=np.ones(nparticles)
                config = np.column_stack((atom_index, traj[j].reshape((-1, 3)).cpu()))
                np.savetxt(pos, config, fmt=['%u', '%.5f', '%.5f', '%.5f'])
            
def set_input_params(params, template, input):
    with open(template, "r") as template:
        with open (input,"w") as output:
            for line in template:
                output.write(line.format(**params))

def subsample(data,nsamples,device="cpu",random=True):
    if random:
        total_n = len(data)
        indices = torch.randint(total_n,[nsamples]).to(device)
        return data.index_select(0,indices)
    else:
        return data[:nsamples]
