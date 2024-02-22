import torch
import math
from gen.simulations import utils
import numpy as np

class SimData():
    def __init__(self, pos_dir=None, device="cpu"):
        self.device = device
        if pos_dir is not None:
            self.traj = utils.load_position(pos_dir).to(device)
    
    def sample(self,nsamples, flatten=False, random=True):
        samples = utils.subsample(self.traj,nsamples, self.device, random=random)
        if flatten:
            return samples.reshape(nsamples,-1)
        else:
            return samples
    def update_data(self,file,append=False):
        traj = self.load_traj(file).float().to(self.device)
        if append:
            self.traj = torch.cat((self.traj,traj),axis=0)
        else:
            self.traj = traj
    def load_traj(self,pos_dir):
        return utils.load_position(pos_dir).to(self.device)
    
class LJ(SimData):
    def __init__(self, pos_dir=None, boxlength=None, device="cuda:0",
                  epsilon=1., sigma=1., cutoff=None, shift=True,
                  periodic=True, max_r=None, dim=3, harm_coeff=0.5):
        super().__init__(pos_dir,device)
        self.epsilon=epsilon
        self.sigma=sigma
        self.cutoff=cutoff
        self.dim = dim
        self.shift=shift
        self.boxlength=boxlength
        self.periodic=periodic
        self.harm_coeff = harm_coeff
        #self.max_r = self.boxlength/2 if max_r is None else max_r

    def potential(self,particle_pos, min_dist=None):
        """
        Calculates Lennard_Jones potential between particles
        Arguments:
        particle_pos: A tensor of shape (n_particles, n_dimensions)
        representing the particle positions
        boxlength: A tensor of shape (1) representing the box length
        epsilon: A float representing epsilon parameter in LJ
        Returns:
        total_potential: A tensor of shape (n_particles, n_dimensions)
        representing the total potential of the system
        """
        pair_vec = self.pair_vec(particle_pos)
        distances = torch.linalg.norm(pair_vec.float(), axis=-1)
        rem_dims = distances.shape[:-2]
        n = distances.shape[-1]
        distances = distances.flatten(start_dim=-2)[...,1:].view(*rem_dims,n-1, n+1)[...,:-1].reshape(*rem_dims,n, n-1)
        scaled_distances = distances
        if min_dist is not None:
            scaled_distances = torch.clamp(scaled_distances,min=min_dist)
        distances_inverse = 1/scaled_distances
        if self.cutoff is not None:
            distances_inverse = distances_inverse-(distances >self.cutoff)*distances_inverse
            pow_6 = torch.pow(self.sigma*distances_inverse, 6)
            if self.shift:
                pow_6_shift = (self.sigma/self.cutoff)**6
                pair_potential = self.epsilon * 4 * (torch.pow(pow_6, 2)
                                        - pow_6 - pow_6_shift**2+pow_6_shift)
            else:
                pair_potential = self.epsilon * 4 * (torch.pow(pow_6, 2)
                                        - pow_6)
        else:
            pow_6 = torch.pow(self.sigma*distances_inverse, 6)
            pair_potential = self.epsilon * 4 * (torch.pow(pow_6, 2)
                                        - pow_6)
        pair_potential = pair_potential *distances_inverse*distances
        total_potential = torch.sum(pair_potential,axis=(-1,-2)) 
        if not self.periodic:
            com = torch.mean(particle_pos,axis=-2)
            rel_pos = particle_pos-com.unsqueeze(1)
            harm_potential = self.harm_coeff*torch.sum(0.5*rel_pos**2,axis=(-2,-1))
            total_potential+=harm_potential
        return total_potential   

    
    def force_clipped(self,particle_pos,max_val=80):
        return self.force(particle_pos).clamp(-max_val,max_val)
    
    def force(self,particle_pos,sig=None, eps=None,min_dist=None):
        """
        Calculates Lennard_Jones force between particles
        Arguments:
            particle_pos: A tensor of shape (n_particles, n_dimensions)
        representing the particle positions
        box_length: A tensor of shape (1) representing the box length
        epsilon: A float representing epsilon parameter in LJ
        
        Returns:
            total_force_on_particle: A tensor of shape (n_particles, n_dimensions)
        representing the total force on a particle
         """
        if eps is None:
            eps = self.epsilon
        if sig is None:
            sig = self.sigma
        pair_vec = self.pair_vec(particle_pos)
        distances = torch.linalg.norm(pair_vec.float(), axis=-1)
        scaled_distances = distances + (distances == 0)
        if min_dist is not None:
            scaled_distances = torch.clamp(scaled_distances,min=min_dist)
        distances_inverse = 1/scaled_distances
        if self.cutoff is not None:
            distances_inverse = distances_inverse-(distances >self.cutoff)*distances_inverse
            pow_6 = torch.pow(sig*distances_inverse, 6)
            force_mag = eps * 24 * (2 * torch.pow(pow_6, 2)
                                    - pow_6)*sig*distances_inverse
        else:
            pow_6 = torch.pow(sig/scaled_distances, 6)
            force_mag = eps * 24 * (2 * torch.pow(pow_6, 2)
                                    - pow_6)*sig*distances_inverse
        force_mag = force_mag * distances_inverse
        force = -force_mag.unsqueeze(-1) * pair_vec
        total_force = torch.sum(force, dim=1)
        com = torch.mean(particle_pos,axis=1)
        rel_pos = particle_pos-com.unsqueeze(1)
        if not self.periodic:
            harm_f = -rel_pos * self.harm_coeff
            total_force += harm_f
        # r = torch.linalg.norm(rel_pos,axis=-1)
        # if self.max_r is not None:
        #      boundary_f = (100*(r>self.max_r)*(self.max_r-r)/(r+1e-6)).unsqueeze(-1)*rel_pos
        #      total_force += boundary_f
        return 2*total_force

    def grad_log_prob(self,particle_pos):
        particle_pos.requires_grad_(True)
        log_prob = self.log_prob(particle_pos)
        return torch.autograd.grad(log_prob,particle_pos,grad_outputs=torch.ones_like(log_prob))[0]
    
    def log_prob(self,x):
        return -self.potential(x)

    def pair_vec(self,particle_pos):
        pair_vec = (particle_pos.unsqueeze(-2) - particle_pos.unsqueeze(-3))
        if self.periodic:
            #to_subtract = ((torch.abs(pair_vec)> 0.5 * self.boxlength)
            #            * torch.sign(pair_vec) * self.boxlength)
            pair_vec = pair_vec - torch.round(pair_vec / self.boxlength) * self.boxlength
        return pair_vec

    def g_r(self,particle_pos, bins=100):
        dim = particle_pos.shape[-1]
        pair_vec = self.pair_vec(particle_pos)
        nsamples = len(pair_vec)
        nparticles = pair_vec.shape[1]
        distances = torch.linalg.norm(pair_vec.float(), axis=-1)
        # remove diagonal zeros
        rem_dims = distances.shape[:-2]
        distances = distances.flatten(start_dim=-2)[...,1:].view(
        *rem_dims,nparticles-1, nparticles+1)[...,:-1].reshape(*rem_dims,nparticles, nparticles-1)
        if self.periodic:
            counts,bins = np.histogram(distances.detach().cpu().numpy(),bins=bins)
            if dim == 2:
                bulk_density = nparticles/(self.boxlength**2)
                areas = math.pi*(bins[1:]**2-bins[:-1]**2)
            elif dim == 3:
                bulk_density = nparticles/(self.boxlength**3)
                areas = 4/3*math.pi*(bins[1:]**3-bins[:-1]**3)
            g_r = counts/nparticles/nsamples/areas/bulk_density
        else:
            com = torch.mean(particle_pos,axis=1)
            dist_from_com = torch.linalg.norm(particle_pos-com.unsqueeze(1),axis=-1)
            com_atom = torch.min(dist_from_com,axis=1)[1]
            distances = distances[torch.arange(distances.shape[0]),com_atom].flatten()
            counts,bins = np.histogram(distances.detach().cpu().numpy(),bins=bins)
            bulk_density = nparticles/(4/3*math.pi*(self.boxlength/2)**3)
            areas = 4*math.pi*((bins[:-1]+bins[1:])/2)**2*(bins[1:]-bins[:-1])
            g_r = counts/nsamples/areas/bulk_density
        return bins, g_r
    
    def apply_constraints(self,particle_pos):
        if self.periodic:
            particle_pos = torch.remainder(particle_pos,self.boxlength)-self.boxlength/2
            return particle_pos
        else:
            com = torch.mean(particle_pos,axis=-2,keepdim=True)
            return particle_pos-com