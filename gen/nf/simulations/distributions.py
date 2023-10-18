#from lammps import lammps, PyLammps, LMP_STYLE_ATOM, LMP_TYPE_ARRAY
import MDAnalysis as MDA
import torch
import numpy as np
import random
from torch.distributions import MultivariateNormal, Normal
from ctypes import c_double, POINTER
import math
from numbers import Number
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

from dm.simulations import utils



class GaussianMixture:
    def __init__(self, centers, std, npoints=None,dim=3,device="cpu"):
        self.dim = dim
        self.device = device
        if isinstance(centers,str):
            self.centers = utils.load_position(centers).reshape(-1,dim).to(self.device)
        else:
            self.centers=torch.tensor(centers).float().to(self.device)    
        self.ncenters=len(self.centers)
        self.var = (torch.tensor(std)**2).float().to(self.device)
        if self.var.dim()==0:
            self.var = self.var.expand(self.ncenters)
        if npoints == None:
            self.nparticles = self.ncenters
        else:
            self.nparticles = npoints
        self.dist=[]
        for i in range(self.ncenters):
            self.dist.append(MultivariateNormal(self.centers[i], self.var[i]*torch.eye(self.dim).to(self.device)))

    def pair_dist(self,pos):
        pair_vec = (pos.unsqueeze(-2) - pos.unsqueeze(-3))
        pair_dist = torch.linalg.norm(pair_vec.float(), axis=-1)
        n = pair_vec.shape[1]
        xindex,yindex = torch.triu_indices(n,n,1).unbind()
        pair_dist = pair_dist[:,xindex,yindex]
        pair_dist = pair_dist.unsqueeze(1)
        if n % 2 ==0:
            pair_dist = pair_dist.reshape(-1,n-1,n//2)
        else:
            pair_dist = pair_dist.reshape(-1,n,(n-1)//2)
        return pair_dist
    
    def sample(self,nsamples,flatten=False):
        with torch.no_grad():
            if isinstance(nsamples,tuple):
                nsamples=nsamples[0]
            which_dist=torch.tensor([random.randint(0,self.ncenters-1) for _ in range(nsamples*self.nparticles)])
            samples = torch.stack([self.dist[which_dist[i]].sample((1,)) for i in range(nsamples*self.nparticles)])
            samples = samples.reshape((nsamples,self.nparticles,self.dim))
            if flatten:
                return samples.reshape((nsamples,-1))
            else:
                return samples

    def log_prob(self,x):
        x=x.reshape(-1,self.dim)
        for i in range(self.ncenters):
            if i==0:
                prob=1/self.ncenters*torch.exp(self.dist[i].log_prob(x))
            else:
                prob= prob+1/self.ncenters*torch.exp(self.dist[i].log_prob(x))
        return torch.sum(torch.log(prob).reshape(-1,self.nparticles),axis=1)

    def potential(self,x):
        return -self.log_prob(x)
    
    def get_potential(self,x=None):
        if x is not None:
            return self.potential(x)
        else:
            return self.potential(self.position)

    
    def force(self,x):
        require_grad = x.requires_grad
        with torch.enable_grad():
            x.requires_grad_(True)
            pot=self.potential(x)
            force = -torch.autograd.grad(pot,x,torch.ones_like(pot),create_graph=True)[0]
        x.requires_grad_(require_grad)
        return force
    
    def neg_force_clipped(self,x):
        return -torch.clip(self.force(x),-50,50)
    
    def set_position(self,position):
        self.position = position.flatten()


    def get_position(self):
        return self.position

    def set_velocity(self,velocity):
        self.velocity = velocity.flatten()

    def integration_step(self,path_len=1,dt=0.005, init_pos=None, init_velocity=None):
        if init_pos is not None:
            self.set_position(init_pos)
        if init_velocity is not None:
            self.set_velocity(init_velocity)
        self.force = self.force(self.position)
        if dt is None:
            dt=0.005
        for _ in range(path_len):
            new_position = self.position + self.velocity*dt+ self.force/2*(dt**2)
            new_force = self.get_force(self.position)
            self.velocity = self.velocity + dt*(self.force + new_force)/2
            self.force = new_force
            self.position = new_position
        potential = self.potential(self.position)
        return self.position, potential

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

class StandardNormal:
    def __init__(self, input_size, var=1, shape=None, device="cpu"):
        self.device = device
        self.input_size = input_size
        self.dist=MultivariateNormal(torch.zeros(input_size).to(self.device),var* torch.eye(input_size).to(self.device))
        if shape is not None:
            self.shape = list(shape)
        else:
            self.shape=[input_size]

    def sample(self,nsamples, flatten=True):
        if isinstance(nsamples,tuple):
            nsamples=nsamples[0]
        return self.dist.sample((nsamples,)).reshape([nsamples]+self.shape).to(self.device)
    
    def log_prob(self,x):
        x = x.reshape(len(x),-1)
        return self.dist.log_prob(x)
    
    def update_var(self,var):
        self.dist=MultivariateNormal(torch.zeros(self.input_size).to(self.device),var* torch.eye(self.input_size).to(self.device))


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale


class EinsteinCrystal:
    def __init__(self, centers=0, dim=3, shape=None, boxlength=None, std=1.0, device="cpu",cutoff=None):
        self.device = device
        if isinstance(centers,str):
            self.centers = utils.load_position(centers).reshape(-1,dim).to(self.device)
        elif isinstance(centers,torch.Tensor):
            self.centers=torch.tensor(centers).float().to(self.device)
        elif isinstance(centers,float) or isinstance(centers,int):
            assert shape is not None
            self.centers = centers* torch.ones(shape).to(self.device)
        else:
            raise ValueError("centers must be a path to a xyz/npy/pt file or a tensor")
        self.natoms = self.centers.shape[0]

        if isinstance(std,str):
            self.std = utils.load_position(std).reshape(-1,dim).to(self.device)
        elif isinstance(std,float):
            self.std = std * torch.ones_like(self.centers).to(self.device)
        elif isinstance(std,torch.Tensor):
            self.std=torch.tensor(std).float().to(self.device)
        else:
            raise ValueError("variance must be a path to an npy/pt file or a tensor")
        self.dim = dim
        if cutoff is not None:
            cutoff_rescaled = cutoff / torch.min(self.std)
            self.std_noise=TruncatedStandardNormal(-cutoff_rescaled,cutoff_rescaled)
        else:
            self.std_noise = Normal(torch.tensor([0.0]).to(self.device), torch.tensor([1.0]).to(self.device)) 
        self.boxlength = boxlength

    def sample(self,nsamples, flatten=True):
        with torch.no_grad():
            if isinstance(nsamples,tuple):
                nsamples=nsamples[0]
            noise = self.std_noise.rsample((nsamples*self.dim*self.natoms,)).reshape(-1,self.natoms,self.dim).to(self.device)
            noise = noise * self.std.unsqueeze(0)
            samples=self.centers+ noise
            if flatten:
                return samples.reshape(nsamples,-1)
            else:
                return samples
    def log_prob(self,x):
        dev_from_lattice=x.reshape(-1,self.natoms,self.dim)-self.centers
        if self.boxlength is not None:
            dev_from_lattice -= ((torch.abs(dev_from_lattice) > 0.5*self.boxlength)
                * torch.sign(dev_from_lattice) * self.boxlength) 
        dev_rescaled = dev_from_lattice/self.std.unsqueeze(0)
        return torch.sum(self.std_noise.log_prob(dev_rescaled.reshape(-1,self.natoms*self.dim)),dim=-1)   

