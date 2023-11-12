import torch
from torchsde import sdeint, sdeint_adjoint
import torch.nn as nn 
import matplotlib.pyplot as plt
from dm.simulations import utils

import numpy as np

class SDE_helper(nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"
    def __init__(self,f,g,subtract_t_from=None, shape=None,**kwargs):
        super().__init__()
        self.f = self.flattened_func_wrapper(f,shape,subtract_t_from=subtract_t_from,**kwargs)
        self.g = self.flattened_func_wrapper(g,shape,subtract_t_from=subtract_t_from)
    def flattened_func_wrapper(self,func,shape=None, subtract_t_from=None,**kwargs):
        def f(t,x):
            if shape is not None:
                x = x.reshape([-1]+list(shape))
            if subtract_t_from is not None:
                t = subtract_t_from-t
                dt_sign = -1
            else:
                dt_sign = 1
            t = t*torch.ones(len(x)).to(x.device)
            return dt_sign*func(t=t,x=x,**kwargs).reshape(len(x),-1)
        return f

   
class SDESolver():
    # solve the sde dx = f(x,t)dt + g(x,t)dw
    def __init__(self,f,g,method="euler",adjoint=False,dt=1e-3):
        self.f = f
        self.g = g 
        self.method = method
        self.solver = sdeint_adjoint if adjoint else sdeint
        self.dt = dt

    def solve(self,x0,t_init:float,t_final,t_range=(0,1),save_sample_traj=True,**kwargs):
        shape = torch.tensor(x0.shape[1:])
        if isinstance(t_final,float):
            t_final = torch.tensor([t_final]).to(x0.device)
        idx = torch.argmax(torch.abs(t_final-t_init))
        all_t = torch.cat([torch.tensor([t_init]).to(x0.device),t_final])
        if t_final[idx] > t_init: #integrating sde forwards given the initial condition x(t0)
            fn = SDE_helper(self.f,self.g,shape=shape,**kwargs)
            sorted_t, sorted_idx = torch.sort(all_t)
            unique_t, inverse_indices = torch.unique_consecutive(sorted_t, return_inverse=True)
        else: #integrating sde backwards given the initial condition x(t1)
            sorted_t, sorted_idx = torch.sort(all_t,descending=True)
            fn = SDE_helper(self.f,self.g,shape=shape,subtract_t_from=t_range[0]+t_range[1],**kwargs)
            unique_t, inverse_indices = torch.unique_consecutive(sorted_t, return_inverse=True)
            unique_t = t_range[0]+t_range[1]-unique_t
        d = torch.arange(len(all_t)).to(x0.device)
        order = sorted_idx.clone().scatter_(0, sorted_idx, d)
        order = order[order!=0]
        x0 = x0.reshape(len(x0),-1)
        if save_sample_traj:
            t = torch.linspace(t_init,float(t_final[0]),100).to(x0.device)
            traj_sample = sdeint(fn, x0, t, dt=self.dt, method=self.method)
            utils.write_coord("forward_traj.xyz",traj_sample.reshape([-1,len(x0)]+list(shape))[:,0],nparticles=20)
            np.save("forward_traj.npy",traj_sample.detach().cpu().numpy())
        traj = sdeint(fn, x0, unique_t, dt=self.dt, method=self.method)
        traj = traj[inverse_indices]
        traj = traj[order]
        diag_indices = torch.arange(len(x0))
        x = traj[diag_indices,diag_indices]
        x = x.reshape([-1]+list(shape))
        return x


class NoiseSchedule():
    def __init__(self,min=0.1,max=20,t_range=(0,1), type="quadratic"):
        '''
        min: noise at time 0
        max: noise at time 1
        t_range: range of integration
        '''
        print("NoiseSchedule: min={}, max={}, t_range={}".format(min,max,t_range))
        self.t_range = t_range
        if type == "linear":
            self.beta_schedule = FixedLinearSchedule(min=min, max=max)
        elif type == "quadratic":
            self.beta_schedule = QuadraticSchedule(min=min, max=max)
    
    def rescale(self,t): #rescale t to be between 0 and 1
        return (t-self.t_range[0])/(self.t_range[1]-self.t_range[0])
    
    def beta(self,t):
        return self.beta_schedule(self.rescale(t))
    
    def alpha_cumprod(self,t, t_init=None):
        if t_init is None:
            t_init = self.t_range[0]
        return torch.exp(-self.beta_schedule.cumulate(self.rescale(t),t_init=self.rescale(t_init)))
    


class QuadraticSchedule(nn.Module):
    def __init__(self, min=0.1,max=20):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, t):
        return self.min + (self.max - self.min) * t ** 2
    
    def _cumulate_from_0(self,t):
        if isinstance(t,float):
            t = torch.tensor([t])
        return 1/3 * t ** 3 * (self.max - self.min) + t * self.min
    
    def cumulate(self,t, t_init=0):
        return self._cumulate_from_0(t) - self._cumulate_from_0(t_init)
    
class FixedLinearSchedule(nn.Module):
    def __init__(self,min=0.1, max=20):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, t):
        return self.min + (self.max - self.min) * t
    
    def _cumulate_from_0(self,t):
        return 0.5 * t ** 2 * (self.max - self.min) + t * self.min
    
    def cumulate(self,t, t_init=0):
        return self._cumulate_from_0(t) - self._cumulate_from_0(t_init)
    
class LearnedLinearSchedule(nn.Module):
    def __init__(self, min=0.1, max=15):
        super().__init__()
        self.b = nn.Parameter(torch.tensor(min).float())
        self.w = nn.Parameter(torch.tensor(max - min).float())

    def forward(self, t):
        return self.b + self.w.abs() * t
    
    def _cumulate_from_0(self,t):
        return 0.5 * t ** 2 * self.w.abs()  + t * self.b
    
    def cumulate(self,t, t_init=0):
        return self._cumulate_from_0(t) - self._cumulate_from_0(t_init)
    

if __name__=="__main__":
    schedule = NoiseSchedule(type="linear",max=20)
    t = torch.linspace(0,1,100)
    beta = schedule.beta(t)
    alpha = schedule.alpha_cumprod(t)
    plt.plot(t,beta.detach().cpu().numpy())
    plt.show()
    plt.close()
    plt.plot(t,torch.sqrt(alpha).detach().cpu().numpy())
    plt.savefig("mean_schedule.png")

