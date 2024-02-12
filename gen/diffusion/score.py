import torch
import torch.nn as nn 
import copy

def _batch_mult(coeff,data):
        return torch.einsum(data, [0,...],coeff,[0], [0,...])

def force_wrapper(distribution):
    def f(x):
        #x_decoded = data_handler.decoder(x.clone())
        x_decoded = x.clone()
        f = distribution.force_clipped(x_decoded)
        #f = data_handler.encoder(f)
        return f
    return f

class Correction(nn.Module):
    def __init__(self,distribution,prior="gaussian",duration=0.05,power=4.0, trainable=True):
        super().__init__()
        self.force = force_wrapper(distribution)
        self.prior = prior
        if trainable:
            self.duration = nn.Parameter(torch.tensor([duration]*4))
            self.duration.requires_grad = False
            self.power = nn.Parameter(torch.tensor([power]*4))
            self.power.requires_grad = False
        else:
            self.duration = duration
            self.power = power

    
    def prior_correction_schedule(self,t):
        schedule = torch.zeros_like(t)
        mask = (t>1-self.duration[0])
        schedule[mask] = ((t[mask] -1 + self.duration[0])/self.duration[0])**self.power[0]
        return schedule
    
    def force_correction_schedule(self,t):
        schedule = torch.zeros_like(t)
        mask = (t<self.duration[1])
        schedule[mask] =  (-1.0/self.duration[1]*t[mask] + 1.0)**self.power[1]
        return schedule
    
    def score_correction_schedule(self,t,initial_factor=0):
        schedule = torch.ones_like(t)
        left = max(0.001,min(self.duration[2],1-self.duration[3]))
        schedule[t<left] = 1-(1-initial_factor)*(-1.0/left*t[t<left] + 1.0)**self.power[2]
        right = min(0.999,max(self.duration[2],1-self.duration[3]))
        schedule[t>right] = 1 - ((t[t>right] -right)/(1-right))**self.power[3]
        return schedule

    
    def forward(self,x,t,score,lattice=None):
        if self.duration is None or self.duration.sum() == 0:
            return score
        else:
            force_correction = _batch_mult(self.force_correction_schedule(t),self.force(x))
            if self.prior == "gaussian":
                prior_correction = _batch_mult(self.prior_correction_schedule(t),-x)
            elif self.prior == "uniform":
                prior_correction = 0
            else:
                raise NotImplementedError
            if lattice is not None:
                score_correction = _batch_mult(self.score_correction_schedule(t,initial_factor=0.2),score)
                confining_force = lattice.scaled_confining_force(x,t)
                print(confining_force[0,-1])
                print(x[0,-1])
                print("score",score[0,-1])
                force_correction += confining_force
            else:
                score_correction = _batch_mult(self.score_correction_schedule(t),score)
            return force_correction+prior_correction+score_correction
    
    def plot_schedule(self):
        t = torch.linspace(0,1,100).to(self.duration.device)
        force_correction = self.force_correction_schedule(t)
        prior_correction = self.prior_correction_schedule(t)
        score_correction = self.score_correction_schedule(t)
        import matplotlib.pyplot as plt
        plt.style.use("my_style")
        t = t.detach().cpu().numpy()
        plt.plot(t,force_correction.detach().cpu().numpy(),label="force")
        plt.plot(t,prior_correction.detach().cpu().numpy(),label="linear")
        plt.plot(t,score_correction.detach().cpu().numpy(),label="score")
        plt.xlabel("t")
        plt.ylabel("correction_coeff")
        plt.legend()
        plt.savefig("correction_schedule.png")
    
def get_force_correction(force):
    def force_correction(x,t):
        return _batch_mult(1.0-t**0.1,force(x))+_batch_mult(t**0.1,-x)
    return force_correction

class Score(torch.nn.Module):
    def __init__(self,sde,nn,correction=None,lattice=None, single_nn=True):
        super().__init__()
        self.sde = sde
        self.correction = correction
        self.lattice = lattice
        if not single_nn:
            self.t_range_list = []
            self.nn = torch.nn.ModuleList()
            for i in range(len(self.sde.sde_list)):
                sde = self.sde.sde_list[i]
                t_range = sde.t_range
                self.nn.append(copy.deepcopy(nn))
                self.t_range_list.append(t_range)
        else:
            self.nn = torch.nn.ModuleList([nn])
            self.t_range_list = [self.sde.t_range]

    def marginal_prob_std(self,t):
        if isinstance(t,float):
            t = torch.tensor([t])
        std = torch.ones_like(t)
        if hasattr(self.sde,"std_fn"):
            std = self.sde.std_fn(t)
        elif hasattr(self.sde,"sde_list"):
            for sde in self.sde.sde_list:
                if hasattr(sde,"std_fn"): 
                    mask = (t>=sde.t_range[0])&(t<=sde.t_range[1])
                    if torch.any(mask):
                        std[mask] = sde.std_fn(t[mask])
        return std

    def forward(self,x,t,return_trace=False,context=None,**nn_kwargs):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(t.shape) == 0:
            t = t*torch.ones(x.shape[0],device=x.device)
        atomic_numbers = torch.zeros((x.shape[0],x.shape[1]),device=x.device).long()
        if context is not None:
            # x_full = torch.ones((x.shape[0],context.shape[1]+x.shape[1],*x.shape[2:]),device=x.device)

            # if len(mask.shape) >1:
            #     mask = mask[0]
            # x_full[:,mask] = x
            # x_full[:,~mask] = context
            # x = x_full
            # atomic_numbers = torch.zeros((x.shape[0],x.shape[1]),device=x.device).long()
            # atomic_numbers[:,mask] = 1
            x = torch.cat((context,x),dim=1)
            atomic_numbers[:,context.shape[1]:] = 1
        if return_trace:
            assert hasattr(self.nn,"trace")
            score, trace = self.nn.trace(x,t)
        else:
            with torch.enable_grad():
                score = torch.zeros_like(x).to(x.device)
                for i, nn in enumerate(self.nn):
                    t_range = self.t_range_list[i]
                    m = (t>=t_range[0])&(t<=t_range[1])
                    if torch.any(m):
                        try:
                            score_i = nn(x[m],t[m],atomic_numbers=atomic_numbers,**nn_kwargs)
                        except:
                            score_i = nn(x[m],t[m])
                        score_i = _batch_mult(1/self.marginal_prob_std(t[m]),score_i)
                        score[m] = score_i

                if self.correction is not None:
                    score = self.correction(x,t,score,lattice=self.lattice) 
        if context is not None:
            score = score[:,context.shape[1]:]
        if return_trace:
            return score, trace
        else:
            return score
        
if __name__=="__main__":
    Correction(None,duration=0.1).plot_schedule()