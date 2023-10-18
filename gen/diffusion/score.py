import torch
import torch.nn as nn 
import copy

def _batch_mult(coeff,data):
        return torch.einsum(data, [0,...],coeff,[0], [0,...])

def neg_force_wrapper(data_handler):
    def f(x):
        #x_decoded = data_handler.decoder(x.clone())
        x_decoded = x.clone()
        f = data_handler.distribution.neg_force_clipped(x_decoded)
        #f = data_handler.encoder(f)
        return f
    return f

class Correction(nn.Module):
    def __init__(self,data_handler,duration=0.1):
        super().__init__()
        self.neg_force = neg_force_wrapper(data_handler)
        self.duration = duration

    
    def linear_correction_schedule(self,t):
        schedule = torch.zeros_like(t)
        mask = (t>1-self.duration)
        schedule[mask] = 1.0/self.duration*(t[mask] -1 + self.duration)
        return schedule
    
    def force_correction_schedule(self,t):
        schedule = torch.zeros_like(t)
        mask = (t<self.duration)
        schedule[mask] = -1.0/self.duration*t[mask] + 1.0
        return schedule
    
    def score_correction_schedule(self,t):
        schedule = torch.ones_like(t)
        left = max(0.001,min(self.duration,1-self.duration))
        schedule[t<left] = 1.0/left*t[t<left]
        right = min(0.999,max(self.duration,1-self.duration))
        schedule[t>right] = -1.0/(1-right)*(t[t>right] -1)
        return schedule
    
    def forward(self,x,t,score):
        if self.duration is None or self.duration == 0:
            return score
        else:
            force_correction = _batch_mult(self.force_correction_schedule(t),self.neg_force(x))
            linear_correction = _batch_mult(self.linear_correction_schedule(t),-x)
            score_correction = _batch_mult(self.score_correction_schedule(t),score)
            return force_correction.detach()+linear_correction.detach()+score_correction
    
    def test_plot(self):
        t = torch.linspace(0,1,100)
        force_correction = self.force_correction_schedule(t)
        linear_correction = self.linear_correction_schedule(t)
        score_correction = self.score_correction_schedule(t)
        import matplotlib.pyplot as plt
        plt.plot(t,force_correction,label="force")
        plt.plot(t,linear_correction,label="linear")
        plt.plot(t,score_correction,label="score")
        plt.legend()
        plt.savefig("correction_schedule.png")
        plt.show()
    
def get_force_correction(force):
    def force_correction(x,t):
        return _batch_mult(1.0-t**0.1,force(x))+_batch_mult(t**0.1,-x)
    return force_correction

class Score(torch.nn.Module):
    def __init__(self,sde,nn,correction=None, single_nn=True):
        super().__init__()
        self.sde = sde
        self.correction = correction
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
                    std[mask] = sde.std_fn(t[mask])
        return std

    def forward(self,x,t,return_trace=False):
        if len(t.shape) == 0:
            t = t*torch.ones(x.shape[0],device=x.device)
        if return_trace:
            assert hasattr(self.nn,"trace")
            score, trace = self.nn.trace(x,t)
        else:
            with torch.enable_grad():
                score = torch.zeros_like(x).to(x.device)
                for i, nn in enumerate(self.nn):
                    t_range = self.t_range_list[i]
                    mask = (t>=t_range[0])&(t<=t_range[1])
                    score_i = nn(x[mask],t[mask])
                    score_i = _batch_mult(1/self.marginal_prob_std(t[mask]),score_i)
                    score[mask] = score_i
                if self.correction is not None:
                    score = self.correction(x,t,score)   
                if torch.max(x)>5:
                    print(torch.max(x))
                    print("warning: x is large")
        if return_trace:
            return score, trace
        else:
            return score
        
if __name__=="__main__":
    Correction(None,duration=0.6).test_plot()