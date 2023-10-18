
import torch
import torch.nn as nn
import math
import copy
from gen.diffusion.sde_utils import SDESolver, NoiseSchedule
from gen.diffusion.utils import ode_sampler, PDFSampler
from gen.diffusion.likelihood import  get_likelihood_fn



class SDE(nn.Module):
    def __init__(self,max_nsteps=1000,schedule="linear",beta_max=20,eps=1e-4,translation_inv=False,t_range=(1e-4,1),**kwargs):
        super().__init__()
        self.t_range = torch.clip(torch.tensor(t_range),min=eps)
        self.schedule = NoiseSchedule(type=schedule,max=beta_max,t_range=self.t_range)
        self.eps = eps
        self.max_nsteps = max_nsteps
        self.translation_inv = translation_inv
        self.reverse_solver = SDESolver(self.reverse_drift,self.diffusion)

    def _batch_mult(self,coeff,data):
        return torch.einsum(data, [0,...],coeff,[0], [0,...])

    def _apply_constraints(self,x):
        if self.translation_inv:
            x = x - torch.mean(x,axis=1).unsqueeze(1)          
        return x
    
    def generate_t(self,nsamples,likelihood_weighting=False):
        return torch.rand(nsamples)*(self.t_range[1]-self.t_range[0])+self.t_range[0]

    def prior_logp(self,z):  
        N = torch.prod(torch.tensor(z.shape[1:]))
        logp = -N/2.*math.log(2*math.pi)-torch.sum(
            z.reshape(len(z),-1)**2,1)/2
        return logp
    
    def reverse_sde(self, x, t, score_fn,ode=False):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        x = self._apply_constraints(x)
        drift, diffusion = self.sde(x, t)
        with torch.enable_grad():
            x.requires_grad_(True)
            score = score_fn(x, t)
        drift = drift - diffusion** 2 * score * (0.5 if ode else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = torch.zeros_like(diffusion) if ode else diffusion
        return drift, diffusion
    
    def reverse_drift(self,x,t,score_fn,ode=False):
        x = self._apply_constraints(x)
        drift, diffusion = self.sde(x, t)
        #print("forward drift",drift[0])
        score = score_fn(x, t)
        #print("x/score",x[0]/score[0])
        drift = drift - diffusion** 2 * score * (0.5 if ode else 1.)
        #print(drift[0])
        return drift
    
    def sde(self,x,t):
        return self.drift(x,t),self.diffusion(x,t)
    
    def forward(self,x,t_final,t_init=0):
        assert torch.any(t_final - t_init >= 0)  
        marginal_prob = getattr(self, "marginal_prob", None)
        if callable(marginal_prob): # for linear sde, marginal prob is Gaussian
            std_noise = torch.randn_like(x).to(x.device)
            std_noise = self._apply_constraints(std_noise)
            mean_coeff, sigma = self.marginal_prob(t_final,t_init)
            mean_coeff, sigma = mean_coeff.to(x.device), sigma.to(x.device)
            dlogpdx = torch.einsum(-std_noise, [0,...],1/sigma,[0], [0,...])
            diffused_x = self._batch_mult(mean_coeff,x
                        )+self._batch_mult(sigma,std_noise)
            if torch.any(torch.isnan(dlogpdx)) or torch.any(torch.isinf(dlogpdx)):
                print(t_final,t_init)
            diffused_x = self._apply_constraints(diffused_x)
            output = {"diffused_x":diffused_x,"score":dlogpdx,"sigma":sigma}
        else:
            assert hasattr(self, "sde_solver")
            with torch.no_grad():
                diffused_x = self.sde_solver.solve(x,t_init=t_init,t_final=t_final)
            diffused_x = self._apply_constraints(diffused_x)
            output = {"diffused_x":diffused_x}
            #utils.write_coord("forward_intermediate.xyz",diffused_x,nparticles=13)
        return output
        
    def backward(self,x,score_fn,t_init=None,t_final=None, method="ode",return_prob=False,return_traj=False):
        if t_init is None:
            t_init = torch.full((len(x),),self.t_range[1]).to(x.device)
        if t_final is None:
            t_final = torch.full((len(x),),self.t_range[0]+self.eps).to(x.device)
        else:
            t_final = torch.clip(t_final, min=self.t_range[0]+self.eps)
        if len(t_init.shape)==0:
            t_init = t_init.unsqueeze(0).expand(len(x),).to(x.device)
        if len(t_final.shape)==0:
            t_final = t_final.unsqueeze(0).expand(len(x),).to(x.device)
        assert torch.any(t_init - t_final >= 0) 

        if method == "ode":
            if return_prob:
                prob = get_likelihood_fn(noise_to_data=True)
                logp, x, _ = prob(self,score_fn, x)
                output = {"logp":logp,"x":x}
            else:
                drift_fn = lambda x, t: self.reverse_drift(x, t,score_fn, ode=True)
                with torch.no_grad():
                    output = ode_sampler(drift_fn, x, (t_init[0],t_final[0]),return_traj=return_traj)  
            
        elif method == "sde":
            assert hasattr(self, "reverse_solver")
            x = x.requires_grad_(False)
            output={"x":self.reverse_solver.solve(x,t_init=t_init[0], t_final=t_final,score_fn=score_fn)}
        
        elif method == "p-c":
            dt = 1/self.max_nsteps
            t = t_init
            if return_traj:
                traj=[]
            while torch.any(t-t_final>0):
                snr = 0.10
                mask = (t>t_final)
                x = x.detach()
                x = self._apply_constraints(x)
                grad = score_fn(x[mask], t[mask])
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.sqrt(torch.prod(torch.tensor(x.shape[1:])))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                x[mask] = x[mask] + langevin_step_size * grad + torch.sqrt(2
                    * langevin_step_size) * self._apply_constraints(torch.randn_like(x[mask]))
                #Predictor step (Euler-Maruyama)
                x = self._apply_constraints(x)
                drift, diffusion = self.reverse_sde(x[mask], t[mask], score_fn)

                x_mean = x[mask] - drift * dt
                x[mask] = x_mean - diffusion* math.sqrt(dt) * self._apply_constraints(torch.randn_like(x[mask]))
                if return_traj:
                    traj.append(x)
                t = t-dt
                output = {"x":x}
                if return_traj:
                    output.update({"traj":torch.stack(traj).transpose(0,1)})  
        return output
    
class LinearSDE(SDE):
    def __init__(self,max_nsteps=1000,schedule="linear",eps=1e-4,**kwargs):
        super().__init__(max_nsteps,schedule,eps=eps,**kwargs)
    
    def time_importance(self,t):
        t = torch.tensor(t)
        drift_coeff, diffusion_coeff= self.sde_coeff(t)
        sigma = self.marginal_prob_std(t)
        return (diffusion_coeff**2/(sigma**2)).float()
    
    def _t_generator(self):
        return PDFSampler(pdf=self.sde.time_importance,
                                    a=self.t_range[0], b=self.t_range[1])
    
    def generate_t(self,nsamples,likelihood_weighting=False):
        if likelihood_weighting:
            return self._t_generator().rvs(size=nsamples)
        else:
            return torch.rand(nsamples)*(self.t_range[1]-self.t_range[0])+self.t_range[0]

    def std_fn(self,t):
        std = torch.ones_like(t)
        mask = (t >= self.t_range[0]) * (t <= self.t_range[1])
        std[mask] = self.marginal_prob(t[mask],t_init=self.t_range[0])[1]
        return std


class VP_SDE(LinearSDE):
    def __init__(self,max_nsteps=1000,beta_max=20,eps=1e-3,schedule="linear",**kwargs):
        super().__init__(max_nsteps,schedule,eps=eps,beta_max=beta_max,**kwargs)
        self.sde_solver = SDESolver(self.drift,self.diffusion)

    def marginal_prob_std(self,t, t_init=0, min=1e-4):
        return torch.clip(torch.sqrt(1-self.schedule.alpha_cumprod(t,t_init))
                          ,min=min)
    def sde(self,x,t):
        if len(t.shape)==0:
            t = t.unsqueeze(0).expand(len(x),).to(x.device)
        x = self._apply_constraints(x)
        drift_coeff, diffusion_coeff = self.sde_coeff(t)
        drift = self._batch_mult(drift_coeff,x)
        diffusion = self._batch_mult(diffusion_coeff,torch.ones_like(x))
        return drift,diffusion
    
    def drift(self,x,t):
        x = self._apply_constraints(x)
        beta = self.schedule.beta(t)
        drift_coeff = -0.5*beta
        drift = self._batch_mult(drift_coeff,x)
        return drift
    
    def diffusion(self,x,t):
        beta = self.schedule.beta(t)
        diffusion_coeff = torch.sqrt(beta)
        diffusion = self._batch_mult(diffusion_coeff,torch.ones_like(x))
        return diffusion
    
    def sde_coeff(self,t):
        beta = self.schedule.beta(t)
        drift_coeff = -0.5*beta
        diffusion_coeff = torch.sqrt(beta)
        return drift_coeff,diffusion_coeff    
    
    def marginal_prob(self,t,t_init=0):
        alpha_cumprod = self.schedule.alpha_cumprod(t,t_init)
        mean_coeff = torch.sqrt(alpha_cumprod)
        sigma = torch.sqrt(1-alpha_cumprod)
        return mean_coeff, sigma


class GeneralSDE(SDE):
    def __init__(self, f, g=None,max_nsteps=300,dt=1e-3,kT=1,beta_max=0.5,schedule="linear",
                 friction=10,eps=0, force_schedule_power=0.5,**kwargs):
        super().__init__(max_nsteps=max_nsteps,eps=eps,schedule=schedule,beta_max=beta_max,**kwargs)
        self.f = f
        self.kT = kT
        self.friction = friction
        if g is None:
            self.g = lambda x: torch.ones_like(x)
        else:
            self.g = g
        self.duration = self.t_range[1]-self.t_range[0]
        if dt is None:
            self.dt = self.duration/self.max_nsteps
        else:
            self.dt = dt
        self.inc_coeff = lambda t: ((t-self.t_range[0])/self.duration)**force_schedule_power
        self.dec_coeff = lambda t: 1-((t-self.t_range[0])/self.duration)**force_schedule_power
        self.sde_solver = SDESolver(self.drift,self.diffusion,dt=self.dt)
        self.reverse_solver = SDESolver(self.reverse_drift,self.diffusion,dt=self.dt)
        

    def drift(self,x,t):
        x = self._apply_constraints(x)
        force_coeff = self.dec_coeff(t)
        drift = self._batch_mult(force_coeff,self.f(x)/self.friction)
        linear_coeff = -0.5*self.schedule.beta(t)*self.inc_coeff(t)
        drift = drift + self._batch_mult(linear_coeff,x)
        return drift
    
    def diffusion(self,x,t):
        eta_init = math.sqrt(2*self.kT/self.friction)
        eta_final = float(torch.sqrt(self.schedule.beta(torch.tensor(self.t_range[1]))))
        #linear interpolation
        diffusion_coeff = eta_init + (eta_final-eta_init)/self.duration*(t-self.t_range[0])
        x = self._apply_constraints(x)
        diffusion = self._batch_mult(diffusion_coeff,self.g(x))
        return diffusion
    
    def sde(self,x,t):
        return self.drift(x,t),self.diffusion(x,t)

class PiecewiseSDE(SDE):
    def __init__(self, device="cuda",**kwargs):
        super().__init__()
        self.setup_sde(**kwargs)
        self.t_init_list = self.knots[:-1].to(device)
        self.t_final_list = self.knots[1:].to(device)

    def setup_sde(self,type=["VP_SDE"],schedule="linear",knots=[],data_handler=None,kT=1,friction=10,translation_inv=False,):
        if isinstance(type, str):
            type = [type]
        self.sde_list = nn.ModuleList()
        knots = torch.tensor([self.eps]+knots+[1.0])
        self.knots = knots
        for i, sde_type in enumerate(type):
            t_range = (knots[i],knots[i+1])
            diffusion_params={"t_range": t_range}
            if sde_type == "GeneralSDE":
                diffusion_params.update({"f": copy.deepcopy(neg_force_wrapper(data_handler)), 
                                    "kT": kT, "friction": friction})
        
            sde_i = eval(sde_type)(schedule=schedule, 
                translation_inv=translation_inv, **dict(diffusion_params))
            self.sde_list.append(sde_i)

    def _sde_idx(self,t,forward=True):         
        t_diff = t.unsqueeze(1)-self.t_init_list.unsqueeze(0)
        t_diff[t_diff<=0] = 1.
        sde_idx = torch.argmin(t_diff,dim=1)
        if forward:
            return sde_idx
        else:
            return sde_idx+1
    
    def drift(self,x,t):
        if len(t.shape)==0:
            t = t.unsqueeze(0).expand(len(x),).to(x.device)
        sde_idx = self._sde_idx(t)
        drift = torch.zeros_like(x)
        for i in range(torch.max(sde_idx)+1):
            mask = (sde_idx==i)
            drift[mask] = self.sde_list[i].drift(x[mask],t[mask])
        return drift
    
    def diffusion(self,x,t):
        if len(t.shape)==0:
            t = t.unsqueeze(0).expand(len(x),).to(x.device)
        sde_idx = self._sde_idx(t)
        diffusion = torch.zeros_like(x)
        for i in range(torch.max(sde_idx)+1):
            mask = (sde_idx==i)
            diffusion[mask] = self.sde_list[i].diffusion(x[mask],t[mask])
        return diffusion

    
    def forward(self,x,t_final): #integrate forward from eps to t_final
        diffused_x = x.clone()
        diffused_x.requires_grad_(False)
        sde_idx = self._sde_idx(t_final) # index of the last sde to use
        output_list = [] # list of output dictionaries from each sde
        for i in range(torch.max(sde_idx)+1):
            mask = (sde_idx>=i)
            t_final_i = torch.clip(t_final[mask],max=self.t_final_list[i])
            output = self.sde_list[i].forward(diffused_x[mask].clone(),t_final_i,t_init=self.t_init_list[i])
            diffused_x[mask] = output["diffused_x"].clone()
            output_list.append({k:v[sde_idx[mask]==i].clone() for k,v in output.items()})
        return output_list, sde_idx
    
    def _backward(self,x,score_fn,t_init=None, **kwargs): #integrate backward from t to eps
        if t_init is None:
            t_init = torch.full((len(x),),1.).to(x.device)
        for i in torch.linspace(len(self.sde_list)-1,0,len(self.sde_list)).int():
            x = self._apply_constraints(x)
            t_init = torch.clip(t_init,max=self.t_final_list[i])
            t_final = self.t_init_list[i]
            with torch.no_grad():
                output_i = self.sde_list[i].backward(x,score_fn,t_init,t_final,**kwargs)
                x = output_i["x"]
                if "traj" in output_i.keys():
                    traj_i = output_i["traj"]
                    try:
                        traj = torch.cat((traj,traj_i),dim=1)
                    except:
                        traj = traj_i
        if kwargs.get("return_traj",False):
            return x , traj
        else:
            return x
    
def neg_force_wrapper(data_handler):
    def f(x):
        #x_decoded = data_handler.decoder(x.clone())
        x_decoded = x.clone()
        f = data_handler.distribution.neg_force_clipped(x_decoded)
        #f = data_handler.encoder(f)
        return f
    return f