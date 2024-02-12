
import torch
import torch.nn as nn
import math
import copy
import numpy as np
from gen.diffusion.sde_utils import SDESolver, NoiseSchedule
from gen.diffusion.utils import ode_sampler, PDFSampler, neg_force_wrapper
from gen.diffusion.likelihood import  ODESampler



class SDE(nn.Module):
    def __init__(self,max_nsteps=500,schedule="linear",beta_max=20,eps=1e-4,
                 periodic_boxlen=None,translation_inv=False,t_range=(1e-4,1),
                 reverse_drift_correction=None,
                 **kwargs):
        super().__init__()
        self.t_range = torch.clip(torch.tensor(t_range),min=eps)
        self.schedule = NoiseSchedule(type=schedule,max=beta_max,t_range=self.t_range)
        self.reverse_drift_correction = reverse_drift_correction
        self.eps = eps
        self.max_nsteps = max_nsteps
        self.translation_inv = translation_inv
        self.periodic_boxlen = periodic_boxlen
        self.reverse_solver = SDESolver(self.reverse_drift,self.diffusion)

    def _batch_mult(self,coeff,data):
        return torch.einsum(data, [0,...],coeff,[0], [0,...])

    def _apply_constraints(self,x,context=None):
        if self.translation_inv:
            x_mean = torch.mean(x,axis=1).unsqueeze(1)
            x = x - x_mean      
        if self.periodic_boxlen is not None:
            x = torch.remainder(x,self.periodic_boxlen)
        return x
    
    def generate_t(self,nsamples,likelihood_weighting=False):
        return torch.rand(nsamples)*(self.t_range[1]-self.t_range[0])+self.t_range[0]

    def prior_logp(self,z):  
        N = torch.prod(torch.tensor(z.shape[1:]))
        logp = -N/2.*math.log(2*math.pi)-torch.sum(
            z.reshape(len(z),-1)**2,1)/2
        return logp
    
    def reverse_sde(self, x, t, score_fn,ode=False,**context_args):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        x = self._apply_constraints(x)
        drift, diffusion = self.sde(x, t)
        with torch.enable_grad():
            #x.requires_grad_(True)
            score = score_fn(x, t,**context_args)
        drift = drift - diffusion** 2 * score * (0.5 if ode else 1.)
        if self.reverse_drift_correction is not None:
            drift = drift + self.reverse_drift_correction(x, t)
        # Set the diffusion function to zero for ODEs.
        diffusion = torch.zeros_like(diffusion) if ode else diffusion
        return drift, diffusion
    
    def reverse_drift(self,x,t,score_fn,ode=False,**context_args):
        x = self._apply_constraints(x)
        drift, diffusion = self.sde(x, t)
        score = score_fn(x, t,**context_args)
        drift = drift - diffusion** 2 * score * (0.5 if ode else 1.)
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
                print("True score becomes nan:",t_final,t_init)
            diffused_x = self._apply_constraints(diffused_x)
            output = {"diffused_x":diffused_x,"score":dlogpdx,"weight":sigma**2}
        else:
            assert hasattr(self, "sde_solver")
            with torch.no_grad():
                diffused_x = self.sde_solver.solve(x,t_init=t_init,t_final=t_final,t_range=self.t_range)
            diffused_x = self._apply_constraints(diffused_x)
            output = {"diffused_x":diffused_x}
        return output

    def logp(self,x, score_fn):
        ode =  ODESampler(self, score_fn)
        x_prior, lj, traj = ode.forward(x)
        prior_logp = self.prior_logp(x_prior)
        logp = prior_logp+lj
        return logp, x_prior, traj

    def backward(self,x,score_fn,t_init=None,t_final=None, method="ode",return_prob=False,return_traj=False,**context_args):
        
        if t_init is None:
            t_init = torch.full((len(x),),self.t_range[1]).to(x.device)
        if t_final is None:
            t_final = torch.full((len(x),),self.t_range[0]+self.eps).to(x.device)
        else:
            t_final = torch.clip(t_final, min=(self.t_range[0]+self.eps).to(t_final.device))
        if len(t_init.shape)==0:
            t_init = t_init.unsqueeze(0).expand(len(x),).to(x.device)
        if len(t_final.shape)==0:
            t_final = t_final.unsqueeze(0).expand(len(x),).to(x.device)
        assert torch.any(t_init - t_final >= 0) 
        if method == "ode":
            if return_prob:
                ode =  ODESampler(self, score_fn)
                prior_logp = self.prior_logp(x)
                x_final, lj = ode.reverse(x)
                logp = prior_logp-lj
                # prob = get_likelihood_fn(noise_to_data=True)
                # logp, x, _ = prob(self,score_fn, x)
                output = {"logp":logp,"x":x_final}
                # nqueries_list = [1,10,100,200]
                # for nqueries in nqueries_list:
                #     ode = ODESampler(self, score_fn, nqueries=nqueries, trace_method="hutch")
                #     prior_logp = self.prior_logp(x)
                #     x, lj = ode.reverse(x)
                #     logp = prior_logp-lj
                #     np.save("logp_{}.npy".format(nqueries),logp.cpu().numpy())
            else:
                        drift_fn = lambda x,t: self.reverse_drift(x,t,score_fn,ode=True,**context_args)
                        output = ode_sampler(drift_fn, x, (t_init[0],t_final[0]),return_traj=return_traj) 
                        # ode =  ODESampler(self, score_fn, return_jacobian=False)
                        # x = ode.reverse(x) 
                        # output = {"x":x}
        elif method == "e-m":
            t_seq = torch.linspace(t_init[0],t_final[0],self.max_nsteps).to(x.device)
            # context = context_args.get("context",None)
            # if context is not None: #Only implimented euler-maruyama for now
            #     if context_idx is None:
            #         context_idx = torch.arange(context.shape[1])
            #     std_noise = torch.randn(len(t_seq),*context.shape).to(x.device)
            #     std_noise = self._apply_constraints(std_noise)
            #     context_traj = context.unsqueeze(0).expand(len(t_seq),*context.shape)
            #     mean_coeff, sigma = self.marginal_prob(t_seq,0)
            #     mean_coeff, sigma = mean_coeff.to(x.device), sigma.to(x.device)
            #     noised_context_traj = self._batch_mult(mean_coeff,context_traj)+self._batch_mult(sigma,std_noise)
            dt = 1/self.max_nsteps
            for i,t in enumerate(t_seq):
                #if context is not None:
                    #x[:,context_idx] = noised_context_traj[i]
                x = self._apply_constraints(x)
                drift, diffusion = self.reverse_sde(x, t, score_fn,**context_args)
                x_mean = x - drift * dt
                x = x_mean - diffusion* math.sqrt(dt) * self._apply_constraints(torch.randn_like(x))
            #if context is not None:
                #x[:,context_idx] = noised_context_traj[i]
            output = {"x":x_mean}    
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
                snr = 0.16
                mask = (t>t_final)
                x = x.detach()
                x = self._apply_constraints(x)
                grad = score_fn(x[mask], t[mask],**context_args)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.sqrt(torch.prod(torch.tensor(x.shape[1:])))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                x[mask] = x[mask] + langevin_step_size * grad + torch.sqrt(2
                    * langevin_step_size) * self._apply_constraints(torch.randn_like(x[mask]))
                #Predictor step (Euler-Maruyama)
                x = self._apply_constraints(x)
                drift, diffusion = self.reverse_sde(x[mask], t[mask], score_fn,**context_args)
                x_mean = x[mask] - drift * dt + diffusion **2 * grad* dt
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
    
    def prior(self, shape):
        return torch.distributions.normal.Normal(torch.zeros(list(shape)), torch.ones(list(shape)))

    def sde(self,x,t):
        if len(t.shape)==0:
            t = t.unsqueeze(0).expand(len(x),).to(x.device)
        x = self._apply_constraints(x)
        return self.drift(x,t),self.diffusion(x,t)
     
    def std_fn(self,t):
        std = torch.ones_like(t)
        mask = (t >= self.t_range[0]) * (t <= self.t_range[1])
        std[mask] = self.marginal_prob(t[mask],t_init=self.t_range[0])[1]
        return std

class ToroidalDiffusion(LinearSDE):
    def __init__(self,sigma_min=0.01,sigma_max=1,ncells=10,
                 max_nsteps=1000,eps=1e-4,**kwargs):
        super().__init__(max_nsteps,eps=eps,**kwargs)
        self.sigma_min = sigma_min*self.periodic_boxlen
        self.sigma_max = sigma_max*self.periodic_boxlen
        self.ncells = ncells
        self.log_ratio = math.log(sigma_max/sigma_min)
        self.sde_solver = SDESolver(self.drift,self.diffusion)
        self.register_buffer("d",torch.arange(-self.ncells,self.ncells+1)*self.periodic_boxlen)
        self.mult_factor = 50000
        self.register_buffer("std_noise", torch.randn(self.mult_factor,1))

    def sigma(self,t):
        return self.sigma_min**(1.-t)*self.sigma_max**(t)
    
    def drift(self,x,t):
        return torch.zeros_like(x)
    
    def diffusion(self,x,t):
        if isinstance(t,float):
            t = torch.tensor(t).to(x.device)
        if len(t.shape)==0:
            t = t.unsqueeze(0).expand(len(x),).to(x.device)
        t_scaled = (t-self.t_range[0])/(self.t_range[1]-self.t_range[0])
        sigma = self.sigma(t_scaled)
        g_t = math.sqrt(2*self.log_ratio)*sigma
        return self._batch_mult(g_t,torch.ones_like(x))

    def prior(self, shape):
        class UniformDistribution(torch.distributions.Uniform):
            def __init__(self,shape,boxlen):
                super().__init__(-0.5*boxlen,0.5*boxlen)
                self.shape = shape
                self.boxlen = boxlen
                self.dim = self.shape[-1]
            def sample(self,nsamples):
                if isinstance(nsamples,tuple):
                    nsamples = nsamples[0]
                return self.rsample((nsamples,)+self.shape)

            def log_prob(self,x):
                return 1/(self.boxlen**self.dim)*torch.ones(len(x)).to(x.device)

        assert self.periodic_boxlen is not None 
        return UniformDistribution(shape,self.periodic_boxlen)            

    def _calc_score(self,displacement,sigma):
        displacement_flat = (displacement).reshape(len(displacement),-1)
        
        d_expanded = self.d[None,None,:].expand(len(displacement_flat),displacement_flat.shape[1],self.d.shape[-1])
        d_expanded = d_expanded.to(displacement.device)
        displacement_flat_expanded = displacement_flat[:,:,None].expand_as(d_expanded)
        prob = torch.exp((displacement_flat_expanded-d_expanded)**2/(-2*sigma[:,None,None]**2))
        prob_weighted= (-displacement_flat_expanded+d_expanded)/sigma[:,None,None]**2*prob
        score = 1/torch.sum(prob,axis=-1)*torch.sum(prob_weighted,axis=-1)
        return score.reshape(displacement.shape)

    def forward(self,x,t_final,t_init=0):
        assert torch.any(t_final - t_init >= 0)  
        sigma = self.sigma(t_final)
        std_noise = torch.randn_like(x).to(x.device)
        noise = self._batch_mult(sigma,std_noise)
        diffused_x = x + noise
        diffused_x = self._apply_constraints(diffused_x)
        score = self._calc_score(diffused_x-x,sigma)
        mult_factor = 1000
        mult_noise = torch.randn(mult_factor*x.shape[0],*x.shape[1:]).to(x.device)
        mult_sigma = sigma.repeat(mult_factor)
        mult_noise = self._batch_mult(mult_sigma,mult_noise)%self.periodic_boxlen
        mult_score = self._calc_score(mult_noise,mult_sigma).reshape(mult_factor,len(x),-1)
        avg_score_norm = torch.mean(torch.sum(mult_score**2,axis=-1),axis=0)
        return {"diffused_x":diffused_x,"score":score,"weight":1/avg_score_norm}
    
    def std_fn(self,t):
        mult_factor = self.mult_factor//len(t)
        mult_noise = self.std_noise[:len(t)*mult_factor].to(t.device)
        sigma = self.sigma(t)
        mult_sigma = sigma.repeat(mult_factor)
        mult_noise = self._batch_mult(mult_sigma,mult_noise)%self.periodic_boxlen
        mult_score = self._calc_score(mult_noise,mult_sigma).reshape(mult_factor,len(t),-1)
        avg_score_norm = torch.sqrt(torch.mean(mult_score**2,axis=(0,-1)))
        return 1/avg_score_norm

class VP_SDE(LinearSDE):
    def __init__(self,max_nsteps=1000,beta_max=20,eps=1e-4,schedule="linear",**kwargs):
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
    def __init__(self, **kwargs):
        super().__init__()
        self.setup_sde(**kwargs)
        self.register_buffer("t_init_list", self.knots[:-1])
        self.register_buffer("t_final_list", self.knots[1:])

    def setup_sde(self,type=["VP_SDE"],schedule="linear",knots=[],data_handler=None,D=1, kT=1,friction=10,periodic_boxlen=None,translation_inv=False,):
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
                translation_inv=translation_inv,periodic_boxlen=periodic_boxlen, **dict(diffusion_params))
            self.sde_list.append(sde_i)

    def prior(self, shape):
        return self.sde_list[-1].prior(shape)

    def _sde_idx(self,t,forward=True):   
        t_diff = t.unsqueeze(1)-self.t_init_list.unsqueeze(0).to(t.device)
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
            if torch.sum(mask)>0:
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
            t_init = torch.clip(t_init,max=self.t_final_list[i].to(t_init.device))
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
    
