import torch
import torch.nn as nn
import numpy as np
from scipy import integrate
from torchdiffeq import odeint as odeint
try:
    from functorch import vmap
except ModuleNotFoundError:
    pass
from gen.diffusion import utils
from gen.diffusion.hessian_trace import hutch_trace, exact_trace



class ODEFunc(nn.Module):
    def __init__(self, dynamics, return_jacobian=True,
                 trace_method="exact",nqueries=100, ode_regularization=0, hutch_noise="gaussian"):
        super().__init__()
        self.dynamics = dynamics
        self.return_jacobian = return_jacobian
        self.hutch_noise = hutch_noise
        self.trace_method = trace_method
        self.nqueries = nqueries
        self.ode_regularization = ode_regularization
        self.num_evals = None
        self._eps = None

    def get_trace(self, x, t):
        def f(x):
            return self.dynamics(x, t)
        if self.trace_method == "hutch":
            print("check convergence:")
            exact = exact_trace(f, x)
            hutch_trace_200 = hutch_trace(f, x, m=200)
            error_200 = torch.mean((exact-hutch_trace_200)**2)
            print("m=200, error=",error_200)
            return hutch_trace(f, x, m=self.nqueries)
        elif self.trace_method == "exact":
            return exact_trace(f, x)
            
    def exact_trace(self,x,t, create_graph=False):
      shape = x.shape
          # x in shape (Batch, Length)
      def _func_sum(x):
        return self.dynamics(x.reshape(shape),t).sum(dim=0).flatten()
      jacobian = torch.autograd.functional.jacobian(_func_sum, x.reshape(x.shape[0],-1), create_graph=create_graph).transpose(0,1)
      return torch.vmap(torch.trace)(jacobian).flatten()
    

    def reset_variables(self, x):
        self.num_evals = 0
        if self.trace_method == "hutch":
            if self.hutch_noise == "gaussian":
                self._eps = torch.randn_like(x)

    def forward(self, t, state):
        x, ldj, reg_term = state
        self.num_evals += 1
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            x = x.detach()
            t.requires_grad_(True)
            if self.ode_regularization > 0:
                raise NotImplementedError
            else:
                dx = self.dynamics(x, t)
                if self.return_jacobian:
                    ldj = self.get_trace(x,t)
                reg_term = torch.zeros_like(ldj)
            
            return dx, ldj, reg_term




class ODESampler(nn.Module):
    def __init__(self, sde, score_fn, return_jacobian=True,
                 num_time_steps=100, trace_method='exact',nqueries=100,
                 integration_method="rk4",
                 ode_regularization=0, hutch_noise='gaussian'):
        super().__init__()
        self.score_fn = score_fn
        self.sde = sde
        self.dynamics = self.get_dynamics()
        self.odefunc = ODEFunc(self.dynamics, trace_method=trace_method,nqueries=nqueries,
                               return_jacobian=return_jacobian,
                               ode_regularization=ode_regularization, hutch_noise=hutch_noise)

        self.integration_method = integration_method
        self.return_jacobian = return_jacobian 
        self._atol = 1e-4
        self._rtol = 1e-4
        self._atol_test = 1e-7
        self._rtol_test = 1e-7
        self.num_time_steps = num_time_steps
        self.set_integration_time(times=list(np.linspace(1e-3, 1, self.num_time_steps)))

    def get_dynamics(self):
        def dynamics(x,t):
          """The drift function of the reverse-time SDE."""
          # Probability  flow ODE is a special case of Reverse SDE
          rsde = self.sde.reverse_sde(x,t,self.score_fn,ode=True)
          return rsde[0]
        return dynamics
    
    def set_integration_time(self, times=list(np.linspace(1e-3, 1, 100)), device=torch.device("cuda:0")):
        self.register_buffer('int_time',
                             torch.tensor(times, device=device).float())
        self.register_buffer('inv_int_time',
                             torch.tensor(list(reversed(times)), device=device).float())

    @property
    def atol(self):
        return self._atol if self.training else self._atol_test

    @property
    def rtol(self):
        return self._rtol if self.training else self._rtol_test
    
    def change_trace_method(self, trace_method="hutch"):
        self.odefunc.trace_method = trace_method

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.sde._apply_constraints(x)
        ldj = x.new_zeros(batch_size)
        reg_term = x.new_zeros(batch_size)
        state = (x, ldj, reg_term)

        self.odefunc.reset_variables(x)

        statet = odeint(self.odefunc, state, self.int_time, method=self.integration_method,
                        options=dict(step_size=2e-2),
                        rtol=self.rtol, atol=self.atol)
        print("num of evals:",self.odefunc.num_evals)
        xt, ldjt, reg_termt = statet
        x, ldj, reg_term = xt[-1], ldjt[-1], reg_termt[-1]
        if self.return_jacobian:
            return x, ldj, xt
        else:
            return x

    def reverse(self, x):
        x = self.sde._apply_constraints(x)
        batch_size = x.shape[0]
        ldj = x.new_zeros(batch_size)
        reg_term = x.new_zeros(batch_size)
        state = (x, ldj, reg_term)
        self.odefunc.reset_variables(x)
        statet = odeint(self.odefunc, state, self.inv_int_time,
                        method=self.integration_method,
                        options=dict(step_size=2e-2),
                        rtol=self.rtol,
                        atol=self.atol)
        xt, ldjt, reg_termt = statet
        x, ldj, reg_term = xt[-1], ldjt[-1], reg_termt[-1]
        np.save("traj.npy",xt.cpu().detach().numpy())
        print("num of evals:",self.odefunc.num_evals)
        if self.return_jacobian:
            return x, ldj
        else:
            return x
