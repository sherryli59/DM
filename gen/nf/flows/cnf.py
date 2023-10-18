import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
try:
    from functorch import vmap
except ModuleNotFoundError:
    pass
import numpy as np

class ODEFunc(nn.Module):
    def __init__(self, dynamics, trace_method="hutch", ode_regularization=0, hutch_noise="gaussian"):
        super().__init__()
        self.dynamics = dynamics
        self.hutch_noise = hutch_noise
        self.trace_method = trace_method
        self.ode_regularization = ode_regularization
        self.num_evals = None
        self._eps = None

    def hutch_trace(self, f, y, e=None, batch_size=1):
        """e is grad outputs
        """
        e_dzdx = torch.autograd.grad(f, y,
                                     grad_outputs=e, create_graph=True)[0]
        e_dzdx = e_dzdx 
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = e_dzdx_e.reshape((batch_size, -1)).sum(-1)
        return approx_tr_dzdx

    def exact_trace_vmap(self, f, y, batch_size=1):
        """Test vmap
        """
        number_atoms = self.node_mask.shape[0]  # includes batch_size
        all_atoms_to_compute = torch.where(self.node_mask == 1)[0]
        all_indices = torch.cat(
            [all_atoms_to_compute + number_atoms * i for i in range(3)])
        basis_vectors = torch.eye(number_atoms * 3).to(f.device)
        basis_vectors = basis_vectors[all_indices]
        all_indices = all_indices.unsqueeze(-1)

        def get_vjp(v, i):
            return torch.autograd.grad(f.flatten(), y, grad_outputs=v, create_graph=True)[0].flatten()[i]
        jacobian_vmap = vmap(get_vjp)(
            basis_vectors, all_indices).reshape((batch_size, -1))
        jacobian_vmap = jacobian_vmap.sum(-1)
        return jacobian_vmap


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
            t.requires_grad_(True)
            if self.ode_regularization > 0:
                raise NotImplementedError
            else:
                if self.trace_method == "hutch":
                    dx = self.dynamics(x, t)
                    ldj = self.hutch_trace(dx, x,
                                           e=self._eps,
                                           batch_size=x.shape[0])
                    reg_term = torch.zeros_like(ldj)
                elif self.trace_method == "exact":
                    dx, ldj = self.dynamics(x, t, return_trace=True)
                else:
                    raise ValueError("Incorrect Trace Method Supplied")

            return dx, ldj, reg_term


class FFJORD(nn.Module):
    def __init__(self, dynamics, num_time_steps=20, trace_method='hutch',
                 integration_method="rk4",
                 ode_regularization=0, hutch_noise='gaussian',
                 translation_inv=True):
        super().__init__()
        self.odefunc = ODEFunc(dynamics, trace_method=trace_method,
                               ode_regularization=ode_regularization, hutch_noise=hutch_noise)

        self.integration_method = integration_method
        self._atol = 1e-4
        self._rtol = 1e-4
        self._atol_test = 1e-7
        self._rtol_test = 1e-7
        self.num_time_steps = num_time_steps
        self.set_integration_time(times=list(np.linspace(1e-3, 1, self.num_time_steps)))
        self.translation_inv = translation_inv

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

    def _apply_constraints(self,x):
        if self.translation_inv:
            x = x - torch.mean(x,axis=1).unsqueeze(1)
        return x

    def initialize_weights(self, n_layers=3, scale=100):
        last_layers = list(
            self.odefunc.dynamics.state_dict().keys())[-n_layers:]
        for layer in last_layers:
            self.odefunc.dynamics.state_dict()[layer] /= scale

    def forward(self, x):
        batch_size = x.shape[0]
        x = self._apply_constraints(x)
        ldj = x.new_zeros(batch_size)
        reg_term = x.new_zeros(batch_size)
        state = (x, ldj, reg_term)

        self.odefunc.reset_variables(x)

        statet = odeint(self.odefunc, state, self.int_time, method=self.integration_method,
                        rtol=self.rtol, atol=self.atol)

        xt, ldjt, reg_termt = statet
        x, ldj, reg_term = xt[-1], ldjt[-1], reg_termt[-1]
        return x, ldj

    def reverse(self, x):
        x = self._apply_constraints(x)
        batch_size = x.shape[0]
        ldj = x.new_zeros(batch_size)
        reg_term = x.new_zeros(batch_size)
        state = (x, ldj, reg_term)
        self.odefunc.reset_variables(x)

        statet = odeint(self.odefunc, state, self.inv_int_time,
                        method=self.integration_method,
                        rtol=self.rtol,
                        atol=self.atol)
        xt, ldjt, reg_termt = statet
        x, ldj, reg_term = xt[-1], ldjt[-1], reg_termt[-1]
        return x, ldj
