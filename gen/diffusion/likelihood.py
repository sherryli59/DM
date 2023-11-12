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


def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""
  def batch_jacobian(func, x, create_graph=False):
        # x in shape (Batch, Length)
    def _func_sum(x):
      return func(x).sum(dim=0)
    return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)

  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t).unsqueeze(0) * eps, dim=tuple(range(1, len(x.shape)+1)))
      grad_fn_eps = torch.autograd.grad(fn_eps, x,torch.ones_like(fn_eps))[0]
    x.requires_grad_(False)
    return torch.mean(torch.sum(grad_fn_eps* eps, dim=tuple(range(2, len(x.shape)+1))),dim=0)

  # def div_fn_2(x,t,eps, n_eps=1):
  #   dup_samples = x.unsqueeze(0).expand(n_eps,
  #        *x.shape).contiguous().view(-1, *x.shape[1:])
  #   dup_t = t.tile(n_eps)
  #   dup_samples = dup_samples.detach().requires_grad_(True)
  #   rand_v = eps
  #   with torch.enable_grad():
  #       score = fn(dup_samples,dup_t)
  #       scorev = torch.sum(score * rand_v)
  #       grad = torch.autograd.grad(scorev, dup_samples, create_graph=True)[0]    
  #   trace = torch.sum((rand_v * grad).view(grad.shape[0],-1), dim=-1)
  #   trace_2 = div_fn(x,t,eps)
  #   print("should be 0:", torch.mean(trace-trace_2))
  #   return trace
  return div_fn


def get_likelihood_fn(hutchinson_type='Gaussian', noise_mult=5, 
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3, noise_to_data=False):
  """Create a function to compute the unbiased log-likelihood estimate of a given x0 point.
  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow = integrate.solve_ivp(ode_func, (1, eps), init, rtol=rtol, atol=atol, method=ode_method)
  """

  def drift_fn(model,score_fn, x, t):
    """The drift function of the reverse-time SDE."""
    # Probability  flow ODE is a special case of Reverse SDE
    rsde = model.reverse_sde(x,t,score_fn,ode=True)
    return rsde[0]

  def div_fn(model, score_fn, x, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn(model, score_fn, xx, tt))(x, t, noise)

  def likelihood_fn(model,score_fn, x0):
    """Compute an unbiased estimate to the log-likelihood.
    Args:
      model: A score model.
      x0: A PyTorch tensor.
    Returns:
      logp: A PyTorch tensor of shape [batch size]. The log-likelihoods on `x0` in bits/dim.
      z: A PyTorch tensor of the same shape as `x0`. The latent representation of `x0` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    shape = x0.shape
    if hutchinson_type == 'Gaussian':
      epsilon = torch.randn([noise_mult]+list(shape)).to(x0.device)
    elif hutchinson_type == 'Rademacher':
      epsilon = torch.randint(size=[noise_mult]+list(shape), low=0, high=2).to(x0.device).float() * 2 - 1.
    else:
      raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

    def ode_func(t, x):
      sample = utils.from_flattened_numpy(x[:-shape[0]], shape).to(x0.device).type(torch.float32)
      #sample = sample- torch.mean(sample,axis=1).unsqueeze(1)
      sample.requires_grad_(True)
      vec_t = torch.ones(sample.shape[0], device=sample.device) * t
      drift = utils.to_flattened_numpy(drift_fn(model, score_fn, sample, vec_t))
      logp_grad = utils.to_flattened_numpy(div_fn(model,score_fn, sample, vec_t, epsilon))
      #print(t,drift[0])
      return np.concatenate([drift, logp_grad], axis=0)
    init = np.concatenate([utils.to_flattened_numpy(x0), np.zeros((shape[0],))], axis=0)
    if noise_to_data:
      span = (1,eps)
    else:
      span = (eps,1)  
    #solution = integrate.solve_ivp(ode_func, span, init, rtol=rtol, atol=atol, method=method, vectorized = True)
    solution = integrate.solve_ivp(ode_func, span, init, rtol=1., atol=1., method=method, max_step=1/1000, vectorized = True)
    nfe = solution.nfev
    zp = solution.y[:, -1]
    z = utils.from_flattened_numpy(zp[:-shape[0]], shape).to(x0.device).type(torch.float32)
    delta_logp = utils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(x0.device).type(torch.float32)
    if noise_to_data:
      prior_logp = model.prior_logp(x0)
      logp = prior_logp - delta_logp
    else:
      prior_logp = model.prior_logp(z)
      logp = prior_logp + delta_logp
    print("prior:",torch.mean(prior_logp))
    print("prior_ref:",torch.mean(model.prior_logp(torch.randn_like(x0))))
    #print(delta_logp)
    return logp, z, nfe

  return likelihood_fn





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


class ODESampler(nn.Module):
    def __init__(self, sde, score_fn, num_time_steps=1000, trace_method='hutch',
                 integration_method="rk4",
                 ode_regularization=0, hutch_noise='gaussian',
                 translation_inv=True):
        super().__init__()
        self.score_fn = score_fn
        self.sde = sde
        self.dynamics = self.get_dynamics()
        self.odefunc = ODEFunc(self.dynamics, trace_method=trace_method,
                               ode_regularization=ode_regularization, hutch_noise=hutch_noise)

        self.integration_method = integration_method
        self._atol = 1e-4
        self._rtol = 1e-4
        self._atol_test = 1e-7
        self._rtol_test = 1e-7
        self.num_time_steps = num_time_steps
        self.set_integration_time(times=list(np.linspace(1e-3, 1, self.num_time_steps)))
        self.translation_inv = translation_inv

    def get_dynamics(self):
        def dynamics(x,t):
          """The drift function of the reverse-time SDE."""
          # Probability  flow ODE is a special case of Reverse SDE
          rsde = self.sde.reverse_sde(x,t,self.score_fn,ode=True)
          #print(t,rsde[0][0,0])
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
                        dtype=torch.float64,
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
                        options=dict(step_size=1e-3),
                        dtype=torch.float64,
                        rtol=self.rtol,
                        atol=self.atol)
        xt, ldjt, reg_termt = statet
        x, ldj, reg_term = xt[-1], ldjt[-1], reg_termt[-1]
        return x, ldj
