import torch
import numpy as np
from scipy import integrate,interpolate
from scipy.stats import rv_continuous


class PDFSampler(rv_continuous): 
    def __init__(self, pdf, a=None, b=None, **kwargs):
        super().__init__(a=a, b=b,  **kwargs)
        self.pdf = pdf
        self.norm_const = integrate.quad(pdf, a, b)[0]
        t = torch.linspace(a,b,1000)
        discrete_cdf = integrate.cumtrapz(self._pdf(t),t)
        discrete_cdf = np.append(0,discrete_cdf)
        self.cdf = interpolate.interp1d(t, discrete_cdf,fill_value="extrapolate")
        self.ppf = interpolate.interp1d(discrete_cdf, t, fill_value="extrapolate") 
    def _cdf(self,x):
        return self.cdf(x)
    def _ppf(self,x):
        return self.ppf(x)
    def _pdf(self, x):
        with torch.no_grad():
            return self.pdf(x)/self.norm_const

def ode_sampler(drift_fn,
                init_x,
                t_range,
                atol=1e-5, 
                rtol=1e-5,
                return_traj=False):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation 
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """
  device = init_x.device
  shape = init_x.shape
  
  def ode_func(t, x):        
    """The ODE function for use by the ODE solver."""
    if torch.any(torch.isnan(torch.tensor(x))):
        raise ValueError("NaN encountered")
    time_steps = np.ones((shape[0],)) * t   
    x = torch.tensor(x, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((x.shape[0], ))    
    drift = drift_fn(x, time_steps)
    return drift.detach().cpu().numpy().reshape((-1,)).astype(np.float64)
  if return_traj:
      t_eval = np.linspace(float(t_range[0]), float(t_range[1]), int(abs(t_range[1]-t_range[0])/0.002))
  else:
      t_eval = None
  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, t_range, init_x.reshape(-1).cpu().numpy(),dt=1e-3, t_eval=t_eval,rtol=rtol, atol=atol, method='RK45')  
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=init_x.device).reshape(shape)
  if return_traj:
      traj = torch.tensor(res.y, device=init_x.device).transpose(0,1).reshape(([-1]+list(shape))).transpose(0,1)
      return {"x":x, "traj":traj}
  return {"x":x} 

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


