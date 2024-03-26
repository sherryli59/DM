import torch
from torch import nn
import numpy as np
from scipy import integrate,interpolate
from scipy.stats import rv_continuous
import copy

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
  res = integrate.solve_ivp(ode_func, t_range, init_x.reshape(-1).cpu().numpy(),first_step=1e-3, max_step=1e-2,dt=1e-3,t_eval=t_eval,rtol=rtol, atol=atol, method='RK45')  
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

def neg_force_wrapper(data_handler):
    def f(x):
        #x_decoded = data_handler.decoder(x.clone())
        x_decoded = x.clone()
        f = data_handler.distribution.neg_force_clipped(x_decoded)
        #f = data_handler.encoder(f)
        return f
    return f

def _batch_mult(coeff,data):
        return torch.einsum(data, [0,...],coeff,[0], [0,...])

class ReverseDriftCorrection(torch.nn.Module):
    def __init__(self, data_handler, t_range, 
                 friction=10, force_schedule_power=0.5):
        super().__init__()
        self.f = copy.deepcopy(neg_force_wrapper(data_handler))
        self.friction = friction
        self.t_range = t_range
        self.duration = t_range[1]-t_range[0]
        self.force_schedule_power = force_schedule_power
    
    def dec_coeff(self,t):
        coeff = torch.zeros_like(t)
        mask = (t>self.t_range[0])&(t<self.t_range[1])
        coeff[mask] = 1-((t[mask]-self.t_range[0])/self.duration)**self.force_schedule_power
        return coeff
    
    def forward(self,x,t):
        if len(t.shape)==0:
            t = t.unsqueeze(0).expand(len(x),).to(x.device)
        force_coeff = self.dec_coeff(t)
        drift = _batch_mult(force_coeff,self.f(x)/self.friction)
        return drift
    
def get_lattice_and_neighbor(n_cells=3):
    lattice = torch.stack(torch.meshgrid(torch.arange(n_cells),
                    torch.arange(n_cells), torch.arange(n_cells)), dim=-1).reshape(-1, 3)
    #find the index of the neighbor to the right of each cell
    neighbor = lattice.clone()
    neighbor[:,0] = (neighbor[:,0]+1)%n_cells
    # match the neighbor index to the lattice index
    neighbor_idx = torch.zeros(n_cells**3)
    for i in range(n_cells**3):
        neighbor_idx[i] = torch.where(torch.all(lattice==neighbor[i],dim=-1))[0]
    return lattice,neighbor_idx

class Lattice():
    def __init__(self, boxlength, n_cells, device="cuda"):
        super().__init__()
        self.boxlength = boxlength
        self.n_cells = n_cells
        self.lattice, self.neighbor_idx = get_lattice_and_neighbor(n_cells)
        self.lattice = self.lattice.float().to(device)
        self.neighbor_idx = self.neighbor_idx.int().to(device)
        self.cell_len = boxlength / n_cells
        self.lattice = self.lattice.float() *self.cell_len - boxlength / 2 + self.cell_len/2

    def polynomial_cutoff(self,x, cutoff, scale,power=3):
        mask = (x<cutoff)
        return mask.float()*scale*(x/cutoff)**power + (1-mask.float())*scale
    
    def scaled_confining_force(self,x,t,lattice_center=(0,0,0),max_force=10,buffer = 0.1):
        max_force = max_force * (1-t)
        # expand max_force to match the size of x
        max_force = max_force.view(len(max_force), *(1,)*(x.ndim-1)).expand(x.shape)
        buffer = buffer * self.cell_len + t * self.boxlength/2
        buffer = buffer.view(len(buffer), *(1,)*(x.ndim-1)).expand(x.shape)
        return self.confining_force(x,lattice_center,max_force,buffer)

    def confining_force(self,x,lattice_center=(0,0,0),max_force=1,buffer = 0.1):
        lattice_center = torch.tensor(lattice_center).to(x.device)
        lattice_center = lattice_center.reshape(*(1,)*(x.ndim-1)+(-1,))
        displacement = x - lattice_center
        direction = -torch.sign(displacement)
        #apply periodic boundary conditions
        displacement -= ((torch.abs(displacement)> 0.5 * self.boxlength)
                        * torch.sign(displacement) * self.boxlength)
        displacement = torch.abs(displacement)
        mask = (displacement>self.cell_len/2).int()
        force = mask*self.polynomial_cutoff(displacement-self.cell_len/2,buffer*self.cell_len,max_force)
        force = force*direction
        return force
    

    def select_adjacent_cells(self,x,cell_idx):
        cell1 = torch.randint(int(cell_idx.max().item()),(1,)).to(cell_idx.device)
        cell2 = self.neighbor_idx[cell1].to(cell_idx.device)
        mask = (cell_idx==cell1).unsqueeze(-1).expand(x.shape)
        x1 = x[mask].reshape(x.shape[0],-1,x.shape[-1])
        mask = (cell_idx==cell2).unsqueeze(-1).expand(x.shape)
        x2 = x[mask].reshape(x.shape[0],-1,x.shape[-1])
        #move x1 to the origin
        displacement_vec = self.lattice[cell1]
        x1 = x1-displacement_vec.unsqueeze(1)
        x2 = x2-displacement_vec.unsqueeze(1)
        boxlength = self.boxlength
        return x1,x2

    def select_sphere_and_context(self,x,cell_idx):
        #randomly select a particle
        nparticles = x.shape[0]
        particle = torch.randint(nparticles,(1,)).to(cell_idx.device)
        #compute the distance to all other particles
        displacement = x - x[particle]
        #apply periodic boundary conditions
        displacement -= ((torch.abs(displacement)> 0.5 * self.boxlength)
                        * torch.sign(displacement) * self.boxlength)
        distance = torch.norm(displacement,dim=-1)
        #only select the top 19 of the context based on distance
        _, idx = torch.topk(distance,20,dim=1,  largest=False)
        mask = torch.zeros_like(distance).bool()
        mask.scatter_(1,idx,True)
        mask = mask.unsqueeze(-1).expand(x.shape)
        core = x[mask].reshape(x.shape[0],-1,x.shape[-1])
        # select the next 100 particles as the context
        _, idx_context = torch.topk(distance,120,dim=1,largest=False)
        mask = torch.zeros_like(distance).bool()
        mask.scatter_(1,idx_context,True)
        mask.scatter_(1,idx,False)
        mask = mask.unsqueeze(-1).expand(x.shape)
        context = x[mask].reshape(x.shape[0],-1,x.shape[-1])
        return core,context




    def select_cell_and_context(self,x,cell_idx):
        cell = torch.randint(int(cell_idx.max().item()),(1,)).to(cell_idx.device)
        mask = (cell_idx==cell).unsqueeze(-1).expand(x.shape)
        x1 = x[mask].reshape(x.shape[0],-1,x.shape[-1])
        context = x[~mask].reshape(x.shape[0],-1,x.shape[-1])
        #move x1 to the origin
        displacement_vec = self.lattice[cell]
        x1 = x1-displacement_vec.unsqueeze(1)
        context = context-displacement_vec.unsqueeze(1)
        #apply periodic boundary conditions
        to_subtract = ((torch.abs(context)> 0.5 * self.boxlength)
                        * torch.sign(context) * self.boxlength)
        context -= to_subtract
        #apply cutoff
        distance = torch.linalg.norm(context,dim=-1)
        #only select the top 20% of the context based on distance
        _, idx = torch.topk(distance,int(0.2*distance.shape[1]),dim=1,  largest=False)
        #convert idx to mask
        mask = torch.zeros_like(distance).bool()
        mask.scatter_(1,idx,True)
        mask = mask.unsqueeze(-1).expand(context.shape)
        context = context[mask].reshape(x.shape[0],-1,x.shape[-1])
        return x1, context


if __name__=="__main__":
    import matplotlib.pyplot as plt
    lattice = Lattice(8.7,3)
    x = torch.linspace(-8.7/2,8.7/2,500).unsqueeze(1).repeat(1,3)
    for t in torch.linspace(0,1,10):
        t = torch.ones(len(x))*t
        force = lattice.scaled_confining_force(x,t)
        #print(force[:,0])
        plt.plot(x[:,0].detach(),force[:,0].detach())
    plt.show()
    # cell_idx = torch.randint(3,(100,))
    # x1,x2 = lattice.select_adjacent_cells(x,cell_idx)
    # plt.scatter(x1[:,0,0],x1[:,0,1])
    # plt.scatter(x2[:,0,0],x2[:,0,1])
    # plt.show()