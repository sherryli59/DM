import torch
import string

def hutch_trace(f,x,m=100):
    repeat_shape = [m]+[1]*(len(x.shape)-1)
    x_mult = x.repeat(repeat_shape).detach()
    x_mult.requires_grad_(True)
    eps = torch.randint(0,2,[m*x.shape[0]]+list(x.shape)[1:]).float()*2-1
    eps = eps.to(x.device)
    #eps = torch.randn([m*x.shape[0]]+list(x.shape)[1:])
    f_e = torch.sum(f(x_mult) * eps)
    grad_f_e = torch.autograd.grad(f_e, x_mult)[0]*eps
    grad_f_e = grad_f_e.reshape(m,x.shape[0],-1)
    return torch.mean(torch.sum(grad_f_e, dim=-1),dim=0)


def exact_trace(f, x,t, create_graph=False):
      shape = x.shape
          # x in shape (Batch, Length)
      def _func_sum(x):
        return f(x.reshape(shape),t).sum(dim=0).flatten()
      jacobian = torch.autograd.functional.jacobian(_func_sum, x.reshape(x.shape[0],-1), create_graph=create_graph).transpose(0,1)
      return torch.vmap(torch.trace)(jacobian).flatten()

def exact_trace_2(f,x,t):
      alphabet = string.ascii_lowercase
      print("x.shape",x.shape)
      dydx = torch.vmap(torch.func.jacrev(f))(x,t)
      print("dydx.shape",dydx.shape)
      num_dims = len(x.shape)-1
      # Check if the number of dimensions is within the allowed range for einsum
      if num_dims > len(alphabet):
            raise ValueError("Number of dimensions exceeds einsum's capability")
      # Construct the einsum string for the given number of dimensions
      # The last two dimensions are supposed to be equal and summed over
      einsum_str = alphabet[num_dims] + alphabet[:num_dims] + alphabet[:num_dims] + '->' + alphabet[num_dims]
      dydx = dydx.reshape(dydx.shape[0],*dydx.shape[2:])
      return torch.einsum(einsum_str,dydx)

def gradient(y, x, grad_outputs=None):
    " Compute dy/dx @ grad_outputs "
    " train_points: [B, DIM]"
    " model: R^{DIM} --> R"
    " grads: [B, DIM]"
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grads = torch.autograd.grad(y, [x], 
                                grad_outputs=grad_outputs,                        
                                retain_graph=True,
                                create_graph=True,
                                allow_unused=False)[0]
    return grads

def partial_t_j(f, x, t, j):
    """
    :param s: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: (dsdt)_j (torch.tensor) of shape [B, 1]
    """
    assert j <= x.shape[-1]
    s = f(x, t)
    v = torch.zeros_like(s)
    v[:, j] = 1.
    dy_j_dx = torch.autograd.grad(
                   s,
                   t,
                   grad_outputs=v,
                   retain_graph=True,
                   create_graph=True,
                   allow_unused=False)[0]  # shape [B, N]
    return dy_j_dx

def batch_div(f, x, t):
    x.requires_grad = True
    def batch_jacobian():
        f_sum = lambda x: torch.sum(f(x, t), axis=0)
        return torch.autograd.functional.jacobian(f_sum, x, create_graph=True, strict=True).permute(1,0,2) 
    jac = batch_jacobian()
    return torch.sum(jac.diagonal(offset=0, dim1=-1, dim2=-2), dim=-1, keepdim=False)
       

    
def hutch_div(score_model, sample, time_steps, repeat=1):      
    """Compute the divergence of the score-based model with Skilling-Hutchinson."""
    with torch.enable_grad():
        sample.requires_grad_(True)
        divs = torch.zeros((sample.shape[0],), device=sample.device, requires_grad=False) #div: [B,]
        for _ in range(repeat):
            epsilon = torch.randn_like(sample)
            score_e = torch.sum(score_model(sample, time_steps) * epsilon)
            grad_score_e = torch.autograd.grad(score_e, sample,
                                                retain_graph=True,
                                                create_graph=True,
                                                allow_unused=False)[0]
            divs += torch.sum(grad_score_e * epsilon, dim=(1))  
        divs = divs/repeat
    return divs

def t_finite_diff(fn, x, t, hs=0.001, hd=0.0005):
    up = hs**2 * fn(x, t+hd) + (hd**2 - hs**2) * fn(x, t) - hd**2 * fn(x, t-hs)
    low = hs * hd * (hd+hs)
    return up/low  

