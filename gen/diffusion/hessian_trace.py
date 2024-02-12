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

def exact_trace(f,x):
      alphabet = string.ascii_lowercase
      dydx = torch.vmap(torch.func.jacrev(f))(x)
      num_dims = len(x.shape)-1
      # Check if the number of dimensions is within the allowed range for einsum
      if num_dims > len(alphabet):
            raise ValueError("Number of dimensions exceeds einsum's capability")
      # Construct the einsum string for the given number of dimensions
      # The last two dimensions are supposed to be equal and summed over
      einsum_str = alphabet[num_dims] + alphabet[:num_dims] + alphabet[:num_dims] + '->' + alphabet[num_dims]
      dydx = dydx.reshape(dydx.shape[0],*dydx.shape[2:])
      return torch.einsum(einsum_str,dydx)
