import torch
import torch.nn as nn
import torch.distributions as D

class GaussianMixture(D.mixture_same_family.MixtureSameFamily):
    def __init__(self, centers, std, npoints=None, device="cuda"):
        centers = torch.tensor(centers,device=device).float()
        std = torch.tensor(std,device=device).float()
        dim = centers.size(-1)
        ncenters = len(centers)
        mix = D.Categorical(torch.ones(ncenters,device=device))
        if std.dim()==1:
            std = std.unsqueeze(-1).expand(-1,dim)
        comp = D.Independent(D.Normal(centers, std), 1)
        D.mixture_same_family.MixtureSameFamily.__init__(self, mix, comp)
        self.npoints = npoints


    def sample(self, nsamples):
        if self.npoints is None:
            shape = torch.Size([nsamples])
        else:
            shape = torch.Size([nsamples,self.npoints])
        return (super().sample(shape))
    
    def potential(self,x):
        return -self.log_prob(x).sum(dim=-1)
    
    def force(self,x):
        require_grad = x.requires_grad
        with torch.enable_grad():
            try:
                x.requires_grad_(True)
            except:
                pass
            pot=self.potential(x)
            force = -torch.autograd.grad(pot,x,torch.ones_like(pot),create_graph=True)[0]
        x.requires_grad_(require_grad)
        return force
    
    def neg_force_clipped(self,x):
        return -torch.clamp(self.force(x),-80,80)

if __name__=="__main__":
    model = GaussianMixture([[0,0,0],[1,1,1]], [0.1,0.2], npoints=10)
    print(model.sample(5).shape)
