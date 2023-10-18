import matplotlib.pyplot as plt
import torch

import os

import tables
from pathlib import Path

def batch_jacobian(func, x, create_graph=False):
        # x in shape (Batch, Length)
    def _func_sum(x):
      return func(x).sum(dim=0)
    return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).permute(1,0,2)

def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, eps=None, noise_mult=1, **kwargs):
    if eps is None:
        eps = torch.randn([noise_mult]+list(x.shape)).to(x.device)
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, **kwargs).unsqueeze(0) * eps, dim=tuple(range(1, len(x.shape)+1)))
      grad_fn_eps = torch.autograd.grad(fn_eps, x,torch.ones_like(fn_eps))[0]

    x.requires_grad_(False)

    return torch.mean(torch.sum(grad_fn_eps* eps, dim=tuple(range(2, len(x.shape)+1))),dim=0)

  return div_fn


        
                    
def reverseKL(model,potential,nsamples):
    x, log_prob = model.sample(nsamples,return_prob=True)
    return torch.mean(potential(x))+torch.mean(log_prob)

def KL(model,x=None, potential=None,nsamples=None,device="cuda:0"):
    if x is None:
        x = potential.sample(nsamples,flatten=True).to(device)
    z, prior_logprob, log_det = model(x)
    logprob = prior_logprob + log_det
    if potential is None:
        return -torch.mean(logprob)
    else:
        return -torch.mean(logprob)+torch.mean(potential.log_prob(x))

def plot_Q(cfg,Q,split=False,save=True):
    if split:
        fig,(ax1,ax2)=plt.subplots(1,2,sharex=True, sharey=True,figsize=(12,6),tight_layout=True)
        ax1.plot(Q[0][:,0], Q[0][:,1],'.',color="darkgray")
        ax1.set_title("trajectory generated by NF")
        ax2.plot(Q[1][:,0], Q[1][:,1],'.',color="darkgray")
        ax2.set_title("trajectory from MD simulation")
        fig.supxlabel("logpx from NF")
        fig.supylabel("-potential (kT)")
    else:
        plt.plot(Q[0][:,0], Q[0][:,1],'.',color="darkblue",label="NF traj")
        plt.plot(Q[1][:,0], Q[1][:,1],'.',color="darkgray",label="MD traj")
        plt.xlabel("logpx from NF")
        plt.ylabel("-potential (kT)")
        plt.legend()
    if save:
        plt.savefig(cfg.output.testing_dir+"Q_%s.png"%cfg.dataset.name)
    else:
        plt.show()
    plt.close()


                
def metropolize_naive(cfg,potential,x,burnin=20):
    x=x.cpu().detach().reshape(-1,cfg.dataset.nparticles,3)
    nsamples=x.size(dim=0)
    index=[False for i in range(nsamples)]
    full_energy_list = potential.potential(x)/cfg.dataset.kT
    frame=x[0]
    energy=full_energy_list[0]
    energy_list=[]
    for i in range(nsamples):
        new_frame=x[i]
        new_energy=full_energy_list[i]
        acc_prob=torch.exp(torch.tensor(energy-new_energy))
        if torch.rand(1)<acc_prob:
            frame=new_frame
            energy=new_energy
            if i>burnin:
                index[i]=True
            energy_list.append(energy)
    print(full_energy_list)
    return x[index],energy_list


def mkdir(dir):
    if not(os.path.exists(dir)):
        os.mkdir(dir)


def plot_loss(training_loss,testing_loss=None,filename=None):
    plt.plot(training_loss,label="training_loss")
    if testing_loss is not None:
        plt.plot(training_loss,label="training_loss")
    plt.legend()
    plt.title("loss vs iteration")
    if filename is not None:
        plt.savefig(filename)
    plt.show()
    plt.close()

def cycle(dl):
    while True:
        for data in dl:
            yield data



def append_to_h5(data):
    f_path = Path(f"./data.h5")
    if not f_path.exists():
        f = tables.open_file(str(f_path), mode='w')
        atom = tables.Float64Atom()
        batches_ea = f.create_earray(f.root, 'batches', atom, shape=(0, *data.shape[1:]))
    else:
        f = tables.open_file(str(f_path), mode='a')
        f.root.batches.append(data.cpu().numpy())
    f.close()