from dm.nn.unet import Unet
from dm.nn.transformer import Transformer
from dm.nn.mlp import TimeEmbeddedMLP
from dm.nn.misc import score_Gaussian
from dm.nn.schnet import SchNet
from dm.nn.vectorfield import VectorField
from dm.nn.egnn import EGNN
from dm.nn.vnn import VNN
from dm.diffusion.sde_utils import NoiseSchedule

import torch
import numpy as np
from omegaconf import OmegaConf
def setup_representation(cfg,type,**kwargs):
       
    if type == "unet":
        rep = Unet(dim=64, channels=cfg.dataset.nchannels,dim_mults=(1, 2, 4, 8))
    elif type == "transformer":
        rep = Transformer(cfg.dataset.dim,cfg.dataset.dim, n_layers=6, n_head=6, d_k=128, d_v=128,
            d_hidden=128)
    elif type =="mlp":
        d_in = torch.prod(torch.tensor(cfg.dataset.shape))
        rep = TimeEmbeddedMLP(d_in,cfg.model.representation.hidden_dim,
                               num_blocks=cfg.model.representation.num_blocks,dropout=0.0
                               )
    elif type =="gaussian analytical":
        rep = score_Gaussian(mean=cfg.dataset.centers,std = cfg.dataset.std, noise_schedule=NoiseSchedule(
        type=cfg.model.sde.schedule))
    elif type =="schnet" or type =="vectorfield":
        atomic_numbers = OmegaConf.select(cfg, "dataset.atomic_numbers")
        if atomic_numbers is None:
            atomic_numbers = torch.ones(cfg.dataset.nparticles).to(cfg.device).long()
        else:
            atomic_numbers = torch.tensor(np.load(cfg.dataset.atomic_numbers)).to(cfg.device)
        max_z = int(torch.max(atomic_numbers).item()+1)
        if type == "schnet":
            rep = SchNet(cutoff=cfg.dataset.cutoff, dim = cfg.dataset.dim,
                                atomic_numbers=atomic_numbers,max_z=max_z, boxlength=cfg.dataset.boxlength)
        else:
            # rep = VectorField(cutoff=cfg.dataset.cutoff, dim = cfg.dataset.dim,
            #                    atomic_numbers=atomic_numbers,max_z=max_z)
            d_max = cfg.dataset.cutoff
            mus = torch.linspace(0, d_max, cfg.model.representation.hidden_dim).to(cfg.device)
            mus.sort()
            gammas = 0.3 * torch.ones(len(mus)).to(cfg.device)
            t_range = kwargs.get("t_range", [0,1])
            print("t_range", t_range)
            mus_time = torch.linspace(t_range[0], t_range[1], 20).to(cfg.device)
            gammas_time = 0.3 * torch.ones(len(mus_time)).to(cfg.device)
            rep = VectorField(cfg.dataset.nparticles, cfg.dataset.dim, mus, gammas, optimize_d_gammas=True, optimize_t_gammas=True,
                                mus_time=mus_time, gammas_time=gammas_time).to(cfg.device)

    elif type =="gnn":
        atomic_numbers = OmegaConf.select(cfg, "dataset.atomic_numbers")
        if atomic_numbers is None:
            atomic_numbers = torch.ones(cfg.dataset.nparticles).to(cfg.device).long()
        else:
            atomic_numbers = torch.tensor(np.load(cfg.dataset.atomic_numbers)).to(cfg.device)
        rep = EGNN(2,cfg.model.representation.hidden_dim, cutoff=cfg.dataset.cutoff, device=cfg.device,
                   atomic_numbers=atomic_numbers,n_layers=cfg.model.representation.num_blocks) 
    elif type == "vnn":
        rep = VNN(n_knn=cfg.dataset.nparticles)
    else:
        raise ValueError("representation type not recognized")  
    return rep