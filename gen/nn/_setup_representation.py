from gen.nn.unet import Unet
from gen.nn.transformer import Transformer
from gen.nn.mlp import TimeEmbeddedMLP
from gen.nn.misc import score_Gaussian
from gen.nn.schnet import SchNet
from gen.nn.vectorfield import VectorField
from gen.nn.egnn import EGNN
from gen.nn.vnn import VNN
from gen.diffusion.sde_utils import NoiseSchedule

import torch
import numpy as np
from omegaconf import OmegaConf
def setup_representation(cfg,type,**kwargs):
    device = cfg.model.device
    if type == "unet":
        rep = Unet(dim=64, channels=cfg.distribution.nchannels,dim_mults=(1, 2, 4, 8))
    elif type == "transformer":
        rep = Transformer(cfg.model.representation.dim,cfg.model.representation.dim, n_layers=6, n_head=6, d_k=128, d_v=128,
            d_hidden=128)
    elif type =="mlp":
        d_in = torch.prod(torch.tensor(cfg.model.sde.shape))
        rep = TimeEmbeddedMLP(d_in,cfg.model.representation.hidden_dim,
                               num_blocks=cfg.model.representation.num_blocks,dropout=0.0
                               )
    elif type =="gaussian analytical":
        rep = score_Gaussian(mean=cfg.distribution.centers,std = cfg.distribution.std, noise_schedule=NoiseSchedule(
        type=cfg.model.sde.schedule))
    elif type =="schnet" or type =="vectorfield":
        atomic_numbers = OmegaConf.select(cfg, "distribution.atomic_numbers")
        if atomic_numbers is None:
            atomic_numbers = torch.ones(cfg.model.representation.nparticles).long()
        else:
            atomic_numbers = torch.tensor(np.load(cfg.distribution.atomic_numbers))
        max_z = int(torch.max(atomic_numbers).item()+1)
        if type == "schnet":
            rep = SchNet(cutoff=cfg.distribution.cutoff, dim = cfg.model.representation.dim,
                                atomic_numbers=atomic_numbers,max_z=max_z, boxlength=cfg.distribution.boxlength)
        else:
            # rep = VectorField(cutoff=cfg.distribution.cutoff, dim = cfg.model.representation.dim,
            #                    atomic_numbers=atomic_numbers,max_z=max_z)
            d_max = cfg.model.representation.cutoff
            mus = torch.linspace(0, d_max, cfg.model.representation.hidden_dim)
            mus.sort()
            gammas = 0.3 * torch.ones(len(mus))
            t_range = kwargs.get("t_range", [0,1])
            mus_time = torch.linspace(t_range[0], t_range[1], 20)
            gammas_time = 0.3 * torch.ones(len(mus_time))
            rep = VectorField(cfg.model.representation.nparticles, cfg.model.representation.dim, mus, gammas, optimize_d_gammas=False, optimize_t_gammas=False,
                                mus_time=mus_time, gammas_time=gammas_time,device=device)

    elif type =="gnn":
        atomic_numbers = OmegaConf.select(cfg, "model.representation.atomic_numbers")
        if atomic_numbers is None:
            atomic_numbers = torch.ones(cfg.model.representation.nparticles).long()
        else:
            atomic_numbers = torch.tensor(np.load(cfg.model.representation.atomic_numbers))
        rep = EGNN(2,cfg.model.representation.hidden_dim, cutoff=cfg.model.representation.cutoff,
                   atomic_numbers=atomic_numbers,n_layers=cfg.model.representation.num_blocks) 
    elif type == "vnn":
        rep = VNN(n_knn=cfg.model.representation.nparticles)
    else:
        raise ValueError("representation type not recognized")  
    return rep