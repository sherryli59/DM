from dm.simulations import distributions, adp, lj
import torch

def setup_distribution(cfg,name,dist_cfg):
    config = dist_cfg
    device = cfg.device
    if name =="GaussianMixture":
        distribution=distributions.GaussianMixture(config.centers,config.std,cfg.distribution.nparticles,cfg.distribution.dim,device=device)
    elif name =="LJ":
        distribution=lj.LJ(pos_dir=cfg.distribution.testing_data, boxlength=cfg.distribution.boxlength, device=device, sigma=cfg.distribution.sigma,
         epsilon=cfg.distribution.epsilon, cutoff=cfg.distribution.cutoff, shift=cfg.distribution.shift, periodic=cfg.distribution.periodic)
    elif name == "LJ_fluid":
        distribution=lj.LJ_polar(pos_dir=cfg.distribution.testing_data, boxlength=cfg.distribution.boxlength, device=device, sigma=cfg.distribution.sigma,
         epsilon=cfg.distribution.epsilon, cutoff=cfg.distribution.cutoff, shift=config.shift)
    elif name =="EinsteinCrystal":
        distribution=distributions.EinsteinCrystal(config.centers,cfg.distribution.shape[-1], shape = config.shape,
                                                   cutoff=config.cutoff, boxlength=cfg.distribution.boxlength,
                                                     std=config.std,device=device)
    elif name == "TruncatedNormal":
        centers = torch.zeros((cfg.distribution.nparticles,cfg.distribution.dim))
        distribution=distributions.EinsteinCrystal(centers,cutoff=cfg.distribution.cutoff, dim=cfg.distribution.dim, std=config.std,device=device)
    elif name =="Normal":
        if config.shape is not None:
            shape = config.shape    
        else:
            shape=(config.nparticles* config.dim,)
        if config.std == None:
            var = 1.0
        else: var = config.std**2
        distribution = distributions.StandardNormal(cfg.distribution.nparticles* cfg.distribution.dim, var,shape,device=device)
    elif name == "Fe":
        distribution = distributions.Fe(cfg.distribution.input_dir, pos_dir=cfg.distribution.testing_data, cell_len=cfg.distribution.cell_len, device=device)
    elif name == "SimData":
        distribution = distributions.SimData(pos_dir=cfg.distribution.testing_data, device=device)
    elif name == "ADP":
        distribution = adp.ADP()
    else:
        print("Distribution not found")
        return None
    return distribution