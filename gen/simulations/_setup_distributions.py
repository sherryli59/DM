from dm.simulations import distributions, adp, lj
import torch

def setup_distribution(cfg,name,dist_cfg):
    device = cfg.device
    config = dist_cfg
    if name =="GaussianMixture":
        distribution=distributions.GaussianMixture(config.centers,config.std,cfg.dataset.nparticles,cfg.dataset.dim,device=device)
    elif name =="LJ":
        distribution=lj.LJ(pos_dir=cfg.dataset.testing_data, boxlength=cfg.dataset.boxlength, device=device, sigma=cfg.dataset.sigma,
         epsilon=cfg.dataset.epsilon, cutoff=cfg.dataset.cutoff, shift=cfg.dataset.shift, periodic=cfg.dataset.periodic)
    elif name == "LJ_fluid":
        distribution=lj.LJ_polar(pos_dir=cfg.dataset.testing_data, boxlength=cfg.dataset.boxlength, device=device, sigma=cfg.dataset.sigma,
         epsilon=cfg.dataset.epsilon, cutoff=cfg.dataset.cutoff, shift=config.shift)
    elif name =="EinsteinCrystal":
        distribution=distributions.EinsteinCrystal(config.centers,cfg.dataset.shape[-1], shape = config.shape,
                                                   cutoff=config.cutoff, boxlength=cfg.dataset.boxlength,
                                                     std=config.std,device=device)
    elif name == "TruncatedNormal":
        centers = torch.zeros((cfg.dataset.nparticles,cfg.dataset.dim))
        distribution=distributions.EinsteinCrystal(centers,cutoff=cfg.dataset.cutoff, dim=cfg.dataset.dim, std=config.std,device=device)
    elif name =="Normal":
        if config.shape is not None:
            shape = config.shape    
        else:
            shape=(config.nparticles* config.dim,)
        if config.std == None:
            var = 1
        else: var = config.std**2
        distribution = distributions.StandardNormal(cfg.dataset.nparticles* cfg.dataset.dim, var,shape,device)
    elif name == "Fe":
        distribution = distributions.Fe(cfg.dataset.input_dir, pos_dir=cfg.dataset.testing_data, cell_len=cfg.dataset.cell_len, device=device)
    elif name == "SimData":
        distribution = distributions.SimData(pos_dir=cfg.dataset.testing_data, device=device)
    elif name == "ADP":
        distribution = adp.ADP()
    else:
        print("Distribution not found")
        return None
    return distribution