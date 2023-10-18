import numpy as np
import logging
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

from dm.nf.nf import setup_nf
from dm.diffusion.diffusion import setup_diffusion

from dm.pipeline.config import get_cfg_defaults
from dm.pipeline.train import Trainer, optimization_manager
from dm.pipeline import utils
from dm.pipeline.load_data import DataHandler

def setup_training(cfg,model):
    optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr, 
                           betas=(cfg.optimizer.beta1, 0.999), eps=cfg.optimizer.eps)
    #optimizer = optim.AdamW(model.parameters(), lr=cfg.lr=cfg.optimizer.lr)
    optimization_fn = optimization_manager(cfg)
    if cfg.train_parameters.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, cfg.train_parameters.lr_scheduler_gamma)
    elif cfg.train_parameters.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train_parameters.max_epochs)
    elif cfg.train_parameters.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logging.getLogger('matplotlib.font_manager').disabled = True
    
    return optimizer, optimization_fn, scheduler, logger


def setup_model(cfg, mode="train"):
    data_handler = DataHandler(cfg,mode=mode)
    dict = {"data_handler": data_handler}
    if cfg.model.type == "nf":
        if cfg.model.sde.type is not None:
            diffusion = setup_diffusion(cfg,data_handler)
        else:
            diffusion = None
        model = setup_nf(cfg,diffusion)
    elif cfg.model.type == "diffusion":
        model = setup_diffusion(cfg,data_handler)
    dict.update({"model": model,"ema_model": copy.deepcopy(model)})
    if mode != "sample":
        optimizer, optimization_fn, scheduler, logger = setup_training(cfg,model)
        dict.update({"optimizer" : optimizer,
                    "optimization_fn" : optimization_fn,
                     "scheduler" : scheduler,
                     "logger" : logger})

    utils.mkdir(cfg.output.training_dir)
    utils.mkdir(cfg.output.testing_dir)
    utils.mkdir(cfg.output.model_dir)
    utils.mkdir(cfg.output.best_model_dir)

    return dict

    
def load_model(cfg, name=None,model_dir=None,model_file=None,mode="train",prefix=""):
    if mode == "sample" or mode == "resume":
        if model_file is None:
            saved = torch.load("%s%s.pth"%(model_dir,name),map_location='cpu')
        else:
            saved = torch.load(model_file,map_location='cpu')
    dict = setup_model(cfg,mode=mode)
    if mode == "sample":
        dict["model"].load_state_dict(saved["model"],strict=False)
        dict["model"].to(cfg.device)
        dict["ema_model"].load_state_dict(saved["ema"],strict=False)
        dict["ema_model"].to(cfg.device)
        loss = saved["loss"]
        dict.update({"loss" : loss})
        # plt.plot(np.arange(0,len(loss)*interval,interval)+1, loss)
        # plt.xlabel("training step")
        # plt.ylabel("loss")
        # plt.ylim(top=0)
        # plt.savefig(cfg.output.testing_dir+cfg.dataset.name+"_loss.png")
        # plt.close()
        np.savetxt(cfg.output.testing_dir+cfg.dataset.name+"_loss.dat",loss.cpu().numpy())
    else:
        start_epoch = 0
        if mode == "resume":
            dict["model"].load_state_dict(saved["model"],strict=False)
            dict["model"].to(cfg.device)
            dict["ema_model"].load_state_dict(saved["ema"],strict=False)
            dict["ema_model"].to(cfg.device)
            try:
                dict["optimizer"].load_state_dict(saved["optim"])
                dict["scheduler"].load_state_dict(saved["scheduler"])
                print("lr:", dict["optimizer"].param_groups[0]["lr"])
            except:
                pass
            if "loss" in saved.keys():
                dict["loss"] = saved["loss"].tolist()
            if "best_loss" in saved.keys():
                dict["best_loss"] = saved["best_loss"]
            start_epoch = saved["epoch"]

        trainer = Trainer(**dict,device = cfg.device,
                          train_batch_size = cfg.train_parameters.batch_size,
                          start_epoch=start_epoch,results_folder=cfg.output.training_dir,
                          gradient_accumulate_every = 1,
                          model_name=prefix+cfg.dataset.name,max_epochs= cfg.train_parameters.max_epochs)
        dict.update({"trainer": trainer})
    
    return dict 


def read_input(dir):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(dir)
    print(cfg)
    return cfg