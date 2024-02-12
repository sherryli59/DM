import torch
from pytorch_lightning import LightningModule, Callback
from typing import Any
import numpy as np


from gen.diffusion import loss_fn
from gen.diffusion.sde import *
from gen.simulations.utils import write_coord

class Resample(Callback):
    #def on_train_start(self,trainer, pl_module):
        #if pl_module.live:
            #trainer.datamodule.dataset.start()
            #trainer.datamodule.reload_data()
    def on_train_epoch_end(self, trainer, pl_module):
        # log best so far train loss
        pl_module.metric_hist['train/loss'].append(
            trainer.callback_metrics['train/loss'])
        pl_module.log('train/loss_best',
                 min(pl_module.metric_hist['train/loss']), prog_bar=False)
        if pl_module.live:
            if trainer.current_epoch>5:
                with torch.no_grad():
                    out = pl_module.sample(trainer.datamodule.dataset.sample_size,batch_size=50, method="ode", return_prob=True)
                proposal = out["x"]
                prob = out["logp"]
                trainer.datamodule.dataset.update_proposals(proposal,prob,pl_module.likelihood_estimator)
            trainer.datamodule.dataset.run()
            trainer.datamodule.reload_data()
    
class BaseModule(LightningModule):
    def __init__(self):
        super().__init__()    
        self.metric_hist = {
            'train/loss': [],
            'val/loss': [],
        }
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
                "monitor": "val/loss",
                "frequency": 1,
            },
        }

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print('detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def on_train_epoch_end(self):
        # log best so far train loss
        self.metric_hist['train/loss'].append(
            self.trainer.callback_metrics['train/loss'])
        self.log('train/loss_best',
                 min(self.metric_hist['train/loss']), prog_bar=False)
        
    def on_validation_epoch_end(self):
        # log best so far val acc and val loss
        self.metric_hist['val/loss'].append(
            self.trainer.callback_metrics['val/loss'])
        self.log('val/loss_best',
                 min(self.metric_hist['val/loss']), prog_bar=False)
        
    
    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log('train/loss', loss, on_step=True,
                  on_epoch=False, prog_bar=True)
        return {'loss': loss}
    
    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log('val/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        return {'loss': loss}
    
    def test_step(self, batch: Any, batch_idx: int):
        return {'loss': self.step(batch)}
    
class DiffusionModel(BaseModule):
    def __init__(self,score,sde,shape,conditional=False,live=False,likelihood_weighting=False,
                 eps = 1e-4, lr: float = 0.001, 
                weight_decay: float = 0.0005,lattice=None, **kwargs: Any):
        super().__init__()    
        self.save_hyperparameters()
        self.eps = eps
        self.shape = shape
        self.likelihood_weighting = likelihood_weighting
        self.sde = sde
        self.score_fn = score.to(self.device)
        self.conditional = conditional
        self.lattice = lattice
        self.live = live
        if live:
            self.likelihood_estimator = lambda x: self.logp(x)
        if conditional:
            self.loss_fn = lambda x,t: loss_fn.conditional_loss(self.sde,self.score_fn,x,t,lattice=self.lattice)
        else:
            if hasattr(self.sde, "sde_list"):
                self.loss_fn = lambda x,t: loss_fn.piecewise_sde_loss(self.sde,self.score_fn,x,t)
            else:
                self.loss_fn = lambda x,t: loss_fn.single_sde_loss(self.sde,self.score_fn,x,t)

    def configure_callbacks(self):
        resample = Resample()
        return [resample]

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log('train/loss', loss, on_step=True,
                  on_epoch=False, prog_bar=True)
        return {'loss': loss}


    def generate_t(self,nsamples):
        if hasattr(self.sde, "sde_list"):
            t = torch.zeros(nsamples).to(self.device)
            n_sdes = len(self.sde.sde_list)
            sde_idx = torch.randint(n_sdes,(nsamples,))
            for i in range(n_sdes):
                mask = sde_idx==i
                t[mask] = self.sde.sde_list[i].generate_t(sum(mask)).to(self.device)
        else:
            t = self.sde.generate_t(nsamples).to(self.device)
        return t
            
    def step(self, batch: Any):
        t = self.generate_t(batch.shape[0])
        if torch.min(t)<1e-4:
            print("Warning: t<1e-4")
            print(t)
        loss = self.loss_fn(batch,t)
        return loss
    
    def test_step(self, batch: Any, batch_idx: int):
        if batch_idx <= 1:
            self.score_fn.correction.plot_schedule()
        if self.conditional:
            cell_idx = batch[...,0]
            x = batch[...,1:]
            #randomly select a cell
            x1, x2 = self.lattice.select_adjacent_cells(x,cell_idx)
            context = x2[0].unsqueeze(0).expand_as(x2)
            out = self.sample(len(batch),context=context,method="e-m")
            conditional_sample = torch.cat((context,out["x"]),dim=1)
            np.save("conditional_sample_traj.npy",conditional_sample.detach().cpu().numpy())
            write_coord("conditional_sample_traj.xyz",conditional_sample.detach().cpu().numpy(),
                        nparticles=conditional_sample.shape[1])
            np.save("simulated_sample_single.npy",batch[0].detach().cpu().numpy())
            write_coord("simulated_sample_single.xyz",batch[0].detach().cpu().numpy(),
                        nparticles=batch[0].shape[0])
        else:
            pass
    
    def logp(self,x):
        with torch.no_grad():
            return self.sde.logp(x,self.score_fn)[0]
    
    def sample(self,nsamples, method="ode",return_traj=False, return_prob=False,batch_size=100,init=None,t_init=None,context=None):
        self.score_fn = self.score_fn.to(self.device)
        if t_init is None:
            t_init = torch.full((nsamples,),1.).to(self.device)
        if init is None:
            shape = self.shape.copy()
            dist = self.sde.prior(shape)
            init = dist.sample((nsamples,)).to(self.device)
        nsamples = init.shape[0]
        if nsamples < batch_size:
            batch_size = nsamples
        for i in range(nsamples//batch_size):
            start = i*batch_size
            end = min((i+1)*batch_size,nsamples)
            context_batch = context[start:end].to(self.device) if context is not None else None
            batch_out = self.sde.backward(init[start:end],score_fn=self.score_fn, context=context_batch,
                                    t_init=t_init[start:end],method=method,return_traj=return_traj,return_prob=return_prob)
            if i==0:
                out = batch_out
            else:
                for key, val in batch_out.items():
                    out[key] = torch.cat((out[key],val),dim=0)
        out.update({"init":init})
        return out
    


