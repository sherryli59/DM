import torch
from pytorch_lightning import LightningModule
from typing import Any

from gen.diffusion import loss_fn
from gen.diffusion.sde import *


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
    def __init__(self,score,sde,shape,data_handler=None,likelihood_weighting=False,
                 eps = 1e-4, lr: float = 0.001,
                weight_decay: float = 0.0005, **kwargs: Any):
        super().__init__()    
        self.save_hyperparameters()
        self.eps = eps
        self.data_handler = data_handler
        self.shape = shape
        self.likelihood_weighting = likelihood_weighting
        self.sde = sde
        self.score_fn = score
        if hasattr(self.sde, "sde_list"):
            self.loss_fn = lambda x,t: loss_fn.piecewise_sde_loss(self.sde,self.score_fn,x,t)
        else:
            self.loss_fn = lambda x,t: loss_fn.single_sde_loss(self.sde,self.score_fn,x,t)

               
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


    def sample(self,nsamples,method="ode",return_traj=False, return_prob=False,batchsize=100,init=None,t_init=None):
        if t_init is None:
            t_init = torch.full((nsamples,),1.).to(self.device)
        if init is None:
            init = torch.randn([nsamples]+list(self.shape)).to(self.device)
        if init is not None:
            nsamples = init.shape[0]
        if nsamples < batchsize:
            batchsize = nsamples
        for i in range(nsamples//batchsize):
            start = i*batchsize
            end = min((i+1)*batchsize,nsamples)
            with torch.no_grad():
                batch_out = self.sde.backward(init[start:end],score_fn=self.score_fn,
                                        t_init=t_init[start:end],method=method,return_traj=return_traj,return_prob=return_prob)
            if i==0:
                out = batch_out
            else:
                for key, val in batch_out.items():
                    out[key] = torch.cat((out[key],val),dim=0)
            out.update({"init":init})
        return out
    
    
    



def force_wrapper(data_handler):
    def f(x):
        x_decoded = data_handler.decoder(x.clone())
        f = -data_handler.distribution.neg_force_clipped(x_decoded)
        f = data_handler.encoder(f)
        return f
    return f

