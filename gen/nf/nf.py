import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning import LightningModule
from typing import Any

from nf.flows.cnf import FFJORD
from nf.flows.autoregressive import RQSAutoregressive
from nf.nn._setup_representation import setup_representation
from nf.simulations._setup_distributions import setup_distribution



class BaseModule(LightningModule):
    def __init__(self):
        super().__init__()    
        self.metric_hist = {
            'train/loss': [],
            'val/loss': [],
        }
    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
    
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
        self.log('train/loss', loss, on_step=False,
                  on_epoch=True, prog_bar=True)
        return {'loss': loss}
    
    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.step(batch)
        self.log('val/loss', loss, on_step=False,
                 on_epoch=True, prog_bar=False)
        return {'loss': loss}
    
    def test_step(self, batch: Any, batch_idx: int):
        return {'loss': self.step(batch)}
    
    
class NormalizingFlow(BaseModule):

    def __init__(self, cfg, lr, weight_decay, is_iaf=True):
        super().__init__()
        self.save_hyperparameters()
        self.setup_nf(cfg)
        self.is_iaf = is_iaf # if iaf, x->z is the one-step forward pass

    def setup_nf(self,cfg):
        if cfg.model.flow.type == "continuous":
            rep = setup_representation(cfg,cfg.model.representation.type)  
            self.flow = FFJORD(rep)
            self.context_dim = 0
        elif cfg.model.flow.type == "autoregressive":
            if "dependecy" in cfg.distribution and cfg.distribution.dependency is not None:
                atom_dependency = torch.tensor(np.load(cfg.distribution.dependency))
            else:
                atom_dependency = None
            if "context_dim" in cfg.model.representation:
                self.context_dim = cfg.model.representation.context_dim
            else:    
                self.context_dim = 0
            d_in = torch.prod(torch.tensor(cfg.distribution.shape))-self.context_dim
            self.flow = RQSAutoregressive(
                features=d_in,
                dependency=atom_dependency,
                context_dim=self.context_dim,
                hidden_features=cfg.model.representation.hidden_dim,
                num_bins=32,
                n_layers=2,
                tails="linear",
                tail_bound=0.5*cfg.distribution.boxlength, 
                num_blocks = cfg.model.representation.num_blocks,
                random_mask=False,
                use_residual_blocks=False)
        self.prior = setup_distribution(cfg, cfg.model.prior.type,cfg.model.prior)
    def forward(self, x, context=None):
        z, log_det = self.flow.forward(x, context)
        return z, log_det

    def reverse(self, z, context=None):
        x, log_det = self.flow.reverse(z,context)
        return x, log_det

    def step(self,x):
        if self.context_dim > 0:
            context = x[:, :self.context_dim].reshape(x.shape[0], -1)
            x = x[:, self.context_dim:].reshape(x.shape[0],-1)
            logprob = self.evaluate(x,context)
        else:
            logprob = self.evaluate(x)
        return -torch.mean(logprob)

    def sample(self, nsamples, context=None, prior=None, return_prob=True,return_base=False):
        if prior is None:
            prior = self.prior
        with torch.no_grad():
            z = prior.sample((nsamples,))
            if self.is_iaf:
                x, log_det = self.reverse(z,context)
            else:
                x, log_det = self.forward(z,context)

        log_pz = prior.log_prob(z)
        log_px = log_pz -log_det
        if self.context_dim > 0:
            context = context.reshape(x.shape[0],-1)
            x = torch.cat([context,x],dim=1)
        if return_base:
            return x,log_px,z,log_pz
        elif return_prob:
            return x,log_px
        else:
            return x

    def evaluate(self,x,context=None, prior=None, return_base=False):
        if self.is_iaf:
            z, log_det = self.forward(x,context)
        else:
            z, log_det = self.inverse(x,context)
        if prior is not None:
            prior_logprob = prior.log_prob(z)
        else:
            prior_logprob = self.prior.log_prob(z)
        log_px = prior_logprob+log_det
        if return_base:
            return log_px, prior_logprob
        else:
            return log_px

