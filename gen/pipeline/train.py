import torch
import copy
from pathlib import Path
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from dm.pipeline import utils

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optimizer.lr,
                  warmup=config.optimizer.warmup,
                  grad_clip=config.optimizer.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn
    
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class Trainer(object):
    def __init__(
        self,
        model,
        optimizer,
        optimization_fn,
        scheduler,
        data_handler,
        *,
        device = "cuda",
        ema_model = None,
        loss = [],
        best_loss = None,
        start_epoch = 0,
        ema_decay = 0.9999,
        train_batch_size = 32,
        max_epochs = 1000,
        gradient_accumulate_every = 1,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 10,
        update_scheduler_every = 500,
        results_folder = './results',
        model_name = "model",
        load_path = None,
        logger = None
    ):
        super().__init__()
        self.data_handler = data_handler
        self.dl = data_handler.dataloader

        if data_handler.test_dataloader is not None:
            self.test_dl = cycle(data_handler.test_dataloader)
        else:
            self.test_dl = None
        self.model = model
        self.ema = EMA(ema_decay)
        if ema_model is not None:
            self.ema_model = ema_model
        else:
            self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.update_scheduler_every = update_scheduler_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_epochs = max_epochs
        self.device = device


        self.optimizer = optimizer
        self.optimization_fn = optimization_fn
        self.scheduler = scheduler
        self.epoch = start_epoch
        
        self.writer = SummaryWriter(log_dir=results_folder,filename_suffix=f'{model_name}')
        self.loss_history = loss
        self.best_loss = best_loss
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        self.model_name = model_name

        self.reset_parameters()

        if load_path != None:
            self.load(load_path)


    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict(),strict=False)

    def step_ema(self,step):
        if step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, loss=None, best_loss=None, itrs=None):
        data = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'optim' : self.optimizer.state_dict(),
            'scheduler' : self.scheduler.state_dict()
        }
        if loss is not None:
            data['loss'] = loss
        if best_loss is not None:
            data['best_loss'] = best_loss
        if itrs is None:
            torch.save(data, str(self.results_folder / f'{self.model_name}.pth'))
        else:
            torch.save(data, str(self.results_folder / f'{self.model_name}_{itrs}.pth'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path)

        self.epoch = data['epoch']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.scheduler.load_state_dict(data['scheduler'])
        self.optimizer.load_state_dict(data['optim'])
        if 'loss' in data.keys():
            self.loss_history = data['loss'].tolist()
        if 'best_loss' in data.keys():
            self.best_loss = data['best_loss']

    def train(self):
        acc_loss = 0
        n_batches = len(self.dl)
        if self.best_loss is None:
            self.best_loss = -torch.log(torch.tensor(0))
        self.best_loss = 64
        while self.epoch < self.max_epochs:
            for i_batch, data in enumerate(self.dl): 
                data = data.to(self.device)
                step = self.epoch*n_batches+i_batch
                loss = self.model.compute_loss(data)
                #lambd = 0.5*(self.epoch-1)/self.max_epochs
                #print("lambda:",lambd)
                lambd = 0
                if lambd > 0:
                    energy_loss = utils.reverseKL(self.model,
                                              self.data_handler.distribution.potential,len(data))
                    loss = (1-lambd)*loss + lambd*energy_loss     
                if isinstance(loss, list): 
                    for i, l in enumerate(loss):
                        self.writer.add_scalar(f'Loss/train_{i}',l, step)
                        #l.backward()   
                        acc_loss += l
                        print(l)
                else:
                    acc_loss += loss
                    print(loss)
                    #loss.backward()
                if step % self.gradient_accumulate_every == 0:
                    loss_normalized = acc_loss / self.gradient_accumulate_every
                    self.writer.add_scalar('Loss/train',loss_normalized, step)
                    self.optimization_fn(self.optimizer, self.model.parameters(), step)
                    self.optimizer.zero_grad()
                    acc_loss = 0

                if step % self.update_ema_every == 0:
                    self.step_ema(step)
                    self.loss_history.append(loss_normalized)

                if step != 0 and step % self.save_and_sample_every == 0:
                    #milestone = step // self.save_and_sample_every
                    #self.save(itrs=milestone)
                    if self.test_dl is not None:
                        test_data = next(self.test_dl)
                        test_data = test_data.to(self.device).requires_grad_(False)
                        test_loss = self.model.compute_loss(test_data)
                        print("test loss:",test_loss)
                        if isinstance(test_loss, list):
                            test_loss = sum(test_loss)
                        self.writer.add_scalar('Loss/test',test_loss, step)
                        if test_loss < self.best_loss:
                            self.best_loss = test_loss
                            print("New best loss : ", self.best_loss)
                        self.save(torch.tensor(self.loss_history),self.best_loss)
                        if step > 5000 and (test_loss - loss_normalized )/test_loss > 0.1:
                            print("Overfitting?")
                    else:
                        self.save(torch.tensor(self.loss_history))
                    torch.cuda.empty_cache()
                if (self.update_scheduler_every is not None) and (step % (self.update_scheduler_every) == 0):
                    self.scheduler.step()
            self.epoch += 1

        print('training completed')
        self.writer.flush()
        self.writer.close()
