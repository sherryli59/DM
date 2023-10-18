import numpy as np
import torch
import torch.nn as nn
from src.sampling import (MonteCarloSampler, stdG, StochasticAllenCahn,
                          StochasticAllenCahn1D, StochasticAllenCahn1D_base)
from src.RNVP import stacked_NVP
import matplotlib.pyplot as plt


class MCMC_w_flow:
    def __init__(
        self,
        model,
        walker,
        epochs,
        init_pos,
        optim,
        scheduler,
        base_dist,
        burnin=5000,
        k_lang=4,
        batch_size=1,
        save_dir=None,
        threshold=0.5,
    ):

        self.model = model
        self.walker = walker
        self.epochs = epochs
        self.batch_size = batch_size
        self.optim = optim
        self.scheduler = scheduler
        self.base_dist = base_dist
        self.burnin = burnin
        self.k_lang = k_lang
        self.losses = []
        self.acc_probs = []
        self.nwalkers = list(init_pos.size())[0]
        self.dim = list(init_pos.view(self.nwalkers, -1).size())[1]
        self.save_dir = save_dir
        self.threshold = threshold
        self.best_prob = 0

        for i in range(self.nwalkers):
            self.walker[i].initialize(init_pos[i])
        self.x = torch.cat(
            [self.walker[i].x.clone().view(1, -1)
             for i in range(self.nwalkers)]
        )
        self.batch = torch.empty((0, self.dim))

    def equilibrate(self):
        for i in range(self.nwalkers):
            for t in range(self.burnin):
                self.walker[i].sample_step(record=False, scheme="MALA")
            self.walker[i].energy = self.walker[i].compute_energy()
            # print(self.walker[i].energy)
        torch.save(
            torch.cat([self.walker[i].x.view(1, -1)
                      for i in range(self.nwalkers)]),
            "starting_pos.pth",
        )

    def resampling(self, log_pz, log_jacob):
        log_p_hat = log_pz + log_jacob
        log_p_star = torch.Tensor(
            [self.walker[i].log_prob(self.x[i]) for i in range(self.nwalkers)]
        )
        z_samp = self.base_dist.get_samples(nsamples=self.nwalkers)
        log_pz_samp = self.base_dist.log_prob(z_samp)
        x_samp = self.model.inverse(z_samp)
        z_samp_1, log_pz_samp_1, log_jacob_samp = self.model(x_samp)
        log_p_hat_samp = log_pz_samp + log_jacob_samp
        log_p_star_samp = torch.Tensor(
            [self.walker[i].log_prob(x_samp[i]) for i in range(self.nwalkers)]
        )
        # print("log_p_hat difference:", torch.mean(log_p_hat - log_p_hat_samp))
        # print("log_p_star difference:", torch.mean(log_p_star_samp - log_p_star))
        acc_prob = torch.exp(log_p_hat - log_p_hat_samp +
                             log_p_star_samp - log_p_star)
        acc = 0
        for i in range(self.nwalkers):
            acc_prob[i] = torch.min(torch.Tensor([1]), acc_prob[i])
            if np.random.rand() < acc_prob[i]:
                self.batch = torch.cat(
                    (self.batch, x_samp[i].clone().unsqueeze(0)))
                self.walker[i].initialize(x_samp[i])
                acc += 1
        mean_prob = acc_prob.mean()
        self.acc_probs.append(mean_prob)
        print("theoretical acceptance ratio", mean_prob)
        if self.save_dir is not None:
            if mean_prob > max(self.threshold, self.best_prob):
                self.best_prob = mean_prob
                self.save_model()

    def minimize_loss(self):
        self.optim.zero_grad()
        randindex = torch.randperm(list(self.batch.size())[0])
        z, log_pz, log_jacob = self.model(self.batch[randindex])
        loss = (-log_pz - log_jacob).mean()
        self.losses.append(loss)
        #print("loss:", loss.data)
        loss.backward()
        self.optim.step()
        self.scheduler.step()

    def save_model(self):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optim": self.optim.state_dict(),
                "losses": self.losses,
                "acc_probs": self.acc_probs,
            },
            self.save_dir,
        )

    def train(self):
        if self.burnin is not None:
            self.equilibrate()
        self.x = torch.cat(
            [self.walker[i].x.clone().view(1, -1)
             for i in range(self.nwalkers)]
        )
        self.x.requires_grad = False
        for k in range(self.epochs * self.batch_size):
            if k % self.batch_size == self.batch_size - 1:

                # minimize the forward KL divergence
                self.minimize_loss()

                self.batch = torch.empty((0, self.dim))

            with torch.no_grad():
                if k % (self.k_lang + 1) == self.k_lang:  # resampling step
                    _, log_pz, log_jacob = self.model(self.x)
                    self.resampling(log_pz, log_jacob)
                else:  # MCMC step
                    for i in range(self.nwalkers):
                        self.walker[i].sample_step(scheme="MALA")
                        self.x[i] = self.walker[i].x.clone().view(-1)
                        self.batch = torch.cat(
                            (self.batch, self.walker[i].x.clone().view(1, -1))
                        )

