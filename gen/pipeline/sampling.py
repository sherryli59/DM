import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class MonteCarloSampler:
    def __init__(self, dim=0, beta=1.0, k=0.0, energy_fn=None, step_size=1e-2):
        self.dim = dim
        self.step_size = step_size
        self.x = torch.zeros(dim)
        self.beta = beta
        self.energy = 0.0
        self.bias_energy = 0.0
        self.k = k
        self.energy_fn = energy_fn
        self.n_per_sweep = 1
        self.samples = torch.empty((0, self.dim))
        self.samp_energy = torch.empty(0)

    def initialize(self, x):
        self.x = x.clone()
        self.energy = self.compute_energy(self.x)

    def compute_energy(self, x=None, args=[], kwargs={}):
        with torch.no_grad():
            if x is None:
                x = self.x.clone()
            return self.energy_fn(x, *args, **kwargs)

    def sample_step(self, record=True, scheme="MALA"):
        with torch.no_grad():
            dx = self.step_size * torch.randn(self.x.shape)
            dE = self.compute_energy(self.x + dx) - self.energy

            if dE < 0.0 or np.random.rand() < np.exp(-self.beta * dE):
                self.x += dx
                self.energy += dE
            if record:
                self.samples = torch.cat(
                    (self.samples, self.x.clone().view(1, -1)))
                self.samp_energy = torch.cat(
                    [self.samp_energy, self.energy.clone().unsqueeze(0)]
                )
            return self.x.view(-1)

    def log_prob(self, x=None):

        return -self.beta * self.compute_energy(x)


class stdG:
    def __init__(self, d=2, nsamples=10000):
        base_mu, base_cov = torch.zeros(d), torch.eye(d)
        self.dist = MultivariateNormal(base_mu, base_cov)
        self.samples = self.get_samples(nsamples)
        self.d = d

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def get_samples(self, nsamples):
        return self.dist.rsample(sample_shape=(nsamples,))


class gaussian_on_lattice:
    def __init__(self, length=1, width=1, height=1):
        self.length = length
        self.width = width
        self.height = height
        self.ncells = length * width * height
        self.lattice = torch.cartesian_prod(torch.arange(
            length), torch.arange(width), torch.arange(height))
        base_mu, base_cov = torch.zeros(3), torch.eye(3)
        self.dist = MultivariateNormal(base_mu, base_cov)

    def log_prob(self, x):
        nparticles=x.shape[1]
        prob=1/self.ncells*torch.stack([torch.exp(self.dist.log_prob(x-center.expand(nparticles,-1))) for center in self.lattice]).sum(dim=0)
        log_prob=torch.log(prob).reshape(-1,nparticles).sum(dim=1)
        return log_prob

    def get_samples(self, nsamples):
        centers = torch.stack((torch.randint(self.length, (nsamples,)), torch.randint(
            self.width, (nsamples,)), torch.randint(self.height, (nsamples,))),dim=1)
        return centers + self.dist.rsample(sample_shape=(nsamples,))


class lattice:
    def __init__(self, n, samples, energy, beta):
        self.n = n
        self.samples = samples
        self.beta = beta
        self.energy = energy

    def get_samples(self, nsamples):
        total_n = list(self.samples.size())[0]
        indices = torch.randint(total_n, [nsamples])
        return self.samples.index_select(0, indices), self.energy.index_select(0, indices)


class StochasticAllenCahn(MonteCarloSampler):
    def __init__(self, n, beta=1.0, D=1.0, k=0.0, k_m=0.0, h=1.0, step_size=5e-2):
        super().__init__()
        self.n = n
        self.dim = n * n
        self.h = h
        self.D = D
        self.k_m = k_m
        self.beta = beta
        self.k = k
        self.step_size = step_size
        self.x = torch.zeros(n, n)
        self.energy = 0
        self.samples = torch.empty((0, self.n ** 2))
        self.samp_energy = torch.empty(0)

    def discrete_laplacian(self, site, bc="periodic"):
        neigh_sum = 0.0

        if bc == "dirichlet":
            if site[0] == 0:
                neigh_sum += -1.0 + self.x[site[0] + 1, site[1]]
            elif site[0] == self.n - 1:
                neigh_sum += self.x[site[0] - 1, site[1]] - 1.0
            else:
                neigh_sum += self.x[site[0] - 1, site[1]] + \
                    self.x[site[0] + 1, site[1]]

            if site[1] == 0:
                neigh_sum += 1.0 + self.x[site[0], site[1] + 1]
            elif site[1] == self.n - 1:
                neigh_sum += self.x[site[0], site[1] - 1] + 1.0
            else:
                neigh_sum += self.x[site[0], site[1] - 1] + \
                    self.x[site[0], site[1] + 1]

        elif bc == "periodic":
            if site[0] == 0:
                neigh_sum += self.x[self.n - 1, site[1]] + \
                    self.x[site[0] + 1, site[1]]
            elif site[0] == self.n - 1:
                neigh_sum += self.x[site[0] - 1, site[1]] + self.x[0, site[1]]
            else:
                neigh_sum += self.x[site[0] - 1, site[1]] + \
                    self.x[site[0] + 1, site[1]]

            if site[1] == 0:
                neigh_sum += self.x[site[0], self.n - 1] + \
                    self.x[site[0], site[1] + 1]
            elif site[1] == self.n - 1:
                neigh_sum += self.x[site[0], site[1] - 1] + self.x[site[0], 0]
            else:
                neigh_sum += self.x[site[0], site[1] - 1] + \
                    self.x[site[0], site[1] + 1]

        neigh_sum -= 4 * self.x[tuple(site)]
        return neigh_sum / self.h ** 2

    def initialize(self, x):
        self.x = x
        self.energy = self.compute_energy()

    def compute_m(self):
        return torch.mean(self.x)

    def compute_gradient(self, x, site):
        if site[0] == self.n - 1:
            dx = (x[0, site[1]] - x[site]) / self.h
        else:
            dx = (x[site[0] + 1, site[1]] - x[site]) / self.h

        if site[1] == self.n - 1:
            dy = (x[site[0], 0] - x[site]) / self.h
        else:
            dy = (x[site[0], site[1] + 1] - x[site]) / self.h
        return torch.Tensor([dx, dy])

    def compute_energy(self, x=None):
        if x is None:
            x = self.x
        energy = 0.0
        with torch.no_grad():
            for i in range(self.n):
                for j in range(self.n):
                    energy += (
                        0.5 * self.D
                        * torch.linalg.norm(
                            self.compute_gradient(
                                x.view((self.n, self.n)), tuple([i, j])
                            )
                        )
                        ** 2
                    )
                    energy += (
                        -0.5 * x.view(self.n, self.n)[i, j] ** 2
                        + 0.25 * x.view(self.n, self.n)[i, j] ** 4
                    )
        return energy * self.h ** 2

    def log_prob(self, x=None):

        return -self.beta * self.compute_energy(x)

    def sample_step(self, record=True, scheme="MALA"):
        with torch.no_grad():
            self.x = self.x.view(self.n, self.n)
            dx = torch.zeros((self.n, self.n))
            randsite = torch.randint(self.n, [2])
            dx[tuple(randsite)] = self.step_size * (
                self.x[tuple(randsite)] - self.x[tuple(randsite)] ** 3
            ) + self.step_size * self.D * self.discrete_laplacian(randsite)
            dx[tuple(randsite)] += (
                self.step_size*np.sqrt(2 / self.beta)*20 * np.random.randn()
            )
            if scheme == "MALA":
                dE = self.compute_energy(self.x + dx) - self.energy
                #print(np.exp(-self.beta * dE))
                if dE < 0.0 or np.random.rand() < np.exp(-self.beta * dE):
                    self.x += dx
                    self.energy += dE
            elif scheme == "simple":
                self.x += dx

            if record:
                self.samples = torch.cat(
                    (self.samples, self.x.clone().view(1, -1)))
                self.samp_energy = torch.cat(
                    [self.samp_energy, self.energy.clone().unsqueeze(0)]
                )
            return self.x.view(-1)


class StochasticAllenCahn1D(StochasticAllenCahn):
    def __init__(self, n, beta=20, a=0.1, h=0.01, step_size=5e-4):
        super().__init__(n, beta=beta, h=h, step_size=step_size)
        self.x = torch.zeros(n)
        self.a = a
        self.energy = 0
        self.samples = torch.empty((0, self.n))

    def discrete_laplacian(self, site, bc="dirichlet"):
        if site == 0:
            return (0 + self.x[site + 1] - 2 * self.x[site]) / self.h ** 2
        elif site == self.n - 1:
            return (self.x[site - 1] + 0 - 2 * self.x[site]) / self.h ** 2
        else:
            return (
                self.x[site - 1] + self.x[site + 1] - 2 * self.x[site]
            ) / self.h ** 2

    def compute_gradient(self, x, site):
        if site == self.n - 1:
            return (0 - x[site]) / self.h
        else:
            return (x[site + 1] - x[site]) / self.h

    def U_1(self, phi):
        return (1 - phi ** 2) ** 2 / (4 * self.a)

    def compute_energy(self, x=None):
        if x is None:
            x = self.x
        energy = 0.0
        with torch.no_grad():
            energy += 0.5 * self.a * (x[0] / self.h) ** 2
            for i in range(self.n):
                energy += 0.5 * self.a * self.compute_gradient(x, i) ** 2
                energy += self.U_1(x[i])
        return energy * self.h

    def F(self, site):
        return self.x[site] - self.x[site] ** 3

    def sample_step(self, record=True, scheme="MALA"):
        with torch.no_grad():
            dx = torch.zeros(self.n)
            randsite = torch.randint(self.n, [1])
            dx[randsite] = self.step_size * self.F(randsite) / self.a
            dx[randsite] += self.step_size * self.a * \
                self.discrete_laplacian(randsite)
            dx[randsite] += np.sqrt(2 * self.step_size /
                                    self.beta) * np.random.randn()
            if scheme == "MALA":
                dE = self.compute_energy(self.x + dx) - self.energy
                if dE < 0.0 or np.random.rand() < np.exp(-self.beta * dE):
                    self.x += dx
                    self.energy += dE
            elif scheme == "simple":
                self.x += dx

            if record:
                self.samples = torch.cat(
                    (self.samples, self.x.clone().view(1, -1)))
                self.samp_energy = torch.cat(
                    [self.samp_energy, self.energy.clone().unsqueeze(0)]
                )
            return self.x.view(-1)


class StochasticAllenCahn1D_base(StochasticAllenCahn1D):
    def F(self, site):
        return -self.x[site]

    def U_1(self, phi):
        return phi ** 2 / (2 * self.a)

    def get_samples(self, nsamples):
        self.samples = torch.load("../saved_models/ac1d_base_samples.pth")
        total_n = list(self.samples.size())[0]
        indices = torch.randint(total_n, [nsamples])
        return self.samples.index_select(0, indices)

    def log_prob(self, x=None):
        if x is None:
            x = self.x.clone()
        prob = []
        for xi in x.view(-1, self.n):
            prob.append(-self.beta * self.compute_energy(xi))
        return torch.Tensor(prob).view(-1)
