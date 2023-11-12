import torch
import torch.nn as nn
import numpy as np
from typing import Union


def tile(a, dim, n_tile):
    """
    Tiles a pytorch tensor along one an arbitrary dimension.

    Parameters
    ----------
    a : PyTorch tensor
        the tensor which is to be tiled
    dim : Integer
        dimension along the tensor is tiled
    n_tile : Integer
        number of tiles

    Returns
    -------
    b : PyTorch tensor
        the tensor with dimension `dim` tiled `n_tile` times
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)]
    )
    order_index = torch.LongTensor(order_index).to(a).long()
    return torch.index_select(a, dim, order_index)
        
def distance_vectors(x, remove_diagonal=True):
    r"""
    Computes the matrix :math:`r` of all distance vectors between
    given input points where

    .. math::
        r_{ij} = x_{i} - y_{j}

    as used in :footcite:`Khler2020EquivariantFE`

    Parameters
    ----------
    x : torch.Tensor
        Tensor of shape `[n_batch, n_particles, n_dimensions]`
        containing input points.
    remove_diagonal : boolean
        Flag indicating whether the all-zero distance vectors
        `x_i - x_i` should be included in the result

    Returns
    -------
    r : torch.Tensor
        Matrix of all distance vectors r.
        If `remove_diagonal=True` this is a tensor of shape
            `[n_batch, n_particles, n_particles, n_dimensions]`.
        Otherwise this is a tensor of shape
            `[n_batch, n_particles, n_particles - 1, n_dimensions]`.

    """
    r = tile(x.unsqueeze(2), 2, x.shape[1])
    r = r - r.permute([0, 2, 1, 3])
    if remove_diagonal:
        r = r[:, torch.eye(x.shape[1], x.shape[1]) == 0].view(
            -1, x.shape[1], x.shape[1] - 1, x.shape[2]
        )
    return r


def rbf_kernels(d: torch.Tensor, mu: Union[torch.Tensor, float], neg_log_gamma: Union[torch.Tensor, float],
                derivative=False) -> torch.Tensor:
    """
    Takes a distance matrix `d` of shape

        `[n_batch, n_particles, n_particles, 1]`

    and maps it onto a normalized radial basis function (RBF) kernel
    representation of shape

        `[n_batch, n_particles, n_particles, n_kernels]`

    via

        `d_{ij} -> f_{ij}

    where

        `f_{ij} = (g_{ij1}, ..., g_{ijK}) / sum_{k} g_{ijk}

    and

        `g_{ijk} = exp(-(d_{ij} - mu_{k})^{2} / gamma^{2})`.

    Parameters
    ----------
    d: PyTorch tensor
        distance matrix of shape `[n_batch, n_particles, n_particles, 1]`
    mu: PyTorch tensor / scalar
        Means of RBF kernels. Either of shape `[1, 1, 1, n_kernels]` or
        scalar
    neg_log_gamma: PyTorch tensor / scalar
        Negative logarithm of bandwidth of RBF kernels. Either same shape as `mu` or scalar.
    derivative: boolean
        Whether the derivative of the rbf kernels is computed.

    Returns
    -------
    kernels: PyTorch tensor
        RBF representation of distance matrix of shape
        `[n_batch, n_particles, n_particles, n_kernels]`
    dkernels: PyTorch tensor
        Corresponding derivatives of shape
        `[n_batch, n_particles, n_particles, n_kernels]`
    """
    inv_gamma = torch.exp(neg_log_gamma)
    rbfs = torch.exp(-(d - mu).pow(2) * inv_gamma.pow(2))

    srbfs = rbfs.sum(dim=-1, keepdim=True)
    kernels = rbfs / (1e-6 + srbfs)
    if derivative:
        drbfs = -2 * (d - mu) * inv_gamma.pow(2) * rbfs
        sdrbfs = drbfs.sum(dim=-1, keepdim=True)
        dkernels = drbfs / (1e-6 + srbfs) - rbfs * sdrbfs / (1e-6 + srbfs ** 2)
    else:
        dkernels = None
    return kernels, dkernels

def distances_from_vectors(r, eps=1e-6):
    """
    Computes the all-distance matrix from given distance vectors.
    
    Parameters
    ----------
    r : torch.Tensor
        Matrix of all distance vectors r.
        Tensor of shape `[n_batch, n_particles, n_other_particles, n_dimensions]`
    eps : Small real number.
        Regularizer to avoid division by zero.
    
    Returns
    -------
    d : torch.Tensor
        All-distance matrix d.
        Tensor of shape `[n_batch, n_particles, n_other_particles]`.
    """
    return (r.pow(2).sum(dim=-1) + eps).sqrt()

class VectorField(nn.Module):
    """
    Equivariant dynamics functions.
    Equivariant dynamics functions that allows an efficient
    and exact divergence computation :footcite:`Khler2020EquivariantFE`.

    References
    ----------
    .. footbibliography::

    """

    def __init__(self,shape, cutoff,
                 mus=None, gammas=None,
                 mus_time=None, gammas_time=None,
                 hidden_dim=500,
                 optimize_d_gammas=False,
                 optimize_t_gammas=False, device="cuda"):
        super().__init__()
        self.device = device
        self._n_particles = shape[0]
        self._n_dimensions = shape[1]
        if mus is None:
            mus,gammas,mus_time,gammas_time = self._init_mus_gammas(cutoff,hidden_dim)
        self._mus = mus.to(device)
        self._neg_log_gammas = -torch.log(gammas)

        self._n_kernels = self._mus.shape[0]

        self._mus_time = mus_time.to(device)
        self._neg_log_gammas_time = -torch.log(gammas_time)

        if self._mus_time is None:
            self._n_out = 1
        else:
            assert self._neg_log_gammas_time is not None and self._neg_log_gammas_time.shape[0] == self._mus_time.shape[
                0]
            self._n_out = self._mus_time.shape[0]

        if optimize_d_gammas:
            self._neg_log_gammas = torch.nn.Parameter(self._neg_log_gammas).to(device)

        if optimize_t_gammas:
            self._neg_log_gammas_time = torch.nn.Parameter(self._neg_log_gammas_time).to(device)

        self._weights = torch.nn.Parameter(
            torch.Tensor(self._n_kernels, self._n_out).normal_() * np.sqrt(1. / self._n_kernels)
        )
        self._bias = torch.nn.Parameter(
            torch.Tensor(1, self._n_out).zero_()
        )

        self._importance = torch.nn.Parameter(
            torch.Tensor(self._n_kernels).zero_()
        )

    def _init_mus_gammas(self,cutoff,hidden_dim,t_range=[0,1]):
        mus = torch.linspace(0, cutoff, hidden_dim)
        mus.sort()
        gammas = 0.3 * torch.ones(len(mus))
        mus_time = torch.linspace(t_range[0], t_range[1], 20)
        gammas_time = 0.3 * torch.ones(len(mus_time))
        return mus,gammas,mus_time,gammas_time
    
    def _force_mag(self, t, d, derivative=False):
        importance = self._importance.to(d.device)
        self._mus = self._mus.to(d.device)
        self._neg_log_gammas = self._neg_log_gammas.to(d.device)
        self._mus_time = self._mus_time.to(t.device)
        self._neg_log_gammas_time = self._neg_log_gammas_time.to(t.device)
        rbfs, d_rbfs = rbf_kernels(d, self._mus, self._neg_log_gammas, derivative=derivative)
        force_mag = (rbfs + importance.pow(2).view(1, 1, 1, -1)) @ self._weights + self._bias
        if torch.any(torch.isnan(force_mag)):
            raise ValueError("force_mag is nan")
        if derivative:
            d_force_mag = (d_rbfs) @ self._weights
        else:
            d_force_mag = None
        if self._mus_time is not None:
            trbfs, _ = rbf_kernels(t.unsqueeze(-1), self._mus_time, self._neg_log_gammas_time)
            trbfs = trbfs.unsqueeze(1).unsqueeze(1)
            force_mag = (force_mag * trbfs).sum(dim=-1, keepdim=True)
            if derivative:
                d_force_mag = (d_force_mag * trbfs).sum(dim=-1, keepdim=True)
        return force_mag, d_force_mag

    def _trace(self,x,t):
        return self.forward(x,t,compute_divergence=True)

    def forward(self, x, t, compute_divergence=False):
        """
        Computes the change of the system `dxs` at state `x` and
        time `t` due to the kernel dynamic. Furthermore, can also compute the exact change of log density
        which is equal to the divergence of the change.

        Parameters
        ----------
        t : PyTorch tensor
            The current time
        x : PyTorch tensor
            The current configuration of the system
        compute_divergence : boolean
            Whether the divergence is computed

        Returns
        -------
        forces, -divergence: PyTorch tensors
            The combined state update of shape `[n_batch, n_dimensions]`
            containing the state update of the system state `dx/dt`
            (`forces`) and the negative exact update of the log density (`-divergence`)

        """
        x= x.to(self.device)
        t = t.to(self.device)
        if len(t.shape) == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        n_batch = x.shape[0]
        x = x.view(n_batch, self._n_particles, self._n_dimensions)
        r = distance_vectors(x)
        d = distances_from_vectors(r).unsqueeze(-1)
        force_mag, d_force_mag = self._force_mag(t, d, derivative=compute_divergence)
        forces = (r * force_mag).sum(dim=-2)
        if compute_divergence:
            divergence = (d * d_force_mag + self._n_dimensions * force_mag).view(n_batch, -1).sum(dim=-1)
            divergence = divergence.unsqueeze(-1)
            return forces, divergence
        else:
            return forces

if __name__=="__main__":
    n_particles = 10
    n_dimension = 2
    d_max = 8
    mus = torch.linspace(0, d_max, 50).cuda()
    mus.sort()
    gammas = 0.3 * torch.ones(len(mus)).cuda()

    mus_time = torch.linspace(0, 1, 10).cuda()
    gammas_time = 0.3 * torch.ones(len(mus_time)).cuda()

    kdyn = VectorField(n_particles, n_dimension, mus, gammas, optimize_d_gammas=True, optimize_t_gammas=True,
                        mus_time=mus_time, gammas_time=gammas_time).cuda()
    
    x = torch.randn(2, n_particles, n_dimension).cuda()
    t = torch.randn(2).cuda()
    print(kdyn(x,t))
    x_reversed = torch.flip(x, dims=[1])
    print(kdyn(x_reversed,t))
    