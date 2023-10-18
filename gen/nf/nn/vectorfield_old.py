from typing import Callable, Dict, Union, Optional
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_
import math
import numpy as np

__all__ = ["VectorField"]


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function.
    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = xavier_uniform_,
        bias_init: Callable = zeros_,
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: umber of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
            weight_init: weight initializer from current weight.
            bias_init: bias initializer from current bias.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y

def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y

def concentrated_spacing(min,center,max,npoints):
    seq = -torch.exp(torch.torch.linspace(4,0,npoints//2))
    first_half = seq/(seq[-1]-seq[0])*(center-min)
    first_half = first_half + min - first_half[0]
    seq = torch.exp(torch.torch.linspace(0,4,npoints-npoints//2))
    second_half = seq/(seq[-1]-seq[0])*(max-center)
    second_half = second_half + center - second_half[0]
    return torch.cat((first_half,second_half),0)

class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, concentrated_at = None, trainable: bool = True
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        if concentrated_at is not None:
            offset = concentrated_spacing(start,concentrated_at,cutoff,n_rbf)
            widths = (torch.abs(offset-concentrated_at)/(cutoff-start)+1)**3/2*torch.ones_like(offset)
        else:
            offset = torch.linspace(start, cutoff, n_rbf)
            center = (start+cutoff)/2
            widths = (torch.abs(offset-center)/(cutoff-start)+1)**2/4*torch.ones_like(offset)

        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
            #self.register_buffer("offsets", offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)

class BesselRBF(nn.Module):
    """
    Sine for radial basis functions with coulomb decay (0th order bessel).
    References:
    .. [#dimenet] Klicpera, Groß, Günnemann:
       Directional message passing for molecular graphs.
       ICLR 2020
    """

    def __init__(self, n_rbf: int, cutoff: float):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselRBF, self).__init__()
        self.n_rbf = n_rbf

        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)

    def forward(self, inputs):
        ax = inputs[..., None] * self.freqs
        sinax = torch.sin(ax)
        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[..., None]
        return y

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
    """SchNet architecture for learning representations of atomistic systems
    References:
    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.
    """

    def __init__(
        self,
        n_atom_basis: int = 10,
        n_radial_basis: int = 50,
        cutoff : float = 5,
        n_filters: int = None,
        max_z: int = 100,
        dim = 3,
        boxlength = None,
        atomic_numbers = None
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            max_z: maximal nuclear charge
            activation: activation function
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.n_filters = n_filters or self.n_atom_basis
        self.cutoff = cutoff
        self.radial_basis = GaussianRBF(n_radial_basis, start=0, cutoff = 5, concentrated_at=1)
        self.time_basis = GaussianRBF(n_atom_basis, start=0, cutoff=1)  
        self.boxlength = boxlength
        self.dim = dim
        self.atomic_numbers = atomic_numbers
        self.W = nn.Parameter(0.01*torch.randn(self.n_atom_basis, n_radial_basis))
        # layers
        self.embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)

    def _phi(self,d_ij,t=None,return_derivatives=False):
        f_ij = self.radial_basis(d_ij)
        # compute atom and pair features
        x = self.embedding(self.atomic_numbers).unsqueeze(0)
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
        x = x* self.time_basis(t)[:,None].expand(d_ij.shape[0], d_ij.shape[1], self.n_atom_basis)
        W = self.W.unsqueeze(0).expand(d_ij.shape[0],-1,-1)
        prefactor = torch.einsum("bit,btr->bir",x,W)
        phi = torch.einsum("bir,bijr->bij",prefactor,f_ij)
        if return_derivatives:
            partial_f_partial_d = self.batch_jacobian(self.radial_basis,d_ij,batch_dim=(0,1,2))
            partial_phi_partial_d = torch.einsum("bir,bijr->bij",prefactor,partial_f_partial_d)

            return phi, partial_phi_partial_d
        else:
            return phi

    def batch_jacobian(self,func, x, batch_dim=(0,), create_graph=False, **kwargs):
            # x in shape (Batch, Length)
        def _func_sum(x):
            return func(x,**kwargs).sum(dim=batch_dim)
        shifted_batch_dim = tuple(torch.tensor(batch_dim)+len(func(x,**kwargs).shape)-len(batch_dim))
        return torch.autograd.functional.jacobian(_func_sum, x, create_graph=create_graph).movedim(shifted_batch_dim,batch_dim)
    
    def _trace_test(self,positions):
        positions.requires_grad_(True)
        jacob = self.batch_jacobian(self.forward,positions)
        #print("actual jacob:",jacob[0])
        jacob = rearrange(jacob, 'b i j k l -> b (i j) (k l)')
        trace = torch.diagonal(jacob, dim1=1, dim2=2).sum(dim=1)
        return trace
    
    def _trace(self,positions,t):
        return self.forward(positions,t,return_trace=True)
    
    def forward(self, positions,t=None,residual=True,return_trace=False):
        if t is None:
            t = torch.ones(positions.shape[0], device=positions.device)
        r_ij = distance_vectors(positions)
        d_ij = distances_from_vectors(r_ij)
        if self.boxlength is not None:
            to_subtract = ((torch.abs(d_ij) > 0.5 * self.boxlength)
                * torch.sign(d_ij) * self.boxlength)
            d_ij = d_ij - to_subtract

        if return_trace:
            phi, partial_phi_partial_d = self._phi(d_ij,t,return_derivatives=True)
            #print("calculated:", partial_phi_partial_d[0]*d_ij[0]+self.dim*phi[0])
            trace = torch.einsum("bij,bij->b",partial_phi_partial_d,d_ij)+torch.sum(self.dim*phi,dim=(1,2))
        else:
            phi = self._phi(d_ij,t,return_derivatives=False)
        score = torch.einsum("bij,bijd->bid", phi,r_ij)
        if residual:
            score = score+positions
        if return_trace:
            #trace = self._trace_test(positions)
            #print("should be 0", trace,trace_2)
            return score, trace
        else:
            return score

if __name__=="__main__":
    positions = torch.randn(2,4,3)
    atomic_numbers = torch.ones(positions.shape[1]).int()
    model = VectorField(atomic_numbers=atomic_numbers,n_radial_basis=50)
    print(model(positions,return_trace=True))
    #positions_flipped = torch.flip(positions,[1])
    #print(model(positions_flipped,return_trace=True))
    #rotate 90 degrees
    #rotation_mat = torch.tensor([[0,1,0],[-1,0,0],[0,0,1]]).float().unsqueeze(0).expand(positions.shape[0],-1,-1)
    #positions_rotated = torch.einsum("bij,bjk->bik",positions,rotation_mat)
    #print(model(positions_rotated,return_trace=True))
    #print(torch.einsum("bij,bjk->bik",model(positions),rotation_mat))