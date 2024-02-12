from typing import Callable, Dict, Union, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_
import math


__all__ = ["SchNet", "SchNetInteraction"]

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

def shifted_softplus(x: torch.Tensor):
    r"""Compute shifted soft-plus activation function.
    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)
    Args:
        x (torch.Tensor): input tensor.
    Returns:
        torch.Tensor: shifted soft-plus of input.
    """
    return F.softplus(x) - math.log(2.0)

def replicate_module(
    module_factory: Callable[[], nn.Module], n: int, share_params: bool
):
    if share_params:
        module_list = nn.ModuleList([module_factory()] * n)
    else:
        module_list = nn.ModuleList([module_factory() for i in range(n)])
    return module_list

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
class GaussianRBFCentered(nn.Module):
    r"""Gaussian radial basis functions centered at the origin."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 1.0, trainable: bool = True
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: width of last Gaussian function, :math:`\mu_{N_g}`
            start: width of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBFCentered, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        widths = torch.linspace(start, cutoff, n_rbf)
        offset = torch.zeros_like(widths)
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
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

def cosine_cutoff(input: torch.Tensor, cutoff: torch.Tensor):
    """ Behler-style cosine cutoff.
        .. math::
           f(r) = \begin{cases}
            0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
              & r < r_\text{cutoff} \\
            0 & r \geqslant r_\text{cutoff} \\
            \end{cases}
        Args:
            cutoff (float, optional): cutoff radius.
        """

    # Compute values of cutoff function
    input_cut = 0.5 * (torch.cos(input * math.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= (input < cutoff).float()
    return input_cut

def mollifier_cutoff(input: torch.Tensor, cutoff: torch.Tensor, eps: torch.Tensor):
    r""" Mollifier cutoff scaled to have a value of 1 at :math:`r=0`.
    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}
    Args:
        cutoff: Cutoff radius.
        eps: Offset added to distances for numerical stability.
    """
    mask = (input + eps < cutoff).float()
    exponent = 1.0 - 1.0 / (1.0 - torch.pow(input * mask / cutoff, 2))
    cutoffs = torch.exp(exponent)
    cutoffs = cutoffs * mask
    return cutoffs

class CosineCutoff(nn.Module):
    r""" Behler-style cosine cutoff module.
    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}
    """

    def __init__(self, cutoff: float):
        """
        Args:
            cutoff (float, optional): cutoff radius.
        """
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff

    def forward(self, input: torch.Tensor):
        return cosine_cutoff(input, self.cutoff)




class SchNetInteraction(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(SchNetInteraction, self).__init__()
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation), Dense(n_filters, n_filters)
        )

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        """Compute interaction output.
        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j
        Returns:
            atom features after interaction
        """
        n_atoms = x.shape[1]
        n_configs = x.shape[0]
        y = self.in2f(x)
        W = self.filter_network(f_ij)
        W = W * rcut_ij[..., None]
        # continuous-filter convolution

        neighbors = torch.tensor([[list(range(n_atoms))] * n_atoms
                                  for _ in range(n_configs)], device=x.device)

        #mask = (torch.eye(n_atoms) == 0).to(y.device)
        #neighbors = neighbors.masked_select(mask).view((-1,
        #                                                n_atoms,
        #                                                n_atoms - 1))
        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, y.size(2))
        nbh = nbh.to(y.device)
        y = torch.gather(y, 1, nbh)
        y = y.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)
        y = y * W
        y = y.sum(dim=2)
        y = self.f2out(y)
        return y


class SchNet(nn.Module):
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

        n_atom_basis: int = 64,
        n_time_basis: int = 64,
        n_interactions: int = 3,
        n_radial_basis: int = 20,
        cutoff : float = 5,
        radial_basis_type: str = "gaussian",
        n_filters: int = None,
        shared_interactions: bool = False,
        max_z: int = 100,
        dim = 3,
        boxlength = None,
        activation: Callable = shifted_softplus,
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
        self.n_time_basis = n_time_basis
        self.n_basis = n_atom_basis + n_time_basis
        self.size = (self.n_basis,)
        self.n_filters = n_filters or self.n_basis
        self.dim = dim
        if radial_basis_type == "bessel":
            self.radial_basis = BesselRBF(n_radial_basis, cutoff)
        elif radial_basis_type == "gaussian":
            self.radial_basis = GaussianRBFCentered(n_radial_basis, cutoff)
            
        self.cutoff_fn = CosineCutoff(cutoff)
        self.cutoff = cutoff
        self.boxlength = boxlength
        self.energy_predictor = nn.Sequential(Dense(self.n_basis, 
                self.n_basis//2, activation=activation),
                Dense(self.n_basis//2, 1, activation=None))
        self.norm = nn.LayerNorm(dim,elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),nn.Linear(n_time_basis, 2 * dim, bias=True))
        self.energy_shift = nn.Sequential(Dense(self.dim, 
                self.n_basis//2, activation=activation),
                Dense(self.n_basis//2, 1, activation=None))
        self.sum = lambda x: torch.einsum("b...->b", x)
        
        # layers
        self.atom_embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)
        self.time_embedding = TimestepEmbedder(self.n_time_basis)

        self.interactions = replicate_module(
            lambda: SchNetInteraction(
                n_atom_basis=self.n_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )
    
    def forward(self, positions,t=None,atomic_numbers=None,box_vector=None):
        try:
            positions.requires_grad_(True)
        except:
            pass
        with torch.enable_grad():
            if t is None:
                t = torch.ones(positions.shape[0], device=positions.device)
            if atomic_numbers is None:
                atomic_numbers = torch.ones(*positions.shape[:2], device=positions.device).long()
            r_ij  = (positions.unsqueeze(-2) - positions.unsqueeze(-3))
            eps = 1e-6
            d_ij = (r_ij.pow(2).sum(dim=-1) + eps).sqrt()
            n_atoms = positions.shape[1]
            mask = (torch.eye(n_atoms) == 0).to(positions.device)
            d_ij = d_ij.masked_select(mask).view((-1, n_atoms, n_atoms - 1))
            distance_to_origin = torch.norm(positions, dim=-1).unsqueeze(-1)
            d_ij = torch.cat((d_ij,distance_to_origin),dim=-1)
            if self.boxlength is not None:
                d_ij = d_ij % self.boxlength
                to_subtract = ((torch.abs(d_ij) > 0.5 * self.boxlength)
                    * torch.sign(d_ij) * self.boxlength)
                d_ij = d_ij - to_subtract
            f_ij = self.radial_basis(d_ij)
            dcut_ij = self.cutoff_fn(d_ij)

            # compute atom and pair features
            x = self.atom_embedding(atomic_numbers)
            t_emb = self.time_embedding(t).unsqueeze(1).expand(-1,n_atoms,-1)
            x = torch.cat((x,t_emb),dim=-1)
            # compute interaction block to update atomic embeddings
            for interaction in self.interactions:
                v = interaction(x, f_ij, dcut_ij)
                x = x + v 
            energy = self.sum(self.energy_predictor(x))
            if box_vector is not None:
                displacement = positions-box_vector.unsqueeze(-2)
                #layer normalization
                displacement = self.norm(displacement)
                #modulation
                shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=-1)
                displacement = displacement * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
                energy = energy + self.sum(
                    self.energy_shift(displacement))
            score = -torch.autograd.grad(energy, positions, 
                        grad_outputs=torch.ones_like(energy),retain_graph=True,create_graph=True)[0]  
            return score
