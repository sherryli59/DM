import torch

def compute_distances(x, n_particles, n_dimensions, remove_duplicates=True):
    """
    Computes the all distances for a given particle configuration x.

    Parameters
    ----------
    x : torch.Tensor
        Positions of n_particles in n_dimensions.
    remove_duplicates : boolean
        Flag indicating whether to remove duplicate distances
        and distances be.
        If False the all distance matrix is returned instead.

    Returns
    -------
    distances : torch.Tensor
        All-distances between particles in a configuration
        Tensor of shape `[n_batch, n_particles * (n_particles - 1) // 2]` if remove_duplicates.
        Otherwise `[n_batch, n_particles , n_particles]`
    """
    x = x.reshape(-1, n_particles, n_dimensions)
    distances = torch.cdist(x, x)
    if remove_duplicates:
        distances = distances[:, torch.triu(torch.ones((n_particles, n_particles)), diagonal=1) == 1]
        distances = distances.reshape(-1, n_particles * (n_particles - 1) // 2)
    return distances

class Energy(torch.nn.Module):
    """
    Base class for all energy models.

    It supports energies defined over:
        - simple vector states of shape [..., D]
        - tensor states of shape [..., D1, D2, ..., Dn]
        - states composed of multiple tensors (x1, x2, x3, ...)
          where each xi is of form [..., D1, D2, ...., Dn]

    Each input can have multiple batch dimensions,
    so a final state could have shape
        ([B1, B2, ..., Bn, D1, D2, ..., Dn],
         ...,
         [B1, B2, ..., Bn, D'1, ..., D'1n]).

    which would return an energy tensor with shape
        ([B1, B2, ..., Bn, 1]).

    Forces are computed for each input by default.
    Here the convention is followed, that forces will have
    the same shape as the input state.

    To define the state shape, the parameter `dim` has to
    be of the following form:
        - an integer, e.g. d = 5
            then each event is a simple vector state
            of shape [..., 5]
        - a non-empty list of integers, e.g. d = [3, 6, 7]
            then each event is a tensor state of shape [..., 3, 6, 7]
        - a list of len > 1 containing non-empty integer lists,
            e.g. d = [[1, 3], [5, 3, 6]]. Then each event is
            a tuple of tensors of shape ([..., 1, 3], [..., 5, 3, 6])

    Parameters:
    -----------
    dim: Union[int, Sequence[int], Sequence[Sequence[int]]]
        The event shape of the states for which energies/forces ar computed.

    """

    def __init__(self, dim: Union[int, Sequence[int], Sequence[Sequence[int]]], **kwargs):

        super().__init__(**kwargs)
        self._event_shapes = _parse_dim(dim)

    @property
    def dim(self):
        if len(self._event_shapes) > 1:
            raise ValueError(
                "This energy instance is defined for multiple events."
                "Therefore there exists no coherent way to define the dimension of an event."
                "Consider using Energy.event_shapes instead."
            )
        elif len(self._event_shapes[0]) > 1:
            warnings.warn(
                "This Energy instance is defined on multidimensional events. "
                "Therefore, its Energy.dim is distributed over multiple tensor dimensions. "
                "Consider using Energy.event_shape instead.",
                UserWarning,
            )
        return int(torch.prod(torch.tensor(self.event_shape, dtype=int)))

    @property
    def event_shape(self):
        if len(self._event_shapes) > 1:
            raise ValueError(
                "This energy instance is defined for multiple events."
                "Therefore therefore there exists no single event shape."
                "Consider using Energy.event_shapes instead."
            )
        return self._event_shapes[0]

    @property
    def event_shapes(self):
        return self._event_shapes

    def _energy(self, *xs, **kwargs):
        raise NotImplementedError()

    def energy(self, *xs, temperature=1.0, **kwargs):
        assert len(xs) == len(
            self._event_shapes
        ), f"Expected {len(self._event_shapes)} arguments but only received {len(xs)}"
        batch_shape = xs[0].shape[: -len(self._event_shapes[0])]
        for i, (x, s) in enumerate(zip(xs, self._event_shapes)):
            assert x.shape[: -len(s)] == batch_shape, (
                f"Inconsistent batch shapes."
                f"Input at index {i} has batch shape {x.shape[:-len(s)]}"
                f"however input at index 0 has batch shape {batch_shape}."
            )
            assert (
                x.shape[-len(s) :] == s
            ), f"Input at index {i} as wrong shape {x.shape[-len(s):]} instead of {s}"
        return self._energy(*xs, **kwargs) / temperature

    def force(
        self,
        *xs: Sequence[torch.Tensor],
        temperature: float = 1.0,
        ignore_indices: Optional[Sequence[int]] = None,
        no_grad: Union[bool, Sequence[int]] = False,
        **kwargs,
    ):
        """
        Computes forces with respect to the input tensors.

        If states are tuples of tensors, it returns a tuple of forces for each input tensor.
        If states are simple tensors / vectors it returns a single forces.

        Depending on the context it might be unnecessary to compute all input forces.
        For this case `ignore_indices` denotes those input tensors for which no forces.
        are to be computed.

        E.g. by setting `ignore_indices = [1]` the result of `energy.force(x, y, z)`
        will be `(fx, None, fz)`.

        Furthermore, the forces will allow for taking high-order gradients by default.
        If this is unwanted, e.g. to save memory it can be turned off by setting `no_grad=True`.
        If higher-order gradients should be ignored for only a subset of inputs it can
        be specified by passing a list of ignore indices to `no_grad`.

        E.g. by setting `no_grad = [1]` the result of `energy.force(x, y, z)`
        will be `(fx, fy, fz)`, where `fx` and `fz` allow for taking higher order gradients
        and `fy` will not.

        Parameters:
        -----------
        xs: *torch.Tensor
            Input tensor(s)
        temperature: float
            Temperature at which to compute forces
        ignore_indices: Sequence[int]
            Which inputs should be skipped in the force computation
        no_grad: Union[bool, Sequence[int]]
            Either specifies whether higher-order gradients should be computed at all,
            or specifies which inputs to leave out when computing higher-order gradients.
        """
        if ignore_indices is None:
            ignore_indices = []

        with torch.enable_grad():
            forces = []
            requires_grad_states = [x.requires_grad for x in xs]

            for i, x in enumerate(xs):
                if i not in ignore_indices:
                    x = x.requires_grad_(True)
                else:
                    x = x.requires_grad_(False)

            energy = self.energy(*xs, temperature=temperature, **kwargs)

            for i, x in enumerate(xs):
                if i not in ignore_indices:
                    if isinstance(no_grad, bool):
                        with_grad = not no_grad
                    else:
                        with_grad = i not in no_grad
                    force = -torch.autograd.grad(
                        energy.sum(), x, create_graph=with_grad,
                    )[0]
                    forces.append(force)
                    x.requires_grad_(requires_grad_states[i])
                else:
                    forces.append(None)

        forces = (*forces,)
        if len(self._event_shapes) == 1:
            forces = forces[0]
        return forces


class DoubleWellEnergy(Energy):
    def __init__(self, dim, a=0, b=-4.0, c=1.0):
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x):
        d = x[..., [0]]
        v = x[..., 1:]
        e1 = self._a * d + self._b * d.pow(2) + self._c * d.pow(4)
        e2 = 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return e1 + e2


class MultiDimensionalDoubleWell(Energy):
    def __init__(self, dim, a=0.0, b=-4.0, c=1.0, transformer=None):
        super().__init__(dim)
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c)
        self.register_buffer("_a", a)
        self.register_buffer("_b", b)
        self.register_buffer("_c", c)
        if transformer is not None:
            self.register_buffer("_transformer", transformer)
        else:
            self._transformer = None

    def _energy(self, x):
        if self._transformer is not None:
            x = torch.matmul(x, self._transformer)
        e1 = self._a * x + self._b * x.pow(2) + self._c * x.pow(4)
        return e1.sum(dim=1, keepdim=True)

class MultiDoubleWellPotential(Energy):
    """Energy for a many particle system with pair wise double-well interactions.
    The energy of the double-well is given via

    .. math::
        E_{DW}(d) = a \cdot (d-d_{\text{offset})^4 + b \cdot (d-d_{\text{offset})^2 + c.

    Parameters
    ----------
    dim : int
        Number of degrees of freedom ( = space dimension x n_particles)
    n_particles : int
        Number of particles
    a, b, c, offset : float
        parameters of the potential
    """

    def __init__(self, dim, n_particles, a, b, c, offset, two_event_dims=True):
        if two_event_dims:
            super().__init__([n_particles, dim // n_particles])
        else:
            super().__init__(dim)
        self._dim = dim
        self._n_particles = n_particles
        self._n_dimensions = dim // n_particles
        self._a = a
        self._b = b
        self._c = c
        self._offset = offset

    def _energy(self, x):
        x = x.contiguous()
        dists = compute_distances(x, self._n_particles, self._n_dimensions)
        dists = dists - self._offset

        energies = self._a * dists ** 4 + self._b * dists ** 2 + self._c
        return energies.sum(-1, keepdim=True)


class GaussianMCMCSampler(IterativeSampler):
    """This is a shortcut implementation of a simple Gaussian MCMC sampler
    that is largely backward-compatible with the old implementation.
    The only difference is that `GaussianMCMCSampler.sample(n)`
    will propagate for `n` strides rather than `n` steps.

    Parameters
    ----------
    energy : bgflow.Energy
        The target energy.
    init_state : Union[torch.Tensor, SamplerState]
    temperature : Union[torch.Tensor, float], optional
        The temperature scaling factor that is broadcasted along the batch dimension.
    noise_std : float, optional
        The Gaussian noise standard deviation.
    stride : int, optional
    n_burnin : int, optional
    box_constraint : Callable, optional
        The function is supplied as a `set_samples_hook` to the SamplerState so that
        boundary conditions are applied to all samples.
    return_hook : Callable, optional
        The function is supplied as a `return_hook` to the Sampler. By default, we combine
        the batch and sample dimensions to keep consistent with the old implementation.
    """
    def __init__(
            self,
            energy,
            init_state,
            temperature=1.,
            noise_std=.1,
            stride=1,
            n_burnin=0,
            box_constraint=None,
            return_hook=None,
            **kwargs
    ):
        # first, some things to ensure backwards compatibility
        # apply the box constraint function whenever samples are set
        set_samples_hook = default_set_samples_hook
        if box_constraint is not None:
            set_samples_hook = lambda samples: [box_constraint(x) for x in samples]
        if not isinstance(init_state, SamplerState):
            init_state = SamplerState(samples=init_state, set_samples_hook=set_samples_hook)
        # flatten batches before returning
        if return_hook is None:
            return_hook = lambda samples: [
                x.reshape(-1, *shape) for x, shape in zip(samples, energy.event_shapes)
            ]
        if "n_stride" in kwargs:
            warnings.warn("keyword n_stride is deprecated, use stride instead", DeprecationWarning)
            stride = kwargs["n_stride"]
        # set up sampler
        super().__init__(
            init_state,
            sampler_steps=[
                MCMCStep(
                    energy,
                    proposal=GaussianProposal(noise_std=noise_std),
                    target_temperatures=temperature,
                ),
            ],
            stride=stride,
            n_burnin=n_burnin,
            return_hook=return_hook
        )


def metropolis_accept(
        current_energies,
        proposed_energies,
        proposal_delta_log_prob
):
    """Metropolis criterion.

    Parameters
    ----------
    current_energies : torch.Tensor
        Dimensionless energy of the current state x.
    proposed_energies : torch.Tensor
        Dimensionless energy of the proposed state x'.
    proposal_delta_log_prob : Union[torch.Tensor, float]
        The difference in log probabilities between the forward and backward proposal.
        This corresponds to    log g(x'|x) - log g(x|x'), where g is the proposal distribution.

    Returns
    -------
    accept : torch.Tensor
        A boolean tensor that is True for accepted proposals and False otherwise
    """
    # log p(x') - log p(x) - (log g(x'|x) - log g(x|x'))
    log_prob = -(proposed_energies - current_energies) - proposal_delta_log_prob
    log_acceptance_ratio = torch.min(
        torch.zeros_like(proposed_energies),
        log_prob,
    )
    log_random = torch.rand_like(log_acceptance_ratio).log()
    accept = log_acceptance_ratio >= log_random
    return accept


class _GaussianMCMCSampler(Energy, Sampler):
    """Deprecated legacy implementation."""
    def __init__(
            self,
            energy,
            init_state=None,
            temperature=1.,
            noise_std=.1,
            n_stride=1,
            n_burnin=0,
            box_constraint=None
    ):
        super().__init__(energy.dim)
        warnings.warn(
            """This implementation of the MC sampler is deprecated. 
Instead try using:
>>> IterativeSampler(
>>>     init_state, [MCMCStep(energy)]
>>> ) 
""",
            DeprecationWarning
        )
        self._energy_function = energy
        self._init_state = init_state
        self._temperature = temperature
        self._noise_std = noise_std
        self._n_stride = n_stride
        self._n_burnin = n_burnin
        self._box_constraint = box_constraint

        self._reset(init_state)

    def _step(self):
        noise = self._noise_std * torch.Tensor(self._x_curr.shape).normal_()
        x_prop = self._x_curr + noise
        e_prop = self._energy_function.energy(x_prop, temperature=self._temperature)
        e_diff = e_prop - self._e_curr
        r = -torch.Tensor(x_prop.shape[0]).uniform_(0, 1).log().view(-1, 1)
        acc = (r > e_diff).float().view(-1, 1)
        rej = 1. - acc
        self._x_curr = rej * self._x_curr + acc * x_prop
        self._e_curr = rej * self._e_curr + acc * e_prop
        if self._box_constraint is not None:
            self._x_curr = self._box_constraint(self._x_curr)
        self._xs.append(self._x_curr)
        self._es.append(self._e_curr)
        self._acc.append(acc.bool())

    def _reset(self, init_state):
        self._x_curr = self._init_state
        self._e_curr = self._energy_function.energy(self._x_curr, temperature=self._temperature)
        self._xs = [self._x_curr]
        self._es = [self._e_curr]
        self._acc = [torch.zeros(init_state.shape[0]).bool()]
        self._run(self._n_burnin)

    def _run(self, n_steps):
        with torch.no_grad():
            for i in range(n_steps):
                self._step()

    def _sample(self, n_samples):
        self._run(n_samples)
        return torch.cat(self._xs[-n_samples::self._n_stride], dim=0)

    def _sample_accepted(self, n_samples):
        samples = self._sample(n_samples)
        acc = torch.cat(self._acc[-n_samples::self._n_stride], dim=0)
        return samples[acc]

    def _energy(self, x):
        return self._energy_function.energy(x)

