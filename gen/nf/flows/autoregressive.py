from nf.flows.made import MADE
from nf.flows.mlp_conditioner import MLPList
from nf.flows.spline import (
    unconstrained_rational_quadratic_spline,
    rational_quadratic_spline
)
import dm.nf.flows.spline as rational_quadratic
import dm.nf.flows.rqs_utils as utils
import numpy as np
import torch
import torch.nn as nn

class AutoregressiveTransform(nn.Module):

    def __init__(self, conditioner):
        super(AutoregressiveTransform, self).__init__()
        self.conditioner = conditioner

    def forward(self, inputs, context=None, order=None):
        if order is not None:
            self.conditioner.update_order(order=order)        
        autoregressive_params = self.conditioner(inputs, context)
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def reverse(self, inputs, context=None, partial_structure=None, partial_dim=[], order=None):
        num_inputs = int(np.prod(inputs.shape[1:]))
        if partial_structure is not None:
            outputs = partial_structure
            num_inputs -= len(partial_dim)
            if order is not None:
                self.conditioner.update_order(order=order)
        else:
            outputs = torch.zeros_like(inputs).to(inputs.device).requires_grad_(True)
        logabsdet = None
        conditioner_list=[]
        output_list=[]
        for i in range(num_inputs):
            autoregressive_params = self.conditioner(outputs, context)
            output_list.append(outputs[0])
            conditioner_list.append(autoregressive_params[0])
            outputs, logabsdet = self._elementwise_inverse(
                inputs, autoregressive_params
            )
        return outputs, logabsdet

class RQSAutoregressiveLayer(AutoregressiveTransform):
    def __init__(
        self,
        features,
        dependency = None,
        num_bins=10,
        tails=None,
        hidden_features=1000,
        num_blocks = 3,
        tail_bound=1.0,
        context_dim = 0,
        min_bin_width=rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=rational_quadratic.DEFAULT_MIN_DERIVATIVE,
        **made_params
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound
        self.context_dim = context_dim
        if dependency is None:
            conditioner = MADE(features=features,hidden_features=hidden_features,
                           output_multiplier = self._output_dim_multiplier(), **made_params)
        else:
            conditioner = MLPList(input_dim=features,dependency=dependency, context_dim=context_dim,
                           output_dim_multiplier = self._output_dim_multiplier(), hidden_dim=hidden_features,
                           num_layers=num_blocks)

        super().__init__(conditioner)

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.conditioner, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.conditioner.hidden_features)
            unnormalized_heights /= np.sqrt(self.conditioner.hidden_features)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, utils.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)

class RQSAutoregressive(nn.Module):
    def __init__(self, n_layers=2, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([RQSAutoregressiveLayer(**kwargs) for _ in range(n_layers)])

    def forward(self, x ,context):
        """(target->base)
        x is target
        z is base
        """
        x = x.reshape(x.shape[0],-1)
        log_det = 0
        for layer in self.layers:
            x, ld = layer.forward(x, context)
            log_det += ld
        z = x
        return z, log_det

    def reverse(self, z, context):
        """ (base->target)
        x is target
        z is base
        """
        z = z.reshape(z.shape[0],-1)
        log_det = 0
        for layer in self.layers[::-1]:
            z, ld = layer.reverse(z, context)
            log_det += ld
        x = z
        return x, log_det

