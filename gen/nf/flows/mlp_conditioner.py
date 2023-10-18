import torch
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim,
     hidden_dim, num_layers, activation=F.tanh,):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.linear_list = nn.ModuleList()
        input_dim = max(self.input_dim,1)
        self.linear_list.append(nn.Linear(input_dim, self.hidden_dim))
        for i in range(self.num_layers - 1):
            self.linear_list.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.linear_list.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, inputs):
        outputs = inputs
        for i in range(self.num_layers):
            outputs = self.linear_list[i](outputs)
            outputs = self.activation(outputs)
        outputs = self.linear_list[-1](outputs)
        return outputs

class MLPList(nn.Module):
    def __init__(self, input_dim, output_dim_multiplier=1, context_dim=0,
    hidden_dim=100, num_layers=2, dependency=None,activation=F.relu):
        super().__init__()
        total_dim = input_dim+context_dim
        if dependency is None:
            self.dependency = torch.tril(torch.ones(total_dim,total_dim),diagonal=-1)
        else:
            self.dependency = torch.tensor(dependency)
        assert total_dim == self.dependency.shape[0]
        self.dependency_indices = [torch.nonzero(self.dependency[i]).flatten() for i in range(total_dim)]
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.output_dim_multiplier = output_dim_multiplier
        self.mlp_list = nn.ModuleList()
        for i in range(self.input_dim):
            ndep = self.dependency_indices[i+context_dim].shape[0]
            if ndep==0:
                dim = 10
            else:
                dim = ndep
            self.mlp_list.append(MLP(input_dim=dim,output_dim=self.output_dim_multiplier,
                hidden_dim=hidden_dim,num_layers=num_layers,activation=activation))
     
    def forward(self, inputs, context=None):

        if len(inputs.shape)>1:
            inputs = inputs.flatten(start_dim=1)
        assert inputs.shape[-1] == self.input_dim
        if context is not None:
            if len(context.shape)>1:
                context = context.flatten(start_dim=1)
            assert context.shape[-1] == self.context_dim

        outputs = torch.zeros(inputs.shape[0],self.input_dim*self.output_dim_multiplier)
        outputs = outputs.to(inputs.device)
        for i in range(self.input_dim):
            out_mask = torch.arange(self.output_dim_multiplier)*self.input_dim+i
            in_mask = self.dependency_indices[i+self.context_dim]
            ndep = in_mask.shape[0]
            if context is not None:
                inputs = torch.cat([context,inputs],dim=1)
            if ndep==0:
                in_data = 0.1*torch.linspace(-5,5,10).to(inputs.device)
            else:
                in_data = inputs[:,in_mask]

            outputs[:,out_mask] = self.mlp_list[i](in_data)           
        return outputs

