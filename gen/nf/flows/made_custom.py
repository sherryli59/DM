import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedLinear(nn.Linear):
    """A linear module with a masked weight matrix."""

    def __init__(
        self,
        mask,
        in_features,
        out_features,
        bias=True,
    ):
        super().__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        print(out_features)
        self.mask = mask

    def forward(self, x):
        return F.linear(x, 1000*self.weight * self.mask, self.bias)


class MaskedFeedforwardBlock(nn.Module):
    """A feedforward block based on a masked linear module.
    NOTE: In this implementation, the number of output features is taken to be equal to
    the number of input features.
    """

    def __init__(
        self,
        mask,
        features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()

        # Batch norm.
        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(features, eps=1e-3)
        else:
            self.batch_norm = None

        # Masked linear.
        self.linear = MaskedLinear(
            mask=mask,
            in_features=features,
            out_features=features,
        )

        # Activation and dropout.
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, inputs):
        if self.batch_norm:
            temps = self.batch_norm(inputs)
        else:
            temps = inputs
        temps = self.linear(temps)
        temps = self.activation(temps)
        outputs = self.dropout(temps)
        return outputs


class MADE(nn.Module):
    """Implementation of MADE.

    """

    def __init__(
        self,
        dependency,
        num_blocks=1,
        features=1,
        hidden_features=100,
        output_multiplier=1,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        device="cpu",
    ):
        super().__init__()
        self.features = features
        self.num_blocks = num_blocks
        self.device = device
        self.in_mask, self.hid_mask, self.out_mask = mask_generator(dependency,
         hidden_dim=hidden_features, n_hid_layers = num_blocks+1, n_edges=10, output_multiplier=output_multiplier)
        self.activation = activation
        self.initial_layer = MaskedLinear(
            in_features=self.features,
            out_features=hidden_features,
            mask = self.in_mask,
        )
        blocks = []
        block_constructor = MaskedFeedforwardBlock
        for i in range(num_blocks):
            blocks.append(
                block_constructor(
                    features=hidden_features, #both input and output features
                    mask = self.hid_mask[i],
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.final_layer = MaskedLinear(
            in_features=hidden_features,
            out_features=features * output_multiplier,
            mask = self.out_mask,
        )

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        temps = self.activation(temps)
        for block in self.blocks:
            temps = block(temps)
        outputs = self.final_layer(temps)
        return outputs



def intersection(A, B):
    combined = torch.cat((A.view(-1), B.view(-1)))
    unique, counts = combined.unique(return_counts=True)
    return unique[counts > 1]

def difference(A, B):
    combined = torch.cat((A.view(-1), B.view(-1)))
    unique, counts = combined.unique(return_counts=True)
    return unique[counts == 1]
def zero(a):
    return (a == 0).nonzero()

def mask_generator(dependency, hidden_dim=100, n_hid_layers = 2, n_edges=5, output_multiplier=1):
    '''
    input:
        dependency: lower triangular matrix of shape (n_units,n_units) with [i,j]=1 if unit i depends on unit j
    return:
    (in_mask, hid_mask, out_mask)
    if hid_mask[l,i,j] = 0 then the weight matrix W_l[i,j] = 0
    '''
    n_units = len(dependency)
    point_to = torch.zeros(n_hid_layers,hidden_dim,n_units)  # point_to[i,j,k] = 1 if node j in layer i is ultimately connected to unit k
    point_from = torch.zeros(n_hid_layers,hidden_dim,n_units)  # point_to[i,j,k] = 1 if node j in layer i is originally from unit k
    mask = torch.zeros(n_hid_layers+1,hidden_dim,hidden_dim)
    for u in range(1,n_units):
        if u>8:
            exit()
        dep_on = torch.nonzero(dependency[u,:u]).squeeze()
        if dep_on.dim()==0:
            dep_on = torch.tensor([dep_on])
        other = zero(dependency[u]).squeeze()  
        for d in dep_on:
            print("other",other)
            print("u to d",u,d)
            prev_selected = torch.tensor([u])
            for l in range(n_hid_layers):
                print("layer",l)
                if other.dim()==0:
                    other = torch.tensor([other])
                avail = zero(torch.sum(point_from[l,:,:d+1],axis=-1) +
                torch.sum(point_to[l,:,other],axis=-1)).squeeze()
                priority = torch.nonzero(point_to[l,:,d]).squeeze()
                priority = intersection(priority,avail)
                avail = torch.cat((priority,difference(avail,priority)))
                #avail = avail[torch.randperm(len(avail))]
                new_edges = 0
                full_cycle = False 
                counter =0 
                prev = prev_selected[0]
                selected_list = []
                while True:
                    selected = zero(mask[l,avail,prev]).squeeze()
                    if len(selected.size())==0:
                        print("warning: num of new edges per new dependence not satisfied")
                        break
                    else:
                        if selected.dim()>0:
                            selected = selected[0]
                    selected_list.append(avail[selected])
                    mask[l,avail[selected],prev] = 1
                    point_to[l,avail[selected],d] = 1
                    point_from[l,avail[selected],u] = 1
                    print("connecting to ",avail[selected],d,"from",prev,u)
                    new_edges +=1
                    counter +=1
                    if counter>=len(prev_selected):
                        full_cycle = True
                        counter = 0
                    if new_edges == n_edges and full_cycle:
                        break
                    prev = prev_selected[counter]
                prev_selected = torch.tensor(selected_list)
            
            mask[-1,d,prev_selected] = 1
    print(torch.nonzero(point_from[-1]))
    print(torch.nonzero(point_to[-1]))
    mask = mask.transpose(1,2)
    mask = torch.flip(mask,[0])
    in_mask = mask[0][:,:n_units]
    if n_hid_layers>1:
        hid_mask = mask[1:-1]
    else:
        hid_mask = None
    out_mask = mask[-1][:n_units]
    if output_multiplier>1:
        out_mask = out_mask.tile((int(output_multiplier),1))
    return (in_mask, hid_mask, out_mask)

if __name__=="__main__":
    input_dim = 5
    dep = torch.zeros(input_dim,input_dim)
    dep[1,0] = 1
    dep[2,0] = 1
    dep[2,1] = 1
    dep[3,1] = 1
    dep[3,2] = 1
    dep[4,2] = 1
    hidden_dim = 50
    output_multiplier = 1
    model = MADE(dep,features=input_dim,hidden_features=hidden_dim, output_multiplier=output_multiplier)
    x = torch.randn(input_dim)
    x.requires_grad = True
    y = model(x)
    print(y)
    print(x)
    print(torch.autograd.grad(y[4],x))