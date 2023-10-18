
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def maskgenerator(dependency, hidden_dim=100, n_hid_layers = 1, n_edges=10, output_multiplier=1):
    '''
    input:
        dependency: lower triangular matrix of shape (n_units,n_units) with [i,j]=1 if unit i depends on unit j
    return:
    (in_mask, hid_mask, out_mask)
    if hid_mask[l,i,j] = 0 then the weight matrix W_l[i,j] = 0
    '''
    n_units = len(dependency)
    node_occ = torch.zeros(n_hid_layers,hidden_dim,n_units)  # node_occ[i,j,k] = 1 if node j in layer i is ultimately connected to unit k
    mask = torch.zeros(n_hid_layers+1,hidden_dim,hidden_dim)
    for u in range(1,n_units):
        dep_on = torch.nonzero(dependency[u,:u]).squeeze()
        if dep_on.dim()==0:
            dep_on = torch.tensor([dep_on])
        other = zero(dependency[u]).squeeze()
        for d in dep_on:
            prev_selected = torch.tensor([u])
            for l in range(n_hid_layers):
                if other.dim()==0:
                    other = torch.tensor([other])
                avail = zero(torch.sum(node_occ[l,:,other],axis=-1)).squeeze()
                #priority = torch.nonzero(node_occ[l,:,d]).squeeze()
                #priority = intersection(priority,avail)
                #avail = torch.cat((priority,difference(avail,priority)))
                avail = avail[torch.randperm(len(avail))]
                new_edges = 0
                full_cycle = False 
                counter =0 
                prev = prev_selected[0]
                selected_list = []
                while True:
                    selected = zero(mask[l,avail,prev]).squeeze()
                    if len(selected)==0:
                        print("warning: num of new edges per new dependence not satisfied")
                        break
                    else:
                        selected = selected[0]
                    selected_list.append(avail[selected])
                    mask[l,avail[selected],prev] = 1
                    node_occ[l,avail[selected],d] = 1
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
    mask = mask.transpose(1,2)
    mask = torch.flip(mask,[0])
    in_mask = mask[0][:,:n_units]
    if n_hid_layers>1:
        hid_mask = mask[1:-1]
    else:
        hid_mask = None
    out_mask = mask[-1][:n_units]
    if output_multiplier>1:
        out_mask = out_mask.repeat_interleave(int(output_multiplier),dim=0)
    return (in_mask, hid_mask, out_mask)

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
    output_multiplier = 3
    mask = maskgenerator(dep,hidden_dim=hidden_dim,n_edges=5,n_hid_layers = 2, output_multiplier=output_multiplier)
    print(mask)
    linear = MaskedLinear(mask[0],input_dim, hidden_dim)
    linear2 =  MaskedFeedforwardBlock(mask[1][0],hidden_dim)
    linear3 = MaskedLinear(mask[2],hidden_dim,input_dim*output_multiplier)
    x = torch.randn(input_dim)
    x.requires_grad = True
    y = linear3(linear2(linear(x)))
    print(y)
    print(torch.autograd.grad(y[3],x))

                        
                    

                

