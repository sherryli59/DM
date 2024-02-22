import math
import torch

class GCL(torch.nn.Module):
    def __init__(self, input_nf, output_nf, hidden_dim, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=torch.nn.SiLU(), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_edge + edges_in_d, hidden_dim),
            act_fn,
            torch.nn.Linear(hidden_dim, hidden_dim),
            act_fn)

        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + input_nf + nodes_att_dim, hidden_dim),
            act_fn,
            torch.nn.Linear(hidden_dim, output_nf))

        if self.attention:
            self.att_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, 1),
                torch.nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def node_model(self, x, edge_index, edge_attr, node_attr, update_features_mask=None):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        messages = self.node_mlp(agg)
        if update_features_mask is not None:
            messages *= update_features_mask
        out = x + messages
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None, update_features_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, update_features_mask=update_features_mask)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class EquivariantUpdate(torch.nn.Module):
    def __init__(self, hidden_dim, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=torch.nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_dim * 2 + edges_in_d
        layer = torch.nn.Linear(hidden_dim, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=1)
        linear1 = torch.nn.Linear(input_edge, hidden_dim)
        torch.nn.init.xavier_uniform_(linear1.weight, gain=1)
        linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(linear2.weight, gain=1)
        self.coord_mlp = torch.nn.Sequential(
            linear1,
            act_fn,
            linear2,
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask, update_coords_mask=None):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)

        if update_coords_mask is not None:
            agg = update_coords_mask * agg
        coord = coord + agg
        return coord

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None,
                node_mask=None, edge_mask=None, update_coords_mask=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask,
                                 update_coords_mask=update_coords_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(torch.nn.Module):
    def __init__(self, hidden_dim, edge_feat_nf=2, device='cuda', act_fn=torch.nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum'):
        super(EquivariantBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_dim, self.hidden_dim, self.hidden_dim, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_dim, edges_in_d=edge_feat_nf, act_fn=torch.nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None, update_coords_mask=None, update_features_mask=None):
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr,
                                               node_mask=node_mask, edge_mask=edge_mask,
                                               update_features_mask=update_features_mask)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr,
                                    node_mask, edge_mask, update_coords_mask=update_coords_mask)
        
        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class EGNN(torch.nn.Module):
    def __init__(self, node_dim=1, hidden_dim=128, cutoff=None, atomic_numbers=None,device='cuda', act_fn=torch.nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum',**kwargs):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = node_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.cutoff = cutoff
        if atomic_numbers is not None:
            self.atomic_numbers = atomic_numbers.to(device)
        else:
            self.atomic_numbers = None
        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = torch.nn.Linear(node_dim, self.hidden_dim)
        self.embedding_out = torch.nn.Linear(self.hidden_dim, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_dim, edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method))
        self.to(self.device)

    def forward(self, x, t, context=None, edge_index=None, node_mask=None, edge_mask=None, update_coords_mask=None, update_features_mask=None):
        n_configs = x.shape[0]
        if edge_index is None:
            batch = torch.arange(x.shape[0], device=x.device).reshape(-1,1).tile((1,x.shape[1])).flatten()
        if self.atomic_numbers is not None:
            h = self.atomic_numbers[None,:,None].expand(n_configs,x.shape[1],1)
            if t.dim() == 0:
                t = t*torch.ones(n_configs).to(x.device)
            t = t[:,None,None].expand_as(h)
            h = torch.cat([h,t],dim=2)
        else:
            h = t[:,None,None].expand(n_configs,x.shape[1],1)
        if context is not None:
            context = context[:,None,:].expand(n_configs,x.shape[1],context.shape[1])
            h = torch.cat([h,context],dim=2)
        x = x.flatten(0,1)
        h = h.flatten(0,1)
        edge_index = get_edges(batch,x,edge_cutoff=self.cutoff)
        distances, _ = coord2diff(x, edge_index)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x = self._modules["e_block_%d" % i](
                h, x, edge_index, node_mask=node_mask, edge_mask=edge_mask,
                edge_attr=distances, update_coords_mask=update_coords_mask,
                update_features_mask=update_features_mask)

        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        x = x.reshape(n_configs,-1,x.shape[-1])
        h = h.reshape(n_configs,-1,h.shape[-1])
        return x


class SinusoidsEmbeddingNew(torch.nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.detach()

def pair_dist(x, y):
    pair_vec = (x.unsqueeze(-2) - y.unsqueeze(-3))
    return torch.linalg.norm(pair_vec.float(), axis=-1)

def get_edges(batch, x,edge_cutoff=None):
    adj = batch[:, None] == batch[None, :]
    if edge_cutoff is not None:
        adj = adj & (pair_dist(x, x) <= edge_cutoff)
    edges = torch.stack(torch.where(adj), dim=0)
    return edges

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

if __name__=="__main__":
    context_dim = 2
    model = EGNN(context_dim+1,256, n_layers=1)
    x = torch.randn(2,2,3).to("cuda")
    x.requires_grad=True
    t = torch.rand(2).to("cuda")
    context = torch.randn(2,context_dim).to("cuda")
    y = model(x,t,context=context)
    x_flipped = torch.flip(x,[1])
    print(model(x_flipped,t,context)-torch.flip(y,[1]))
    #rotate 90 degrees
    rotation_mat = torch.tensor([[0,1,0],[-1,0,0],[0,0,1]]).float().unsqueeze(0).expand(x.shape[0],-1,-1)
    rotation_mat = rotation_mat.to("cuda")
    x_rotated = torch.einsum("bij,bjk->bik",x,rotation_mat)
    print(model(x_rotated,t,context)-torch.einsum("bij,bjk->bik",y,rotation_mat))