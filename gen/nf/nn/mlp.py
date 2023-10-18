import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import pi

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class TimeEmbeddedMLPBlock(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, d_out, encoder=None, dropout=0):
        super().__init__()
        if encoder is None:
            self.encoder = nn.Linear(d_in, d_hid) 
        else:
            self.encoder = encoder
        self.linear_1 = nn.Linear(d_hid, d_hid)
        self.linear_2 = nn.Linear(2*d_hid, d_out) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x,t):
        x = x.reshape(x.shape[0],-1)
        x = torch.cat((self.encoder(x),self.linear_1(t)),dim=-1)
        x = self.linear_2(F.gelu(x,approximate="tanh"))
        x = self.dropout(x)
        return x

class MLP(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, d_out, dropout=0, residual=False):
        super().__init__()
        self.linear_1 = nn.Linear(d_in, d_hid) 
        self.linear_2 = nn.Linear(d_hid, d_out) 
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x):

        residual = x
        x = self.linear_1(x)
        x = self.linear_2(F.gelu(x,approximate="tanh"))
        x = self.dropout(x)

        if self.residual:
            x += residual

        return x

class TimeEmbeddedMLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out=None, t_embed_dim=256, dropout=0, num_blocks=2, residual=False):
        super().__init__()
        self.t_embedder = nn.Sequential(GaussianFourierProjection(embed_dim=t_embed_dim),
            nn.Linear(t_embed_dim, d_hid))
        if d_out is None:
            d_out = d_in
        self.first_layer = TimeEmbeddedMLPBlock(d_in,d_hid,d_hid, dropout=dropout)
        self.mlp = nn.ModuleList([MLP(d_hid,d_hid,d_hid, residual=residual) for _ in range(num_blocks-1)])
        self.final_layer = MLP(d_hid,d_hid,d_out,residual=False)
        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self,x,t):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        t_embed = self.act(self.t_embedder(t)) 
        shape = x.shape
        x = self.first_layer(x,t_embed)
        for layer in self.mlp:
            x = layer(x)
        x = self.final_layer(x)
        x = x.reshape(shape)
        return x