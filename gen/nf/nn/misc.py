import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # FIXME

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0, residual=True):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) 
        self.w_2 = nn.Linear(d_hid, d_in) 
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self, x):

        residual = x

        x = self.w_2(F.gelu(self.w_1(x),approximate="tanh"))
        x = self.dropout(x)

        if self.residual:
            x += residual

        return x

class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
      super().__init__()
      self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
      return self.dense(x)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


  


    
def pair_dist(pos):
    pair_vec = (pos.unsqueeze(-2) - pos.unsqueeze(-3))
    pair_dist = torch.linalg.norm(pair_vec.float(), axis=-1)
    n = pair_vec.shape[1]
    xindex,yindex = torch.triu_indices(n,n,1).unbind()
    pair_dist = pair_dist[:,xindex,yindex]
    pair_dist = pair_dist.unsqueeze(1)
    if n % 2 ==0:
        pair_dist = pair_dist.reshape(-1,n-1,n//2)
    else:
        pair_dist = pair_dist.reshape(-1,n,(n-1)//2)
    return pair_dist

class TimeEmbeddedMLP_2(nn.Module):
  def __init__(self, d_in, d_hid, marginal_prob_std=None, embed_dim=256,d_out=None, residual=False, dropout=0, num_blocks=2):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    if d_out is None:
        d_out = d_in
    self.first_layer =MLP(d_in,d_hid,d_hid, dropout)
    self.time_layer = nn.ModuleList([Dense(embed_dim,d_hid) for _ in range(num_blocks-1)])
    self.mlp = nn.ModuleList([MLP(d_hid,d_hid,d_hid, residual=residual) for _ in range(num_blocks-1)])
    self.final_layer = MLP(d_hid,d_hid,d_out,residual=residual)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std

  def update_marginal_prob_std(self, marginal_prob_std:callable):
    self.marginal_prob_std = marginal_prob_std

  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t)) 
    shape = x.shape
    x = x.reshape(shape[0],-1)
    output = self.first_layer(x)
    for t_layer, layer in zip(self.time_layer, self.mlp):
        output += t_layer(embed)
        output = layer(output)

    output = self.final_layer(output)
    if self.marginal_prob_std is not None:
        output = output / self.marginal_prob_std(t)[:, None]
    output = output.reshape(shape) 
    return output



    


class score_Gaussian(nn.Module):
    def __init__(self, mean, std, noise_schedule):
        super().__init__()     
        self.true_mean = torch.tensor(mean)
        self.mean = nn.Parameter(torch.tensor(mean))
        self.std = torch.tensor(std).squeeze()
        self.noise_schedule = noise_schedule

    def _batch_mult(self,coeff,data):
            return torch.einsum(data, [0,...],coeff,[0], [0,...])
    
    def forward(self,x,t):
        print(self.mean)
        alpha = self.noise_schedule.alpha_cumprod(t)
        score = -(x-self.mean[None].to(x.device))
        score = self._batch_mult(1/(self.std**2+1-alpha),score)
        return score
    

if __name__=="__main__":
    model = TimeEmbeddedMLP(5,500)
    x = torch.ones(10,5,2)
    t = torch.ones(10)
    print(model(x,t).shape)