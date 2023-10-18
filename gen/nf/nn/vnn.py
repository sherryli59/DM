import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from typing import Union
from numpy import pi
import torch.nn.functional as F


EPS = 1e-6


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

def rbf_kernels(d: torch.Tensor, mu: Union[torch.Tensor, float], neg_log_gamma: Union[torch.Tensor, float],) -> torch.Tensor:
    inv_gamma = torch.exp(neg_log_gamma)
    rbfs = torch.exp(-(d - mu).pow(2) * inv_gamma.pow(2))
    srbfs = rbfs.sum(dim=-1, keepdim=True)
    kernels = rbfs / (1e-6 + srbfs)
    return kernels

class VNN(nn.Module):
    def __init__(self, n_knn=10,pooling="mean", feature_transform=True, t_hid_dim=128):
        super(VNN, self).__init__()
        output_channels = [64//3, 64//3, 128//3]
        self.t_embedder = nn.ModuleList([nn.Sequential(GaussianFourierProjection(embed_dim=t_hid_dim),
            nn.Linear(t_hid_dim, out)) for out in output_channels])
        self.n_knn = n_knn
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        conv2_in_channels = (64//3)*(1+(feature_transform==True))
        self.conv2 = VNLinearLeakyReLU(conv2_in_channels, 128//3, dim=4, negative_slope=0.0)
        
        self.conv3 = VNLinear(128//3, 1024//3)
        self.bn3 = VNBatchNorm(1024//3, dim=4)
        self.out = VNLinear(1024//3, 1)
        
        if pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif pooling == 'mean':
            self.pool = mean_pool
        
        self.feature_transform = feature_transform
        
        if self.feature_transform:
            self.fstn = STNkd(pooling=pooling, d=64//3)

    def forward(self, x,t=None):
        if t is not None:
            t_embed = [t_embedder(t) for t_embedder in self.t_embedder]
        else:
            t_embed = [None]*4
        x = x.transpose(2, 1)
        B, D, N = x.size()
        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn)
        x = self.conv_pos(feat)
        x = self.pool(x)
        x = self.conv1(x,t_embed[0])
        if self.feature_transform:
            x_global = self.fstn(x,t_embed[1]).unsqueeze(-1).repeat(1,1,1,N)
            x = torch.cat((x, x_global), 1)

        x = self.conv2(x, t_embed[2])
        x = self.bn3(self.conv3(x))
        x = self.out(x).squeeze()
        return x.transpose(1, 2).contiguous()
    


class STNkd(nn.Module):
    def __init__(self, pooling="max", d=64):
        super(STNkd, self).__init__()
        
        self.conv1 = VNLinearLeakyReLU(d, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128//3, 1024//3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU(1024//3, 512//3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512//3, 256//3, dim=3, negative_slope=0.0)
        
        if pooling == 'max':
            self.pool = VNMaxPool(1024//3)
        elif pooling == 'mean':
            self.pool = mean_pool
        
        self.fc3 = VNLinear(256//3, d)
        self.d = d

    def forward(self, x, t_embed=None):
        batchsize = x.size()[0]
        x = self.conv1(x, t_embed)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x_out = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        return x_out


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)
        
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, out_channels, bias=False)
      

    def forward(self, x, t_embed=None):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        if t_embed is not None:
            t_embed = t_embed.unsqueeze(1).unsqueeze(1)
            p = (t_embed * self.map_to_feat(x.transpose(1,-1))).transpose(1,-1)
        else:
            p = self.map_to_feat(x.transpose(1,-1)).transpose(1,-1)
        # BatchNorm
        p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (p*d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out



class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x


class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        return x_std, z0
    

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)
    idx_base = torch.arange(0, batch_size).to(x.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)
    
    feature = torch.cat((feature-x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
  
    return feature


if __name__=="__main__":
    model = VNN().to("cuda")
    x = torch.randn(2,10,3).to("cuda")
    t = torch.rand(2).to("cuda")
    y = model(x,t)
    print(y)
    x_flipped = torch.flip(x,[1])
    print(model(x_flipped,t)-torch.flip(y,[1]))
    #rotate 90 degrees
    rotation_mat = torch.tensor([[0,1,0],[-1,0,0],[0,0,1]]).float().unsqueeze(0).expand(x.shape[0],-1,-1)
    rotation_mat = rotation_mat.to("cuda")
    x_rotated = torch.einsum("bij,bjk->bik",x,rotation_mat)
    print(model(x_rotated,t)-torch.einsum("bij,bjk->bik",y,rotation_mat))
