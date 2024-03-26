import torch
from gen.diffusion.diff import gradient, t_finite_diff

def sliced_score_matching(score_fn, samples, t, n_eps=1):
    dup_samples = samples.unsqueeze(0).expand(n_eps,
         *samples.shape).contiguous().view(-1, *samples.shape[1:])
    dup_t = t.tile(n_eps)
    dup_samples = dup_samples.detach().requires_grad_(True)
    rand_v = torch.randn_like(dup_samples)
    with torch.enable_grad():
        score = score_fn(dup_samples,dup_t)
        norm = torch.sum((score * score).view(score.shape[0],-1), dim=-1) 
        scorev = torch.sum(score * rand_v)
        grad = torch.autograd.grad(scorev, dup_samples, create_graph=True)[0]    
        
    trace = torch.sum((rand_v * grad).view(grad.shape[0],-1), dim=-1)
    #print(norm.view(n_eps, -1).std(dim=0))
    norm = norm.view(n_eps, -1).mean(dim=0)
    trace = trace.view(n_eps, -1).mean(dim=0)
    loss = norm/ 2. + trace
    loss = loss/torch.prod(torch.tensor(samples.shape[1:]))
    return loss.mean()


def hutch_trace(f, y, e=None, batch_size=1):
        """e is grad outputs
        """
        e_dzdx = torch.autograd.grad(f, y,
                                     grad_outputs=e, create_graph=True)[0]
        e_dzdx = e_dzdx 
        e_dzdx_e = e_dzdx * e
        approx_tr_dzdx = e_dzdx_e.reshape((batch_size, -1)).sum(-1)
        return approx_tr_dzdx



def loss_from_sde_output(sde_output,score_fn,t):
    diffused_x = sde_output["diffused_x"]
    if "score" in sde_output: # denoising score matching
        with torch.enable_grad():
            diffused_x = diffused_x.detach().requires_grad_(True)
            predicted_score = score_fn(diffused_x,t)
        score = sde_output["score"]
        weights = sde_output["weight"]
        score_diff = predicted_score-score.detach()
        score_diff = score_diff.reshape(len(score_diff),-1)
        loss = torch.mean(weights*torch.mean(score_diff**2,axis=-1))
        #loss = sliced_score_matching(sde.score_fn,diffused_x,t,n_eps=1)
    else:
        if hasattr(score_fn.nn,"trace"):
            print("has trace")
            predicted_score, trace = score_fn(diffused_x,t,return_trace=True)
            norm = torch.sum((predicted_score**2).view(
                predicted_score.shape[0],-1), dim=-1)
            loss = norm/2.+trace
        else: # sliced score matching
            loss = sliced_score_matching(score_fn,diffused_x,t, n_eps=5)
    return loss

def fp_regularizer(sde,score_fn,x,t,alpha=0.15,beta=0.01,m=2):
    batch_size = x.shape[0]
    with torch.enable_grad():
        x = x.requires_grad_(True)
        dlogp_dt = sde.dlogp_dt(x,t,score_fn)
    ds_dt = gradient(dlogp_dt,x)
    emp_ds_dt = t_finite_diff(score_fn,x,t)
    _ , g = sde.sde(x,t)
    weights = g**2
    weights[t<1e-3] = 0
    res = torch.linalg.norm((weights*(ds_dt-emp_ds_dt)).reshape(batch_size,-1),dim=1)
    D = torch.prod(torch.tensor(x.shape[1:]))
    res = res**m/D**m
    loss = alpha*torch.mean(res) + beta*torch.mean(torch.abs(dlogp_dt))
    print("fp loss",torch.mean(res), torch.mean(torch.abs(dlogp_dt)))
    return loss

def single_sde_loss(sde,score_fn,x,t,fp_reg=False):
    sde_output = sde(x,t)
    loss = loss_from_sde_output(sde_output,score_fn,t)
    if fp_reg:
        loss = loss + fp_regularizer(sde,score_fn,x,t)
    return loss

def conditional_loss(sde,score_fn,x,t,lattice):
    #assume the first column is cell index
    cell_idx = x[...,0]
    x = x[...,1:]
    #randomly select two adjacent cells
    x1,x2 = lattice.select_sphere_and_context(x,cell_idx)
    sde_output = sde(x1,t)
    diffused_x = sde_output["diffused_x"]
    with torch.enable_grad():
        diffused_x = diffused_x.detach().requires_grad_(True)
        predicted_score = score_fn(diffused_x,t,context=x2,lattice=lattice)
        #predicted_score = score_fn(diffused_x,t)
    score = sde_output["score"]
    weights = sde_output["weight"]
    score_diff = predicted_score-score.detach()
    score_diff = score_diff.reshape(len(score_diff),-1)
    loss = torch.mean(weights*torch.mean(score_diff**2,axis=-1))
    return loss


def conditional_loss_half(sde,score_fn,x,t):
    #randomly partition the particles into two groups
    nparticles = x.shape[1]
    mask = torch.randperm(nparticles)<nparticles//2
    x1 = x[:,mask]
    x2 = x[:,~mask] #x2 is the context
    sde_output = sde(x1,t)
    diffused_x = sde_output["diffused_x"]
    with torch.enable_grad():
        diffused_x = diffused_x.detach().requires_grad_(True)
        predicted_score = score_fn(diffused_x,t,context=x2,mask=mask)
    score = sde_output["score"]
    weights = sde_output["weight"]
    score_diff = predicted_score-score.detach()
    score_diff = score_diff.reshape(len(score_diff),-1)
    loss = torch.mean(weights*torch.mean(score_diff**2,axis=-1))
    return loss




def piecewise_sde_loss(sde,score_fn,x,t):
    sde_output, sde_idx  = sde(x,t)
    nsamples = len(x)
    loss = 0
    prev_x = x.clone()
    for i in sde_idx.unique():
        mask = (sde_idx==i)
        xi = sde_output[i]["diffused_x"]
        ti = t[mask]
        output = sde_output[i]
        lossi = loss_from_sde_output(output,score_fn,ti)
        #lossi = sliced_score_matching(score_fn,xi, ti, n_eps=5)
        loss = loss + lossi*len(xi)/nsamples
        prev_x[mask] = xi
    return loss

def piecewise_sde_loss_separate(sde,score_fn,x,t):
    sde_output,sde_idx = sde(x,t)
    loss = []
    for i in sde_idx.unique():
        mask = (sde_idx==i)
        output = sde_output[i]
        ti = t[mask]
        lossi = loss_from_sde_output(output,score_fn,ti)
        loss.append(lossi)
    return loss


