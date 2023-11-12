import torch

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
        weights = sde_output["sigma"]**2
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
            loss = sliced_score_matching(score_fn,diffused_x,t, n_eps=3)
    return loss

def single_sde_loss(sde,score_fn,x,t):
    sde_output = sde(x,t)
    loss = loss_from_sde_output(sde_output,score_fn,t)
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


