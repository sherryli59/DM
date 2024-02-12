import torch

def hutchpp(A, d, m):
    """https://arxiv.org/abs/2010.09649

    A is the LinearOperator whose trace to estimate
    d is the input dimension
    m is the number of queries (larger m yields better estimates)
    """
    S = torch.randn(d, m // 3)
    G = torch.randn(d, m // 3)
    Q, _ = torch.qr(A.matvec(S))
    proj = G - Q @ (Q.T @ G)
    return torch.trace(Q.T @ A.matvec(Q)) + (3./m)*torch.trace(proj.T @ A.matvec(proj))

