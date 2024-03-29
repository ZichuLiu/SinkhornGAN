# --------------------------------------------------------------------------------------------
#    Simplistic implementation of the Sinkhorn divergences, with a vanilla PyTorch backend
# --------------------------------------------------------------------------------------------

import numpy as np
import torch


#######################################################################################################################
# Elementary operations .....................................................................
#######################################################################################################################

def scal(α, f):
    return torch.dot(α.view(-1), f.view(-1))


def lse(v_ij):
    """[lse(v_ij)]_i = log sum_j exp(v_ij), with numerical accuracy."""
    V_i = torch.max(v_ij, 1)[0].view(-1, 1)
    return V_i + (v_ij - V_i).exp().sum(1).log().view(-1, 1)


def dist_matrix(x_i, y_j, p):
    x_y = x_i.unsqueeze(1) - y_j.unsqueeze(0)
    if p == 1:
        return x_y.norm(dim=2)
    elif p == 2:
        return (x_y ** 2).sum(2)
    else:
        return x_y.norm(dim=2) ** (p / 2)


#######################################################################################################################
# Sinkhorn iterations .....................................................................
#######################################################################################################################

def sink(α_i, x_i, β_j, y_j, p=2, eps=.5, nits=200, **kwargs):
    ε = eps  # Python supports Unicode. So fancy!
    # Sinkhorn loop with A = a/eps , B = b/eps ....................................................

    α_i_log, β_j_log = α_i.log(), β_j.log()  # Precompute the logs of the measures' weights
    B_i, A_j = torch.zeros_like(α_i), torch.zeros_like(β_j)  # Sampled influence fields

    Cxy_e = dist_matrix(x_i, y_j, p) / ε

    for i in range(nits):
        A_j = -lse((B_i + α_i_log).view(1, -1) - Cxy_e.t())  # a(y)/ε = Smin_ε,x~α [ C(x,y) - b(x) ]  / ε
        B_i = -lse((A_j + β_j_log).view(1, -1) - Cxy_e)  # b(x)/ε = Smin_ε,y~β [ C(x,y) - a(y) ]  / ε

    return ε * A_j.view(-1), ε * B_i.view(-1)


def sym_sink(α_i, x_i, y_j=None, p=2, eps=.5, nits=200, **kwargs):
    ε = eps  # Python supports Unicode. So fancy!
    # Sinkhorn loop ......................................................................

    α_i_log = α_i.log()
    A_i = torch.zeros_like(α_i)
    Cxx_e = dist_matrix(x_i, x_i, p)

    for i in range(nits - 1):
        A_i = 0.5 * (A_i - lse(
            (A_i + α_i_log).view(1, -1) - Cxx_e))  # a(x)/ε = .5*(a(x)/ε + Smin_ε,y~α [ C(x,y) - a(y) ] / ε)

    a_x = -ε * lse((A_i + α_i_log).view(1, -1) - Cxx_e).view(-1)  # a(x) = Smin_e,z~α [ C(x,z) - a(z) ]

    if y_j is None:
        return None, a_x
    else:  # extrapolate "a" to the point cloud "y_j"
        Cyx_e = dist_matrix(y_j, x_i, p)
        a_y = - ε * lse((A_i + α_i_log).view(1, -1) - Cyx_e).view(-1)  # a(z) = Smin_e,z~α [ C(y,z) - a(z) ]
        return a_y, a_x


#######################################################################################################################
# Derived Functionals .....................................................................
#######################################################################################################################

def regularized_ot(α, x, β, y, **params):  # OT_ε
    a_y, b_x = sink(α, x, β, y, **params)
    return scal(α, b_x) + scal(β, a_y)


def hausdorff_divergence(α, x, β, y, **params):  # H_ε
    a_y, a_x = sym_sink(α, x, y, **params)
    b_x, b_y = sym_sink(β, y, x, **params)
    return .5 * (scal(α, b_x - a_x) + scal(β, a_y - b_y))


def KEOPS_sinkhorn_divergence(*args, **params):  # S_ε

    α, x, β, y = process_args(*args)
    a_y, b_x = sink(α, x, β, y, **params)
    _, a_x= sym_sink(α, x, **params)
    _, b_y= sym_sink(β, y, **params)
    return scal(α, b_x - a_x) + scal(β, a_y - b_y)


def process_args(*args):
    if len(args) == 4:
        α, x, β, y = args
        return α, x, β, y
    elif len(args) == 2:
        x, y = args
        α = generate_weights(x)
        β = generate_weights(y)
        α = α.unsqueeze(1)
        β = β.unsqueeze(1)
        return α, x, β, y
    else:
        raise ValueError(
            "A SamplesLoss accepts two (x, y), four (α, x, β, y) or six (l_x, α, x, l_y, β, y)  arguments.")


def generate_weights(x):
    if x.dim() == 2:  #
        N = x.shape[0]
        return torch.ones(N).type_as(x) / N
    elif x.dim() == 3:
        B, N, _ = x.shape
        return torch.ones(B, N).type_as(x) / N
    else:
        raise ValueError("Input samples 'x' and 'y' should be encoded as (N,D) or (B,N,D) (batch) tensors.")