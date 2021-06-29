#!/usr/bin/env python
"""
sinkhorn_pointcloud.py

Discrete OT : Sinkhorn algorithm for point cloud marginals.

"""

import torch
from torch.autograd import Variable


def sinkhorn_normalized(x, y, epsilon, n, niter, p=1):
    Cxy = cost_matrix(x, y, p)
    Wxy, pi = sinkhorn_loss(x, y, epsilon, n, niter, C_Matrix=Cxy)
    Cxx = cost_matrix(x, x, p)
    Wxx, pi_x = sinkhorn_loss(x, x, epsilon, n, niter, C_Matrix=Cxx)
    Cyy = cost_matrix(y, y, p)
    Wyy, pi_y = sinkhorn_loss(y, y, epsilon, n, niter, C_Matrix=Cyy)
    return 2 * Wxy - Wxx - Wyy, (pi, pi_x, pi_y)


def Wasserstein1(x, y, epsilon, n, niter, p=1):
    Cxy = cost_matrix(x, y, p)
    Wxy, pi = sinkhorn_loss(x, y, epsilon, n, niter, C_Matrix=Cxy)
    return Wxy


def mixed_sinkhorn_normalized(x, y, fx, fy, epsilon, n, niter, p=2, ratio = 0.,nfac=1.0):
    if len(x.shape) > 2:
        x = x.view(x.shape[0], -1).cuda()
        y = y.view(y.shape[0], -1).cuda()

    # Cxy = cost_matrix(x, y, p) + nfac * RKHS_Norm(fx, fy)
    # Cxx = cost_matrix(x, x, p) + nfac * RKHS_Norm(fx, fx)
    # Cyy = cost_matrix(y, y, p) + nfac * RKHS_Norm(fy, fy)

    Cxy = 10 * ratio * cost_matrix(x, y, p) + 10 * (1 - ratio) * cost_matrix(fx, fy, p)
    Cxx = 10 * ratio * cost_matrix(x, x, p) + 10 * (1 - ratio) * cost_matrix(fx, fx, p)
    Cyy = 10 * ratio * cost_matrix(y, y, p) + 10 * (1 - ratio) * cost_matrix(fy, fy, p)

    # Cxy = cost_matrix(x, y, p)
    # Cxx = cost_matrix(x, x, p)
    # Cyy = cost_matrix(y, y, p)

    # Cxy = 10*cost_matrix(fx, fy, p)
    # Cxx = 10*cost_matrix(fx, fx, p)
    # Cyy = 10*cost_matrix(fy, fy, p)

    # Cxy = nfac * RKHS_Norm(fx, fy)
    # Cxx = nfac * RKHS_Norm(fx, fx)
    # Cyy = nfac * RKHS_Norm(fy, fy)

    Wxy, pi = sinkhorn_loss(x, y, epsilon, n, niter, C_Matrix=Cxy)
    Wxx, pi_x = sinkhorn_loss(x, x, epsilon, n, niter, C_Matrix=Cxx)
    Wyy, pi_y = sinkhorn_loss(y, y, epsilon, n, niter, C_Matrix=Cyy)

    # Wxx = 0
    # Wyy = 0
    return 2 * Wxy - Wxx - Wyy, pi


def mixed_sinkhorn_normalized_unbiased(x, y, fx, fy, epsilon, n, niter, p=2, nfac=1.0):
    for i in range(len(x)):
        if len(x[i].shape) > 2:
            x[i] = x[i].view(x[i].shape[0], -1).cuda()
            y[i] = y[i].view(y[i].shape[0], -1).cuda()

    # Cxy = cost_matrix(x, y, p) + nfac * RKHS_Norm(fx, fy)
    # Cxx = cost_matrix(x, x, p) + nfac * RKHS_Norm(fx, fx)
    # Cyy = cost_matrix(y, y, p) + nfac * RKHS_Norm(fy, fy)

    Cxy_0 = 0.8 * cost_matrix(x[0], y[0], p) + 0.2 * cost_matrix(fx[0], fy[0], p)
    Cxy_1 = 0.8 * cost_matrix(x[1], y[1], p) + 0.2 * cost_matrix(fx[1], fy[1], p)
    Cxx = 0.8 * cost_matrix(x[2], x[3], p) + 0.2 * cost_matrix(fx[2], fx[3], p)
    Cyy = 0.8 * cost_matrix(y[2], y[3], p) + 0.2 * cost_matrix(fy[2], fy[3], p)

    # Cxy_0 = 2*cost_matrix(fx[0], fy[0], p)
    # Cxy_1 = 2*cost_matrix(fx[1], fy[1], p)
    # Cxx = 2*cost_matrix(fx[2], fx[3], p)
    # Cyy = 2*cost_matrix(fy[2], fy[3], p)

    # Cxy = cost_matrix(x, y, p)
    # Cxx = cost_matrix(x, x, p)
    # Cyy = cost_matrix(y, y, p)

    # Cxy = 2*cost_matrix(fx, fy, p)
    # Cxx = 2*cost_matrix(fx, fx, p)
    # Cyy = 2*cost_matrix(fy, fy, p)

    # Cxy = nfac * RKHS_Norm(fx, fy)
    # Cxx = nfac * RKHS_Norm(fx, fx)
    # Cyy = nfac * RKHS_Norm(fy, fy)

    Wxy_0, pi_0 = sinkhorn_loss(x[0], y[0], epsilon, n, niter, C_Matrix=Cxy_0)
    Wxy_1, pi_1 = sinkhorn_loss(x[1], y[1], epsilon, n, niter, C_Matrix=Cxy_1)
    Wxx, pi_x = sinkhorn_loss(x[2], x[3], epsilon, n, niter, C_Matrix=Cxx)
    Wyy, pi_y = sinkhorn_loss(y[2], y[3], epsilon, n, niter, C_Matrix=Cyy)

    # Wxx = 0
    # Wyy = 0
    return Wxy_0 + Wxy_1 - Wxx - Wyy, pi_0


def mixed_sinkhorn_normalized_unbiased_1(x, y, fx, fy, epsilon, n, niter, p=2, nfac=1.0):
    for i in range(len(x)):
        if len(x[i].shape) > 2:
            x[i] = x[i].view(x[i].shape[0], -1).cuda()
            y[i] = y[i].view(y[i].shape[0], -1).cuda()

    # Cxy = cost_matrix(x, y, p) + nfac * RKHS_Norm(fx, fy)
    # Cxx = cost_matrix(x, x, p) + nfac * RKHS_Norm(fx, fx)
    # Cyy = cost_matrix(y, y, p) + nfac * RKHS_Norm(fy, fy)

    Cxy = 0.8 * cost_matrix(x[0], y[0], p) + 0.2 * cost_matrix(fx[0], fy[0], p)
    Cxx = 0.8 * cost_matrix(x[1], x[1], p) + 0.2 * cost_matrix(fx[1], fx[1], p)
    Cyy = 0.8 * cost_matrix(y[1], y[1], p) + 0.2 * cost_matrix(fy[1], fy[1], p)

    # Cxy_0 = 2*cost_matrix(fx[0], fy[0], p)
    # Cxy_1 = 2*cost_matrix(fx[1], fy[1], p)
    # Cxx = 2*cost_matrix(fx[2], fx[3], p)
    # Cyy = 2*cost_matrix(fy[2], fy[3], p)

    # Cxy = cost_matrix(x, y, p)
    # Cxx = cost_matrix(x, x, p)
    # Cyy = cost_matrix(y, y, p)

    # Cxy = 2*cost_matrix(fx, fy, p)
    # Cxx = 2*cost_matrix(fx, fx, p)
    # Cyy = 2*cost_matrix(fy, fy, p)

    # Cxy = nfac * RKHS_Norm(fx, fy)
    # Cxx = nfac * RKHS_Norm(fx, fx)
    # Cyy = nfac * RKHS_Norm(fy, fy)

    Wxy, pi = sinkhorn_loss(x[0], y[0], epsilon, n, niter, C_Matrix=Cxy)
    Wxx, pi_x = sinkhorn_loss(x[1], x[1], epsilon, n, niter, C_Matrix=Cxx)
    Wyy, pi_y = sinkhorn_loss(y[1], y[1], epsilon, n, niter, C_Matrix=Cyy)

    # Wxx = 0
    # Wyy = 0
    return 2 * Wxy - Wxx - Wyy, pi


def ERWD_normalized(x, y, fx, fy, epsilon, n, niter, p=2, nfac=1.0):
    if len(x.shape) > 2:
        x = x.view(x.shape[0], -1).cuda()
        y = y.view(y.shape[0], -1).cuda()
    gamma = .9
    Cxy = gamma * RKHS_Norm(x, y) + (1-gamma) * RKHS_Norm(fx, fy)
    # Cxy = gamma * cost_matrix(x, y, p) + (1 - gamma) * cost_matrix(fx, fy, p)
    Wxy, pi = sinkhorn_loss(x, y, epsilon, n, niter, C_Matrix=Cxy)
    return Wxy, pi


def sinkhorn_loss(x, y, epsilon, n, niter, C_Matrix):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    # The Sinkhorn algorithm takes as input three variables :

    C = C_Matrix  # Wasserstein cost function
    # both marginals are fixed with equal weights
    mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10 ** (-3)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.logsumexp(A, dim=1, keepdim=True)

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err.data.tolist() < thresh):
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return cost, pi


def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1).cuda()
    y_lin = y.unsqueeze(0).cuda()
    c = torch.sum((torch.abs(x_col - y_lin)).cuda() ** p, 2)
    return c


def RBF_Kernel(fx, fy, gamma):
    "Returns the matrix of $exp(-gamma * |x_i-y_j|^2)$."
    x_col = fx.unsqueeze(1).cuda()
    y_lin = fy.unsqueeze(0).cuda()
    c = torch.norm(torch.abs(x_col - y_lin), p=1, dim=2)
    # c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    # return c
    RBF_K = torch.exp(-gamma * c)
    return RBF_K


def RKHS_Norm(x, y, gamma=0.5):
    Kxy = RBF_Kernel(x, y, gamma)
    return 1 + 1 - 2 * Kxy
