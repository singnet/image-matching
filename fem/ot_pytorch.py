# modified from https://github.com/rythei/PyTorchOT

import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


def sink(M, r, c, reg, numItermax=1000, epsilon=1e-9):

    a = r
    b = c
    # init data
    Nini = len(a)
    Nfin = len(b)

    u = Variable(torch.ones(Nini) / Nini).to(M)
    v = Variable(torch.ones(Nfin) / Nfin).to(M)

    # print(reg)

    K = torch.exp(-M / reg)
    # print(np.min(K))

    Kp = (1 / a).view(-1, 1) * K
    cpt = 0
    err = 1
    while (err > epsilon and cpt < numItermax):
        uprev = u
        vprev = v
        #print(T(K).size(), u.view(u.size()[0],1).size())
        KtransposeU = K.t().matmul(u)
        v = torch.div(b, KtransposeU)
        u = 1. / Kp.matmul(v)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            transp = u.view(-1, 1) * (K * v)
            err = (torch.sum(transp) - b).norm(1).pow(2)


        cpt += 1

    P = u.view((-1, 1)) * K * v.view((1, -1))
    return P, torch.sum(P * M)


def sink_stabilized(M, r, c, reg, numItermax=1000, tau=1e2, epsilon=1e-9, warmstart=None, print_period=20):
    """
    Compute transport matrix P and total cost
    Parameters
    ----------
    M : torch.Tensor
        distance matrix len(r) x len(c)
    r : torch.Tensor
        source amounts(equals to sum of rows in M)
    c : torch.Tensor
        target amounts(equals to sum of columns in M)
    """
    M = M - M.min() + 0.0000000001
    a = r
    b = c

    # init data
    na = len(a)
    nb = len(b)

    # we assume that no distances are null except those of the diagonal of
    # distances
    if warmstart is None:
        alpha, beta = Variable(torch.zeros(na)).to(M), Variable(torch.zeros(nb)).to(M)
    else:
        alpha, beta = warmstart

    u, v = Variable(torch.ones(na) / na).to(M), Variable(torch.ones(nb) / nb).to(M)

    def get_K(alpha, beta):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg)

    def get_Gamma(alpha, beta, u, v):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg + torch.log(u.view((na, 1))) + torch.log(v.view((1, nb))))

    # print(np.min(K))

    K = get_K(alpha, beta)

    loop = 1
    cpt = 0
    err = 1
    transp = None

    while loop:
        # sinkhorn update
        v = torch.div(b, (K.t().matmul(u) + 1e-16))
        u = torch.div(a, (K.matmul(v) + 1e-16))

        # remove numerical problems and store them in K
        if torch.max(torch.abs(u)) > tau or torch.max(torch.abs(v)) > tau:
            alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)
            u, v = Variable(torch.ones(na) / na).to(M), Variable(torch.ones(nb) / nb).to(M)
            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            transp_prev = transp
            transp = get_Gamma(alpha, beta, u, v)
            if transp_prev is not None:
                err = (transp - transp_prev).abs().max()

        if err <= epsilon:
            loop = False

        if cpt >= numItermax:
            loop = False

        #if np.any(np.isnan(u)) or np.any(np.isnan(v)):
        #    # we have reached the machine precision
        #    # come back to previous solution and quit loop
        #    print('Warning: numerical errors at iteration', cpt)
        #    u = uprev
        #    v = vprev
        #    break

        cpt += 1

    P = u.view((-1, 1)) * K * v.view((1, -1))
    cost = torch.sum(get_Gamma(alpha, beta, u, v)*M)
    return P, cost


def pairwise_distances(x, y, method='l1'):
    n = x.size()[0]
    m = y.size()[0]
    d = x.size()[1]

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    if method == 'l1':
        dist = torch.abs(x - y).sum(2)
    else:
        dist = torch.pow(x - y, 2).sum(2)

    return dist.float()


def dmat(x,y):
    mmp1 = torch.stack([x] * x.size()[0])
    mmp2 = torch.stack([y] * y.size()[0]).transpose(0, 1)
    mm = torch.sum((mmp1 - mmp2) ** 2, 2).squeeze()

    return mm
