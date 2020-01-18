import torch
import numpy
np = numpy


# preferences, need to be converted to costs
# row i = cost of moving each item from c to place i
# making cost non-negative will not changes solution matrix P
preference = numpy.asarray([[2, 2, 1 , 0 ,0],
                            [0,-2,-2,-2,  2],
                            [1, 2, 2, 2, -1],
                            [2, 1, 0, 1, -1],
                            [0.5, 2, 2, 1, 0],
                            [0,  1,1, 1, -1],
                            [-2, 2, 2, 1, 1],
                            [2, 1, 2, 1, -1]])

# how much do we have place awailable at place
r = (3,3,3,4,2,2,2,1)
r = torch.from_numpy(numpy.asarray(r)).float()

# how mach do we need to transfer from each place
c = (4,2,6,4,4)
c = torch.from_numpy(numpy.asarray(c)).float()
x = torch.from_numpy(preference).float()


# from here https://michielstock.github.io/OptimalTransport/
def compute_optimal_transport(M, r, c, lam, epsilon=1e-8):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm

    Inputs:
        - M : cost matrix (n x m)
        - r : vector of marginals (n, )
        - c : vector of marginals (m, )
        - lam : strength of the entropic regularization
        - epsilon : convergence parameter

    Outputs:
        - P : optimal transport matrix (n x m)
        - dist : Sinkhorn distance
    """
    n, m = M.shape
    P = torch.exp(- lam * M)
    P = P / P.sum()
    u = torch.ones(n)
    # normalize this matrix
    i = 0
    while torch.max(torch.abs(u - P.sum(dim=1))) > epsilon:
        u = P.sum(dim=1)
        P *= (r / u).reshape((-1, 1))
        P *= (c / P.sum(0)).reshape((1, -1))
        i += 1
    print(i)
    return P, torch.sum(P * M)


def optimal_transport(M, r, c, lam, epsilon=1e-8):
    n, m = M.shape
    # not very stable
    Kinit = torch.exp(-M.double() * lam)
    K = torch.diag(1./r.double()).mm(Kinit)

    u = r
    v = c
    vprev = v * 2
    i = 0
    while(torch.abs(v - vprev).sum() > epsilon):
        vprev = v
        # changing order affects convergence a little bit
        v = c / K.T.matmul(u.double())
        u = r / K.matmul(v)
        i += 1

    P = torch.diag(u) @ K @ torch.diag(v)
    return P, torch.sum(P * M)


# see https://arxiv.org/pdf/1612.02273.pdf
# https://arxiv.org/pdf/1712.03082.pdf
# but instead i multiply by lam like in code above
def optimal_transport_np(M, r, c, lam, epsilon=1e-8):
    n, m = M.shape
    Kinit = np.exp(- M * lam)
    K = np.diag(1./r).dot(Kinit)
    u = r
    v = c
    vprev = v * 2
    i = 0
    while(np.abs(v - vprev).sum() > epsilon):
        vprev = v
        v = c / K.T.dot(u)
        u = r / K.dot(v)
        i += 1
    print(i)
    P = np.diag(u) @ K @ np.diag(v)
    return P, np.sum(P * M)


if __name__ == '__main__':
    print('numpy')
    P, cost = compute_optimal_transport(x * -1, r, c, 0.2, epsilon=0.001)
    print(P)
    print('ot_pytorch')
    # from ot_pytorch import sink, sink_stabilized
    # P2, dist2 = sink(x * -1, r, c, reg=0.2)
    # P3, dist3 = sink_stabilized(x * -1, r, c, reg=0.2, epsilon=0.001)
    #import pdb;pdb.set_trace()
    print('torch')
    P1, cost1 = optimal_transport(x * -1, r, c, 5)
    print(P1)
    print('shifted')
    # shifting cost above zero will not change the solution P
    x = x * -1
    x = x - x.min()
    P, cost = optimal_transport_np(x.numpy(), r.numpy(), c.numpy(), 5)
    print(P)
    P1, cost1 = optimal_transport_np(x.numpy(), r.numpy(), c.numpy(), 5, epsilon=0.01)
    print(P1)

