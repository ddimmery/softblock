# This code is adapted from 
# http://www.nathankallus.com/papers/OptimalAPriori.py.txt

from functools import reduce

import mosek
import mosek.fusion
from mosek.fusion import *

from cvxopt import solvers
from cvxopt import matrix

import numpy as np

from scipy.spatial.distance import pdist, squareform

solvers.options['show_progress'] = False


def safeinvcov(x):
    """
    Safely invert the sample covariance matrix
    """
    n,d = x.shape
    if d==1:
        return array(1./np.var(x)).reshape((1,1))
    else:
        covar = np.cov(x,rowvar=0)
        if np.linalg.det(covar)==0.:
            return scipy.linalg.pinv(covar)
        else:
            return scipy.linalg.inv(covar)

def LinearKernel(x, normalize=False, eps=1e-8):
    """
    Compute the Gram matrix for the linear kernel

    Args:
        x:          n by d array of covariates
        normalize:  whether to normalize the data
    Returns:
        d by d Gram matrix
    """
    if normalize:
        s = safeinvcov(x)
        xc = x - x.mean(0)
        return np.dot(np.dot(xc,s),xc.T) + np.diag([eps] * x.shape[0])
    else:
        return np.dot(x,x.T)

def GaussianKernel(x, s=1., normalize=True, eps=1e-8):
    """
    Compute the Gram matrix for the Gaussian kernel

    Args:
        x:          n by d array of covariates
        s:          bandwidth
        normalize:  whether to normalize the data
    Returns:
        d by d Gram matrix
    """
    pairwise_dists = squareform(pdist(x, 'mahalanobis')**2 if normalize
                                else pdist(x, 'sqeuclidean'))
    return np.exp(-pairwise_dists / s**2)  + np.diag([eps] * x.shape[0])

def PolynomialKernel(x, deg=2, normalize=False):
    """
    Compute the Gram matrix for the polynomial kernel

    Args:
        x:          n by d array of covariates
        deg:        degree
        normalize:  whether to normalize the data
    Returns:
        d by d Gram matrix
    """
    if normalize:
        s = safeinvcov(x)
        xc = x - x.mean(0)
        return ((np.dot(np.dot(xc,s),xc.T)/float(deg)+1.)**deg)
    else:
        return ((np.dot(x,x.T)/float(deg)+1.)**deg)

def ExpKernel(x, normalize=False):
    """
    Compute the Gram matrix for the exponential kernel

    Args:
        x:          n by d array of covariates
        normalize:  whether to normalize the data
    Returns:
        d by d Gram matrix
    """
    if normalize:
        s = safeinvcov(x)
        xc = x - x.mean(0)
        return np.exp(np.dot(np.dot(xc,s),xc.T))
    else:
        return np.exp(np.dot(x,x.T))


def QuadMatch(K, B = 1):
    """
    Return the top B solutions in increasing objective value to the following
    quadratic optimization problem (with symmetry eliminated)
    minimize    u^T K u
    subject to  u in {-1, +1}^n
                sum_i u_i = 0
                u_1 = -1

    Args:
        K: d by d array representing a PSD matrix
        B: number of solutions
    Returns:
        list of lists of +/-1 denoting assignment
    """
    allocations = []
    obj_value = []
    n=len(K)
    k=n/2
    K1 = np.dot(np.ones((1,n)),K)
    K11 = np.dot(K1,np.ones((n,1)))
    result = solvers.coneqp(
        P = matrix(4. * K[1:, 1:]),
        q = matrix(-4 * K1[0, 1:]),
        A = matrix(np.ones((1, n - 1))),
        b = matrix(np.array(k).reshape(-1,1)),
    )
    allocation = [1 if zz > 0.5 else 0 for zz in np.array(result['x']).flatten()]
    allocations.append([0] + allocation)
    obj_value.append(result['primal objective'])
    for b in range(B-1):
        try:
            result = solvers.coneqp(
                P = matrix(4. * K[1:, 1:]),
                q = matrix(-4 * K1[0, 1:]),
                A = matrix(np.ones((1, n - 1))),
                b = matrix(np.array(k).reshape(-1,1)),
                G = matrix(np.array(allocations).astype(float)[:, 1:]),
                h = matrix((k - 1) * np.ones((b + 1, 1)))
            )
            allocation = [1 if zz > 0.5 else 0 for zz in np.array(result['x']).flatten()]
            allocations.append([0] + allocation)
            obj_value.append(result['primal objective'])
        except:
            break
    return [[2 * zz - 1 for zz in z] for z in allocations]


def PSOD(K):
    """
    Return the single top solution to the following quadratic optimization
    problem (with symmetry eliminated)
    minimize    u^T K u
    subject to  u in {-1, +1}^n
                sum_i u_i = 0
                u_1 = -1

    Args:
        K: d by d array representing a PSD matrix
    Returns:
        list of +/-1 denoting assignment (first is always -1 so always
        randomize the sign; see PSODDraw below)
    """
    return QuadMatch(K,1)[0]


def SDPHeuristic(K, us, mixbound = 0.05):
    """
    Solve the semi-definite optimization problem in Algorithm 4.2
    """
    n = len(K)
    (l,v)=np.linalg.eig(K)
    l=np.real(l)
    v=np.real(v)
    l[l<0]=0
    Ksqrt=np.dot(np.dot(v,np.diag(np.sqrt(l))),v.T)
    zz = us[0]
    ZZs=[
        Matrix.dense(
            np.dot(np.dot(Ksqrt, np.outer(zz,zz)), Ksqrt).astype(float).tolist()
        )
        for zz in us
    ]
    I=Matrix.diag([1.,]*n)
    with Model("match") as M:
        M.setSolverParam('numThreads', 1)
        t = M.variable('t', len(ZZs), Domain.greaterThan(0.0)
            if mixbound==None else Domain.inRange(0.0, mixbound))
        z = M.variable("z",Domain.greaterThan(0.0))
        sum1cons=M.constraint(Expr.sum(t), Domain.equalsTo(1.0))
        opnormcons=M.constraint("z>=opnorm", Expr.sub(Expr.mul(z,I),
            reduce(Expr.add, (Expr.mul(t.index(i), ZZs[i])
            for i in range(len(ZZs)))) ), Domain.inPSDCone(n))
        M.objective(ObjectiveSense.Minimize, z)
        M.acceptedSolutionStatus(AccSolutionStatus.Anything)
        M.solve()
        return (z.level()[0], t.level(), M.getPrimalSolutionStatus())


def MSODHeuristic(K, B):
    """
    Compute the MSOD as per heuristic Algorithm 4.3

    Args:
        K: d by d array representing a PSD matrix
        B: number of top solutions to use
    Returns:
        weights for each assignment vector,
        list of lists of +/-1 denoting assignment
        (first of each assignment is always -1 so always randomize the sign;
         see MSODDraw below)
    """
    # print('MSODHeuristic: getting top', B,'solutions')
    us = QuadMatch(K, B)
    # print('MSODHeuristic: solving SDP')
    res = SDPHeuristic(K, us)
    if type(res) != tuple:
        return res
    z2,t,stat2 = res
    return (t, us)


def MSODDraw(msod):
    """
    Draw a random assignment from the MSOD

    Args:
        msod: the output of MSODHeuristic
    Returns:
        list of +/-1 denoting assignment
    """
    cs = np.cumsum(msod[0])/np.sum(msod[0])
    idx = np.sum(cs < np.random.rand())
    return (np.array(msod[1][idx])*((-1)**np.random.randint(2))).tolist()


def PSODDraw(psod):
    """
    Draw a random assignment from the PSOD

    Args:
        psod: the output of MSODHeuristic
    Returns:
        list of +/-1 denoting assignment
    """
    return (np.array(psod)*((-1)**np.random.randint(2))).tolist()
