"""Scratch: which reach recurrence reproduces the old symbolic?"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from mojo_potpourri3d._solver import HalfedgeMesh, IntrinsicGeometry  # noqa: E402
from mojo_potpourri3d._lib import f64  # noqa: E402
from conftest import icosphere  # noqa: E402

sub = int(sys.argv[1]) if len(sys.argv) > 1 else 2
V, F = icosphere(sub)
mesh = HalfedgeMesh(np.ascontiguousarray(F, dtype=np.int64), V.shape[0])
g = IntrinsicGeometry(mesh, f64(V))
ti, tj, tv = g.cotan_laplacian_triplets()
n = mesh.n_vertices

# same RCM + dedup the port does, done here in numpy
import scipy.sparse as sp  # noqa: E402

A = sp.coo_matrix((tv, (ti, tj)), shape=(n, n)).tocsr()
A = A + A.T
Ap = A.indptr.astype(np.int64)
Ai = A.indices.astype(np.int64)
Ax = A.data.astype(np.float64)


def reach_over_U():
    S = [set() for _ in range(n)]
    for k in range(n):
        reach = {int(Ai[p]) for p in range(Ap[k], Ap[k + 1]) if Ai[p] > k}
        for j in range(k):
            if k in S[j]:
                reach |= {i for i in S[j] if i > k}
        S[k] = reach
    return S


def reach_over_A():
    S = [set() for _ in range(n)]
    for k in range(n - 1, -1, -1):
        reach = set()
        for p in range(Ap[k], Ap[k + 1]):
            j = int(Ai[p])
            if j > k:
                reach.add(j)
                reach |= S[j]
        S[k] = reach
    return S


def reach_numeric():
    """The pattern of numerically nonzero L entries, by dense elimination."""
    M = np.zeros((n, n))
    for j in range(n):
        for p in range(Ap[j], Ap[j + 1]):
            M[Ai[p], j] = Ax[p]
    Lm = np.zeros((n, n))
    D = np.zeros(n)
    S = [set() for _ in range(n)]
    for k in range(n):
        Dk = M[k, k] - sum(Lm[k, j] ** 2 * D[j] for j in range(k))
        D[k] = Dk
        for i in range(k + 1, n):
            v = (M[i, k] - sum(Lm[i, j] * Lm[k, j] * D[j] for j in range(k))) / Dk
            Lm[i, k] = v
            if v != 0.0:
                S[k].add(i)
    return S


U = reach_over_U()
A_ = reach_over_A()
N = reach_numeric() if n <= 700 else None
print("n", n, "nnzA", len(Ax))
print("over U  :", sum(len(s) for s in U))
print("over A  :", sum(len(s) for s in A_))
print("numeric :", sum(len(s) for s in N) if N else "skipped")
print("U == A :", U == A_)
print("A == num:", (A_ == N) if N else "skipped")
print("U == num:", (U == N) if N else "skipped")

