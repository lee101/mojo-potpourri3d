"""What fill is available from better orderings? Diagnostic only."""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import mojo_potpourri3d as pp3d  # noqa: E402
from conftest import icosphere  # noqa: E402
from mojo_potpourri3d._lib import addr, lib  # noqa: E402


def ldl_nnz(Ap, Ai, n):
    """nnz(L) for a symmetric pattern, column-compressed with ascending rows."""
    parent = [-1] * n
    for k in range(n):
        for p in range(Ap[k], Ap[k + 1]):
            i = Ai[p]
            if i > k:
                while i != -1 and i < k:
                    inext = parent[i]
                    parent[i] = k
                    i = inext
    c = [0] * n
    for k in range(n):
        cnt = 0
        for p in range(Ap[k], Ap[k + 1]):
            if Ai[p] > k:
                cnt += 1
        p = parent[k]
        while p != -1:
            cnt += c[p]
            p = parent[p]
        c[k] = cnt
    return sum(c)


def main(sub):
    V, F = icosphere(sub)
    L = pp3d.cotan_laplacian(V, F)
    A = sp.csc_matrix(L)
    n = A.shape[0]
    print(f"sub={sub} n={n} nnzA={A.nnz}")

    for spec in ("NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"):
        t0 = time.perf_counter()
        lu = spl.splu(A, permc_spec=spec)
        print(f"  {spec:16s} L nnz = {lu.L.nnz:9d}  factor = {lu.L.nnz/A.nnz:6.2f}  {time.perf_counter()-t0:.2f}s")

    # our own RCM
    s = pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False)
    ti, tj, tv = s.geom.cotan_laplacian_triplets()
    ti = np.ascontiguousarray(ti, dtype=np.int64)
    tj = np.ascontiguousarray(tj, dtype=np.int64)
    Ap = np.zeros(n + 1, dtype=np.int64)
    np.add.at(Ap, tj + 1, 1)
    Ap = np.cumsum(Ap)
    cap = 2 * ti.size + 1
    Ai = np.zeros(cap, dtype=np.int64)
    Ax = np.zeros(cap, dtype=np.float64)
    Axr = np.zeros(cap, dtype=np.float64)
    Axi = np.zeros(cap, dtype=np.float64)
    ApNew = np.zeros(n + 1, dtype=np.int64)
    perm = np.zeros(n, dtype=np.int64)
    iperm = np.zeros(n, dtype=np.int64)
    count = np.zeros(256, dtype=np.int64)
    degree = np.zeros(n, dtype=np.int64)
    visited = np.zeros(n, dtype=np.int64)
    queue = np.zeros(n, dtype=np.int64)
    t0 = time.perf_counter()
    nnzA = lib().mpp3d_permute_upper(
        n, addr(ti), addr(tj), addr(tv), addr(tv), ti.size,
        addr(degree), addr(visited), addr(queue), addr(perm), addr(iperm),
        addr(count), addr(Ap), addr(Ai), addr(Ax), addr(Axr), addr(Axi), addr(ApNew),
    )
    Ap = ApNew
    print(f"  port rcm          L nnz = {ldl_nnz(Ap, Ai[:nnzA], n):9d}  ({time.perf_counter()-t0:.2f}s permute)")


    print("  debug nnzA", nnzA, "Ap[:5]", Ap[:5], "Ai[:8]", Ai[:8])
if __name__ == "__main__":
    main(4)
    main(5)
