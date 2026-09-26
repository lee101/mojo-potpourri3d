"""Phase-level timing of Factorization. Diagnostic only, not part of the bench."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import mojo_potpourri3d as pp3d  # noqa: E402
from conftest import icosphere  # noqa: E402
from mojo_potpourri3d._lib import addr, lib  # noqa: E402
from mojo_potpourri3d._solver import Factorization  # noqa: E402


def prof(sub: int) -> None:
    V, F = icosphere(sub)
    t0 = time.perf_counter()
    g = pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False).geom
    t_geom = time.perf_counter() - t0

    ti, tj, tv = g.cotan_laplacian_triplets()
    n = g.mesh.n_vertices
    nnz_t = ti.size

    t0 = time.perf_counter()
    f = Factorization.__new__(Factorization)
    f.n = n
    f.complex = False
    f.ti = np.ascontiguousarray(ti, dtype=np.int64)
    f.tj = np.ascontiguousarray(tj, dtype=np.int64)
    f.tr = np.ascontiguousarray(tv, dtype=np.float64)
    f.tim = None
    t_prep = time.perf_counter() - t0

    # replay the constructor body with timers
    L = lib()
    cap = 2 * nnz_t + 1
    f.Ap = np.zeros(n + 1, dtype=np.int64)
    f.Ai = np.zeros(cap, dtype=np.int64)
    f.Ax = np.zeros(cap, dtype=np.float64)
    f.Axi = np.zeros(cap, dtype=np.float64)
    f.ApNew = np.zeros(n + 1, dtype=np.int64)
    f.perm = np.zeros(n, dtype=np.int64)
    f.iperm = np.zeros(n, dtype=np.int64)
    count = np.zeros(256, dtype=np.int64)
    degree = np.zeros(n, dtype=np.int64)
    visited = np.zeros(n, dtype=np.int64)
    queue = np.zeros(n, dtype=np.int64)
    imag = np.zeros(max(nnz_t, 1), dtype=np.float64)

    t0 = time.perf_counter()
    nnzA = L.mpp3d_permute_upper(
        n, addr(f.ti), addr(f.tj), addr(f.tr), addr(imag), nnz_t,
        addr(degree), addr(visited), addr(queue), addr(f.perm), addr(f.iperm),
        addr(count), addr(f.Ap), addr(f.Ai), addr(f.Ax), addr(f.Axi), addr(f.ApNew),
    )
    t_perm = time.perf_counter() - t0
    f.Ap = f.ApNew
    f.Ai = np.ascontiguousarray(f.Ai[:nnzA])
    f.Ax = np.ascontiguousarray(f.Ax[:nnzA])
    f.Axi = np.ascontiguousarray(f.Axi[:nnzA])

    L = lib()
    f.S_p = np.zeros(n + 1, dtype=np.int64)
    f.R_p = np.zeros(n + 1, dtype=np.int64)
    U_head = np.zeros(n, dtype=np.int64)
    mark = np.zeros(n, dtype=np.int64)
    cov = np.zeros(n, dtype=np.int64)
    gather = np.zeros(n, dtype=np.int64)
    aux = np.zeros(n, dtype=np.int64)
    cnt = np.zeros(n, dtype=np.int64)
    info = np.zeros(2, dtype=np.int64)
    cursor = np.zeros(n, dtype=np.int64)
    cap2 = max(nnzA, 1)
    tries = 0
    t0 = time.perf_counter()
    while True:
        f.S_i = np.zeros(cap2, dtype=np.int64)
        f.R_i = np.zeros(cap2, dtype=np.int64)
        f.R_pos = np.zeros(cap2, dtype=np.int64)
        U_i = np.zeros(cap2, dtype=np.int64)
        U_next = np.zeros(cap2, dtype=np.int64)
        got = L.mpp3d_ldl_symbolic(
            n, addr(f.Ap), addr(f.Ai), addr(f.S_p), addr(f.S_i),
            addr(f.R_p), addr(f.R_i), addr(f.R_pos), addr(U_head), addr(U_i),
            addr(U_next), addr(mark), addr(cov), addr(gather), addr(aux),
            addr(cnt), addr(cursor), addr(info), cap2,
        )
        tries += 1
        if got >= 0:
            nnzL = got
            break
        cap2 = max((2 * max(int(info[0]), 1) * n) // max(int(info[1]), 1), cap2 + 1)
        print('   retry cap', cap2, 'from write', int(info[0]), 'cols', int(info[1]))
    t_sym = time.perf_counter() - t0

    upto = np.zeros(n, dtype=np.int64)
    f.Lx = np.zeros(nnzL, dtype=np.float64)
    f.D = np.zeros(n, dtype=np.float64)
    f.Y = np.zeros(n, dtype=np.float64)
    f.W = np.zeros(max(int((f.R_p[1:] - f.R_p[:-1]).max()), 1), dtype=np.float64)

    t0 = time.perf_counter()
    L.mpp3d_ldl_numeric(
        n, nnzL, addr(f.Ap), addr(f.Ai), addr(f.Ax),
        addr(f.S_p), addr(f.S_i), addr(f.R_p), addr(f.R_i), addr(f.R_pos), addr(upto),
        addr(f.Lx), addr(f.D), addr(f.Y), addr(f.W),
    )
    t_num = time.perf_counter() - t0

    b = np.zeros(n, dtype=np.float64)
    b[0] = 1.0
    t0 = time.perf_counter()
    for _ in range(5):
        f.solve_vector(b)
    t_solve = (time.perf_counter() - t0) / 5

    print(f"sub={sub} n={n} faces={g.mesh.n_faces} nnzT={nnz_t} nnzA={nnzA} nnzL={nnzL} retries={tries}")
    print(f"  geometry(all caches) {t_geom*1e3:9.2f} ms")
    print(f"  triplet build        {t_prep*1e3:9.2f} ms")
    print(f"  permute_upper        {t_perm*1e3:9.2f} ms")
    print(f"  ldl_symbolic         {t_sym*1e3:9.2f} ms")
    print(f"  ldl_numeric          {t_num*1e3:9.2f} ms")
    print(f"  solve_vector         {t_solve*1e3:9.2f} ms")
    fill = nnzL / max(nnzA, 1)
    print(f"  fill nnzL/nnzA = {fill:.2f}")


if __name__ == "__main__":
    prof(4)
    prof(5)
