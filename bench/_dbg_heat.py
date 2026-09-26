"""Scratch: run the symbolic on the real heat operator and validate it."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from conftest import icosphere  # noqa: E402
from mojo_potpourri3d._lib import addr, f64, lib  # noqa: E402
from mojo_potpourri3d._solver import HalfedgeMesh, IntrinsicGeometry  # noqa: E402

sub = int(sys.argv[1]) if len(sys.argv) > 1 else 2
V, F = icosphere(sub)
mesh = HalfedgeMesh(np.ascontiguousarray(F, dtype=np.int64), V.shape[0])
g = IntrinsicGeometry(mesh, f64(V))
ti, tj, tv = g.cotan_laplacian_triplets()
n = mesh.n_vertices
idx = np.arange(n, dtype=np.int64)
ti = np.ascontiguousarray(np.concatenate([ti, idx]), dtype=np.int64)
tj = np.ascontiguousarray(np.concatenate([tj, idx]), dtype=np.int64)
tv = np.ascontiguousarray(np.concatenate([tv, g.vertex_dual_areas]), dtype=np.float64)
nnz_t = ti.size

cap = 2 * nnz_t + 1
Ap0 = np.zeros(n + 1, dtype=np.int64)
Ai = np.zeros(cap, dtype=np.int64)
Ax = np.zeros(cap, dtype=np.float64)
Axi = np.zeros(cap, dtype=np.float64)
ApNew = np.zeros(n + 1, dtype=np.int64)
perm = np.zeros(n, dtype=np.int64)
iperm = np.zeros(n, dtype=np.int64)
count = np.zeros(256, dtype=np.int64)
degree = np.zeros(n, dtype=np.int64)
visited = np.zeros(n, dtype=np.int64)
queue = np.zeros(n, dtype=np.int64)
nnzA = lib().mpp3d_permute_upper(
    n, addr(ti), addr(tj), addr(tv), addr(tv), nnz_t, addr(degree), addr(visited),
    addr(queue), addr(perm), addr(iperm), addr(count), addr(Ap0), addr(Ai),
    addr(Ax), addr(Axi), addr(ApNew),
)
Ap = ApNew
Ai = np.ascontiguousarray(Ai[:nnzA])
Ax = np.ascontiguousarray(Ax[:nnzA])
print("n", n, "nnzA", nnzA)

L = lib()
S_p = np.zeros(n + 1, dtype=np.int64)
R_p = np.zeros(n + 1, dtype=np.int64)
U_head = np.zeros(n, dtype=np.int64)
mark = np.zeros(n, dtype=np.int64)
cov = np.zeros(n, dtype=np.int64)
gather = np.zeros(n, dtype=np.int64)
aux = np.zeros(n, dtype=np.int64)
cnt = np.zeros(n, dtype=np.int64)
cursor = np.zeros(n, dtype=np.int64)
info = np.zeros(2, dtype=np.int64)
c = max(nnzA, 1)
tries = 0
while True:
    S_i = np.zeros(c, dtype=np.int64)
    R_i = np.zeros(c, dtype=np.int64)
    R_pos = np.zeros(c, dtype=np.int64)
    U_i = np.zeros(c, dtype=np.int64)
    U_next = np.zeros(c, dtype=np.int64)
    got = L.mpp3d_ldl_symbolic(
        n, addr(Ap), addr(Ai), addr(S_p), addr(S_i), addr(R_p), addr(R_i),
        addr(R_pos), addr(U_head), addr(U_i), addr(U_next), addr(mark), addr(cov),
        addr(gather), addr(aux), addr(cnt), addr(cursor), addr(info), c,
    )
    tries += 1
    print("cap", c, "->", got)
    if got >= 0:
        break
    c = max((2 * max(int(info[0]), 1) * n) // max(int(info[1]), 1), c + 1)
nnzL = got
print("nnzL", nnzL, "R_p[n]", int(R_p[n]), "tries", tries)

S = [set() for _ in range(n)]
for k in range(n):
    reach = {int(Ai[p]) for p in range(Ap[k], Ap[k + 1]) if Ai[p] > k}
    for j in range(k):
        if k in S[j]:
            reach |= {i for i in S[j] if i > k}
    S[k] = reach
tot = sum(len(s) for s in S)
print("ref total", tot, "match", tot == nnzL)
bad = 0
for k in range(n):
    got_c = sorted(int(x) for x in S_i[S_p[k]:S_p[k + 1]])
    if got_c != sorted(S[k]):
        bad += 1
        if bad <= 3:
            print("col", k, "got", got_c[:24])
            print("col", k, "ref", sorted(S[k])[:24])
print("bad columns", bad)
for k in range(n):
    for q in range(R_p[k], R_p[k + 1]):
        j = int(R_i[q])
        pos = int(R_pos[q])
        if not (0 <= pos < nnzL and int(S_i[pos]) == k):
            print("bad row entry", k, j, pos)
            raise SystemExit(1)
print("row index ok")

upto = np.zeros(n, dtype=np.int64)
Lx = np.zeros(nnzL, dtype=np.float64)
D = np.zeros(n, dtype=np.float64)
Y = np.zeros(n, dtype=np.float64)
W = np.zeros(max(int((R_p[1:] - R_p[:-1]).max()), 1), dtype=np.float64)
L.mpp3d_ldl_numeric(
    n, nnzL, addr(Ap), addr(Ai), addr(Ax), addr(S_p), addr(S_i),
    addr(R_p), addr(R_i), addr(R_pos), addr(upto), addr(Lx), addr(D), addr(Y), addr(W),
)
B = np.zeros(n)
B[0] = 1.0
Bp = np.ascontiguousarray(B[perm])
X = np.zeros(n)
L.mpp3d_ldl_solve(n, addr(S_p), addr(S_i), addr(Lx), addr(D), addr(Bp), addr(X), 1)
M = np.zeros((n, n))
for j in range(n):
    for p in range(Ap[j], Ap[j + 1]):
        M[int(Ai[p]), j] = Ax[p]
print("solve err", np.abs(M @ X - Bp).max() / np.abs(Bp).max())
Xo = np.empty(n); Xo[perm] = X
print("unpermuted err", np.abs(M[np.ix_(np.argsort(perm), np.argsort(perm))] @ Xo - B).max())
