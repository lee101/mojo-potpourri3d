"""Scratch: inspect the symbolic/numeric factor on a small SPD matrix."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_potpourri3d._lib import addr, lib  # noqa: E402

rng = np.random.default_rng(7)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
M = rng.normal(size=(n, n))
A = M @ M.T + 4.0 * np.eye(n)
rows, cols = np.nonzero(A)
vals = A[rows, cols]

order = np.lexsort((rows, cols))
Ap = np.zeros(n + 1, dtype=np.int64)
Ap[1:] = np.cumsum(np.bincount(cols[order], minlength=n))
Ai = np.ascontiguousarray(rows[order], dtype=np.int64)
Ax = np.ascontiguousarray(vals[order], dtype=np.float64)
nnzA = int(Ap[n])

L = lib()
S_p = np.zeros(n + 1, dtype=np.int64)
R_p = np.zeros(n + 1, dtype=np.int64)
cnt = np.zeros(n, dtype=np.int64)
cursor = np.zeros(n, dtype=np.int64)
cap = nnzA
while True:
    S_i = np.zeros(cap, dtype=np.int64)
    R_i = np.zeros(cap, dtype=np.int64)
    R_pos = np.zeros(cap, dtype=np.int64)
    got = L.mpp3d_ldl_symbolic(
        n, addr(Ap), addr(Ai), addr(S_p), addr(S_i), addr(R_p), addr(R_i),
        addr(R_pos), addr(cnt), addr(cursor), cap,
    )
    print(f"cap={cap} -> {got}")
    if got >= 0:
        break
    cap = max(2 * (-got - 1), cap + 1)

nnzL = got
print("S_p", S_p)
print("S_i", S_i[:nnzL])
print("R_p", R_p)
print("R_i", R_i[: int(R_p[n])])
print("R_pos", R_pos[: int(R_p[n])])

# reference column pattern
S = [set() for _ in range(n)]
for k in range(n):
    reach = set(int(Ai[p]) for p in range(Ap[k], Ap[k + 1]) if Ai[p] > k)
    for j in range(k):
        if k in S[j]:
            reach |= {i for i in S[j] if i > k}
    S[k] = reach
ref = np.zeros(n + 1, dtype=np.int64)
tot = 0
for k in range(n):
    ref[k] = tot
    tot += len(S[k])
ref[n] = tot
print("ref S_p", ref, "tot", tot)
ok = np.array_equal(S_p, ref)
print("columns match:", ok)
if not ok:
    for k in range(n):
        got_c = set(int(x) for x in S_i[S_p[k]:S_p[k + 1]])
        if got_c != S[k]:
            print("  col", k, "got", sorted(got_c), "ref", sorted(S[k]))
    raise SystemExit(1)
for k in range(n):
    for q in range(R_p[k], R_p[k + 1]):
        j = int(R_i[q])
        pos = int(R_pos[q])
        if int(S_i[pos]) != k or not (0 <= pos < S_p[j + 1] and S_p[j] <= pos):
            print("  bad row entry", k, j, pos, "row there:", S_i[pos] if 0 <= pos < nnzL else None)
            raise SystemExit(1)
print("row index ok")

Lx = np.zeros(nnzL, dtype=np.float64)
D = np.zeros(n, dtype=np.float64)
Y = np.zeros(n, dtype=np.float64)
W = np.zeros(max(int((R_p[1:] - R_p[:-1]).max()), 1), dtype=np.float64)
L.mpp3d_ldl_numeric(
    n, nnzL, addr(Ap), addr(Ai), addr(Ax), addr(S_p), addr(S_i),
    addr(R_p), addr(R_i), addr(R_pos), addr(Lx), addr(D), addr(Y), addr(W),
)
print("numeric ok")
B = rng.normal(size=(n, 2))
Bp = np.ascontiguousarray(B[np.argsort(np.argsort(np.arange(n)))])
X = np.zeros((n, 2))
L.mpp3d_ldl_solve(n, addr(S_p), addr(S_i), addr(Lx), addr(D), addr(B), addr(X), 2)
print("solve max err", np.abs(X - np.linalg.solve(A, B)).max())
