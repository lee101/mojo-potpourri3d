"""Benchmarks against upstream potpourri3d. Prints a markdown table.

Run with `pixi run bench`: the pixi task holds a machine-wide flock so that concurrent factory
jobs cannot distort the numbers.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import potpourri3d as pp3d  # noqa: E402

import conftest  # noqa: E402
from mojo_potpourri3d import mesh as mm  # noqa: E402
from mojo_potpourri3d import point_cloud as pc  # noqa: E402

REPEATS = 5


def timeit(fn, repeats=REPEATS):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def row(name, n, ours, theirs, unit="ms"):
    ratio = ours / theirs if theirs > 0 else float("nan")
    return (f"| {name} | {n} | {ours:.3f} | {theirs:.3f} | {ratio:.2f}x |")


def environment():
    """The machine the numbers were taken on, so a reader can judge them."""
    import platform
    import re

    cpu = "unknown CPU"
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
        m = re.search(r"CPU MHz\s*:\s*([\d.]+)", text)
        if m:
            cpu += f" @ {float(m.group(1)) / 1000:.2f}GHz"
    except OSError:
        pass
    return (
        f"{platform.system()} {platform.release()} {platform.machine()}, {cpu}, "
        f"{os.cpu_count()} cores"
    )


def main():
    print(f"## Environment\n\n```\n{environment()}\n```\n")
    meshes = [
        ("icosphere(1)", conftest.icosphere(1)),
        ("icosphere(2)", conftest.icosphere(2)),
        ("icosphere(3)", conftest.icosphere(3)),
        ("open grid 32", conftest.open_grid(32)),
        ("open grid 64", conftest.open_grid(64)),
    ]

    for robust, title in ((False, "Heat method distance (use_robust=False)"),
                          (True, "Heat method distance (use_robust=True, upstream's default)")):
        print(f"## {title}\n")
        print("| mesh | vertices | mojo (ms) | upstream (ms) | ours / upstream |")
        print("|---|---|---|---|---|")
        for name, (V, F) in meshes:
            n = V.shape[0]

            def up():
                pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=robust).compute_distance(0)

            def our():
                mm.MeshHeatMethodDistanceSolver(V, F, use_robust=robust).compute_distance(0)

            ours = timeit(our) * 1e3
            print(row(name, n, ours, timeit(up) * 1e3))

    print("\n## Multi-source distance (three sources, use_robust=False)\n")
    print("| mesh | vertices | mojo (ms) | upstream (ms) | ours / upstream |")
    print("|---|---|---|---|---|")
    for name, (V, F) in meshes:
        n = V.shape[0]
        srcs = [0, 1, 2]
        print(
            row(
                name,
                n,
                timeit(lambda: mm.MeshHeatMethodDistanceSolver(V, F, use_robust=False)
                       .compute_distance_multisource(srcs)) * 1e3,
                timeit(lambda: pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False)
                       .compute_distance_multisource(srcs)) * 1e3,
            )
        )

    print("\n## Setup only (operator build + factorisation, no query)\n")
    print("| mesh | vertices | mojo (ms) | upstream (ms) | ours / upstream |")
    print("|---|---|---|---|---|")
    for name, (V, F) in meshes[:3]:
        n = V.shape[0]
        print(
            row(
                name,
                n,
                timeit(lambda: mm.MeshHeatMethodDistanceSolver(V, F, use_robust=False)) * 1e3,
                timeit(lambda: pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False)) * 1e3,
            )
        )

    print("\n## Mesh operators\n")
    print("| op | mesh | mojo (ms) | upstream (ms) | ours / upstream |")
    print("|---|---|---|---|---|")
    for name, (V, F) in meshes[:3]:
        n = V.shape[0]
        for op, our, up in [
            ("cotan_laplacian", lambda: mm.cotan_laplacian(V, F), lambda: pp3d.cotan_laplacian(V, F)),
            ("face_areas", lambda: mm.face_areas(V, F), lambda: pp3d.face_areas(V, F)),
            ("vertex_areas", lambda: mm.vertex_areas(V, F), lambda: pp3d.vertex_areas(V, F)),
        ]:
            print(row(f"{op} / {name}", n, timeit(our) * 1e3, timeit(up) * 1e3))

    print("\n## Point cloud local triangulation and tangent frames\n")
    print("| cloud | points | local triangulation (ms) | upstream (ms) | ours / upstream |")
    print("|---|---|---|---|---|")
    for n in (300, 1000, 3000):
        rng = np.random.default_rng(0)
        P = np.ascontiguousarray(rng.standard_normal((n, 3)))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        ours = timeit(lambda: pc.PointCloudLocalTriangulation(P).get_local_triangulation()) * 1e3
        theirs = timeit(
            lambda: pp3d.PointCloudLocalTriangulation(P).get_local_triangulation()) * 1e3
        print(row(f"random sphere", n, ours, theirs))

    print("\n## Sparse Cholesky (5-point Laplacian, heat operator shape)\n")
    print("| grid | vertices | nnz(L) | factorise (ms) | solve (ms) |")
    print("|---|---|---|---|---|")
    import ctypes

    import scipy.sparse as sp

    dll = ctypes.CDLL(str(Path(__file__).resolve().parents[1] / "dist" / "libmojo-potpourri3d.so"))
    dll.pp3d_chol_analyze.argtypes = [ctypes.c_int64] * 2 + [ctypes.c_void_p] * 4
    dll.pp3d_chol_analyze.restype = ctypes.c_int64
    dll.pp3d_chol_factorize.argtypes = [ctypes.c_int64] * 2 + [ctypes.c_void_p] * 8
    dll.pp3d_chol_factorize.restype = ctypes.c_int64
    dll.pp3d_chol_solve.argtypes = [ctypes.c_int64] * 2 + [ctypes.c_void_p] * 7
    dll.pp3d_chol_solve.restype = ctypes.c_int64

    for k in (20, 40, 60):
        n = k * k
        A = np.zeros((n, n))
        idx = lambda i, j: (i % k) * k + (j % k)  # noqa: E731
        for i in range(k):
            for j in range(k):
                A[idx(i, j), idx(i, j)] = 4
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    A[idx(i, j), idx(i + di, j + dj)] = -1
        A += 0.5 * np.eye(n)
        U = sp.triu(sp.csc_matrix(A)).tocsc()
        ap = np.zeros(n + 1, dtype=np.int64)
        ai, ax = [], []
        for j in range(n):
            for p in range(U.indptr[j], U.indptr[j + 1]):
                ai.append(U.indices[p])
                ax.append(U.data[p])
            ap[j + 1] = len(ai)
        ap = np.ascontiguousarray(ap)
        ai = np.array(ai, dtype=np.int64)
        ax = np.array(ax, dtype=np.float64)
        perm = np.zeros(n, dtype=np.int64)
        nnzl = dll.pp3d_chol_analyze(n, 1, ap.ctypes.data, ai.ctypes.data, ax.ctypes.data, perm.ctypes.data)
        lp = np.zeros(n + 1, dtype=np.int64)
        li = np.zeros(max(nnzl, 1), dtype=np.int64)
        lx = np.zeros(max(nnzl, 1), dtype=np.float64)
        dx = np.zeros(n, dtype=np.float64)
        b = np.random.RandomState(0).rand(n)
        x = np.zeros(n)

        def fact():
            dll.pp3d_chol_factorize(
                n, 1, ap.ctypes.data, ai.ctypes.data, ax.ctypes.data, perm.ctypes.data,
                lp.ctypes.data, li.ctypes.data, lx.ctypes.data, dx.ctypes.data)

        def solve():
            dll.pp3d_chol_solve(
                n, 1, lp.ctypes.data, li.ctypes.data, lx.ctypes.data, dx.ctypes.data,
                perm.ctypes.data, b.ctypes.data, x.ctypes.data)

        print(f"| {k}x{k} | {n} | {nnzl} | {timeit(fact)*1e3:.3f} | {timeit(solve)*1e3:.3f} |")


if __name__ == "__main__":
    main()
