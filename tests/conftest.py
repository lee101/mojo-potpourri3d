import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import potpourri3d as pp3d  # noqa: E402  (the reference implementation)
import mojo_potpourri3d as mpp3d  # noqa: E402


def grid_mesh(n: int):
    """A flat n x n vertex grid, two triangles per cell. Has a boundary."""
    xs = np.linspace(0, 1, n)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(n * n)], axis=1)
    idx = np.arange(n * n).reshape(n, n)
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            tris.append([idx[i, j], idx[i + 1, j], idx[i + 1, j + 1]])
            tris.append([idx[i, j], idx[i + 1, j + 1], idx[i, j + 1]])
    return np.ascontiguousarray(V), np.ascontiguousarray(np.array(tris, dtype=np.int64))


def icosphere(sub: int = 2):
    """A closed, manifold, well-conditioned mesh: every edge has two faces."""
    t = (1 + 5 ** 0.5) / 2
    V = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
            [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
            [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
            [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
            [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    for _ in range(sub):
        cache = {}
        nf = []
        verts = list(V)

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key in cache:
                return cache[key]
            p = (verts[a] + verts[b]) / 2
            p /= np.linalg.norm(p)
            verts.append(p)
            cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        F = np.array(nf, dtype=np.int64)
        V = np.array(verts)
    return np.ascontiguousarray(V), np.ascontiguousarray(F)


@pytest.fixture(scope="session")
def closed_mesh():
    return icosphere(2)


@pytest.fixture(scope="session")
def boundary_mesh():
    return grid_mesh(10)


def rel_err(a, b):
    scale = max(np.abs(np.asarray(b)).max(), 1e-30)
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / scale)
