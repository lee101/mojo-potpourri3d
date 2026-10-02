import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def icosphere(sub):
    """Subdivided icosahedron, the standard closed test mesh."""
    t = (1.0 + 5.0**0.5) / 2.0
    V = np.array(
        [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
         [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]],
        dtype=np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array(
        [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
         [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
         [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], dtype=np.int64)
    for _ in range(sub):
        new_v, new_f, mid = [], [], {}

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in mid:
                p = V[a] + V[b]
                new_v.append(p / np.linalg.norm(p))
                mid[key] = len(V) + len(new_v) - 1
            return mid[key]

        for f in F:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_f += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        V = np.vstack([V, np.array(new_v)])
        F = np.array(new_f, dtype=np.int64)
    return np.ascontiguousarray(V), np.ascontiguousarray(F)


def is_edge_manifold(F) -> bool:
    """Every edge of a closed triangulation must be shared by exactly two faces."""
    from collections import Counter

    c = Counter()
    for f in F:
        for k in range(3):
            a, b = int(f[k]), int(f[(k + 1) % 3])
            c[(min(a, b), max(a, b))] += 1
    return all(v == 2 for v in c.values())


def open_grid(k):
    """Flat k-by-k grid, z = 0, with free boundary."""
    xs = np.linspace(0, 1, k)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), np.zeros(k * k)], axis=1)
    idx = lambda i, j: i * k + j  # noqa: E731
    F = []
    for i in range(k - 1):
        for j in range(k - 1):
            F.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)])
            F.append([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)])
    return np.ascontiguousarray(V), np.ascontiguousarray(np.array(F, dtype=np.int64))


@pytest.fixture(scope="session")
def ico2():
    return icosphere(2)


@pytest.fixture(scope="session")
def grid8():
    return open_grid(8)


def upstream():
    return pytest.importorskip("potpourri3d")
