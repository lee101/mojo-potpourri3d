"""mojo-potpourri3d against potpourri3d on the same inputs.

    pixi run bench

Prints a markdown table. Always run it through the pixi task: the task holds a
machine-wide flock so a concurrent factory job cannot distort the numbers.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import functools

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))

import mojo_potpourri3d as ours  # noqa: E402
import potpourri3d as theirs  # noqa: E402
from conftest import icosphere  # noqa: E402


@functools.lru_cache(maxsize=None)
def mesh(sub: int):
    return icosphere(sub)


def timeit(fn, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn
    return deco


@case("face_areas (icosphere, 2562 v)")
def _():
    V, F = mesh(4)
    return (
        lambda: ours.face_areas(V, F),
        lambda: theirs.face_areas(V, F),
    )


@case("vertex_areas (icosphere, 2562 v)")
def _():
    V, F = mesh(4)
    return (
        lambda: ours.vertex_areas(V, F),
        lambda: theirs.vertex_areas(V, F),
    )


@case("cotan_laplacian (icosphere, 2562 v)")
def _():
    V, F = mesh(4)
    return (
        lambda: ours.cotan_laplacian(V, F),
        lambda: theirs.cotan_laplacian(V, F),
    )


@case("Heat method: construct (icosphere, 2562 v)")
def _():
    V, F = mesh(4)
    return (
        lambda: ours.MeshHeatMethodDistanceSolver(V, F, use_robust=False),
        lambda: theirs.MeshHeatMethodDistanceSolver(V, F, use_robust=False),
    )


@case("Heat method: 1 distance query (icosphere, 2562 v)")
def _():
    V, F = mesh(4)
    a = ours.MeshHeatMethodDistanceSolver(V, F, use_robust=False)
    b = theirs.MeshHeatMethodDistanceSolver(V, F, use_robust=False)
    return (
        lambda: a.compute_distance(0),
        lambda: b.compute_distance(0),
    )


@case("Heat method: 32 distance queries (icosphere, 2562 v)")
def _():
    V, F = mesh(4)
    a = ours.MeshHeatMethodDistanceSolver(V, F, use_robust=False)
    b = theirs.MeshHeatMethodDistanceSolver(V, F, use_robust=False)
    srcs = np.arange(32, dtype=np.int64)
    return (
        lambda: [a.compute_distance(s) for s in srcs],
        lambda: [b.compute_distance(s) for s in srcs],
    )


@case("Heat method: construct, robust Laplacian (icosphere, 2562 v)")
def _():
    V, F = mesh(4)
    return (
        lambda: ours.MeshHeatMethodDistanceSolver(V, F, use_robust=True),
        lambda: theirs.MeshHeatMethodDistanceSolver(V, F, use_robust=True),
    )


@case("Vector heat: construct (icosphere, 2562 v)")
def _():
    V, F = mesh(5)
    return (
        lambda: ours.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False),
        lambda: theirs.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False),
    )


@case("Vector heat: extend_scalar (icosphere, 2562 v)")
def _():
    V, F = mesh(4)
    a = ours.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    b = theirs.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    return (
        lambda: a.extend_scalar([0, 100, 400], [1.0, 2.0, 3.0]),
        lambda: b.extend_scalar([0, 100, 400], [1.0, 2.0, 3.0]),
    )


@case("Vector heat: transport_tangent_vectors (icosphere, 2562 v)")
def _():
    V, F = mesh(4)
    a = ours.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    b = theirs.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    inds = [0, 100, 400]
    vecs = [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]]
    return (
        lambda: a.transport_tangent_vectors(inds, vecs),
        lambda: b.transport_tangent_vectors(inds, vecs),
    )


@case("get_connection_laplacian (icosphere, 2562 v)")
def _():
    V, F = mesh(4)
    a = ours.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    b = theirs.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    return (
        lambda: a.get_connection_laplacian(),
        lambda: b.get_connection_laplacian(),
    )


@case("Heat method: construct (icosphere, 10242 v)")
def _():
    V, F = mesh(5)
    return (
        lambda: ours.MeshHeatMethodDistanceSolver(V, F, use_robust=False),
        lambda: theirs.MeshHeatMethodDistanceSolver(V, F, use_robust=False),
    )


@case("Heat method: 1 distance query (icosphere, 10242 v)")
def _():
    V, F = mesh(5)
    a = ours.MeshHeatMethodDistanceSolver(V, F, use_robust=False)
    b = theirs.MeshHeatMethodDistanceSolver(V, F, use_robust=False)
    return (
        lambda: a.compute_distance(0),
        lambda: b.compute_distance(0),
    )


@case("Vector heat: 1 transport query (icosphere, 10242 v)")
def _():
    V, F = mesh(5)
    a = ours.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    b = theirs.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    return (
        lambda: a.transport_tangent_vector(0, [1.0, 0.0]),
        lambda: b.transport_tangent_vector(0, [1.0, 0.0]),
    )


@functools.lru_cache(maxsize=None)
def cloud(n: int):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.ascontiguousarray(v)


@case("Point cloud: local triangulation (2000 points)")
def _():
    P = cloud(2000)
    return (
        lambda: ours.PointCloudLocalTriangulation(P, True).get_local_triangulation(),
        lambda: theirs.PointCloudLocalTriangulation(P, True).get_local_triangulation(),
    )


@case("Point cloud: construct (2000 points)")
def _():
    P = cloud(2000)
    return (
        lambda: ours.PointCloudHeatSolver(P),
        lambda: theirs.PointCloudHeatSolver(P),
    )


@case("Point cloud: 1 distance query (2000 points)")
def _():
    P = cloud(2000)
    a = ours.PointCloudHeatSolver(P)
    b = theirs.PointCloudHeatSolver(P)
    return (
        lambda: a.compute_distance(0),
        lambda: b.compute_distance(0),
    )


def main() -> None:
    print(f"machine: {platform.platform()}", flush=True)
    print(f"python:  {platform.python_version()}  numpy: {np.__version__}")
    print()
    print("| case | mojo-potpourri3d | potpourri3d (geometry-central) | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, maker in CASES:
        f_ours, f_theirs = maker()
        repeat = 1 if "10242" in name or "2000 points" in name else 3
        f_ours()  # warm the library build and any lazy imports
        t_ours = timeit(f_ours, repeat)
        t_theirs = timeit(f_theirs, repeat)
        speed = t_theirs / t_ours if t_ours > 0 else math.inf
        print(
            f"| {name} | {t_ours*1e3:.2f} ms | {t_theirs*1e3:.2f} ms | {speed:.2f}x |",
            flush=True,
        )


if __name__ == "__main__":
    main()
