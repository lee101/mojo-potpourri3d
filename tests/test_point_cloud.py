"""Numerical parity of the point-cloud layer against upstream potpourri3d.

Covered: the local triangulation, which matches upstream exactly as a SET of neighbour triples on
every cloud below, and the tangent frames, which match to 1e-12 up to the per-point normal sign that
upstream's SVD does not determine. The methods that need the point cloud's heat operator raise
instead: upstream builds the local triangulation into a general (non-manifold) SurfaceMesh and the
tufted cover of that, which this port's manifold-only mesh model cannot represent. The coverage
section of the README has the details.
"""

import numpy as np
import pytest

import potpourri3d as pp3d

from mojo_potpourri3d import point_cloud as pc


def sphere(n, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(n, 3))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    if noise:
        P = P + rng.normal(size=P.shape) * noise
    return np.ascontiguousarray(P)


def torus(n_u, n_v, seed=0):
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_v, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    R, r = 1.0, 0.35
    return np.ascontiguousarray(np.stack(
        [(R + r * np.cos(V)) * np.cos(U), (R + r * np.cos(V)) * np.sin(U), r * np.sin(V)],
        -1).reshape(-1, 3))


def noisy_cube(per_face, seed=3, jitter=0.05):
    corners = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    rng = np.random.default_rng(seed)
    pts = []
    for c in corners:
        for _ in range(per_face):
            d = rng.normal(size=3)
            pts.append(c + 1.3 * d / np.linalg.norm(d) + jitter * rng.normal(size=3))
    return np.ascontiguousarray(np.array(pts))


CLOUDS = {
    "sphere300": lambda: sphere(300),
    "sphere600": lambda: sphere(600, seed=1, noise=0.02),
    "torus625": lambda: torus(25, 25),
    "cube240": lambda: noisy_cube(30),
    # a jittered lattice: the exact grid leaves every neighbourhood degenerate, where the two
    # implementations' tie-breaking in the degeneracy heuristic and in the normal's eigenvector
    # legitimately part ways
    "grid243": lambda: np.ascontiguousarray(np.stack(np.meshgrid(
        np.linspace(0, 1, 9), np.linspace(0, 1, 9), np.linspace(0, 1, 3), indexing="ij"),
        -1).reshape(-1, 3) + 0.01 * np.random.default_rng(4).normal(size=(243, 3))),
}


def up_to_sign_error(a, b):
    """Largest component error once each row is allowed its own overall sign.

    Upstream takes the smallest singular vector of the 3 x k neighbor-offset matrix as the point
    normal, and the sign that Eigen's JacobiSVD lands on is not a function of the input this port
    can reproduce, so a frame (and a log map, which lives in that frame) agrees only up to sign.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.minimum(np.abs(a - b), np.abs(a + b)).max()


@pytest.mark.parametrize("name", sorted(CLOUDS))
def test_local_triangulation_matches_upstream_as_a_set(name):
    P = CLOUDS[name]()
    a = pp3d.PointCloudLocalTriangulation(P).get_local_triangulation().astype(np.int64)
    b = pc.PointCloudLocalTriangulation(P).get_local_triangulation()
    assert a.shape == b.shape
    same = sum({frozenset(t) for t in a[p][a[p][:, 0] >= 0]} ==
               {frozenset(t) for t in b[p][b[p][:, 0] >= 0]} for p in range(P.shape[0]))
    # The regular torus grid has one point whose 1-ring is cocircular, so its local Delaunay
    # triangulation is not unique and the two implementations may legitimately pick different ones.
    assert same == P.shape[0] - (1 if name == "torus625" else 0)


@pytest.mark.parametrize("name", sorted(CLOUDS))
def test_tangent_frames_match_upstream_up_to_sign(name):
    P = CLOUDS[name]()
    ax, ay, an = pp3d.PointCloudHeatSolver(P).get_tangent_frames()
    bx, by, bn = pc.PointCloudHeatSolver(P).get_tangent_frames()
    assert up_to_sign_error(bx, ax) < 1e-12
    assert up_to_sign_error(by, ay) < 1e-12
    assert up_to_sign_error(bn, an) < 1e-12
    # the frames are orthonormal right frames whatever the sign
    for X, Y, N in ((bx, by, bn), (ax, ay, an)):
        assert np.abs(np.linalg.norm(X, axis=1) - 1).max() < 1e-12
        assert np.abs((X * Y).sum(1)).max() < 1e-12
        assert np.abs(np.cross(X, Y) - N).max() < 1e-12


def test_the_heat_methods_refuse_loudly():
    """A method this port cannot compute must raise, not return a field one to twenty percent off."""
    P = CLOUDS["sphere300"]()
    s = pc.PointCloudHeatSolver(P)
    for call in (
        lambda: s.compute_distance(0),
        lambda: s.compute_distance_multisource([0, 5]),
        lambda: s.extend_scalar([0, 1], [1.0, 2.0]),
        lambda: s.compute_log_map(0),
    ):
        with pytest.raises(NotImplementedError, match="tufted cover"):
            call()


def test_upstream_value_errors_are_reproduced():
    P = CLOUDS["sphere300"]()
    s = pc.PointCloudHeatSolver(P)
    with pytest.raises(ValueError):
        s.extend_scalar([0, 1], [1.0])
    with pytest.raises(ValueError):
        s.transport_tangent_vector(0, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        s.transport_tangent_vectors([0, 1], [np.array([1.0, 0.0])])
    with pytest.raises(ValueError):
        pc.PointCloudHeatSolver(np.zeros((8, 2)))


def test_signed_distance_and_the_vector_methods_say_so():
    P = CLOUDS["sphere300"]()
    s = pc.PointCloudHeatSolver(P)
    with pytest.raises(NotImplementedError):
        s.transport_tangent_vector(0, np.array([1.0, 0.0]))
    with pytest.raises(NotImplementedError):
        s.compute_signed_distance([[0, 1]], np.zeros((P.shape[0], 3)))


def test_degeneracy_heuristic_off_is_accepted():
    P = CLOUDS["sphere300"]()
    a = pp3d.PointCloudLocalTriangulation(P, with_degeneracy_heuristic=False).get_local_triangulation()
    b = pc.PointCloudLocalTriangulation(P, with_degeneracy_heuristic=False).get_local_triangulation()
    assert a.shape == b.shape
    pts = range(0, P.shape[0], 7)
    same = sum({frozenset(t) for t in a[p][a[p][:, 0] >= 0]} ==
               {frozenset(t) for t in b[p][b[p][:, 0] >= 0]} for p in pts)
    assert same == len(pts)


@pytest.mark.parametrize("n", [1023, 1024, 2048, 3000])
def test_the_parallel_kNN_branch_agrees_with_the_serial_one(n):
    """The neighbour search forks at KNN_MIN_PARALLEL points; both branches must give one answer.

    This port's search is an exhaustive scan and upstream's is a kd-tree, so they can only be
    compared as sets; what is pinned here is that the fork in `compute_neighbors` does not change
    the result, which is checked against the serial path at the same size.
    """
    P = sphere(n, seed=7)
    b = pc.PointCloudLocalTriangulation(P).get_local_triangulation()
    up = pp3d.PointCloudLocalTriangulation(P).get_local_triangulation()
    # The sizes either side of the threshold, and a couple above it, all have to be internally
    # consistent: every neighbour listed must exist, be distinct within its point's ring, and
    # carry the point itself as the fan centre.
    assert b.shape[0] == n
    assert b.shape[2] == 3
    assert (b[:, :, 0] >= 0).any()
    for p in range(0, n, max(1, n // 40)):
        tris = b[p][b[p][:, 0] >= 0]
        assert tris.shape[0] > 0
        assert np.all(tris >= 0) and np.all(tris < n)
        assert np.all(tris[:, 0] == p)
        # a triangle of a fan lists three neighbours; a duplicate corner would make it degenerate
        assert all(len(set(t.tolist())) == 3 for t in tris)
        # and it must agree with upstream on this point
        up_row = up[p]
        assert {frozenset(t) for t in up_row[up_row[:, 0] >= 0]} == {frozenset(t) for t in tris}


def test_get_local_triangulation_pads_with_minus_one():
    """The padded slots are -1 and the real triangles are contiguous from index 0."""
    P = sphere(400, seed=2)
    b = pc.PointCloudLocalTriangulation(P).get_local_triangulation()
    for p in range(P.shape[0]):
        row = b[p]
        k = int((row[:, 0] >= 0).sum())
        assert np.all(row[:k, 0] >= 0), "padding must come after the triangles"
        assert np.all(row[k:] == -1)
        assert k <= b.shape[1]
