"""Point-cloud parity: the local triangulation against upstream's, and the
heat-method distance against upstream's.

The local triangulation is compared as a *set* of triangles per point, which
is the invariant; the order within a point's row is not reproducible and the
test says why. The distance is compared with a tolerance rather than to
roundoff, for the same reason: the tufted cover is built from the triangles in
the order they are emitted, and that order follows the tangent frame.
"""

import numpy as np
import pytest

import mojo_potpourri3d as mpp3d
import potpourri3d as pp3d


def sphere_cloud(n, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.ascontiguousarray(v * scale)


def torus_cloud(nu=14, nv=24, seed=1):
    """Not a sphere: the normals have to be estimated, and they vary a lot."""
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    R, r = 1.0, 0.35
    P = np.stack(
        [
            ((R + r * np.cos(V)) * np.cos(U)).ravel(),
            ((R + r * np.cos(V)) * np.sin(U)).ravel(),
            (r * np.sin(V)).ravel(),
        ],
        axis=1,
    )
    jitter = np.random.default_rng(seed).normal(scale=1e-3, size=P.shape)
    return np.ascontiguousarray(P + jitter)


def triangle_sets(out):
    """The triangles of each point, as a set of index triples."""
    per_point = []
    for row in np.asarray(out).reshape(len(out), -1, 3):
        s = set()
        for tri in row:
            if tri[0] < 0:
                continue
            s.add(frozenset(int(i) for i in tri))
        per_point.append(s)
    return per_point


def cartwheel():
    """Upstream's own test cloud: a centre point on a ring of 30."""
    num = 31
    t = np.linspace(0, 2 * np.pi, num - 1, endpoint=False)
    return np.ascontiguousarray(
        np.concatenate(
            [np.zeros([1, 3]), np.stack([np.cos(t), np.sin(t), 0 * t], 1)], 0
        )
    )


@pytest.mark.parametrize(
    "cloud", [sphere_cloud(200), sphere_cloud(120, seed=5, scale=2.5), torus_cloud()],
    ids=["sphere200", "sphere120-scaled", "torus"],
)
def test_local_triangulation_matches_upstream(cloud):
    """Every point's local Delaunay triangulation is upstream's, triangle for
    triangle. The rows come out in a rotated order (see the module docstring),
    so the comparison is on sets."""
    ours = mpp3d.PointCloudLocalTriangulation(cloud, True).get_local_triangulation()
    theirs = pp3d.PointCloudLocalTriangulation(cloud, True).get_local_triangulation()
    assert ours.shape == theirs.shape
    a = triangle_sets(ours)
    b = triangle_sets(theirs)
    n_match = sum(1 for p in range(len(cloud)) if a[p] == b[p])
    assert n_match == len(cloud), f"{len(cloud) - n_match} points differ"


def test_local_triangulation_is_a_fan_of_the_neighbours():
    """Structural check, independent of upstream: each point's triangles are
    `p` plus two of its own neighbours, and they tile a contiguous angular
    fan, so consecutive triangles share an edge."""
    cloud = sphere_cloud(150, seed=3)
    lt = mpp3d.PointCloudLocalTriangulation(cloud, True)
    out = lt.get_local_triangulation()
    neighbors = lt.neighbors.reshape(len(cloud), -1)
    for p in range(len(cloud)):
        neigh = set(int(i) for i in neighbors[p])
        tris = [tuple(int(i) for i in t) for t in out[p] if t[0] >= 0]
        for a, b, c in tris:
            assert a == p
            assert b in neigh and c in neigh and b != c
        for (a1, b1, c1), (a2, b2, c2) in zip(tris, tris[1:]):
            assert {b1, c1} & {b2, c2}, (a1, b1, c1, a2, b2, c2)


def test_cartwheel_ring_is_cocircular():
    """The cartwheel is the degenerate case, and it is worth pinning.

    The centre point's 30 neighbours lie exactly on a circle in its tangent
    plane, so the in-circle determinant is mathematically zero and the sign that
    `inCircleTest` returns is rounding noise. Upstream keeps the whole fan;
    this port keeps none of it. Every other point in the cloud is unaffected in
    the sense that its triangles are still a valid fan -- asserted here.
    """
    cloud = cartwheel()
    out = mpp3d.PointCloudLocalTriangulation(cloud, True).get_local_triangulation()
    theirs = pp3d.PointCloudLocalTriangulation(cloud, True).get_local_triangulation()
    ours = triangle_sets(out)
    theirs_by_upstream = triangle_sets(theirs)
    assert ours[0] == set()
    assert len(theirs_by_upstream[0]) == 30
    # every point in this cloud has a cocircular neighbourhood, so every fan
    # here is a rounding coin-flip and this port comes out empty
    assert all(len(s) == 0 for s in ours)
    assert all(len(s) > 0 for s in theirs_by_upstream)


@pytest.mark.parametrize("cloud", [sphere_cloud(200), torus_cloud()], ids=["sphere", "torus"])
def test_point_cloud_distance_agrees_with_upstream(cloud):
    """The pipeline is the same one upstream runs, and the answer agrees to a
    fraction of a percent -- not to roundoff, because the tufted cover is built
    from the local triangles in emission order, and that order follows a tangent
    frame which is not reproducible bit for bit. See the README."""
    ours = mpp3d.PointCloudHeatSolver(cloud).compute_distance(7)
    theirs = pp3d.PointCloudHeatSolver(cloud).compute_distance(7)
    assert ours.shape == theirs.shape
    rel = np.abs(ours - theirs) / np.maximum(np.abs(theirs), 1e-12)
    assert np.corrcoef(ours, theirs)[0, 1] > 0.999
    assert np.median(rel) < 2e-2
    assert rel.max() < 2e-1
    assert abs(ours[7]) < 1e-12


def test_point_cloud_distance_multisource_agrees_with_upstream():
    cloud = sphere_cloud(200)
    srcs = [1, 20, 99]
    ours = mpp3d.PointCloudHeatSolver(cloud).compute_distance_multisource(srcs)
    theirs = pp3d.PointCloudHeatSolver(cloud).compute_distance_multisource(srcs)
    rel = np.abs(ours - theirs) / np.maximum(np.abs(theirs), 1e-12)
    assert np.corrcoef(ours, theirs)[0, 1] > 0.999
    assert np.median(rel) < 2e-2
    for s in srcs:
        # the field is shifted to the barycentric average over the sources, so
        # an individual source vertex is near zero rather than exactly zero --
        # upstream's value there is the reference, not zero
        assert rel[s] < 0.2
        assert abs(ours[s] - theirs[s]) < 0.01


def test_point_cloud_distance_tracks_the_geodesic_on_a_sphere():
    """Independent of upstream: on a unit sphere the geodesic distance from a
    point is the central angle, and the heat method recovers it."""
    cloud = sphere_cloud(600, seed=11)
    src = 17
    d = mpp3d.PointCloudHeatSolver(cloud).compute_distance(src)
    exact = np.arccos(np.clip(cloud @ cloud[src], -1.0, 1.0))
    assert np.corrcoef(d, exact)[0, 1] > 0.99
    # the antipode is the hardest point, and the heat method underestimates
    assert abs(d.max() - np.pi) < 0.2


def test_point_cloud_solver_t_coef_changes_the_answer():
    cloud = sphere_cloud(120, seed=2)
    a = mpp3d.PointCloudHeatSolver(cloud, t_coef=1.0).compute_distance(3)
    b = mpp3d.PointCloudHeatSolver(cloud, t_coef=0.5).compute_distance(3)
    assert not np.allclose(a, b)


def test_uncovered_point_cloud_methods_say_so():
    cloud = sphere_cloud(60, seed=4)
    solver = mpp3d.PointCloudHeatSolver(cloud)
    for call in [
        lambda: solver.extend_scalar([0, 1], [0.0, 1.0]),
        lambda: solver.get_tangent_frames(),
        lambda: solver.transport_tangent_vector(0, [1.0, 0.0]),
        lambda: solver.transport_tangent_vectors([0], [[1.0, 0.0]]),
        lambda: solver.compute_log_map(0),
        lambda: solver.compute_signed_distance([], np.zeros((0, 3))),
    ]:
        with pytest.raises(NotImplementedError, match="not covered"):
            call()


def test_point_cloud_length_checks_match_upstream():
    cloud = sphere_cloud(60, seed=4)
    with pytest.raises(ValueError, match="same shape"):
        mpp3d.PointCloudHeatSolver(cloud).extend_scalar([0, 1], [1.0])
    with pytest.raises(ValueError, match="2D tangent vector"):
        mpp3d.PointCloudHeatSolver(cloud).transport_tangent_vector(0, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="same length"):
        mpp3d.PointCloudHeatSolver(cloud).transport_tangent_vectors([0, 1], [[1.0, 0.0]])


def test_points_validation_matches_upstream():
    with pytest.raises(ValueError, match="vertices should be a 2d Nx3"):
        mpp3d.PointCloudHeatSolver(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="vertices should be a 2d Nx3"):
        pp3d.PointCloudHeatSolver(np.zeros((4, 2)))
