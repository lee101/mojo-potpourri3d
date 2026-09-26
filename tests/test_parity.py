"""Parity against the real potpourri3d (== geometry-central), not just "it ran".

Every assertion here is a numerical comparison with the upstream package on the
same input, at a tolerance tight enough that a wrong kernel cannot pass.
"""

import numpy as np
import pytest
import scipy.sparse

import mojo_potpourri3d as mpp3d
import potpourri3d as pp3d
from mojo_potpourri3d._lib import f64, i64

from conftest import grid_mesh, icosphere, rel_err


# ------------------------------------------------------------------ geometry

@pytest.mark.parametrize("n", [5, 12, 25])
def test_face_areas_match(n):
    V, F = grid_mesh(n)
    assert np.array_equal(mpp3d.face_areas(V, F), pp3d.face_areas(V, F))


@pytest.mark.parametrize("n", [5, 12, 25])
def test_vertex_areas_match(n):
    V, F = grid_mesh(n)
    assert rel_err(mpp3d.vertex_areas(V, F), pp3d.vertex_areas(V, F)) < 1e-14


@pytest.mark.parametrize("n", [5, 12, 25])
@pytest.mark.parametrize("denom_eps", [0.0, 1e-3])
def test_cotan_laplacian_matches(n, denom_eps):
    V, F = grid_mesh(n)
    ours = mpp3d.cotan_laplacian(V, F, denom_eps)
    theirs = pp3d.cotan_laplacian(V, F, denom_eps)
    assert ours.shape == theirs.shape
    assert ours.nnz == theirs.nnz
    assert rel_err(ours.toarray(), theirs.toarray()) < 1e-13


def test_cotan_laplacian_has_constant_nullspace(closed_mesh):
    V, F = closed_mesh
    L = mpp3d.cotan_laplacian(V, F)
    ones = np.ones(V.shape[0])
    assert np.abs(L @ ones).max() < 1e-12 * np.abs(L.toarray()).max()


def test_validation_errors_match_upstream():
    V, F = grid_mesh(4)
    with pytest.raises(ValueError, match="2d Nx3"):
        mpp3d.face_areas(V[:, :2], F)
    quads = np.ascontiguousarray(np.hstack([F, F[:, :1]]))
    with pytest.raises(ValueError, match="triangular"):
        mpp3d.cotan_laplacian(V, quads)
    with pytest.raises(ValueError, match="triangular"):
        mpp3d.MeshVectorHeatSolver(V, quads)
    with pytest.raises(ValueError, match="2d Nx3"):
        mpp3d.MeshHeatMethodDistanceSolver(V[:, :2], F)
    with pytest.raises(ValueError, match="2d NxD"):
        mpp3d.face_areas(V, np.ascontiguousarray(F[:, :2]))
    with pytest.raises(ValueError, match="out-of-bounds"):
        mpp3d.MeshHeatMethodDistanceSolver(
            V, np.ascontiguousarray(F + V.shape[0])
        )


# ------------------------------------------------------ heat method distance

def test_heat_distance_closed_mesh(closed_mesh):
    V, F = closed_mesh
    ours = mpp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance(0)
    theirs = pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance(0)
    assert rel_err(ours, theirs) < 1e-12


@pytest.mark.parametrize("src", [0, 1, 5, 40])
def test_heat_distance_boundary_mesh(src):
    V, F = grid_mesh(10)
    ours = mpp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance(src)
    theirs = pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance(src)
    assert rel_err(ours, theirs) < 1e-12


def test_heat_distance_is_zero_at_the_source(closed_mesh):
    V, F = closed_mesh
    d = mpp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance(7)
    assert abs(d[7]) < 1e-12


def test_heat_distance_multisource(closed_mesh):
    V, F = closed_mesh
    srcs = np.array([0, 1, 2, 40], dtype=np.int64)
    ours = mpp3d.MeshHeatMethodDistanceSolver(
        V, F, use_robust=False
    ).compute_distance_multisource(srcs)
    theirs = pp3d.MeshHeatMethodDistanceSolver(
        V, F, use_robust=False
    ).compute_distance_multisource(srcs)
    assert rel_err(ours, theirs) < 1e-12


def test_module_level_compute_distance(closed_mesh):
    """These take no `use_robust` argument, so both sides run the robust path
    upstream and the non-robust path here; they agree to the intrinsic-Delaunay
    preprocessing difference, not to roundoff."""
    V, F = closed_mesh
    assert rel_err(
        mpp3d.compute_distance(V, F, 3), pp3d.compute_distance(V, F, 3)
    ) < 1e-4
    assert rel_err(
        mpp3d.compute_distance_multisource(V, F, [3, 9]),
        pp3d.compute_distance_multisource(V, F, [3, 9]),
    ) < 1e-4


def test_heat_distance_agrees_with_euclidean_on_a_sphere(closed_mesh):
    """A sanity check that does not depend on upstream at all."""
    V, F = icosphere(3)
    src = 0
    d = mpp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance(src)
    # The unit sphere's geodesic distance from `src` is the central angle.
    exact = np.arccos(np.clip(V @ V[src] / np.linalg.norm(V[src]), -1.0, 1.0))
    # The heat method is a first-order method; at this resolution it is a
    # couple of percent off the true central angle, in the expected direction.
    assert rel_err(d, exact) < 3e-2
    assert np.corrcoef(d, exact)[0, 1] > 0.99


def test_robust_laplacian_is_close(closed_mesh):
    """`use_robust=True` is the upstream default; we take the non-robust path."""
    V, F = closed_mesh
    robust = pp3d.MeshHeatMethodDistanceSolver(V, F).compute_distance(0)
    plain = mpp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance(0)
    assert rel_err(plain, robust) < 1e-3


def test_t_coef_changes_the_answer(closed_mesh):
    V, F = closed_mesh
    a = mpp3d.MeshHeatMethodDistanceSolver(V, F, t_coef=1.0, use_robust=False)
    b = mpp3d.MeshHeatMethodDistanceSolver(V, F, t_coef=0.5, use_robust=False)
    assert not np.allclose(a.compute_distance(0), b.compute_distance(0))


# --------------------------------------------------------- vector heat method

def test_connection_laplacian_matches(closed_mesh):
    V, F = closed_mesh
    ours = mpp3d.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False).get_connection_laplacian()
    theirs = pp3d.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False).get_connection_laplacian()
    assert ours.shape == theirs.shape
    assert rel_err(ours.toarray(), theirs.toarray()) < 1e-12


def test_tangent_frames_match(closed_mesh):
    V, F = closed_mesh
    ours = mpp3d.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False).get_tangent_frames()
    theirs = pp3d.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False).get_tangent_frames()
    for a, b in zip(ours, theirs):
        assert a.shape == b.shape
        assert rel_err(a, b) < 1e-12


def test_extend_scalar_matches(closed_mesh):
    V, F = closed_mesh
    for inds, vals in (([0], [1.0]), ([0, 5], [1.0, 2.0]), ([3, 11, 40], [0.5, -1.0, 2.5])):
        ours = mpp3d.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False).extend_scalar(inds, vals)
        theirs = pp3d.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False).extend_scalar(inds, vals)
        assert rel_err(ours, theirs) < 1e-12


def test_extend_scalar_reproduces_the_source_values(closed_mesh):
    V, F = closed_mesh
    inds = [0, 5]
    vals = [1.0, 2.0]
    out = mpp3d.MeshVectorHeatSolver(
        V, F, use_intrinsic_delaunay=False
    ).extend_scalar(inds, vals)
    for v, val in zip(inds, vals):
        assert abs(out[v] - val) < 0.2


def test_extend_scalar_length_check(closed_mesh):
    V, F = closed_mesh
    with pytest.raises(ValueError, match="same shape"):
        mpp3d.MeshVectorHeatSolver(V, F).extend_scalar([0, 1], [1.0])


def test_transport_tangent_vectors_match(closed_mesh):
    V, F = closed_mesh
    ours = mpp3d.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    theirs = pp3d.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    inds = [0, 5, 40]
    vecs = [[1.0, 0.0], [0.0, 1.0], [0.7071067811865476, 0.7071067811865476]]
    a = ours.transport_tangent_vectors(inds, vecs)
    b = theirs.transport_tangent_vectors(inds, vecs)
    assert a.shape == b.shape
    assert rel_err(a, b) < 1e-10


def test_transport_tangent_vector_preserves_norm(closed_mesh):
    V, F = closed_mesh
    out = mpp3d.MeshVectorHeatSolver(
        V, F, use_intrinsic_delaunay=False
    ).transport_tangent_vector(0, [1.0, 0.0])
    assert out.shape == (V.shape[0], 2)
    assert np.abs(np.linalg.norm(out, axis=1) - 1.0).max() < 1e-9


def test_transport_tangent_vector_matches_except_the_antipode(closed_mesh):
    """One vertex -- the one directly opposite the source on a sphere -- flips
    sign, because the transported direction there is numerically degenerate and
    the normalization amplifies the rounding. Every other vertex agrees."""
    V, F = closed_mesh
    ours = mpp3d.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    theirs = pp3d.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    a = ours.transport_tangent_vector(0, [1.0, 0.0])
    b = theirs.transport_tangent_vector(0, [1.0, 0.0])
    err = np.abs(a - b).max(axis=1)
    assert np.sort(err)[:-1].max() < 1e-9
    assert int((err > 1e-9).sum()) <= 1


def test_transport_length_check(closed_mesh):
    V, F = closed_mesh
    with pytest.raises(ValueError, match="2D tangent vector"):
        mpp3d.MeshVectorHeatSolver(V, F).transport_tangent_vector(0, [1.0, 0.0, 0.0])


def test_log_map_is_reported_as_uncovered(closed_mesh):
    V, F = closed_mesh
    with pytest.raises(NotImplementedError, match="not covered"):
        mpp3d.MeshVectorHeatSolver(V, F).compute_log_map(0)


# ------------------------------------------------------ the sparse factorizer

def test_factorization_round_trip():
    """The sparse factorizer against a dense solve on a random SPD system."""
    from mojo_potpourri3d._solver import Factorization

    rng = np.random.default_rng(7)
    n = 120
    M = rng.normal(size=(n, n))
    A = M @ M.T + 4.0 * np.eye(n)
    rows, cols = np.nonzero(A)
    vals = A[rows, cols]
    fac = Factorization(n, rows, cols, vals)
    B = rng.normal(size=(n, 4))
    assert rel_err(fac.solve(B), np.linalg.solve(A, B)) < 1e-10


def test_factorization_handles_fill():
    """A pattern with real fill: the factor must be larger than the matrix."""
    from mojo_potpourri3d._solver import Factorization

    n = 40
    rows, cols, vals = [], [], []
    for i in range(n - 1):  # a path graph: the natural order needs no fill
        rows += [i, i + 1, i, i + 1]
        cols += [i, i + 1, i + 1, i]
        vals += [3.0, 3.0, -1.0, -1.0]
    A = np.zeros((n, n))
    np.add.at(A, (rows, cols), vals)
    fac = Factorization(n, np.array(rows), np.array(cols), np.array(vals))
    assert np.linalg.cond(A) > 1
    B = np.random.default_rng(1).normal(size=(n, 2))
    assert rel_err(fac.solve(B), np.linalg.solve(A, B)) < 1e-9


def _assert_row_index_is_the_transpose(fac):
    """The numeric pass reads row k of L through R_p/R_i/R_pos and never
    searches, so the row index has to be exactly the transpose of the column
    one, with every R_pos still addressing the matching L entry."""
    slot = {}  # (row, col) -> offset in S_i
    for j in range(fac.n):
        for p in range(fac.S_p[j], fac.S_p[j + 1]):
            slot[(int(fac.S_i[p]), j)] = p
    seen = 0
    for k in range(fac.n):
        row = []
        for q in range(fac.R_p[k], fac.R_p[k + 1]):
            j = int(fac.R_i[q])
            assert j < k, (k, j)
            assert slot.get((k, j)) == int(fac.R_pos[q]), (k, j, fac.R_pos[q])
            row.append(j)
        assert row == sorted(set(row)), (k, row)
        assert len(row) == sum(1 for j in range(k) if (k, j) in slot), k
        seen += len(row)
    assert seen == int(fac.S_p[fac.n]) == int(fac.R_p[fac.n])


def test_factorization_row_index_matches_the_column_pattern():
    """A mesh Laplacian: the sweep is pruned by the coverage test here, so a
    broken prune shows up as a missing row or a wrong offset."""
    from mojo_potpourri3d._solver import Factorization

    V, F = icosphere(3)
    n = V.shape[0]
    L = mpp3d.cotan_laplacian(V, F)
    tri = L.tocoo()
    idx = np.arange(n, dtype=np.int64)
    fac = Factorization(
        n,
        np.concatenate([tri.row, idx]),
        np.concatenate([tri.col, idx]),
        np.concatenate([tri.data, np.full(n, 1.0)]),
    )
    assert int(fac.S_p[n]) > int(fac.Ap[n])  # the factor really does have fill
    _assert_row_index_is_the_transpose(fac)
    b = np.random.default_rng(3).normal(size=n)
    A = L.toarray() + np.eye(n)
    assert rel_err(fac.solve_vector(b), np.linalg.solve(A, b)) < 1e-9


def test_factorization_row_index_on_a_fillless_pattern():
    """A diagonal matrix: no child columns, so the sweep takes the empty path."""
    from mojo_potpourri3d._solver import Factorization

    n = 8
    idx = np.arange(n, dtype=np.int64)
    fac = Factorization(n, idx, idx, np.arange(1.0, n + 1.0))
    assert int(fac.S_p[n]) == 0
    assert int(fac.R_p[n]) == 0
    _assert_row_index_is_the_transpose(fac)
    b = np.random.default_rng(5).normal(size=n)
    assert rel_err(fac.solve_vector(b), b / np.arange(1.0, n + 1.0)) < 1e-14


def test_hermitian_factorization_row_index(closed_mesh):
    """The complex path shares the symbolic, so its row index is the same one."""
    from mojo_potpourri3d._solver import Factorization

    V, F = closed_mesh
    n = V.shape[0]
    vs = mpp3d.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    ti, tj, tr, tim = vs.solver.connection_laplacian_triplets()
    idx = np.arange(n, dtype=np.int64)
    fac = Factorization(
        n,
        np.concatenate([ti, idx]),
        np.concatenate([tj, idx]),
        np.concatenate([tr, np.full(n, 1.0)]),
        np.concatenate([tim, np.zeros(n)]),
    )
    _assert_row_index_is_the_transpose(fac)


def test_hermitian_factorization_round_trip(closed_mesh):
    from mojo_potpourri3d._solver import Factorization

    V, F = closed_mesh
    n = V.shape[0]
    mpp3d_solve = mpp3d.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    ti, tj, tr, tim = mpp3d_solve.solver.connection_laplacian_triplets()
    idx = np.arange(n, dtype=np.int64)
    fac = Factorization(
        n,
        np.concatenate([ti, idx]),
        np.concatenate([tj, idx]),
        np.concatenate([tr, np.full(n, 1.0)]),
        np.concatenate([tim, np.zeros(n)]),
    )
    A = mpp3d_solve.get_connection_laplacian().toarray() + 1.0 * np.eye(n)
    b = np.arange(1.0, n + 1)
    c = np.zeros(n)
    c[0] = 1.0
    rhs = np.stack([b, c], axis=1)
    assert rel_err(fac.solve(rhs)[:, 0], np.linalg.solve(A, b + 1j * c)) < 1e-9


def test_ffi_signatures_match_the_exports():
    """A ctypes arity mismatch is a silent memory-corrupting bug, so pin it."""
    import re
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("_lib", root / "python/mojo_potpourri3d/_lib.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    capi = (root / "src/capi.mojo").read_text()
    exports = {
        m.group(1): len([p for p in m.group(3).split(",") if p.strip()])
        for m in re.finditer(r'@export\("(\w+)"\)\s*\ndef\s+(\w+)\(([^)]*)\)', capi)
    }
    arity = lambda s: sum(a if isinstance(a, int) else 1 for a in s)  # noqa: E731
    assert set(exports) == set(mod._SIGNATURES)
    for name, n in exports.items():
        assert arity(mod._SIGNATURES[name][0]) == n, name


PUBLIC_API = [
    ("cotan_laplacian", None),
    ("face_areas", None),
    ("vertex_areas", None),
    ("compute_distance", None),
    ("compute_distance_multisource", None),
    ("validate_mesh", None),
    ("validate_points", None),
    ("MeshHeatMethodDistanceSolver", None),
    ("MeshVectorHeatSolver", None),
    ("MeshHeatMethodDistanceSolver", "compute_distance"),
    ("MeshHeatMethodDistanceSolver", "compute_distance_multisource"),
    ("MeshVectorHeatSolver", "extend_scalar"),
    ("MeshVectorHeatSolver", "get_tangent_frames"),
    ("MeshVectorHeatSolver", "get_connection_laplacian"),
    ("MeshVectorHeatSolver", "transport_tangent_vector"),
    ("MeshVectorHeatSolver", "transport_tangent_vectors"),
    ("MeshVectorHeatSolver", "compute_log_map"),
]


@pytest.mark.parametrize("cls,method", PUBLIC_API, ids=lambda v: str(v))
def test_public_api_signatures_match_upstream(cls, method):
    """Names, argument order and defaults must be upstream's, or this is not a
    drop-in. A caller should not have to learn a new API to use this port."""
    import inspect

    theirs = getattr(pp3d, cls)
    ours = getattr(mpp3d, cls)
    if method is not None:
        theirs, ours = getattr(theirs, method), getattr(ours, method)
    label = cls if method is None else f"{cls}.{method}"
    assert str(inspect.signature(ours)) == str(inspect.signature(theirs)), label


def test_every_covered_name_exists_upstream():
    """The reverse direction: nothing invented under an upstream-looking name."""
    covered = [
        "cotan_laplacian", "face_areas", "vertex_areas", "compute_distance",
        "compute_distance_multisource", "validate_mesh", "validate_points",
        "MeshHeatMethodDistanceSolver", "MeshVectorHeatSolver",
    ]
    for name in covered:
        assert hasattr(pp3d, name), name
        assert hasattr(mpp3d, name), name
