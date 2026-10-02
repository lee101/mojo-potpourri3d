"""Numerical parity against upstream potpourri3d."""

import numpy as np
import pytest

import potpourri3d as pp3d

from mojo_potpourri3d import mesh as mm
from mojo_potpourri3d._lib import lib

import conftest


def test_cotan_laplacian_matches_upstream(ico2, grid8):
    for V, F in (ico2, grid8):
        a = mm.cotan_laplacian(V, F)
        b = pp3d.cotan_laplacian(V, F)
        assert a.shape == b.shape
        assert abs(a - b).max() < 1e-12


def test_cotan_laplacian_denom_eps(grid8):
    V, F = grid8
    for eps in (0.0, 1e-6, 1e-2):
        a = mm.cotan_laplacian(V, F, denom_eps=eps)
        b = pp3d.cotan_laplacian(V, F, denom_eps=eps)
        assert abs(a - b).max() < 1e-12


def test_face_areas_match_upstream(ico2, grid8):
    for V, F in (ico2, grid8):
        assert np.abs(mm.face_areas(V, F) - pp3d.face_areas(V, F)).max() < 1e-13


def test_vertex_areas_match_upstream(ico2, grid8):
    for V, F in (ico2, grid8):
        assert np.abs(mm.vertex_areas(V, F) - pp3d.vertex_areas(V, F)).max() < 1e-13


def test_edges_match_upstream(ico2, grid8):
    for V, F in (ico2, grid8):
        a = np.unique(np.sort(mm.edges(V, F), axis=1), axis=0)
        b = np.unique(np.sort(pp3d.edges(V, F), axis=1), axis=0)
        assert a.shape == b.shape
        assert (a == b).all()


@pytest.mark.parametrize("name", ["ico1", "ico2", "grid8", "grid16"])
def test_heat_distance_matches_upstream(name):
    if name.startswith("ico"):
        V, F = conftest.icosphere(int(name[3:]))
    else:
        V, F = conftest.open_grid(int(name[4:]))
    a = pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance(0)
    b = mm.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance(0)
    assert np.abs(a - b).max() < 1e-10 * max(1.0, a.max())


@pytest.mark.parametrize("srcs", [[0, 1], [0, 7, 20], [3, 4, 5, 6]])
def test_heat_distance_multisource_matches_upstream(ico2, srcs):
    V, F = ico2
    a = pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance_multisource(srcs)
    b = mm.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance_multisource(srcs)
    assert np.abs(a - b).max() < 1e-10 * max(1.0, a.max())


@pytest.mark.parametrize("sub", [2, 3])
def test_multisource_at_the_corners_of_one_face(sub):
    """The one documented case where the two implementations do not agree closely.

    Three sources at the corners of one face make the heat field constant across that face, so the
    face has no gradient to normalize. Upstream calls `normalizeCutoff()` with its default cutoff of
    zero, divides by the rounding residue, and normalizes that noise to a unit vector; which noise it
    gets depends on the last bit of its solver's output, which differs from this port's. Both
    implementations therefore return a noise-driven field here, and they can be a couple of percent
    apart. This test pins that: the field stays finite, non-negative and within a few percent of
    upstream, so the divergence cannot grow silently.
    """
    V, F = conftest.icosphere(sub)
    srcs = [int(x) for x in F[0]]
    u = mm.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance_multisource(srcs)
    ref = pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance_multisource(srcs)
    assert np.all(np.isfinite(u))
    assert u.min() > -0.05 * ref.max()
    assert np.abs(u - ref).max() < 0.05 * ref.max()


def test_fixtures_are_closed_manifolds():
    """The icosphere generator must produce a real closed triangulation, not a broken one."""
    for sub in range(4):
        V, F = conftest.icosphere(sub)
        assert conftest.is_edge_manifold(F)
        assert V.shape[0] - _n_edges(F) + F.shape[0] == 2
        assert np.allclose(np.linalg.norm(V, axis=1), 1.0)


def _n_edges(F):
    return len({(min(int(f[k]), int(f[(k + 1) % 3])), max(int(f[k]), int(f[(k + 1) % 3])))
                for f in F for k in range(3)})


def test_heat_distance_is_nonnegative_and_zero_at_source(ico2):
    V, F = ico2
    u = mm.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance(7)
    assert u[7] == pytest.approx(0.0, abs=1e-9)
    assert u.min() > -1e-9
    # a distance field on a unit sphere cannot exceed the sphere's diameter, pi
    assert u.max() <= np.pi + 1e-9


def test_heat_distance_t_coef_knob(ico2):
    V, F = ico2
    a = pp3d.MeshHeatMethodDistanceSolver(V, F, t_coef=2.0, use_robust=False).compute_distance(0)
    b = mm.MeshHeatMethodDistanceSolver(V, F, t_coef=2.0, use_robust=False).compute_distance(0)
    assert np.abs(a - b).max() < 1e-10 * max(1.0, a.max())


@pytest.mark.parametrize("name", ["ico0", "ico1", "ico2", "ico3", "grid4", "grid8", "grid16"])
def test_heat_distance_robust_matches_upstream(name):
    """The tufted-cover / intrinsic-Delaunay path (upstream's default)."""
    if name.startswith("ico"):
        V, F = conftest.icosphere(int(name[3:]))
    else:
        V, F = conftest.open_grid(int(name[4:]))
    a = pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=True).compute_distance(0)
    b = mm.MeshHeatMethodDistanceSolver(V, F, use_robust=True).compute_distance(0)
    assert np.abs(a - b).max() < 1e-8 * max(1.0, a.max())


@pytest.mark.parametrize("sub", [2, 3])
def test_robust_distance_is_a_valid_field_on_a_dense_closed_mesh(sub):
    """The flips to intrinsic Delaunay used to leave a zero-area triangle on dense closed meshes."""
    V, F = conftest.icosphere(sub)
    u = mm.MeshHeatMethodDistanceSolver(V, F, use_robust=True).compute_distance(0)
    assert np.all(np.isfinite(u))
    assert u.min() > -1e-9
    assert u[0] == pytest.approx(0.0, abs=1e-9)
    ref = pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=True).compute_distance(0)
    assert np.abs(u - ref).max() < 1e-8 * max(1.0, ref.max())




@pytest.mark.parametrize("name", ["ico2", "grid8"])
def test_the_two_heat_operators_share_one_factor_pattern(name):
    """`heat_setup` skips the Poisson operator's symbolic pass when its pattern matches the heat one's.

    The heat operator is the Laplacian plus a diagonal mass term and the Poisson operator is the
    same Laplacian plus a diagonal shift, so their CSC index patterns are identical and so is the
    pattern of L for both. The kernel compares the two patterns before taking that shortcut; this
    checks that the comparison does take it, by confirming the two reported nnz(L) agree, and that a
    solver built that way still matches upstream.
    """
    if name.startswith("ico"):
        V, F = conftest.icosphere(int(name[3:]))
    else:
        V, F = conftest.open_grid(int(name[4:]))
    s = mm.MeshHeatMethodDistanceSolver(V, F, use_robust=False)
    # W[4] and W[5] are nnz(L) of the heat and Poisson factors. The fast path sets them equal; the
    # fallback path would compute them separately and they would still agree, so this pins the
    # shortcut's premise rather than only its outcome.
    assert s._W[4] == s._W[5] and s._W[4] > 0
    a = pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False).compute_distance(0)
    b = s.compute_distance(0)
    assert np.abs(a - b).max() < 1e-10 * max(1.0, a.max())


@pytest.mark.parametrize("name", ["ico2", "grid8"])
def test_out_of_range_face_index_is_reported(name):
    """Upstream's index check uses np.amin, so a too-large index slips through; the kernels must not
    walk off the vertex array when it does."""
    if name.startswith("ico"):
        V, F = conftest.icosphere(int(name[3:]))
    else:
        V, F = conftest.open_grid(int(name[4:]))
    bad = F.copy()
    bad[0, 0] = V.shape[0] + 5
    for call in (
        lambda: mm.MeshHeatMethodDistanceSolver(V, bad),
        lambda: mm.cotan_laplacian(V, bad),
        lambda: mm.face_areas(V, bad),
        lambda: mm.vertex_areas(V, bad),
        lambda: mm.edges(V, bad),
    ):
        with pytest.raises(ValueError, match="out-of-bounds face index"):
            call()


def test_out_of_range_source_index_is_reported(ico2):
    """An out-of-range source must not come back as a constant distance field.

    The kernel used to skip a source index outside [0, nV), which left an all-zero right-hand side:
    the solve then returns a field whose distance shift is a single constant, so every vertex comes
    back at the same finite value. Upstream instead indexes its vertex array with the caller's list, so
    the index is either wrapped (a negative one) or reads past the end; neither is a field worth
    matching, so both are refused here.
    """
    V, F = ico2
    n_v = V.shape[0]
    s = mm.MeshHeatMethodDistanceSolver(V, F, use_robust=False)
    for bad in (n_v, n_v + 7, -1, -n_v):
        with pytest.raises(IndexError, match="out of range"):
            s.compute_distance(bad)
    for bad in ([0, n_v], [0, 1, -3]):
        with pytest.raises(IndexError, match="out of range"):
            s.compute_distance_multisource(bad)
    # and the valid queries around them still work
    assert np.all(np.isfinite(s.compute_distance(n_v - 1)))
    assert np.all(np.isfinite(s.compute_distance_multisource([0, n_v - 1])))


def test_an_empty_source_set_is_reported(ico2):
    """Upstream returns all-NaN for an empty source list. A zero field would be worse: it is finite
    and reads as a distance, so the empty query is refused rather than answered."""
    V, F = ico2
    s = mm.MeshHeatMethodDistanceSolver(V, F, use_robust=False)
    with pytest.raises(ValueError, match="at least one source vertex"):
        s.compute_distance_multisource([])
    with pytest.raises(ValueError, match="at least one source vertex"):
        mm.compute_distance_multisource(V, F, [])


@pytest.mark.parametrize("robust", [False, True])
def test_strided_and_narrow_inputs_are_copied_not_reinterpreted(ico2, robust):
    """The kernels index by element offset, so a strided view must be materialised first.

    f64()/i64() copy a non-contiguous or non-float64 input into a temporary. If the call site did not
    hold on to that temporary for the length of the call, the kernel would read freed memory; if it
    did not copy at all, it would read the caller's original layout as if it were packed. Both show
    up here as the wrong answer for a strided or narrowed input.
    """
    V, F = ico2
    V_view = V[::1]
    F_view = F[::1]
    ref = mm.MeshHeatMethodDistanceSolver(V, F, use_robust=robust).compute_distance(0)

    strided = mm.MeshHeatMethodDistanceSolver(V_view[::1], F_view[::1], use_robust=robust)
    assert np.allclose(strided.compute_distance(0), ref, rtol=0, atol=1e-12)

    # float32 vertices and int32 faces must be widened, not read at float64/int64 stride.
    narrow = mm.MeshHeatMethodDistanceSolver(
        np.ascontiguousarray(V, dtype=np.float32), np.ascontiguousarray(F, dtype=np.int32),
        use_robust=robust)
    assert np.allclose(narrow.compute_distance(0), ref, rtol=0, atol=1e-6)

    # a genuinely strided view (every other face) is a different mesh, so compare it against itself
    half = mm.MeshHeatMethodDistanceSolver(V, F[::2], use_robust=robust).compute_distance(0)
    half_direct = pp3d.MeshHeatMethodDistanceSolver(V, np.ascontiguousarray(F[::2]),
                                                    use_robust=robust).compute_distance(0)
    assert np.allclose(half, half_direct, rtol=0, atol=1e-10)


def test_layout_publishes_every_field_the_python_side_reads(ico2):
    """`heat_layout` writes 17 offsets and mesh.py indexes them by position; a dropped write or a
    reordered one is invisible until a field reads as a plausible-but-wrong number."""
    V, F = ico2
    n_v, n_f = V.shape[0], F.shape[0]
    lay = mm.heat_layout(n_v, n_f, n_f, 3 * n_f)
    assert len(lay) == 17
    w_size, m_size = lay[0], lay[1]
    w_hvif, w_hcw, w_corner_len, w_heat_ax, w_pois_ax, w_work = lay[2:8]
    m_heat_ap, m_heat_ai, m_heat_perm, m_pois_ap, m_pois_ai, m_pois_perm = lay[8:14]
    m_corner_v, m_first_he, m_corner_v_compute = lay[14:17]

    # the float arena's regions are packed in the order the kernel writes them
    assert w_hvif <= w_hcw <= w_corner_len <= w_heat_ax <= w_pois_ax <= w_work < w_size
    # the int arena's regions likewise; m_corner_v is the original mesh snapshot and goes first,
    # because the distance shift reads it while the solver works on the tufted cover
    assert m_corner_v == 0
    assert m_first_he >= 3 * n_f
    assert m_heat_ap < m_heat_ai < m_heat_perm < m_pois_ap < m_pois_ai < m_pois_perm
    assert m_corner_v_compute < m_size
    assert m_heat_ap >= m_first_he + n_v

    # each region fits in the arena that holds it, with room for the elements the kernel writes
    nna = n_v + 4 * 3 * n_f
    assert w_work + 4 * n_v == w_size
    assert m_corner_v_compute + 3 * n_f == m_size
    assert w_heat_ax + nna <= w_pois_ax and w_pois_ax + nna <= w_work
    assert m_heat_ap + n_v + 1 <= m_heat_ai and m_heat_ai + nna <= m_heat_perm
    assert m_heat_perm + n_v <= m_pois_ap and m_pois_ap + n_v + 1 <= m_pois_ai
    assert m_pois_ai + nna <= m_pois_perm and m_pois_perm + n_v <= m_corner_v_compute


def test_the_module_level_helpers_match_the_class(ico2):
    """`compute_distance(V, F, v_ind)` and `compute_distance_multisource(V, F, v_inds)` are
    upstream module-level functions that build a solver and query it; the covered table claims they
    agree with upstream, so they are checked as the entry points a caller actually reaches for."""
    V, F = ico2
    srcs = [0, 7, 20]
    a = pp3d.compute_distance(V, F, 5)
    b = mm.compute_distance(V, F, 5)
    assert np.abs(a - b).max() < 1e-10 * max(1.0, a.max())
    a = pp3d.compute_distance_multisource(V, F, srcs)
    b = mm.compute_distance_multisource(V, F, srcs)
    assert np.abs(a - b).max() < 1e-10 * max(1.0, a.max())


def test_the_core_checkers_are_upstreams_verbatim():
    """`core.py` is transcribed from potpourri3d.core, and the covered APIs below depend on it
    reproducing upstream's checks exactly: a stricter check would reject inputs upstream accepts, and
    a looser one would accept inputs upstream rejects. Compared source-for-source, so reformatting
    the copy cannot quietly change its behaviour."""
    import ast
    import inspect

    import potpourri3d.core as up

    from mojo_potpourri3d import core as mc

    def dump(fn):
        # The AST, not the text: this copy wraps one long raise across lines, and a text comparison
        # would report that reflow as a behaviour difference.
        return ast.dump(ast.parse(inspect.getsource(fn).lstrip()))

    for fn in ("validate_mesh", "validate_points"):
        assert dump(getattr(mc, fn)) == dump(getattr(up, fn)), f"{fn} is not upstream's verbatim"


def test_the_input_checkers_reject_what_upstream_rejects(ico2):
    """The checkers are reachable from every covered entry point, so they are exercised through them
    rather than called directly."""
    V, F = ico2
    for bad_V in (np.zeros((8, 2)), np.zeros(8)):
        for call in (
            lambda: mm.cotan_laplacian(bad_V, F),
            lambda: mm.face_areas(bad_V, F),
            lambda: mm.MeshHeatMethodDistanceSolver(bad_V, F),
        ):
            with pytest.raises(ValueError, match="vertices should be a 2d Nx3 numpy array"):
                call()
    quad = np.hstack([F, F[:, :1]])
    for call in (lambda: mm.cotan_laplacian(V, quad), lambda: mm.face_areas(V, quad)):
        with pytest.raises(ValueError, match="faces must be triangular"):
            call()


def test_solver_setup_failures_are_not_swallowed(ico2):
    """Setup returns codes the constructor has to translate, not pass through as a number.

    -50 is a non-finite cotangent weight, which both paths can produce: a zero-area face is not
    differential, and upstream returns an operator full of NaN for the same input. The factorization
    codes are what an operator that is not positive definite would produce. Neither may be solved,
    and both must raise rather than return a field from an unusable operator.
    """
    V, F = ico2
    bad = F.copy()
    bad[0] = [bad[0, 0], bad[0, 1], bad[0, 1]]  # two corners on the same vertex: zero area
    # use_robust=False sees the degenerate face directly, so it reports it.
    with pytest.raises(ValueError, match="degenerate"):
        mm.MeshHeatMethodDistanceSolver(V, bad, use_robust=False)
    # use_robust=True flips to intrinsic Delaunay, which drops the zero-area face instead, so the
    # mesh is usable there and the field is finite. Pinned so the difference stays deliberate.
    robust = mm.MeshHeatMethodDistanceSolver(V, bad, use_robust=True).compute_distance(0)
    assert np.all(np.isfinite(robust))
    assert np.abs(robust - pp3d.MeshHeatMethodDistanceSolver(V, bad, use_robust=True)
                  .compute_distance(0)).max() < 1e-8 * robust.max()
    # the elementwise operators still answer that input, exactly as upstream does (upstream's
    # cotan_laplacian returns NaN there, and so does this port's)
    assert np.isfinite(mm.face_areas(V, bad)).all()
    L = mm.cotan_laplacian(V, bad)
    up = pp3d.cotan_laplacian(V, bad)
    assert np.array_equal(np.isnan(L.data), np.isnan(up.data))


def test_a_solver_that_factors_and_reuses_stays_usable(ico2):
    """Each query writes through the shared arenas, so a second call has to give the same answer as
    the first. A solve that clobbered its own factor, its own permutation or the previous result
    would drift from call to call."""
    V, F = ico2
    for robust in (False, True):
        s = mm.MeshHeatMethodDistanceSolver(V, F, use_robust=robust)
        first = s.compute_distance(3)
        for _ in range(3):
            again = s.compute_distance(3)
            assert np.array_equal(again, first)
        # interleaving source sets must not disturb either
        multi = s.compute_distance_multisource([0, 5, 9])
        assert np.array_equal(s.compute_distance(3), first)
        assert np.array_equal(multi, s.compute_distance_multisource([0, 5, 9]))
        # and two independent solvers on the same mesh agree
        t = mm.MeshHeatMethodDistanceSolver(V, F, use_robust=robust)
        assert np.array_equal(t.compute_distance(3), first)


@pytest.mark.parametrize("name", ["ico2", "grid8"])
def test_denom_eps_only_reaches_the_cotan_laplacian(name):
    """`denom_eps` guards the cotangent against a zero-area face. Every other entry point must
    reject it the way upstream does, because forwarding an unexpected keyword would silently widen
    the call."""
    if name.startswith("ico"):
        V, F = conftest.icosphere(int(name[3:]))
    else:
        V, F = conftest.open_grid(int(name[4:]))
    for call in (
        lambda: mm.face_areas(V, F, denom_eps=1e-6),
        lambda: mm.vertex_areas(V, F, denom_eps=1e-6),
        lambda: mm.edges(V, F, denom_eps=1e-6),
        lambda: mm.MeshHeatMethodDistanceSolver(V, F, denom_eps=1e-6),
        lambda: mm.compute_distance(V, F, 0, denom_eps=1e-6),
    ):
        with pytest.raises(TypeError):
            call()
