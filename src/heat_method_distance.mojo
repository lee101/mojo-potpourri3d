# geometry-central surface/heat_method_distance.cpp: HeatMethodDistanceSolver.
#
# Upstream keeps the two factorizations alive in PositiveDefiniteSolver objects for the lifetime of
# the solver object. A Mojo function cannot hold state across the C ABI, so this module drives the
# same three steps a caller would, parking the persistent arrays in two caller-owned arenas whose
# layout is fixed by `HeatLayout` below. python/mojo_potpourri3d/_lib.py mirrors that layout to size
# the buffers.
#
#   W   float64 arena
#   M   int64 arena
#   LI  int64 arena for the symbolic factor patterns
#   LF  float64 arena for the numeric factors
#
# With useRobustLaplacian the geometry is the intrinsic tufted cover of the input mesh
# ([Sharp & Crane 2020]), mollified and flipped to intrinsic Delaunay. Both sheets of the cover use
# the input mesh's vertices, so the distance field is returned on the original vertex set.

from std.math import sqrt, isfinite
from vector_util import Vector2, Bary3, v2_dot, v2_normalize_cutoff
from surface_mesh import SurfaceMesh, build_surface_mesh, faces_in_range
from intrinsic_geometry_interface import compute_face_areas, compute_vertex_dual_areas, \
    compute_halfedge_cotan_weights, compute_edge_cotan_weights, compute_halfedge_vectors_in_face, \
    compute_cotan_laplacian, \
    compute_vertex_lumped_mass_matrix, build_vertex_halfedges
from tufted_laplacian import build_intrinsic_tufted_cover, mollify_intrinsic
from simple_idt import flip_to_delaunay
from sparse_matrix import Triplets, triplets_to_csc
from sparse import chol_analyze, chol_factorize, chol_solve
from ffi import addr_of


# The heat operator is the Laplacian plus a diagonal mass term and the Poisson operator is the
# same Laplacian plus a diagonal shift, so the two carry the same CSC index pattern and differ
# only in values. Comparing the two patterns costs one linear pass and saves a full symbolic
# factorization when they agree.
def _same_pattern(n: Int, ap_a: Int, ai_a: Int, ap_b: Int, ai_b: Int) -> Bool:
    var pa = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=ap_a)
    var ia = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=ai_a)
    var pb = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=ap_b)
    var ib = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=ai_b)
    var c = 0
    while c <= n:
        if pa.unsafe_load(c) != pb.unsafe_load(c):
            return False
        c += 1
    var total = pa.unsafe_load(n)
    var p = 0
    while p < total:
        if ia.unsafe_load(p) != ib.unsafe_load(p):
            return False
        p += 1
    return True


struct HeatLayout:
    var nV: Int
    var nF: Int
    var nFc: Int
    var nna: Int
    var w_hvif: Int
    var w_hcw: Int
    var w_corner_len: Int
    var w_heat_ax: Int
    var w_pois_ax: Int
    var w_work: Int
    var m_heat_ap: Int
    var m_heat_ai: Int
    var m_heat_perm: Int
    var m_pois_ap: Int
    var m_pois_ai: Int
    var m_pois_perm: Int
    var m_corner_v: Int
    var m_first_he: Int
    var m_corner_v_compute: Int

    def __init__(out self, n_v: Int, n_f: Int, n_fc: Int, n_ec: Int):
        self.nV = n_v
        self.nF = n_f
        self.nFc = n_fc
        self.nna = n_v + 4 * n_ec
        self.w_hvif = 16
        self.w_hcw = 16 + 6 * n_fc
        self.w_corner_len = 16 + 9 * n_fc
        self.w_heat_ax = 16 + 9 * n_fc + 3 * n_f
        self.w_pois_ax = self.w_heat_ax + self.nna
        self.w_work = self.w_pois_ax + self.nna
        # Every region is packed in order, each sized by the element count the kernel writes, so the
        # two solvers' arrays cannot overlap however big the operands are. The snapshot of the
        # ORIGINAL mesh goes first, because it is taken before the solver knows how big the cover is.
        self.m_corner_v = 0
        self.m_first_he = 3 * n_f
        self.m_corner_v_compute = 4 * n_f + 5 * n_v + 2 + 2 * self.nna
        self.m_heat_ap = 3 * n_f + n_v
        self.m_heat_ai = 4 * n_f + 2 * n_v + 1
        self.m_heat_perm = 4 * n_f + 2 * n_v + 1 + self.nna
        self.m_pois_ap = 4 * n_f + 3 * n_v + 1 + self.nna
        self.m_pois_ai = 4 * n_f + 4 * n_v + 2 + self.nna
        self.m_pois_perm = 4 * n_f + 4 * n_v + 2 + 2 * self.nna

    def w_size(self) -> Int:
        return self.w_work + 4 * self.nV

    def m_size(self) -> Int:
        return self.m_corner_v_compute + 3 * self.nFc

    # Each operator's factor lives in its own pair of caller buffers, so no offsets are needed.
    def li_size(self, nnzl: Int) -> Int:
        return self.nV + 1 + nnzl

    def lf_size(self, nnzl: Int) -> Int:
        return nnzl + self.nV



# Publishes the arena layout so the caller can size its buffers without duplicating the formulas.
# Writes 16 int64 values: the two buffer sizes and the offsets of every region inside them.
def heat_layout(
    n_v: Int, n_f: Int, n_fc: Int, n_ec: Int, out_addr: Int
) -> Int:
    if out_addr == 0:
        return -1
    var out = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=out_addr)
    var lay = HeatLayout(n_v, n_f, n_fc, n_ec)
    out.unsafe_store(0, lay.w_size())
    out.unsafe_store(1, lay.m_size())
    out.unsafe_store(2, lay.w_hvif)
    out.unsafe_store(3, lay.w_hcw)
    out.unsafe_store(4, lay.w_corner_len)
    out.unsafe_store(5, lay.w_heat_ax)
    out.unsafe_store(6, lay.w_pois_ax)
    out.unsafe_store(7, lay.w_work)
    out.unsafe_store(8, lay.m_heat_ap)
    out.unsafe_store(9, lay.m_heat_ai)
    out.unsafe_store(10, lay.m_heat_perm)
    out.unsafe_store(11, lay.m_pois_ap)
    out.unsafe_store(12, lay.m_pois_ai)
    out.unsafe_store(13, lay.m_pois_perm)
    out.unsafe_store(14, lay.m_corner_v)
    out.unsafe_store(15, lay.m_first_he)
    out.unsafe_store(16, lay.m_corner_v_compute)
    return 0


# HeatMethodDistanceSolver::HeatMethodDistanceSolver. Builds the geometry the solver reads from
# (with the robust laplacian that means the tufted cover), then the two operators and their
# symbolic factorizations. Returns the float64 slot count the factor arena needs, 0 on a solver
# failure, or -1 on a malformed input.
def heat_setup(
    n_v: Int,
    n_f: Int,
    verts_addr: Int,
    faces_addr: Int,
    t_coef: Float64,
    use_robust: Int,
    w_addr: Int,
    m_addr: Int,
    li_addr: Int,
) -> Int:
    if n_v <= 0 or n_f <= 0 or verts_addr == 0 or faces_addr == 0 or w_addr == 0 or m_addr == 0:
        return -1
    if not faces_in_range(
        Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=faces_addr), n_v, n_f
    ):
        return -2
    var verts = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=verts_addr)
    var faces = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=faces_addr)
    var w = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=w_addr)
    var m = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=m_addr)
    var li = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=li_addr)
    return heat_setup_impl(n_v, n_f, verts, faces, t_coef, use_robust, w, m, li)


def heat_setup_impl(
    n_v: Int,
    n_f: Int,
    verts: Pointer[Float64, AnyOrigin[mut=True]],
    faces: Pointer[Int, AnyOrigin[mut=True]],
    t_coef: Float64,
    use_robust: Int,
    w: Pointer[Float64, AnyOrigin[mut=True]],
    m: Pointer[Int, AnyOrigin[mut=True]],
    li: Pointer[Int, AnyOrigin[mut=True]],
) -> Int:
    var mesh = build_surface_mesh(verts, faces, n_v, n_f)
    var lay = HeatLayout(n_v, n_f, 2 * n_f if use_robust else n_f, 0)

    # The distance shift below reads the ORIGINAL geometry, not the tufted one, so snapshot it
    # before the mesh is consumed by the cover construction.
    var fan = build_vertex_halfedges(mesh)
    var i = 0
    while i < 3 * n_f:
        m.unsafe_store(lay.m_corner_v + i, mesh.F[i])
        w.unsafe_store(lay.w_corner_len + i, mesh.EL[mesh.EID[i]])
        i += 1
    i = 0
    while i < n_v:
        m.unsafe_store(lay.m_first_he + i, fan.vhe[i])
        i += 1

    # Compute the tufted mesh / geometry, if using the robust laplacian
    var compute: SurfaceMesh
    if use_robust:
        var t = build_intrinsic_tufted_cover(mesh^)
        t = mollify_intrinsic(t^, 1e-5)
        compute = flip_to_delaunay(t^, 1e-6)
    else:
        compute = mesh^

    lay = HeatLayout(n_v, n_f, compute.nF, compute.nE)

    # Compute mean edge length and set shortTime
    var mean_edge_length = 0.0
    var e = 0
    while e < compute.nE:
        mean_edge_length += compute.EL[e]
        e += 1
    mean_edge_length /= Float64(compute.nE)
    var short_time = t_coef * mean_edge_length * mean_edge_length

    # Mass matrix / Laplacian
    var face_areas = compute_face_areas(compute)
    var dual_areas = compute_vertex_dual_areas(compute, face_areas)
    var mass = compute_vertex_lumped_mass_matrix(compute, dual_areas)
    var ecw = compute_edge_cotan_weights(compute, face_areas)
    var ec = 0
    while ec < compute.nE:
        # A non-finite cotangent weight means a degenerate face survived the flip (or the input had
        # one to begin with). It cannot be differentiated away, and the operator would carry a NaN
        # into both solves. Reported rather than solved; `denom_eps` is the supported way to guard
        # against it, and upstream returns a NaN-filled operator for the same input.
        if not isfinite(ecw[ec]):
            return -50
        ec += 1
    var cotan = compute_cotan_laplacian(compute, ecw)

    # Heat operator, M + shortTime * L
    var heat_triplets = Triplets()
    var v = 0
    while v < compute.nV:
        heat_triplets.add(v, v, mass[v])
        v += 1
    var k = 0
    while k < len(cotan.rows):
        heat_triplets.add(cotan.rows[k], cotan.cols[k], short_time * cotan.vr[k])
        k += 1
    var nnz_heat = triplets_to_csc(
        compute.nV, 1, heat_triplets, m, lay.m_heat_ap, m, lay.m_heat_ai, w, lay.w_heat_ax)
    # Poisson operator. NOTE: the cotan-Laplacian is only positive semi-definite, which Eigen's
    # Cholesky refuses, so upstream shifts it by 1e-6 on the diagonal.
    # L + 1e-6 * identity, with no mass term.
    var pois_triplets = Triplets()
    k = 0
    while k < len(cotan.rows):
        pois_triplets.add(cotan.rows[k], cotan.cols[k], cotan.vr[k])
        k += 1
    v = 0
    while v < compute.nV:
        pois_triplets.add(v, v, 1e-6)
        v += 1
    var nnz_pois = triplets_to_csc(
        compute.nV, 1, pois_triplets, m, lay.m_pois_ap, m, lay.m_pois_ai, w, lay.w_pois_ax)

    # The corner->vertex map of the mesh the operators were built on. It differs from the snapshot
    # above whenever the robust laplacian doubles the mesh into its tufted cover.
    i = 0
    while i < 3 * compute.nF:
        m.unsafe_store(lay.m_corner_v_compute + i, compute.F[i])
        i += 1

    # Per-face data the divergence step re-reads on every solve
    var hvif = compute_halfedge_vectors_in_face(compute, face_areas)
    var hcw = compute_halfedge_cotan_weights(compute, face_areas)
    i = 0
    while i < 3 * compute.nF:
        w.unsafe_store(lay.w_hvif + 2 * i, hvif[i].x)
        w.unsafe_store(lay.w_hvif + 2 * i + 1, hvif[i].y)
        w.unsafe_store(lay.w_hcw + i, hcw[i])
        i += 1

    w.unsafe_store(0, Float64(n_v))
    w.unsafe_store(1, Float64(n_f))
    w.unsafe_store(2, Float64(compute.nF))
    w.unsafe_store(9, Float64(compute.nE))
    w.unsafe_store(3, short_time)
    w.unsafe_store(6, Float64(nnz_heat))
    w.unsafe_store(7, Float64(nnz_pois))

    var nnzl_heat = chol_analyze(
        compute.nV, 1, addr_of(m) + lay.m_heat_ap * 8, addr_of(m) + lay.m_heat_ai * 8,
        addr_of(w) + lay.w_heat_ax * 8, addr_of(m) + lay.m_heat_perm * 8)
    if nnzl_heat <= 0:
        return -11 * nnzl_heat - 2
    # The Poisson operator is the same Laplacian with a different diagonal shift, so its index pattern
    # is the heat operator's pattern entry for entry. chol_analyze is a full symbolic factorization,
    # so running it a second time on an identical pattern repeats that work for nothing. The pattern
    # is compared first and the second analyze only runs when it is NOT identical, which keeps this
    # correct for any input where the two operators happen to differ.
    var nnzl_pois = 0
    if _same_pattern(
        compute.nV, addr_of(m) + lay.m_heat_ap * 8, addr_of(m) + lay.m_heat_ai * 8,
        addr_of(m) + lay.m_pois_ap * 8, addr_of(m) + lay.m_pois_ai * 8
    ):
        # Same pattern, so the same ordering and the same nnz(L) as the heat operator.
        nnzl_pois = nnzl_heat
        var q = 0
        while q < compute.nV:
            m.unsafe_store(lay.m_pois_perm + q, m.unsafe_load(lay.m_heat_perm + q))
            q += 1
    else:
        nnzl_pois = chol_analyze(
            compute.nV, 1, addr_of(m) + lay.m_pois_ap * 8, addr_of(m) + lay.m_pois_ai * 8,
            addr_of(w) + lay.w_pois_ax * 8, addr_of(m) + lay.m_pois_perm * 8)
        if nnzl_pois <= 0:
            return -11 * nnzl_pois - 3
    w.unsafe_store(4, Float64(nnzl_heat))
    w.unsafe_store(5, Float64(nnzl_pois))
    w.unsafe_store(8, Float64(2 * lay.li_size(max(nnzl_heat, nnzl_pois)) + 2 * lay.lf_size(max(nnzl_heat, nnzl_pois))))
    return 0


# Cholesky-factorizes both operators. Call once, right after the setup call has sized the arenas.
def heat_factor(n_v: Int, n_f: Int, n_fc: Int, n_ec: Int, w_addr: Int, m_addr: Int, li_addr: Int,
                lf_addr: Int, li_cap: Int, lf_cap: Int) -> Int:
    if w_addr == 0 or m_addr == 0 or li_addr == 0 or lf_addr == 0:
        return -1
    var w = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=w_addr)
    var m = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=m_addr)
    var li = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=li_addr)
    var lf = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=lf_addr)
    var li2 = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=li_addr + li_cap * 8)
    var lf2 = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=lf_addr + lf_cap * 8)
    var lay = HeatLayout(n_v, n_f, n_fc, n_ec)
    var nnzl_heat = Int(w.unsafe_load(4))
    var nnzl_pois = Int(w.unsafe_load(5))

    var r1 = chol_factorize(
        n_v, 1, addr_of(m) + lay.m_heat_ap * 8, addr_of(m) + lay.m_heat_ai * 8, addr_of(w) + lay.w_heat_ax * 8,
        addr_of(m) + lay.m_heat_perm * 8, addr_of(li), addr_of(li) + (n_v + 1) * 8,
        addr_of(lf), addr_of(lf) + nnzl_heat * 8, nnzl_heat)
    if r1 != 0:
        return r1
    # The Poisson operator is the same Laplacian with a different diagonal, so L has the same pattern
    # for both. The first factorization just wrote that pattern into `li`; handing it back as the
    # borrowed pattern for the second skips a symbolic pass over the same matrix. The borrow is
    # checked as it is read and falls back to recomputing if it does not hold, so a mismatch is
    # correct, just not faster.
    var r2 = chol_factorize(
        n_v, 1, addr_of(m) + lay.m_pois_ap * 8, addr_of(m) + lay.m_pois_ai * 8, addr_of(w) + lay.w_pois_ax * 8,
        addr_of(m) + lay.m_pois_perm * 8, addr_of(li2), addr_of(li2) + (n_v + 1) * 8,
        addr_of(lf2), addr_of(lf2) + nnzl_pois * 8, nnzl_pois,
        addr_of(li), addr_of(li) + (n_v + 1) * 8)
    return r2


# HeatMethodDistanceSolver::computeDistance(std::vector<Vertex>)
def heat_compute_distance(
    n_v: Int,
    n_f: Int,
    n_fc: Int,
    n_ec: Int,
    w_addr: Int,
    m_addr: Int,
    li_addr: Int,
    lf_addr: Int,
    li2_addr: Int,
    lf2_addr: Int,
    srcs_addr: Int,
    n_srcs: Int,
    out_addr: Int,
) -> Int:
    if w_addr == 0 or m_addr == 0 or li_addr == 0 or lf_addr == 0 or srcs_addr == 0 or out_addr == 0:
        return -1
    var w = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=w_addr)
    var m = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=m_addr)
    var li = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=li_addr)
    var lf = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=lf_addr)
    var li2 = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=li2_addr)
    var lf2 = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=lf2_addr)
    var srcs = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=srcs_addr)
    var out = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=out_addr)
    var lay = HeatLayout(n_v, n_f, n_fc, n_ec)

    # === Build RHS
    var rhs = w.unsafe_offset(lay.w_work)
    var v = 0
    while v < n_v:
        rhs.unsafe_store(v, 0.0)
        v += 1
    # Every source index is checked rather than skipped. Upstream indexes its vertex array with the
    # caller's list directly, so a negative index walks off the front and a too-large one past the
    # end; dropping the bad index instead would leave a right-hand side of all zeros and return a
    # silent constant field, which is a number that is finite and wrong. It is reported instead.
    var s = 0
    while s < n_srcs:
        var sv = srcs.unsafe_load(s)
        if sv < 0 or sv >= n_v:
            return -3
        rhs.unsafe_store(sv, rhs.unsafe_load(sv) + 1.0)
        s += 1
    if n_srcs <= 0:
        # An empty source set has no zero to shift to: upstream returns a field of NaNs. Reporting
        # the empty query keeps this from handing back a finite all-zero distance field instead.
        return -4

    var status = 0
    var dist = heat_compute_distance_rhs(n_v, n_fc, w, m, li, lf, li2, lf2, lay, rhs, status)
    if status != 0:
        return status

    # === Shift distance to put zero at the source set. Helper to measure distance between two
    # points given their barycentric coordinates (Shindler & Chen 2012, section 3.2).
    s = 0
    var dist_diff_at_source = 0.0
    var weight_sum = 0.0
    while s < n_srcs:
        var sv = srcs.unsafe_load(s)
        # Already range-checked above, so sv is a real vertex here.
        var he0 = m.unsafe_load(lay.m_first_he + sv)
        if he0 >= 0:
                var h1 = he0 - he0 % 3 + (he0 % 3 + 1) % 3
                var h2 = h1 - h1 % 3 + (h1 % 3 + 1) % 3
                var l0 = w.unsafe_load(lay.w_corner_len + he0)
                var l1 = w.unsafe_load(lay.w_corner_len + h1)
                var l2 = w.unsafe_load(lay.w_corner_len + h2)
                var bary = Bary3(1.0, 0.0, 0.0)
                var i = 0
                while i < 3:
                    var target = Bary3(0.0, 0.0, 0.0)
                    if i == 0:
                        target = Bary3(1.0, 0.0, 0.0)
                    elif i == 1:
                        target = Bary3(0.0, 1.0, 0.0)
                    else:
                        target = Bary3(0.0, 0.0, 1.0)
                    var b_vec = target - bary
                    var d2 = 0.0
                    var j = 0
                    while j < 3:
                        var el = l0
                        if j == 1:
                            el = l1
                        elif j == 2:
                            el = l2
                        var en = l0
                        if (j + 1) % 3 == 1:
                            en = l1
                        elif (j + 1) % 3 == 2:
                            en = l2
                        d2 += el * el * b_vec[j] * b_vec[(j + 1) % 3]
                        j += 1
                    if not (d2 <= 0.0):
                        d2 = 0.0
                    var expected = sqrt(-d2)
                    var hi = he0 - he0 % 3 + (he0 % 3 + i) % 3
                    var act = dist.unsafe_load(m.unsafe_load(lay.m_corner_v + hi))
                    var wt = bary[i]
                    dist_diff_at_source += (act - expected) * wt
                    weight_sum += wt
                    i += 1
        s += 1
    if weight_sum != 0.0:
        dist_diff_at_source /= weight_sum

    var shift = -dist_diff_at_source
    v = 0
    while v < n_v:
        out.unsafe_store(v, dist.unsafe_load(v) + shift)
        v += 1
    return 0


# HeatMethodDistanceSolver::computeDistanceRHS
def heat_compute_distance_rhs(
    n_v: Int,
    n_fc: Int,
    w: Pointer[Float64, AnyOrigin[mut=True]],
    m: Pointer[Int, AnyOrigin[mut=True]],
    li: Pointer[Int, AnyOrigin[mut=True]],
    lf: Pointer[Float64, AnyOrigin[mut=True]],
    li2: Pointer[Int, AnyOrigin[mut=True]],
    lf2: Pointer[Float64, AnyOrigin[mut=True]],
    lay: HeatLayout,
    rhs: Pointer[Float64, AnyOrigin[mut=True]],
    var status: Int,
) -> Pointer[Float64, AnyOrigin[mut=True]]:
    # The RHS is the caller's `rhs` region, which must not overlap the solution: chol_solve reads b
    # and writes x in the same call, and a forward substitution that overwrites x[i] before reading
    # b[i] would silently corrupt the solve. The last three nV blocks are the working vectors.
    var work = w.unsafe_offset(lay.w_work + lay.nV)
    var heat_vec = work
    var divergence = work.unsafe_offset(n_v)
    var dist_vec = work.unsafe_offset(2 * n_v)

    # === Solve heat
    var r = chol_solve(
        n_v, 1, addr_of(li), addr_of(li) + (n_v + 1) * 8,
        addr_of(lf), addr_of(lf) + Int(w.unsafe_load(4)) * 8, addr_of(m) + lay.m_heat_perm * 8,
        addr_of(rhs), addr_of(heat_vec))
    status = r
    if r != 0:
        return heat_vec

    # === Normalize in each face and evaluate divergence
    var v = 0
    while v < n_v:
        divergence.unsafe_store(v, 0.0)
        v += 1
    var f = 0
    while f < n_fc:
        var h0 = 3 * f
        var grad_u_dir = Vector2(0.0, 0.0)
        var i = 0
        while i < 3:
            var he = h0 + i
            var hn = he - he % 3 + (he % 3 + 1) % 3
            var e_perp = Vector2(w.unsafe_load(lay.w_hvif + 2 * hn + 1) * -1.0,
                                 w.unsafe_load(lay.w_hvif + 2 * hn))
            grad_u_dir = grad_u_dir + e_perp * heat_vec.unsafe_load(m.unsafe_load(lay.m_corner_v_compute + he))
            i += 1
        # normalizeCutoff() with its default mag = 0, as upstream calls it. A cutoff would have to sit
        # below the rounding of the heat solve to be safe: the gradient really does pass through zero
        # at critical points of the heat field, and those faces carry a real (if small) contribution.
        grad_u_dir = v2_normalize_cutoff(grad_u_dir)
        i = 0
        while i < 3:
            var he = h0 + i
            var v_in = Vector2(w.unsafe_load(lay.w_hvif + 2 * he), w.unsafe_load(lay.w_hvif + 2 * he + 1))
            var val = w.unsafe_load(lay.w_hcw + he) * v2_dot(v_in, grad_u_dir)
            var tail = m.unsafe_load(lay.m_corner_v_compute + he)
            var tip = m.unsafe_load(
                lay.m_corner_v_compute + (he - he % 3 + (he % 3 + 1) % 3)
            )
            divergence.unsafe_store(tail, divergence.unsafe_load(tail) + val)
            divergence.unsafe_store(tip, divergence.unsafe_load(tip) - val)
            i += 1
        f += 1

    # === Integrate divergence to get distance
    r = chol_solve(
        n_v, 1, addr_of(li2), addr_of(li2) + (n_v + 1) * 8,
        addr_of(lf2), addr_of(lf2) + Int(w.unsafe_load(5)) * 8, addr_of(m) + lay.m_pois_perm * 8,
        addr_of(divergence), addr_of(dist_vec))
    if r != 0:
        return divergence
    return dist_vec
