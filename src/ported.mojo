# The single compilation unit for the port. Everything upstream lives in geometry-central and
# potpourri3d; the modules below mirror that layout, one file per upstream file and in upstream
# order, and this file pulls them into one shared library.
#
# The kernels keep upstream's names, argument order and branch structure. Two things are forced by
# the dialect and are confined to this file and to the ctypes glue, never to a kernel's structure:
#
#   * `@export` symbols are only emitted from the file passed to `mojo build`, so every kernel
#     module ends in a plain `def` that takes `Int` addresses and the `@export` lives here.
#   * buffers cross the C ABI as `Int` addresses and are rebuilt as Pointers inside the wrapper.
#
# The persistent solver state lives in caller-owned arenas whose layout each kernel module fixes in
# a `*Layout` struct; python/mojo_potpourri3d/_lib.py mirrors those layouts to size the buffers.

from sparse import chol_analyze, chol_factorize, chol_solve
from heat_method_distance import heat_setup, heat_factor, heat_compute_distance, heat_layout
from mesh import cotan_laplacian, face_areas, edges, vertex_areas
from point_position_geometry import point_cloud_layout, point_cloud_prepare


@export("pp3d_chol_analyze")
def _chol_analyze(
    n: Int, w: Int, ap_addr: Int, ai_addr: Int, ax_addr: Int, perm_addr: Int
) abi("C") -> Int:
    return chol_analyze(n, w, ap_addr, ai_addr, ax_addr, perm_addr)


@export("pp3d_chol_factorize")
def _chol_factorize(
    n: Int,
    w: Int,
    ap_addr: Int,
    ai_addr: Int,
    ax_addr: Int,
    perm_addr: Int,
    lp_addr: Int,
    li_addr: Int,
    lx_addr: Int,
    dx_addr: Int,
    lx_cap: Int = -1,
    reuse_lp_addr: Int = 0,
    reuse_li_addr: Int = 0,
) abi("C") -> Int:
    return chol_factorize(
        n, w, ap_addr, ai_addr, ax_addr, perm_addr, lp_addr, li_addr, lx_addr, dx_addr,
        lx_cap, reuse_lp_addr, reuse_li_addr)


@export("pp3d_chol_solve")
def _chol_solve(
    n: Int,
    w: Int,
    lp_addr: Int,
    li_addr: Int,
    lx_addr: Int,
    dx_addr: Int,
    perm_addr: Int,
    b_addr: Int,
    x_addr: Int,
) abi("C") -> Int:
    return chol_solve(n, w, lp_addr, li_addr, lx_addr, dx_addr, perm_addr, b_addr, x_addr)


@export("pp3d_heat_layout")
def _heat_layout(n_v: Int, n_f: Int, n_fc: Int, n_ec: Int, out_addr: Int) abi("C") -> Int:
    return heat_layout(n_v, n_f, n_fc, n_ec, out_addr)


@export("pp3d_heat_setup")
def _heat_setup(
    n_v: Int,
    n_f: Int,
    verts_addr: Int,
    faces_addr: Int,
    t_coef: Float64,
    use_robust: Int,
    w_addr: Int,
    m_addr: Int,
    li_addr: Int,
) abi("C") -> Int:
    return heat_setup(
        n_v, n_f, verts_addr, faces_addr, t_coef, use_robust, w_addr, m_addr, li_addr)


@export("pp3d_heat_factor")
def _heat_factor(
    n_v: Int, n_f: Int, n_fc: Int, n_ec: Int, w_addr: Int, m_addr: Int, li_addr: Int, lf_addr: Int,
    li_cap: Int, lf_cap: Int
) abi("C") -> Int:
    return heat_factor(n_v, n_f, n_fc, n_ec, w_addr, m_addr, li_addr, lf_addr, li_cap, lf_cap)


@export("pp3d_heat_compute_distance")
def _heat_compute_distance(
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
) abi("C") -> Int:
    return heat_compute_distance(
        n_v, n_f, n_fc, n_ec, w_addr, m_addr, li_addr, lf_addr, li2_addr, lf2_addr, srcs_addr, n_srcs, out_addr)


@export("pp3d_cotan_laplacian")
def _cotan_laplacian(
    n_v: Int,
    n_f: Int,
    verts_addr: Int,
    faces_addr: Int,
    denom_eps: Float64,
    rows_addr: Int,
    cols_addr: Int,
    data_addr: Int,
) abi("C") -> Int:
    return cotan_laplacian(
        n_v, n_f, verts_addr, faces_addr, denom_eps, rows_addr, cols_addr, data_addr)


@export("pp3d_face_areas")
def _face_areas(
    n_v: Int, n_f: Int, verts_addr: Int, faces_addr: Int, out_addr: Int
) abi("C") -> Int:
    return face_areas(n_v, n_f, verts_addr, faces_addr, out_addr)


@export("pp3d_vertex_areas")
def _vertex_areas(
    n_v: Int, n_f: Int, verts_addr: Int, faces_addr: Int, out_addr: Int
) abi("C") -> Int:
    return vertex_areas(n_v, n_f, verts_addr, faces_addr, out_addr)


@export("pp3d_edges")
def _edges(n_v: Int, n_f: Int, faces_addr: Int, out_addr: Int) abi("C") -> Int:
    return edges(n_v, n_f, faces_addr, out_addr)


@export("pp3d_pc_layout")
def _pc_layout(n_p: Int, n_n: Int, out_addr: Int) abi("C") -> Int:
    return point_cloud_layout(n_p, n_n, out_addr)


@export("pp3d_pc_prepare")
def _pc_prepare(
    n_p: Int, n_n: Int, with_deg: Int, verts_addr: Int, w_addr: Int, m_addr: Int
) abi("C") -> Int:
    return point_cloud_prepare(n_p, n_n, with_deg, verts_addr, w_addr, m_addr)
