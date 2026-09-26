"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` rejects parametric
functions and a pointer with an inferred origin is parametric. The Python side
owns all memory, including scratch, so nothing here allocates and nothing here
can leak.

Each entry point takes the whole halfedge connectivity (`he_vertex`, `he_next`,
`he_twin`, `he_face`, `v_halfedge`, `f_halfedge`) even when a particular kernel
only reads part of it, because a `Pointer` cannot be null.
"""

from mojopp3d.embedded_geometry import (
    compute_corner_angles_embedded,
    compute_edge_lengths,
    compute_face_normals,
    compute_vertex_normals,
    compute_vertex_tangent_basis,
)
from mojopp3d.intrinsic_geometry import (
    compute_corner_angles,
    compute_corner_scaled_angles,
    compute_cotan_laplacian,
    compute_edge_cotan_weights,
    compute_face_areas,
    compute_halfedge_cotan_weights,
    compute_halfedge_vectors_in_face,
    compute_halfedge_vectors_in_vertex,
    compute_transport_vectors_along_halfedge,
    compute_vertex_dual_areas,
    compute_vertex_angle_sums,
    compute_vertex_connection_laplacian,
)
from mojopp3d.mesh import HalfedgeMesh, build_halfedge_mesh

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]


def fp(address: Int) -> Ptr:
    return Ptr(unsafe_from_address=address)


def ip(address: Int) -> IPtr:
    return IPtr(unsafe_from_address=address)


def mesh(
    he_vertex: Int,
    he_next: Int,
    he_twin: Int,
    he_face: Int,
    v_halfedge: Int,
    f_halfedge: Int,
    n_he: Int,
    n_vertices: Int,
    n_faces: Int,
) -> HalfedgeMesh:
    return HalfedgeMesh(
        he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge,
        n_he, n_vertices, n_faces,
    )


@export("mpp3d_build_halfedge_mesh")
def mpp3d_build_halfedge_mesh(
    F: Int,
    n_faces: Int,
    n_vertices: Int,
    he_vertex: Int,
    he_next: Int,
    he_twin: Int,
    he_face: Int,
    v_halfedge: Int,
    f_halfedge: Int,
    keys: Int,
    vals: Int,
    tmpk: Int,
    tmpv: Int,
    count: Int,
    twin_in: Int,
) abi("C") -> Int:
    return build_halfedge_mesh(
        ip(F), n_faces, n_vertices, ip(he_vertex), ip(he_next), ip(he_twin),
        ip(he_face), ip(v_halfedge), ip(f_halfedge), ip(keys), ip(vals), ip(tmpk),
        ip(tmpv), ip(count), twin_in,
    )


@export("mpp3d_compute_edge_lengths")
def mpp3d_compute_edge_lengths(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    positions: Int, edge_lengths: Int,
) abi("C"):
    compute_edge_lengths(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(positions), fp(edge_lengths),
    )


@export("mpp3d_compute_face_areas")
def mpp3d_compute_face_areas(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    edge_lengths: Int, face_areas: Int,
) abi("C"):
    compute_face_areas(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(edge_lengths), fp(face_areas),
    )


@export("mpp3d_compute_vertex_dual_areas")
def mpp3d_compute_vertex_dual_areas(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    face_areas: Int, vertex_dual_areas: Int,
) abi("C"):
    compute_vertex_dual_areas(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(face_areas), fp(vertex_dual_areas),
    )


@export("mpp3d_compute_corner_angles")
def mpp3d_compute_corner_angles(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    edge_lengths: Int, corner_angles: Int,
) abi("C"):
    compute_corner_angles(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(edge_lengths), fp(corner_angles),
    )


@export("mpp3d_compute_vertex_angle_sums")
def mpp3d_compute_vertex_angle_sums(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    corner_angles: Int, vertex_angle_sums: Int,
) abi("C"):
    compute_vertex_angle_sums(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(corner_angles), fp(vertex_angle_sums),
    )


@export("mpp3d_compute_corner_scaled_angles")
def mpp3d_compute_corner_scaled_angles(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    corner_angles: Int, vertex_angle_sums: Int, corner_scaled_angles: Int,
) abi("C"):
    compute_corner_scaled_angles(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(corner_angles), fp(vertex_angle_sums), fp(corner_scaled_angles),
    )


@export("mpp3d_compute_halfedge_cotan_weights")
def mpp3d_compute_halfedge_cotan_weights(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    edge_lengths: Int, face_areas: Int, halfedge_cotan_weights: Int,
) abi("C"):
    compute_halfedge_cotan_weights(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(edge_lengths), fp(face_areas), fp(halfedge_cotan_weights),
    )


@export("mpp3d_compute_edge_cotan_weights")
def mpp3d_compute_edge_cotan_weights(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    edge_lengths: Int, face_areas: Int, edge_cotan_weights: Int,
) abi("C"):
    compute_edge_cotan_weights(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(edge_lengths), fp(face_areas), fp(edge_cotan_weights),
    )


@export("mpp3d_compute_halfedge_vectors_in_face")
def mpp3d_compute_halfedge_vectors_in_face(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    edge_lengths: Int, face_areas: Int, halfedge_vec: Int,
) abi("C"):
    compute_halfedge_vectors_in_face(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(edge_lengths), fp(face_areas), fp(halfedge_vec),
    )


@export("mpp3d_compute_halfedge_vectors_in_vertex")
def mpp3d_compute_halfedge_vectors_in_vertex(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    edge_lengths: Int, corner_scaled_angles: Int, halfedge_vec: Int,
) abi("C"):
    compute_halfedge_vectors_in_vertex(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(edge_lengths), fp(corner_scaled_angles), fp(halfedge_vec),
    )


@export("mpp3d_compute_transport_vectors_along_halfedge")
def mpp3d_compute_transport_vectors_along_halfedge(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    halfedge_vec: Int, transport: Int,
) abi("C"):
    compute_transport_vectors_along_halfedge(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(halfedge_vec), fp(transport),
    )


@export("mpp3d_compute_cotan_laplacian")
def mpp3d_compute_cotan_laplacian(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    edge_cotan_weights: Int, tri_i: Int, tri_j: Int, tri_v: Int,
) abi("C") -> Int:
    return compute_cotan_laplacian(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(edge_cotan_weights), ip(tri_i), ip(tri_j), fp(tri_v),
    )


@export("mpp3d_compute_corner_angles_embedded")
def mpp3d_compute_corner_angles_embedded(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    positions: Int, corner_angles: Int,
) abi("C"):
    compute_corner_angles_embedded(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(positions), fp(corner_angles),
    )


@export("mpp3d_compute_face_normals")
def mpp3d_compute_face_normals(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    positions: Int, face_normals: Int,
) abi("C"):
    compute_face_normals(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(positions), fp(face_normals),
    )


@export("mpp3d_compute_vertex_normals")
def mpp3d_compute_vertex_normals(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    face_normals: Int, corner_angles: Int, vertex_normals: Int,
) abi("C"):
    compute_vertex_normals(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(face_normals), fp(corner_angles), fp(vertex_normals),
    )


@export("mpp3d_compute_vertex_tangent_basis")
def mpp3d_compute_vertex_tangent_basis(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    positions: Int, vertex_normals: Int, halfedge_vec: Int, tangent_basis: Int,
) abi("C"):
    compute_vertex_tangent_basis(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(positions), fp(vertex_normals), fp(halfedge_vec), fp(tangent_basis),
    )


# ---------------------------------------------------------------- linalg / sparse

from mojopp3d.api import cotan_laplacian_triplets, face_areas, vertex_areas
from mojopp3d.linalg import (
    ldl_numeric,
    ldl_numeric_c,
    ldl_solve,
    ldl_solve_c,
    ldl_symbolic,
    rcm,
)
from mojopp3d.sparse import adjacency_pattern, permute_upper


@export("mpp3d_rcm")
def mpp3d_rcm(n: Int, Ap: Int, Ai: Int, degree: Int, visited: Int, queue: Int, order: Int) abi("C"):
    rcm(n, ip(Ap), ip(Ai), ip(degree), ip(visited), ip(queue), ip(order))


@export("mpp3d_permute_upper")
def mpp3d_permute_upper(
    n: Int, ti: Int, tj: Int, txr: Int, txi: Int, nnzT: Int,
    degree: Int, visited: Int, queue: Int, perm: Int, iperm: Int,
    Ap: Int, Ai: Int, Axr: Int, Axi: Int, ApNew: Int,
    keys: Int, aux: Int, keybuf: Int, auxbuf: Int,
) abi("C") -> Int:
    return permute_upper(
        n, ip(ti), ip(tj), fp(txr), fp(txi), nnzT, ip(degree), ip(visited), ip(queue),
        ip(perm), ip(iperm), ip(Ap), ip(Ai), fp(Axr), fp(Axi), ip(ApNew),
        ip(keys), ip(aux), ip(keybuf), ip(auxbuf),
    )


@export("mpp3d_ldl_symbolic")
def mpp3d_ldl_symbolic(
    n: Int, Ap: Int, Ai: Int, S_p: Int, S_i: Int,
    R_p: Int, R_i: Int, R_pos: Int,
    U_head: Int, U_i: Int, U_next: Int,
    mark: Int, cov: Int, gather: Int, aux: Int, cnt: Int, cursor: Int,
    info: Int, limit: Int,
) abi("C") -> Int:
    return ldl_symbolic(
        n, ip(Ap), ip(Ai), ip(S_p), ip(S_i), ip(R_p), ip(R_i), ip(R_pos),
        ip(U_head), ip(U_i), ip(U_next), ip(mark), ip(cov), ip(gather),
        ip(aux), ip(cnt), ip(cursor), ip(info), limit,
    )


@export("mpp3d_ldl_numeric")
def mpp3d_ldl_numeric(
    n: Int, nnzL: Int, Ap: Int, Ai: Int, Ax: Int,
    S_p: Int, S_i: Int, R_p: Int, R_i: Int, R_pos: Int, upto: Int,
    Lx: Int, D: Int, Y: Int, W: Int,
) abi("C"):
    ldl_numeric(
        n, nnzL, ip(Ap), ip(Ai), fp(Ax), ip(S_p), ip(S_i), ip(R_p), ip(R_i),
        ip(R_pos), ip(upto), fp(Lx), fp(D), fp(Y), fp(W),
    )


@export("mpp3d_ldl_solve")
def mpp3d_ldl_solve(n: Int, S_p: Int, S_i: Int, Lx: Int, D: Int, B: Int, X: Int, nrhs: Int) abi("C"):
    ldl_solve(n, ip(S_p), ip(S_i), fp(Lx), fp(D), fp(B), fp(X), nrhs)


@export("mpp3d_ldl_numeric_c")
def mpp3d_ldl_numeric_c(
    n: Int, nnzL: Int, Ap: Int, Ai: Int, Axr: Int, Axi: Int,
    S_p: Int, S_i: Int, R_p: Int, R_i: Int, R_pos: Int, upto: Int,
    Lxr: Int, Lxi: Int, Dr: Int, Di: Int, Yr: Int, Yi: Int, Wr: Int, Wi: Int,
) abi("C"):
    ldl_numeric_c(
        n, nnzL, ip(Ap), ip(Ai), fp(Axr), fp(Axi), ip(S_p), ip(S_i), ip(R_p),
        ip(R_i), ip(R_pos), ip(upto),
        fp(Lxr), fp(Lxi), fp(Dr), fp(Di), fp(Yr), fp(Yi), fp(Wr), fp(Wi),
    )


@export("mpp3d_ldl_solve_c")
def mpp3d_ldl_solve_c(
    n: Int, S_p: Int, S_i: Int, Lxr: Int, Lxi: Int, Dr: Int, Di: Int,
    Br: Int, Bi: Int, Xr: Int, Xi: Int, nrhs: Int,
) abi("C"):
    ldl_solve_c(n, ip(S_p), ip(S_i), fp(Lxr), fp(Lxi), fp(Dr), fp(Di), fp(Br), fp(Bi), fp(Xr), fp(Xi), nrhs)


# ------------------------------------------------- potpourri3d.mesh free functions

@export("mpp3d_face_areas")
def mpp3d_face_areas(V: Int, F: Int, n_faces: Int, dst: Int) abi("C") -> Int:
    return face_areas(fp(V), ip(F), n_faces, fp(dst))


@export("mpp3d_vertex_areas")
def mpp3d_vertex_areas(V: Int, F: Int, n_faces: Int, n_vertices: Int, scratch: Int, dst: Int) abi("C") -> Int:
    return vertex_areas(fp(V), ip(F), n_faces, n_vertices, fp(scratch), fp(dst))


@export("mpp3d_cotan_laplacian_triplets")
def mpp3d_cotan_laplacian_triplets(
    V: Int, F: Int, n_faces: Int, denom_eps: Float64, mat_i: Int, mat_j: Int, mat_data: Int,
) abi("C") -> Int:
    return cotan_laplacian_triplets(fp(V), ip(F), n_faces, denom_eps, ip(mat_i), ip(mat_j), fp(mat_data))


@export("mpp3d_compute_vertex_connection_laplacian")
def mpp3d_compute_vertex_connection_laplacian(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    edge_cotan_weights: Int, transport: Int, tri_i: Int, tri_j: Int, tri_r: Int, tri_im: Int,
) abi("C") -> Int:
    return compute_vertex_connection_laplacian(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(edge_cotan_weights), fp(transport), ip(tri_i), ip(tri_j), fp(tri_r), fp(tri_im),
    )


@export("mpp3d_adjacency_pattern")
def mpp3d_adjacency_pattern(n: Int, ti: Int, tj: Int, nnzT: Int, Ap: Int, Ai: Int) abi("C") -> Int:
    return adjacency_pattern(n, ip(ti), ip(tj), nnzT, ip(Ap), ip(Ai))


# --------------------------------------------------- HeatMethodDistanceSolver

from mojopp3d.heat_method import (
    build_rhs,
    compute_divergence,
    shift_distance,
    source_face_corner,
)


@export("mpp3d_heat_build_rhs")
def mpp3d_heat_build_rhs(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    sources: Int, n_sources: Int, rhs: Int,
) abi("C"):
    build_rhs(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        ip(sources), n_sources, fp(rhs),
    )


@export("mpp3d_heat_compute_divergence")
def mpp3d_heat_compute_divergence(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    halfedge_vec_in_face: Int, halfedge_cotan_weights: Int, heat_vec: Int, divergence: Int,
) abi("C"):
    compute_divergence(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(halfedge_vec_in_face), fp(halfedge_cotan_weights), fp(heat_vec), fp(divergence),
    )


@export("mpp3d_heat_shift_distance")
def mpp3d_heat_shift_distance(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    dist_vec: Int, edge_lengths: Int, sources: Int, n_sources: Int, shift: Int,
) abi("C") -> Float64:
    return shift_distance(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        fp(dist_vec), fp(edge_lengths), ip(sources), n_sources, fp(shift),
    )


@export("mpp3d_source_face_corner")
def mpp3d_source_face_corner(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int, v: Int, dst: Int,
) abi("C"):
    var r = source_face_corner(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces), v
    )
    ip(dst).unsafe_store(0, Int64(r[0]))
    ip(dst).unsafe_store(1, Int64(r[1]))


# ------------------------------------------------------ VectorHeatMethodSolver

from mojopp3d.vector_heat_method import extend_scalar_rhs, transport_rhs


@export("mpp3d_vector_extend_scalar_rhs")
def mpp3d_vector_extend_scalar_rhs(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    sources: Int, values: Int, n_sources: Int, data_rhs: Int, indicator_rhs: Int,
) abi("C"):
    extend_scalar_rhs(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        ip(sources), fp(values), n_sources, fp(data_rhs), fp(indicator_rhs),
    )


@export("mpp3d_vector_transport_rhs")
def mpp3d_vector_transport_rhs(
    he_vertex: Int, he_next: Int, he_twin: Int, he_face: Int, v_halfedge: Int,
    f_halfedge: Int, n_he: Int, n_vertices: Int, n_faces: Int,
    sources: Int, vectors: Int, n_sources: Int, rhs_re: Int, rhs_im: Int,
) abi("C"):
    transport_rhs(
        mesh(he_vertex, he_next, he_twin, he_face, v_halfedge, f_halfedge, n_he, n_vertices, n_faces),
        ip(sources), fp(vectors), n_sources, fp(rhs_re), fp(rhs_im),
    )


# ------------------------------------------- general mesh / tufted cover / flips

from mojopp3d.general_mesh import (
    GenMesh,
    build_general_mesh,
    duplicate_face as gen_duplicate_face,
    invert_orientation as gen_invert_orientation,
    separate_to_new_edge as gen_separate_to_new_edge,
    flip as gen_flip,
    write_faces,
    write_halfedge_edge_lengths,
    write_twins,
)
from mojopp3d.tufted import (
    build_intrinsic_tufted_cover,
    flip_to_delaunay,
    mollify_intrinsic,
)


def gmesh(
    he_vertex: Int, he_next: Int, he_face: Int, he_edge: Int, he_orient: Int,
    he_sibling: Int, e_halfedge: Int, f_halfedge: Int, counts: Int,
) -> GenMesh:
    return GenMesh(
        he_vertex, he_next, he_face, he_edge, he_orient, he_sibling,
        e_halfedge, f_halfedge, counts,
    )


@export("mpp3d_build_general_mesh")
def mpp3d_build_general_mesh(
    F: Int, n_faces: Int, n_vertices: Int, he_vertex: Int, he_next: Int,
    he_face: Int, he_edge: Int, he_orient: Int, he_sibling: Int,
    e_halfedge: Int, f_halfedge: Int, counts: Int, keys: Int, vals: Int,
    tmpk: Int, tmpv: Int, bucket: Int,
) abi("C"):
    build_general_mesh(
        ip(F), n_faces, n_vertices,
        gmesh(
            he_vertex, he_next, he_face, he_edge, he_orient, he_sibling,
            e_halfedge, f_halfedge, counts,
        ),
        ip(keys), ip(vals), ip(tmpk), ip(tmpv), ip(bucket),
    )


@export("mpp3d_general_write_faces")
def mpp3d_general_write_faces(
    he_vertex: Int, he_next: Int, he_face: Int, he_edge: Int, he_orient: Int,
    he_sibling: Int, e_halfedge: Int, f_halfedge: Int, counts: Int, F: Int,
) abi("C"):
    write_faces(
        gmesh(
            he_vertex, he_next, he_face, he_edge, he_orient, he_sibling,
            e_halfedge, f_halfedge, counts,
        ),
        ip(F),
    )


@export("mpp3d_general_duplicate_face")
def mpp3d_general_duplicate_face(
    he_vertex: Int, he_next: Int, he_face: Int, he_edge: Int, he_orient: Int,
    he_sibling: Int, e_halfedge: Int, f_halfedge: Int, counts: Int, f: Int,
) abi("C") -> Int:
    return gen_duplicate_face(
        gmesh(
            he_vertex, he_next, he_face, he_edge, he_orient, he_sibling,
            e_halfedge, f_halfedge, counts,
        ),
        f,
    )


@export("mpp3d_general_invert_orientation")
def mpp3d_general_invert_orientation(
    he_vertex: Int, he_next: Int, he_face: Int, he_edge: Int, he_orient: Int,
    he_sibling: Int, e_halfedge: Int, f_halfedge: Int, counts: Int, f: Int,
) abi("C"):
    gen_invert_orientation(
        gmesh(
            he_vertex, he_next, he_face, he_edge, he_orient, he_sibling,
            e_halfedge, f_halfedge, counts,
        ),
        f,
    )


@export("mpp3d_general_separate_to_new_edge")
def mpp3d_general_separate_to_new_edge(
    he_vertex: Int, he_next: Int, he_face: Int, he_edge: Int, he_orient: Int,
    he_sibling: Int, e_halfedge: Int, f_halfedge: Int, counts: Int,
    he_a: Int, he_b: Int,
) abi("C") -> Int:
    return gen_separate_to_new_edge(
        gmesh(
            he_vertex, he_next, he_face, he_edge, he_orient, he_sibling,
            e_halfedge, f_halfedge, counts,
        ),
        he_a, he_b,
    )


@export("mpp3d_general_flip")
def mpp3d_general_flip(
    he_vertex: Int, he_next: Int, he_face: Int, he_edge: Int, he_orient: Int,
    he_sibling: Int, e_halfedge: Int, f_halfedge: Int, counts: Int, e: Int,
) abi("C") -> Int:
    if gen_flip(
        gmesh(
            he_vertex, he_next, he_face, he_edge, he_orient, he_sibling,
            e_halfedge, f_halfedge, counts,
        ),
        e,
    ):
        return 1
    return 0


@export("mpp3d_mollify_intrinsic")
def mpp3d_mollify_intrinsic(
    he_vertex: Int, he_next: Int, he_face: Int, he_edge: Int, he_orient: Int,
    he_sibling: Int, e_halfedge: Int, f_halfedge: Int, counts: Int,
    edge_lengths: Int, n_edges: Int, relative_factor: Float64,
) abi("C") -> Float64:
    return mollify_intrinsic(
        gmesh(
            he_vertex, he_next, he_face, he_edge, he_orient, he_sibling,
            e_halfedge, f_halfedge, counts,
        ),
        fp(edge_lengths), n_edges, relative_factor,
    )


@export("mpp3d_build_intrinsic_tufted_cover")
def mpp3d_build_intrinsic_tufted_cover(
    he_vertex: Int, he_next: Int, he_face: Int, he_edge: Int, he_orient: Int,
    he_sibling: Int, e_halfedge: Int, f_halfedge: Int, counts: Int,
    edge_lengths: Int, n_orig_faces: Int, n_orig_edges: Int, other_sheet: Int,
    is_front: Int, is_orig_edge: Int, edge_faces: Int, max_faces: Int,
    max_he: Int, max_edges: Int,
) abi("C") -> Int:
    return build_intrinsic_tufted_cover(
        gmesh(
            he_vertex, he_next, he_face, he_edge, he_orient, he_sibling,
            e_halfedge, f_halfedge, counts,
        ),
        fp(edge_lengths), n_orig_faces, n_orig_edges, ip(other_sheet),
        ip(is_front), ip(is_orig_edge), ip(edge_faces), max_faces, max_he,
        max_edges,
    )


@export("mpp3d_flip_to_delaunay")
def mpp3d_flip_to_delaunay(
    he_vertex: Int, he_next: Int, he_face: Int, he_edge: Int, he_orient: Int,
    he_sibling: Int, e_halfedge: Int, f_halfedge: Int, counts: Int,
    edge_lengths: Int, n_edges: Int, queue: Int, in_queue: Int,
    queue_cap: Int, delaunay_eps: Float64,
) abi("C") -> Int:
    return flip_to_delaunay(
        gmesh(
            he_vertex, he_next, he_face, he_edge, he_orient, he_sibling,
            e_halfedge, f_halfedge, counts,
        ),
        fp(edge_lengths), n_edges, ip(queue), ip(in_queue), queue_cap,
        delaunay_eps,
    )


@export("mpp3d_general_write_twins")
def mpp3d_general_write_twins(
    he_vertex: Int, he_next: Int, he_face: Int, he_edge: Int, he_orient: Int,
    he_sibling: Int, e_halfedge: Int, f_halfedge: Int, counts: Int,
    twins: Int, index_of: Int,
) abi("C"):
    write_twins(
        gmesh(
            he_vertex, he_next, he_face, he_edge, he_orient, he_sibling,
            e_halfedge, f_halfedge, counts,
        ),
        ip(twins), ip(index_of),
    )


@export("mpp3d_general_write_halfedge_edge_lengths")
def mpp3d_general_write_halfedge_edge_lengths(
    he_vertex: Int, he_next: Int, he_face: Int, he_edge: Int, he_orient: Int,
    he_sibling: Int, e_halfedge: Int, f_halfedge: Int, counts: Int,
    edge_lengths: Int, dst: Int,
) abi("C"):
    write_halfedge_edge_lengths(
        gmesh(
            he_vertex, he_next, he_face, he_edge, he_orient, he_sibling,
            e_halfedge, f_halfedge, counts,
        ),
        fp(edge_lengths), fp(dst),
    )


# ------------------------------------------------ point cloud local triangulation

from mojopp3d.local_triangulation import (
    build_local_triangulations,
    compute_neighbors,
    compute_normals,
    compute_tangent_coordinates,
)


@export("mpp3d_pc_neighbors")
def mpp3d_pc_neighbors(
    points: Int, n: Int, k: Int, neighbors: Int, keys: Int, vals: Int,
) abi("C"):
    compute_neighbors(
        fp(points), n, k, ip(neighbors), fp(keys), ip(vals)
    )


@export("mpp3d_pc_normals")
def mpp3d_pc_normals(
    points: Int, neighbors: Int, n: Int, k: Int, normals: Int, a: Int, v: Int,
) abi("C"):
    compute_normals(
        fp(points), ip(neighbors), n, k, fp(normals), fp(a), fp(v)
    )


@export("mpp3d_pc_tangent_coordinates")
def mpp3d_pc_tangent_coordinates(
    points: Int, normals: Int, neighbors: Int, n: Int, k: Int, coords: Int,
) abi("C"):
    compute_tangent_coordinates(
        fp(points), fp(normals), ip(neighbors), n, k, fp(coords)
    )


@export("mpp3d_pc_local_triangulation")
def mpp3d_pc_local_triangulation(
    coords: Int, neighbors: Int, n: Int, k: Int, heuristic: Int, tri: Int,
    offsets: Int, pts: Int, angles: Int, sort_inds: Int,
) abi("C") -> Int:
    return build_local_triangulations(
        fp(coords), ip(neighbors), n, k, heuristic, ip(tri), ip(offsets),
        fp(pts), fp(angles), ip(sort_inds),
    )
