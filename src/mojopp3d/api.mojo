"""The potpourri3d Python-level helpers, in the order of `potpourri3d/mesh.py`.

These are NumPy in upstream rather than C++, so they are a port of the Python
loop bodies: `cotan_laplacian`, `face_areas` and `vertex_areas` keep the
expression order upstream writes. The COO/CSR assembly stays in Python because
upstream returns a `scipy.sparse.csr_matrix`.
"""

from std.math import sqrt

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]


def _triplet(mat_i: IPtr, mat_j: IPtr, mat_data: Ptr, n: Int, i: Int64, j: Int64, v: Float64):
    mat_i.unsafe_store(Int64(n), i)
    mat_j.unsafe_store(Int64(n), j)
    mat_data.unsafe_store(n, v)


# potpourri3d.face_areas
def face_areas(V: Ptr, F: IPtr, n_faces: Int, dst: Ptr) -> Int:
    for t in range(n_faces):
        var i0 = F.unsafe_load(Int64(3 * t))
        var i1 = F.unsafe_load(Int64(3 * t + 1))
        var i2 = F.unsafe_load(Int64(3 * t + 2))

        var v_ij_x = V.unsafe_load(3 * i1) - V.unsafe_load(3 * i0)
        var v_ij_y = V.unsafe_load(3 * i1 + 1) - V.unsafe_load(3 * i0 + 1)
        var v_ij_z = V.unsafe_load(3 * i1 + 2) - V.unsafe_load(3 * i0 + 2)
        var v_ik_x = V.unsafe_load(3 * i2) - V.unsafe_load(3 * i0)
        var v_ik_y = V.unsafe_load(3 * i2 + 1) - V.unsafe_load(3 * i0 + 1)
        var v_ik_z = V.unsafe_load(3 * i2 + 2) - V.unsafe_load(3 * i0 + 2)

        var c_x = v_ij_y * v_ik_z - v_ij_z * v_ik_y
        var c_y = v_ij_z * v_ik_x - v_ij_x * v_ik_z
        var c_z = v_ij_x * v_ik_y - v_ij_y * v_ik_x
        dst.unsafe_store(t, 0.5 * sqrt(c_x * c_x + c_y * c_y + c_z * c_z))
    return n_faces


# potpourri3d.vertex_areas
def vertex_areas(
    V: Ptr, F: IPtr, n_faces: Int, n_vertices: Int, scratch: Ptr, dst: Ptr
) -> Int:
    _ = face_areas(V, F, n_faces, scratch)
    for v in range(n_vertices):
        dst.unsafe_store(v, 0.0)
    for t in range(n_faces):
        var a = scratch.unsafe_load(t)
        for k in range(3):
            var v = F.unsafe_load(Int64(3 * t + k))
            dst.unsafe_store(v, dst.unsafe_load(v) + a)
    for v in range(n_vertices):
        dst.unsafe_store(v, dst.unsafe_load(v) / 3.0)
    return n_vertices


# potpourri3d.cotan_laplacian, emitting the COO triplets upstream concatenates.
def cotan_laplacian_triplets(
    V: Ptr, F: IPtr, n_faces: Int, denom_eps: Float64,
    mat_i: IPtr, mat_j: IPtr, mat_data: Ptr,
) -> Int:
    var n = 0
    for i in range(3):

        # Gather indices and compute cotan weight (via dot() / cross() formula)
        for t in range(n_faces):
            var inds_i = F.unsafe_load(Int64(3 * t + i))
            var inds_j = F.unsafe_load(Int64(3 * t + (i + 1) % 3))
            var inds_k = F.unsafe_load(Int64(3 * t + (i + 2) % 3))

            var vec_ki_x = V.unsafe_load(3 * inds_i) - V.unsafe_load(3 * inds_k)
            var vec_ki_y = V.unsafe_load(3 * inds_i + 1) - V.unsafe_load(3 * inds_k + 1)
            var vec_ki_z = V.unsafe_load(3 * inds_i + 2) - V.unsafe_load(3 * inds_k + 2)
            var vec_kj_x = V.unsafe_load(3 * inds_j) - V.unsafe_load(3 * inds_k)
            var vec_kj_y = V.unsafe_load(3 * inds_j + 1) - V.unsafe_load(3 * inds_k + 1)
            var vec_kj_z = V.unsafe_load(3 * inds_j + 2) - V.unsafe_load(3 * inds_k + 2)

            var dots = vec_ki_x * vec_kj_x + vec_ki_y * vec_kj_y + vec_ki_z * vec_kj_z
            var c_x = vec_ki_y * vec_kj_z - vec_ki_z * vec_kj_y
            var c_y = vec_ki_z * vec_kj_x - vec_ki_x * vec_kj_z
            var c_z = vec_ki_x * vec_kj_y - vec_ki_y * vec_kj_x
            var cross_mags = sqrt(c_x * c_x + c_y * c_y + c_z * c_z)
            var cotans = 0.5 * dots / (cross_mags + denom_eps)

            # Add the four matrix entries from this weight
            _triplet(mat_i, mat_j, mat_data, n, inds_i, inds_i, cotans)
            n += 1
            _triplet(mat_i, mat_j, mat_data, n, inds_j, inds_j, cotans)
            n += 1
            _triplet(mat_i, mat_j, mat_data, n, inds_i, inds_j, -cotans)
            n += 1
            _triplet(mat_i, mat_j, mat_data, n, inds_j, inds_i, -cotans)
            n += 1
    return n
