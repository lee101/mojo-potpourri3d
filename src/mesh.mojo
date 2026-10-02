# potpourri3d mesh.py: the module-level helpers. These are numpy/scipy in upstream, so the
# arithmetic is transcribed loop for loop and the caller still assembles the sparse matrix; the
# vectorised form is a straight elementwise map, so the loop is kept.

from std.math import sqrt
from vector_util import Vector3, v3_cross
from surface_mesh import SurfaceMesh, build_connectivity, faces_in_range


# potpourri3d.mesh.cotan_laplacian
def cotan_laplacian(
    n_v: Int,
    n_f: Int,
    verts_addr: Int,
    faces_addr: Int,
    denom_eps: Float64,
    rows_addr: Int,
    cols_addr: Int,
    data_addr: Int,
) -> Int:
    if verts_addr == 0 or faces_addr == 0:
        return -1
    var verts = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=verts_addr)
    var faces = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=faces_addr)
    if not faces_in_range(faces, n_v, n_f):
        return -2
    var rows = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=rows_addr)
    var cols = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=cols_addr)
    var data = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=data_addr)

    var cursor = 0
    var i = 0
    while i < 3:
        # Gather indices and compute cotan weight (via dot() / cross() formula)
        var f = 0
        while f < n_f:
            var inds_i = faces[3 * f + i]
            var inds_j = faces[3 * f + (i + 1) % 3]
            var inds_k = faces[3 * f + (i + 2) % 3]
            var ki = Vector3(
                verts[3 * inds_i] - verts[3 * inds_k],
                verts[3 * inds_i + 1] - verts[3 * inds_k + 1],
                verts[3 * inds_i + 2] - verts[3 * inds_k + 2],
            )
            var kj = Vector3(
                verts[3 * inds_j] - verts[3 * inds_k],
                verts[3 * inds_j + 1] - verts[3 * inds_k + 1],
                verts[3 * inds_j + 2] - verts[3 * inds_k + 2],
            )
            var dots = v3_dot(ki, kj)
            var cr = v3_cross(ki, kj)
            var cross_mags = sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z)
            var cotans = 0.5 * dots / (cross_mags + denom_eps)

            # Add the four matrix entries from this weight
            rows.unsafe_store(cursor, inds_i)
            cols.unsafe_store(cursor, inds_i)
            data.unsafe_store(cursor, cotans)
            cursor += 1
            rows.unsafe_store(cursor, inds_j)
            cols.unsafe_store(cursor, inds_j)
            data.unsafe_store(cursor, cotans)
            cursor += 1
            rows.unsafe_store(cursor, inds_i)
            cols.unsafe_store(cursor, inds_j)
            data.unsafe_store(cursor, -cotans)
            cursor += 1
            rows.unsafe_store(cursor, inds_j)
            cols.unsafe_store(cursor, inds_i)
            data.unsafe_store(cursor, -cotans)
            cursor += 1
            f += 1
        i += 1
    return 0


# potpourri3d.mesh.face_areas
def face_areas(n_v: Int, n_f: Int, verts_addr: Int, faces_addr: Int, out_addr: Int) -> Int:
    if verts_addr == 0 or faces_addr == 0 or out_addr == 0:
        return -1
    var verts = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=verts_addr)
    var faces = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=faces_addr)
    if not faces_in_range(faces, n_v, n_f):
        return -2
    var out = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=out_addr)
    var f = 0
    while f < n_f:
        var a = faces[3 * f]
        var b = faces[3 * f + 1]
        var c = faces[3 * f + 2]
        var ij = Vector3(
            verts[3 * b] - verts[3 * a],
            verts[3 * b + 1] - verts[3 * a + 1],
            verts[3 * b + 2] - verts[3 * a + 2],
        )
        var ik = Vector3(
            verts[3 * c] - verts[3 * a],
            verts[3 * c + 1] - verts[3 * a + 1],
            verts[3 * c + 2] - verts[3 * a + 2],
        )
        var cr = v3_cross(ij, ik)
        out.unsafe_store(f, 0.5 * sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z))
        f += 1
    return 0


# potpourri3d.mesh.edges
def edges(n_v: Int, n_f: Int, faces_addr: Int, out_addr: Int) -> Int:
    # Returns the number of edges found; `out` receives 2 ints per edge.
    if faces_addr == 0 or out_addr == 0:
        return -1
    var faces = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=faces_addr)
    if not faces_in_range(faces, n_v, n_f):
        return -2
    var out = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=out_addr)
    var mesh = build_connectivity(faces, n_v, n_f)
    var e = 0
    while e < mesh.nE:
        var h = mesh.EHA[e]
        out.unsafe_store(2 * e, mesh.F[h])
        out.unsafe_store(2 * e + 1, mesh.F[mesh.nextHe(h)])
        e += 1
    return mesh.nE


def v3_dot(a: Vector3, b: Vector3) -> Float64:
    return a.x * b.x + a.y * b.y + a.z * b.z


# potpourri3d.mesh.vertex_areas
def vertex_areas(n_v: Int, n_f: Int, verts_addr: Int, faces_addr: Int, out_addr: Int) -> Int:
    if verts_addr == 0 or faces_addr == 0 or out_addr == 0:
        return -1
    var verts = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=verts_addr)
    var faces = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=faces_addr)
    if not faces_in_range(faces, n_v, n_f):
        return -2
    var out = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=out_addr)
    var v = 0
    while v < n_v:
        out.unsafe_store(v, 0.0)
        v += 1
    var f = 0
    while f < n_f:
        var a = faces[3 * f]
        var b = faces[3 * f + 1]
        var c = faces[3 * f + 2]
        var ij = Vector3(
            verts[3 * b] - verts[3 * a],
            verts[3 * b + 1] - verts[3 * a + 1],
            verts[3 * b + 2] - verts[3 * a + 2],
        )
        var ik = Vector3(
            verts[3 * c] - verts[3 * a],
            verts[3 * c + 1] - verts[3 * a + 1],
            verts[3 * c + 2] - verts[3 * a + 2],
        )
        var cr = v3_cross(ij, ik)
        var area = 0.5 * sqrt(cr.x * cr.x + cr.y * cr.y + cr.z * cr.z)
        # vertex_area += np.bincount(F[:,i], face_area, minlength=nV)
        out.unsafe_store(a, out.unsafe_load(a) + area)
        out.unsafe_store(b, out.unsafe_load(b) + area)
        out.unsafe_store(c, out.unsafe_load(c) + area)
        f += 1
    v = 0
    while v < n_v:
        out.unsafe_store(v, out.unsafe_load(v) / 3.0)
        v += 1
    return 0