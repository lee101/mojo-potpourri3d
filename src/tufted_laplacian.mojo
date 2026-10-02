# geometry-central surface/tufted_laplacian.cpp and surface/intrinsic_mollification.cpp.
#
# The intrinsic tufted cover of a manifold mesh is two copies of every face, the second reversed,
# glued to the first around every original edge in a cyclic alternation of sheets. Around a boundary
# edge that alternation has length one, which is exactly the "tuft". Upstream builds this by
# mutating a halfedge mesh (duplicateFace / invertOrientation / separateToNewEdge); the port writes
# the resulting corner array and twin pairs directly, which is the same combinatorics with the
# intermediate mutation removed.

from vector_util import triangle_area
from surface_mesh import SurfaceMesh


# buildIntrinsicTuftedCover
def build_intrinsic_tufted_cover(mesh: SurfaceMesh) -> SurfaceMesh:
    var n_f = mesh.nF
    var n_v = mesh.nV
    var n_c = 3 * n_f
    var total = 2 * n_c

    # Face f's front copy keeps its corner order; its back copy is the reversed cycle.
    var f = List[Int](length=total, fill=0)
    var h = 0
    while h < n_c:
        f[h] = mesh.F[h]
        f[n_c + h] = mesh.F[h - h % 3 + (2 - h % 3) % 3]
        h += 1

    # Group the corners by original edge, in ascending corner order. Upstream walks
    # e.adjacentHalfedges(), which is the same set; the order only matters past two faces, which
    # this port does not support.
    var count = List[Int](length=mesh.nE, fill=0)
    var e = 0
    while e < mesh.nE:
        count[e] = 0
        e += 1
    h = 0
    while h < n_c:
        count[mesh.EID[h]] += 1
        h += 1
    var start = List[Int](length=mesh.nE + 1, fill=0)
    e = 0
    while e < mesh.nE:
        start[e + 1] = start[e] + count[e]
        e += 1
    var cursor = List[Int](length=mesh.nE, fill=0)
    e = 0
    while e < mesh.nE:
        cursor[e] = start[e]
        e += 1
    var group = List[Int](length=n_c, fill=0)
    h = 0
    while h < n_c:
        var eid = mesh.EID[h]
        group[cursor[eid]] = h
        cursor[eid] += 1
        h += 1

    # Sequentially connect the faces: new edge i joins the front of g_i to the back of g_{i+1}.
    # `back(g)` is the corner of g's face in the back sheet running the same edge the other way,
    # which is the sheet's orientation-reversed halfedge.
    var ft = List[Int](length=total, fill=0)
    var eid_new = List[Int](length=total, fill=0)
    var el = List[Float64]()
    var eha = List[Int]()
    e = 0
    while e < mesh.nE:
        var g_lo = start[e]
        var g_hi = start[e + 1]
        var k = 0
        while k < g_hi - g_lo:
            var a = group[g_lo + k]
            var c_next = group[g_lo + (k + 1) % (g_hi - g_lo)]
            var b = n_c + 3 * (c_next // 3) + ((1 - c_next % 3) % 3)
            var n_edge = len(el)
            ft[a] = b
            ft[b] = a
            eid_new[a] = n_edge
            eid_new[b] = n_edge
            el.append(mesh.EL[e])
            eha.append(a)
            k += 1
        e += 1

    var n_b = 0
    h = 0
    while h < total:
        if ft[h] // 3 == h // 3:
            n_b += 1
        h += 1

    var out = SurfaceMesh(n_v, 2 * n_f, len(el), n_b)
    out.F = f^
    out.FT = ft^
    out.EID = eid_new^
    out.EL = el^
    out.EHA = eha^
    return out^


# mollifyIntrinsic
def mollify_intrinsic(var mesh: SurfaceMesh, relative_factor: Float64) -> SurfaceMesh:
    var edge_sum = 0.0
    var e = 0
    while e < mesh.nE:
        edge_sum += mesh.EL[e]
        e += 1
    var mean_edge = edge_sum / Float64(mesh.nE)
    return mollify_intrinsic_absolute(mesh^, mean_edge * relative_factor)


# mollifyIntrinsicAbsolute
def mollify_intrinsic_absolute(var mesh: SurfaceMesh, absolute_factor: Float64) -> SurfaceMesh:
    var mollify_eps = 0.0
    var h = 0
    while h < 3 * mesh.nF:
        if mesh.isInterior(h):
            var l_a = mesh.EL[mesh.EID[h]]
            var l_b = mesh.EL[mesh.EID[mesh.nextHe(h)]]
            var l_c = mesh.EL[mesh.EID[mesh.nextHe(mesh.nextHe(h))]]
            var this_eps = l_c - l_a - l_b + absolute_factor
            mollify_eps = max(mollify_eps, this_eps)
        h += 1
    var e = 0
    while e < mesh.nE:
        mesh.EL[e] += mollify_eps
        e += 1
    return mesh^
