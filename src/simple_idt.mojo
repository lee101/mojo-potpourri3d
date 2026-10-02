# geometry-central surface/simple_idt.cpp: flipToDelaunay with FlipType::Euclidean, the default
# used by both the robust heat method distance solver and the point-cloud tufted triangulation.
# Edges are pushed once and re-queued when a flip changes their cotan weight; the queue, the
# inQueue flags and the flip itself are a direct transcription of upstream's loop.

from std.math import isnan, isinf
from vector_util import triangle_area
from surface_mesh import SurfaceMesh


def _face_area(mesh: SurfaceMesh, f: Int) -> Float64:
    var h = 3 * f
    var a = mesh.EL[mesh.EID[h]]
    h = mesh.nextHe(h)
    var b = mesh.EL[mesh.EID[h]]
    h = mesh.nextHe(h)
    var c = mesh.EL[mesh.EID[h]]
    return triangle_area(a, b, c)


# halfedgeCotanWeight. Upstream returns 0 for a non-interior halfedge, but in its halfedge array a
# boundary edge has a real halfedge AND a ghost; here the two are the same self-twinned corner, which
# is also a corner of a real face. It therefore gets that face's cotangent weight, which is what the
# real halfedge gets upstream and what the edge total needs.
def _halfedge_cotan_weight(mesh: SurfaceMesh, he: Int) -> Float64:
    if True:
        var h = he
        var l_ij = mesh.EL[mesh.EID[h]]
        h = mesh.nextHe(h)
        var l_jk = mesh.EL[mesh.EID[h]]
        h = mesh.nextHe(h)
        var l_ki = mesh.EL[mesh.EID[h]]
        var area_v = _face_area(mesh, he // 3)
        var cot_value = (-l_ij * l_ij + l_jk * l_jk + l_ki * l_ki) / (4.0 * area_v)
        return cot_value / 2
    return 0.0


# edgeCotanWeight
def _edge_cotan_weight(mesh: SurfaceMesh, e: Int) -> Float64:
    var he = mesh.EHA[e]
    return _halfedge_cotan_weight(mesh, he) + _halfedge_cotan_weight(mesh, mesh.FT[he])


# shouldFlipEdge, FlipType::Euclidean
def _should_flip_edge(mesh: SurfaceMesh, e: Int, delaunay_eps: Float64) -> Bool:
    if not mesh.isInterior(mesh.EHA[e]):
        return False
    var c_weight = _edge_cotan_weight(mesh, e)
    return c_weight < -delaunay_eps


# Returns the Delaunay-flipped mesh. Upstream also returns the flip count; no caller in this port
# reads it.
def flip_to_delaunay(var mesh: SurfaceMesh, delaunay_eps: Float64 = 1e-6) -> SurfaceMesh:
    # upstream also returns the flip count; no caller in this port reads it
    var in_queue = List[Bool](length=mesh.nE, fill=False)
    var queue = List[Int]()
    var e = 0
    while e < mesh.nE:
        in_queue[e] = True
        queue.append(e)
        e += 1

    var n_flips = 0
    var head = 0
    while head < len(queue):
        var cur = queue[head]
        head += 1
        in_queue[cur] = False

        # flipEdgeIfNotDelaunay
        var he = mesh.EHA[cur]
        if he // 3 == mesh.FT[he] // 3:
            continue
        if not _should_flip_edge(mesh, cur, delaunay_eps):
            continue
        var n1 = mesh.nextHe(he)
        var p1 = mesh.prevHe(he)
        var t = mesh.FT[he]
        var n2 = mesh.nextHe(t)
        var p2 = mesh.prevHe(t)
        var new_length = mesh.flipEdge(cur)
        if isnan(new_length) or isinf(new_length):
            continue
        mesh.EL[cur] = new_length
        n_flips += 1

        # Add neighbours to queue, as they may need flipping now
        for k in range(4):
            var n_e = 0
            if k == 0:
                n_e = mesh.EID[n1]
            elif k == 1:
                n_e = mesh.EID[p1]
            elif k == 2:
                n_e = mesh.EID[n2]
            else:
                n_e = mesh.EID[p2]
            if not in_queue[n_e]:
                queue.append(n_e)
                in_queue[n_e] = True

    return mesh^
