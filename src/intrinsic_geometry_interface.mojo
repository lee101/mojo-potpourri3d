# geometry-central surface/intrinsic_geometry_interface.cpp, one function per upstream
# `IntrinsicGeometryInterface::compute*`, in upstream order. The upstream class caches its results in
# per-element data arrays behind `require*`/`unrequire*` handles; here each `compute*` returns the
# array it would have cached, and the callers hold the arrays they need for the duration of a solve.
#
# Element index conventions (identical to upstream):
#   face        f
#   corner      h == 3f + i        (upstream: the halfedge leaving that corner)
#   halfedge    h
#   edge        e, reached as EID[h]
#   vertex      v

from std.math import sqrt, acos
from vector_util import Vector2, v2_from_angle, v2_undefined
from surface_mesh import SurfaceMesh
from sparse_matrix import Triplets


# Outgoing halfedges per vertex, plus a valid starting halfedge for the CCW fan orbit. For a
# boundary vertex upstream picks `v.halfedge()`, which can be the one halfedge whose edge is on the
# boundary; the orbit then terminates immediately and leaves the rest of the fan undefined. We start
# from an interior outgoing halfedge when one exists, so the whole fan is covered.
# The flat outgoing-halfedge fan: `VHF[VHS[v] : VHS[v+1]]` are v's outgoing halfedges, `VHE[v]` is
# a valid start for the CCW orbit, and `VHC` is the fill cursor.
struct VertexFan:
    var vhf: List[Int]
    var vhs: List[Int]
    var vhe: List[Int]

    def __init__(out self, n_v: Int, n_h: Int):
        self.vhf = List[Int](length=n_h, fill=0)
        self.vhs = List[Int](length=n_v + 1, fill=0)
        self.vhe = List[Int](length=n_v, fill=0)


# `v.halfedge()` upstream is vHalfedgeArr. SurfaceMesh's constructor overwrites it for every halfedge
# it builds, so by last-write-wins it ends up as the LAST halfedge with that vertex as its tail; and
# ManifoldSurfaceMesh's boundary-loop resolution then overwrites it again for every boundary vertex, so
# that it is a real halfedge lying on a boundary edge at that vertex. In the port's flat array a
# boundary halfedge is exactly a self-twinned corner, so the boundary rule can be applied directly.
def build_vertex_halfedges(mesh: SurfaceMesh) -> VertexFan:
    var n_v = mesh.nV
    var v = 0
    var count = List[Int](length=n_v, fill=0)
    var h = 0
    while h < 3 * mesh.nF:
        count[mesh.F[h]] += 1
        h += 1
    var fan = VertexFan(n_v, 3 * mesh.nF)
    v = 0
    while v < n_v:
        fan.vhs[v + 1] = fan.vhs[v] + count[v]
        v += 1
    var cursor = List[Int](length=n_v, fill=0)
    v = 0
    while v < n_v:
        cursor[v] = fan.vhs[v]
        v += 1
    h = 0
    while h < 3 * mesh.nF:
        var vv = mesh.F[h]
        fan.vhf[cursor[vv]] = h
        cursor[vv] += 1
        h += 1
    var on_boundary = List[Int](length=n_v, fill=-1)
    h = 0
    while h < 3 * mesh.nF:
        if not mesh.isInterior(h):
            on_boundary[mesh.F[h]] = h
        h += 1
    v = 0
    while v < n_v:
        var k = fan.vhs[v]
        var pick = -1
        while k < fan.vhs[v + 1]:
            pick = fan.vhf[k]
            k += 1
        if on_boundary[v] >= 0:
            pick = on_boundary[v]
        fan.vhe[v] = pick
        v += 1
    return fan^


# computeFaceAreas
def compute_face_areas(mesh: SurfaceMesh) -> List[Float64]:
    var fa = List[Float64](length=mesh.nF, fill=0.0)
    var f = 0
    while f < mesh.nF:
        var h0 = 3 * f
        var a = mesh.EL[mesh.EID[h0]]
        var b = mesh.EL[mesh.EID[mesh.nextHe(h0)]]
        var c = mesh.EL[mesh.EID[mesh.nextHe(mesh.nextHe(h0))]]
        # Herons formula
        var s = (a + b + c) / 2.0
        var arg = max(0.0, s * (s - a) * (s - b) * (s - c))
        fa[f] = sqrt(arg)
        f += 1
    return fa^


# computeVertexDualAreas
def compute_vertex_dual_areas(mesh: SurfaceMesh, fa: List[Float64]) -> List[Float64]:
    var vda = List[Float64](length=mesh.nV, fill=0.0)
    var h = 0
    while h < 3 * mesh.nF:
        vda[mesh.F[h]] += fa[h // 3] / 3.0
        h += 1
    return vda^


# computeCornerAngles
def compute_corner_angles(mesh: SurfaceMesh) -> List[Float64]:
    var ca = List[Float64](length=3 * mesh.nF, fill=0.0)
    var h = 0
    while h < 3 * mesh.nF:
        var l_opp = mesh.EL[mesh.EID[mesh.nextHe(h)]]
        var l_a = mesh.EL[mesh.EID[h]]
        var l_b = mesh.EL[mesh.EID[mesh.nextHe(mesh.nextHe(h))]]
        var q = (l_a * l_a + l_b * l_b - l_opp * l_opp) / (2.0 * l_a * l_b)
        q = min(1.0, max(-1.0, q))
        ca[h] = acos(q)
        h += 1
    return ca^


# computeVertexAngleSums
def compute_vertex_angle_sums(mesh: SurfaceMesh, ca: List[Float64]) -> List[Float64]:
    var vas = List[Float64](length=mesh.nV, fill=0.0)
    var h = 0
    while h < 3 * mesh.nF:
        vas[mesh.F[h]] += ca[h]
        h += 1
    return vas^


# computeCornerScaledAngles
def compute_corner_scaled_angles(mesh: SurfaceMesh, ca: List[Float64], vas: List[Float64]) -> List[Float64]:
    # The branch is on the VERTEX being on the boundary, not on the corner: a boundary vertex also
    # owns interior corners, and upstream scales every one of its angles by pi / angleSum.
    var v_boundary = List[Bool](length=mesh.nV, fill=False)
    var hb = 0
    while hb < 3 * mesh.nF:
        if not mesh.isInterior(hb):
            v_boundary[mesh.F[hb]] = True
        hb += 1
    var csa = List[Float64](length=3 * mesh.nF, fill=0.0)
    var pi = 3.14159265358979323846
    var h = 0
    while h < 3 * mesh.nF:
        var v = mesh.F[h]
        if v_boundary[v]:
            var s = pi / vas[v]
            csa[h] = s * ca[h]
        else:
            var s = 2.0 * pi / vas[v]
            csa[h] = s * ca[h]
        h += 1
    return csa^


# computeHalfedgeCotanWeights
def compute_halfedge_cotan_weights(mesh: SurfaceMesh, fa: List[Float64]) -> List[Float64]:
    var w = List[Float64](length=3 * mesh.nF, fill=0.0)
    var h = 0
    while h < 3 * mesh.nF:
        if True:
            var h1 = mesh.nextHe(h)
            var h2 = mesh.nextHe(h1)
            var l_ij = mesh.EL[mesh.EID[h]]
            var l_jk = mesh.EL[mesh.EID[h1]]
            var l_ki = mesh.EL[mesh.EID[h2]]
            var area = fa[h // 3]
            var cot_value = (-l_ij * l_ij + l_jk * l_jk + l_ki * l_ki) / (4.0 * area)
            w[h] = cot_value / 2
        else:
            w[h] = 0.0
        h += 1
    return w^


# computeEdgeCotanWeights
def compute_edge_cotan_weights(mesh: SurfaceMesh, fa: List[Float64]) -> List[Float64]:
    var hw = compute_halfedge_cotan_weights(mesh, fa)
    var w = List[Float64](length=mesh.nE, fill=0.0)
    # Every halfedge is a real corner of a face, so summing over halfedges visits each interior
    # edge's two faces once and each boundary edge's single face once.
    var h = 0
    while h < 3 * mesh.nF:
        w[mesh.EID[h]] += hw[h]
        h += 1
    return w^


# computeHalfedgeVectorsInFace
def compute_halfedge_vectors_in_face(mesh: SurfaceMesh, fa: List[Float64]) -> List[Vector2]:
    var hv = List[Vector2]()
    var i0 = 0
    while i0 < 3 * mesh.nF:
        hv.append(v2_undefined())
        i0 += 1
    var f = 0
    while f < mesh.nF:
        var he_ab = 3 * f
        var he_bc = mesh.nextHe(he_ab)
        var he_ca = mesh.nextHe(he_bc)

        var l_ab = mesh.EL[mesh.EID[he_ab]]
        var l_bc = mesh.EL[mesh.EID[he_bc]]
        var l_ca = mesh.EL[mesh.EID[he_ca]]

        var p_b = Vector2(l_ab, 0.0)
        var t_area = fa[f]

        # width and height of the right triangle formed by the altitude from C
        var hh = 2.0 * t_area / l_ab
        var w = sqrt(max(0.0, l_ca * l_ca - hh * hh))
        if l_bc * l_bc > (l_ab * l_ab + l_ca * l_ca):
            w *= -1.0
        var p_c = Vector2(w, hh)

        var v_bc = p_c - p_b
        var v_ca = -p_c
        hv[he_ab] = p_b^
        hv[he_bc] = v_bc^
        hv[he_ca] = v_ca^^
        f += 1
    return hv^


# computeTransportVectorsAcrossHalfedge
def compute_transport_vectors_across_halfedge(mesh: SurfaceMesh, hvif: List[Vector2]) -> List[Vector2]:
    var tv = List[Vector2]()
    var i0 = 0
    while i0 < 3 * mesh.nF:
        tv.append(v2_undefined())
        i0 += 1
    var e = 0
    while e < mesh.nE:
        var he_a = mesh.EHA[e]
        if mesh.isInterior(he_a):
            var he_b = mesh.FT[he_a]
            var vec_a = hvif[he_a].copy()
            var vec_b = hvif[he_b].copy()
            var rot = (-vec_b / vec_a).unit()
            tv[he_a] = rotc^
            tv[he_b] = rotc.inv()
        e += 1
    return tv^


# computeHalfedgeVectorsInVertex
def compute_halfedge_vectors_in_vertex(mesh: SurfaceMesh, csa: List[Float64], vhe: List[Int]) -> List[Vector2]:
    var hv = List[Vector2]()
    var i0 = 0
    while i0 < 3 * mesh.nF:
        hv.append(v2_undefined())
        i0 += 1
    var v = 0
    while v < mesh.nV:
        var first = vhe[v]
        if first >= 0:
            var coord_sum = 0.0
            var curr = first
            while True:
                hv[curr] = v2_from_angle(coord_sum) * mesh.EL[mesh.EID[curr]]
                coord_sum += csa[curr]
                # Upstream breaks on `!currHe.isInterior()`, which in its halfedge array is the ghost
                # sitting between the last and first corners of the fan. Here a boundary halfedge is
                # the self-twinned corner, which is also a real corner of the fan, so the orbit has to
                # stop on the step that would leave the fan instead of on the current halfedge.
                var nxt = mesh.FT[mesh.nextHe(mesh.nextHe(curr))]
                if nxt == first or mesh.F[nxt] != v:
                    break
                curr = nxt
        v += 1
    return hv^


# computeTransportVectorsAlongHalfedge. A boundary edge has no twin halfedge in the mesh, so it
# keeps the identity rotation; upstream gets a genuine rotation from a ghost halfedge it stores in
# the same face. This is the one place the port diverges, and only for open meshes.
def compute_transport_vectors_along_halfedge(mesh: SurfaceMesh, hviv: List[Vector2]) -> List[Vector2]:
    var tv = List[Vector2]()
    var i0 = 0
    while i0 < 3 * mesh.nF:
        tv.append(Vector2(1.0, 0.0))
        i0 += 1
    var e = 0
    while e < mesh.nE:
        var he_a = mesh.EHA[e]
        if mesh.isInterior(he_a):
            var he_b = mesh.FT[he_a]
            var rotc = (-hviv[he_b] / hviv[he_a]).unit()
            tv[he_a] = rotc.copy()
            tv[he_b] = rotc.inv()
        e += 1
    return tv^


# computeCotanLaplacian, returned as triplets in upstream's emission order
def compute_cotan_laplacian(mesh: SurfaceMesh, ecw: List[Float64]) -> Triplets:
    var t = Triplets()
    var h = 0
    while h < 3 * mesh.nF:
        # Take each edge once: an interior edge from the lower of its two halfedge ids, a boundary
        # edge from its single one.
        if h <= mesh.FT[h]:
            var he = h
            var v_tail = mesh.F[he]
            var v_head = mesh.F[mesh.nextHe(he)]
            var weight = ecw[mesh.EID[he]]
            t.add(v_tail, v_tail, weight)
            t.add(v_head, v_head, weight)
            t.add(v_tail, v_head, -weight)
            t.add(v_head, v_tail, -weight)
        h += 1
    return t^


# computeVertexLumpedMassMatrix, as its diagonal
def compute_vertex_lumped_mass_matrix(mesh: SurfaceMesh, vda: List[Float64]) -> List[Float64]:
    var m = List[Float64](length=mesh.nV, fill=0.0)
    var v = 0
    while v < mesh.nV:
        m[v] = vda[v]
        v += 1
    return m^


# computeVertexConnectionLaplacian, as triplets over complex entries stored interleaved (re, im),
# which is how Eigen stores a std::complex<double> matrix
def compute_vertex_connection_laplacian(
    mesh: SurfaceMesh, ecw: List[Float64], tval: List[Vector2]
) -> Triplets:
    var t = Triplets()
    var h = 0
    while h < 3 * mesh.nF:
        var i_tail = mesh.F[h]
        var i_tip = mesh.F[mesh.nextHe(h)]
        var weight = ecw[mesh.EID[h]]
        var rot = tval[mesh.FT[h]].copy()
        t.add(i_tail, i_tail, weight)
        t.add_complex(i_tail, i_tip, -weight * rot.x, -weight * rot.y)
        h += 1
    return t^
