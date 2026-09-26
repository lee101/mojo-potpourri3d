"""IntrinsicGeometryInterface, in the order of
`geometry-central/src/surface/intrinsic_geometry_interface.cpp`.

Every quantity is derived from edge lengths and face areas, so the module opens
with those two and then walks the list of caches in the order the upstream file
declares them. `pi` is spelled `PI` to match.
"""

from std.math import sqrt, acos

from mojopp3d.mesh import HalfedgeMesh
from mojopp3d.vector import Vec2, Vec3

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

comptime PI = 3.141592653589793


def _clamp(q: Float64, low: Float64, high: Float64) -> Float64:
    if q < low:
        return low
    if q > high:
        return high
    return q


# Face areas (IntrinsicGeometryInterface::computeFaceAreas). Heron, clamped.
def compute_face_areas(mesh: HalfedgeMesh, edge_lengths: Ptr, face_areas: Ptr):
    for f in range(mesh.n_faces):
        var he = mesh.first_halfedge(f)
        var a = edge_lengths.unsafe_load(he)
        var b = edge_lengths.unsafe_load(mesh.next(he))
        var c = edge_lengths.unsafe_load(mesh.next(mesh.next(he)))

        var s = (a + b + c) / 2.0
        var arg = s * (s - a) * (s - b) * (s - c)
        if arg < 0.0:
            arg = 0.0
        face_areas.unsafe_store(f, sqrt(arg))


# Vertex dual areas (computeVertexDualAreas).
def compute_vertex_dual_areas(mesh: HalfedgeMesh, face_areas: Ptr, vertex_dual_areas: Ptr):
    for v in range(mesh.n_vertices):
        vertex_dual_areas.unsafe_store(v, 0.0)
    for f in range(mesh.n_faces):
        var A = face_areas.unsafe_load(f)
        var he = mesh.first_halfedge(f)
        vertex_dual_areas.unsafe_store(mesh.vertex(he), vertex_dual_areas.unsafe_load(mesh.vertex(he)) + A / 3.0)
        vertex_dual_areas.unsafe_store(mesh.vertex(mesh.next(he)), vertex_dual_areas.unsafe_load(mesh.vertex(mesh.next(he))) + A / 3.0)
        vertex_dual_areas.unsafe_store(mesh.vertex(mesh.next(mesh.next(he))), vertex_dual_areas.unsafe_load(mesh.vertex(mesh.next(mesh.next(he)))) + A / 3.0)


# Corner angles (computeCornerAngles), indexed by the halfedge at the corner.
def compute_corner_angles(mesh: HalfedgeMesh, edge_lengths: Ptr, corner_angles: Ptr):
    for he in range(mesh.n_he):
        var heOpp = mesh.next(he)
        var heB = mesh.next(heOpp)

        var lOpp = edge_lengths.unsafe_load(heOpp)
        var lA = edge_lengths.unsafe_load(he)
        var lB = edge_lengths.unsafe_load(heB)

        var q = (lA * lA + lB * lB - lOpp * lOpp) / (2.0 * lA * lB)
        q = _clamp(q, -1.0, 1.0)
        corner_angles.unsafe_store(he, acos(q))


# Vertex angle sums (computeVertexAngleSums).
def compute_vertex_angle_sums(
    mesh: HalfedgeMesh, corner_angles: Ptr, vertex_angle_sums: Ptr
):
    for v in range(mesh.n_vertices):
        vertex_angle_sums.unsafe_store(v, 0.0)
    for he in range(mesh.n_he):
        vertex_angle_sums.unsafe_store(mesh.vertex(he), vertex_angle_sums.unsafe_load(mesh.vertex(he)) + corner_angles.unsafe_load(he))


# Corner scaled angles (computeCornerScaledAngles). Rescale the cone angles so
# they wrap exactly once around each vertex. gc branches on the *vertex's*
# boundary flag, so every corner of a boundary vertex scales to pi -- testing
# the corner's own predecessor halfedge instead would leave the interior
# corners of a valence-3 boundary vertex scaled to 2*pi.
def compute_corner_scaled_angles(
    mesh: HalfedgeMesh,
    corner_angles: Ptr,
    vertex_angle_sums: Ptr,
    corner_scaled_angles: Ptr,
):
    for he in range(mesh.n_he):
        var v = mesh.vertex(he)
        var s = 2.0 * PI / vertex_angle_sums.unsafe_load(v)
        if _vertex_is_boundary(mesh, v):
            s = PI / vertex_angle_sums.unsafe_load(v)
        corner_scaled_angles.unsafe_store(he, s * corner_angles.unsafe_load(he))


# gc's `Vertex::isBoundary()`: a vertex is a boundary vertex when any halfedge
# in its 1-ring is its own twin. gc's `computeHalfedgeVectorsInVertex` does not
# stop at a boundary halfedge -- `Halfedge::isInterior()` there tests the
# *face* against the boundary loops, which a face soup has none of -- so its
# 1-ring walk closes on the implicit twin, and so must this one.
def _vertex_is_boundary(mesh: HalfedgeMesh, v: Int) -> Bool:
    var first_he = mesh.first_outgoing(v)
    if first_he < 0:
        return False
    var curr_he = first_he
    while True:
        if mesh.is_boundary(curr_he):
            return True
        curr_he = mesh.twin(mesh.next(mesh.next(curr_he)))
        if curr_he == first_he:
            return False


# Halfedge cotan weights (computeHalfedgeCotanWeights). Every halfedge here
# belongs to a real face -- which is what gc's `Halfedge::isInterior()` means,
# see `SurfaceMesh::heIsInterior` -- so boundary halfedges are weighted too.
def compute_halfedge_cotan_weights(
    mesh: HalfedgeMesh, edge_lengths: Ptr, face_areas: Ptr, halfedge_cotan_weights: Ptr
):
    for he in range(mesh.n_he):
        var l_ij = edge_lengths.unsafe_load(he)
        var l_jk = edge_lengths.unsafe_load(mesh.next(he))
        var l_ki = edge_lengths.unsafe_load(mesh.next(mesh.next(he)))

        var area = face_areas.unsafe_load(mesh.face(he))
        var cotValue = (-l_ij * l_ij + l_jk * l_jk + l_ki * l_ki) / (4.0 * area)
        halfedge_cotan_weights.unsafe_store(he, cotValue / 2.0)


# Edge cotan weights (computeEdgeCotanWeights), stored on both halfedges. An
# edge with only one adjacent face -- a boundary edge -- takes that face's
# contribution alone, which is what gc's `e.adjacentInteriorHalfedges()` yields.
def compute_edge_cotan_weights(
    mesh: HalfedgeMesh, edge_lengths: Ptr, face_areas: Ptr, edge_cotan_weights: Ptr
):
    for he in range(mesh.n_he):
        var l_ij = edge_lengths.unsafe_load(he)
        var l_jk = edge_lengths.unsafe_load(mesh.next(he))
        var l_ki = edge_lengths.unsafe_load(mesh.next(mesh.next(he)))
        var area = face_areas.unsafe_load(mesh.face(he))
        var cotValue = (-l_ij * l_ij + l_jk * l_jk + l_ki * l_ki) / (4.0 * area)
        var cotSum = cotValue / 2.0

        var twin = mesh.twin(he)
        if twin != he:
            var l2_ij = edge_lengths.unsafe_load(twin)
            var l2_jk = edge_lengths.unsafe_load(mesh.next(twin))
            var l2_ki = edge_lengths.unsafe_load(mesh.next(mesh.next(twin)))
            var area2 = face_areas.unsafe_load(mesh.face(twin))
            var cotValue2 = (-l2_ij * l2_ij + l2_jk * l2_jk + l2_ki * l2_ki) / (4.0 * area2)
            cotSum += cotValue2 / 2.0
        # Both halfedges of an edge carry the same weight.
        edge_cotan_weights.unsafe_store(he, cotSum)


# Halfedge vectors in face (computeHalfedgeVectorsInFace): the flat
# isometry of each triangle, with the first halfedge along +x.
def compute_halfedge_vectors_in_face(
    mesh: HalfedgeMesh, edge_lengths: Ptr, face_areas: Ptr, halfedge_vec: Ptr
):
    # Exterior halfedges are left undefined upstream; zero them so the buffer
    # is deterministic.
    for he in range(mesh.n_he):
        halfedge_vec.unsafe_store(2 * he, 0.0)
        halfedge_vec.unsafe_store(2 * he + 1, 0.0)
    for f in range(mesh.n_faces):
        var heAB = mesh.first_halfedge(f)
        var heBC = mesh.next(heAB)
        var heCA = mesh.next(heBC)

        var lAB = edge_lengths.unsafe_load(heAB)
        var lBC = edge_lengths.unsafe_load(heBC)
        var lCA = edge_lengths.unsafe_load(heCA)

        # Vector2 pA{0., 0.}; used implicitly
        var pB = Vec2(lAB, 0.0)
        # pC is the hard one: width and height of the altitude from C
        var tArea = face_areas.unsafe_load(f)
        var h = 2.0 * tArea / lAB
        var w2 = lCA * lCA - h * h
        if w2 < 0.0:
            w2 = 0.0
        var w = sqrt(w2)

        # Take the closer of the positive and negative solutions
        if lBC * lBC > (lAB * lAB + lCA * lCA):
            w = -w

        var pC = Vec2(w, h)
        halfedge_vec.unsafe_store(2 * heAB, pB.x)
        halfedge_vec.unsafe_store(2 * heAB + 1, pB.y)
        var v = pC - pB
        halfedge_vec.unsafe_store(2 * heBC, v.x)
        halfedge_vec.unsafe_store(2 * heBC + 1, v.y)
        v = -pC
        halfedge_vec.unsafe_store(2 * heCA, v.x)
        halfedge_vec.unsafe_store(2 * heCA + 1, v.y)


# Halfedge vectors in vertex (computeHalfedgeVectorsInVertex). Walks the
# 1-ring of each vertex counter-clockwise, accumulating scaled cone angles.
def compute_halfedge_vectors_in_vertex(
    mesh: HalfedgeMesh, edge_lengths: Ptr, corner_scaled_angles: Ptr, halfedge_vec: Ptr
):
    for v in range(mesh.n_vertices):
        var coord_sum = 0.0

        var first_he = mesh.first_outgoing(v)
        if first_he < 0:
            continue
        var curr_he = first_he
        while True:
            var dir = Vec2.from_angle(coord_sum)
            var len = edge_lengths.unsafe_load(curr_he)
            halfedge_vec.unsafe_store(2 * curr_he, dir.x * len)
            halfedge_vec.unsafe_store(2 * curr_he + 1, dir.y * len)
            coord_sum += corner_scaled_angles.unsafe_load(curr_he)
            curr_he = mesh.twin(mesh.next(mesh.next(curr_he)))
            if curr_he == first_he:
                break


# Transport vectors along halfedge (computeTransportVectorsAlongHalfedge). The
# rotation carrying a vector at the head of an edge into the frame at its tail.
def compute_transport_vectors_along_halfedge(
    mesh: HalfedgeMesh, halfedge_vec: Ptr, transport: Ptr
):
    for he in range(mesh.n_he):
        var twin = mesh.twin(he)
        if twin == he:
            # gc iterates over edges, so a boundary edge is its own twin and
            # `unit(-vecB / vecA)` with vecB == vecA is exactly -1.
            transport.unsafe_store(2 * he, -1.0)
            transport.unsafe_store(2 * he + 1, 0.0)
            continue
        if he > twin:
            continue
        var vecA = Vec2(halfedge_vec.unsafe_load(2 * he), halfedge_vec.unsafe_load(2 * he + 1))
        var vecB = Vec2(halfedge_vec.unsafe_load(2 * twin), halfedge_vec.unsafe_load(2 * twin + 1))
        var rot = (-vecB).cdiv(vecA).normalize()
        transport.unsafe_store(2 * he, rot.x)
        transport.unsafe_store(2 * he + 1, rot.y)
        transport.unsafe_store(2 * twin, rot.x)
        transport.unsafe_store(2 * twin + 1, -rot.y)


# Cotan Laplacian (computeCotanLaplacian), emitted as COO triplets.
def compute_cotan_laplacian(
    mesh: HalfedgeMesh, edge_cotan_weights: Ptr, tri_i: IPtr, tri_j: IPtr, tri_v: Ptr
) -> Int:
    var n = 0
    for he in range(mesh.n_he):
        var twin = mesh.twin(he)
        if twin >= 0:
            if he > twin:
                continue
        var tail = mesh.vertex(he)
        var head = mesh.vertex(mesh.next(he))
        var weight = edge_cotan_weights.unsafe_load(he)

        tri_i.unsafe_store(Int64(n), Int64(tail))
        tri_j.unsafe_store(Int64(n), Int64(tail))
        tri_v.unsafe_store(n, weight)
        n += 1
        tri_i.unsafe_store(Int64(n), Int64(head))
        tri_j.unsafe_store(Int64(n), Int64(head))
        tri_v.unsafe_store(n, weight)
        n += 1
        tri_i.unsafe_store(Int64(n), Int64(tail))
        tri_j.unsafe_store(Int64(n), Int64(head))
        tri_v.unsafe_store(n, -weight)
        n += 1
        tri_i.unsafe_store(Int64(n), Int64(head))
        tri_j.unsafe_store(Int64(n), Int64(tail))
        tri_v.unsafe_store(n, -weight)
        n += 1
    return n


# Vertex connection Laplacian (computeVertexConnectionLaplacian), emitted as
# complex COO triplets. The operator is complex *symmetric*, not Hermitian.
def compute_vertex_connection_laplacian(
    mesh: HalfedgeMesh,
    edge_cotan_weights: Ptr,
    transport: Ptr,
    tri_i: IPtr,
    tri_j: IPtr,
    tri_r: Ptr,
    tri_im: Ptr,
) -> Int:
    var n = 0
    for he in range(mesh.n_he):
        var i_tail = mesh.vertex(he)
        var i_tip = mesh.vertex(mesh.next(he))

        var weight = edge_cotan_weights.unsafe_load(he)
        var twin = mesh.twin(he)
        var rot = Vec2(transport.unsafe_load(2 * twin), transport.unsafe_load(2 * twin + 1))

        tri_i.unsafe_store(Int64(n), Int64(i_tail))
        tri_j.unsafe_store(Int64(n), Int64(i_tail))
        tri_r.unsafe_store(n, weight)
        tri_im.unsafe_store(n, 0.0)
        n += 1
        tri_i.unsafe_store(Int64(n), Int64(i_tail))
        tri_j.unsafe_store(Int64(n), Int64(i_tip))
        tri_r.unsafe_store(n, -weight * rot.x)
        tri_im.unsafe_store(n, -weight * rot.y)
        n += 1
    return n
