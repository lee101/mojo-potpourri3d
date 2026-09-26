"""EmbeddedGeometryInterface, in the order of
`geometry-central/src/surface/embedded_geometry_interface.cpp`.

Only the quantities the heat-method solvers touch: edge lengths, face normals,
corner angles, vertex normals and the vertex tangent basis (the last of which
`get_tangent_frames` hands straight back to the caller).
"""

from std.math import sqrt, acos

from mojopp3d.mesh import HalfedgeMesh
from mojopp3d.vector import Vec2, Vec3

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]


def _clamp(q: Float64, low: Float64, high: Float64) -> Float64:
    if q < low:
        return low
    if q > high:
        return high
    return q


def _pos(positions: Ptr, v: Int) -> Vec3:
    return Vec3(positions.unsafe_load(3 * v), positions.unsafe_load(3 * v + 1), positions.unsafe_load(3 * v + 2))


# Edge lengths (computeEdgeLengths), stored per halfedge.
def compute_edge_lengths(mesh: HalfedgeMesh, positions: Ptr, edge_lengths: Ptr):
    for he in range(mesh.n_he):
        var tail = mesh.vertex(he)
        var tip = mesh.vertex(mesh.next(he))
        var dx = positions.unsafe_load(3 * tip) - positions.unsafe_load(3 * tail)
        var dy = positions.unsafe_load(3 * tip + 1) - positions.unsafe_load(3 * tail + 1)
        var dz = positions.unsafe_load(3 * tip + 2) - positions.unsafe_load(3 * tail + 2)
        edge_lengths.unsafe_store(he, sqrt(dx * dx + dy * dy + dz * dz))


# Face normals (computeFaceNormals).
def compute_face_normals(mesh: HalfedgeMesh, positions: Ptr, face_normals: Ptr):
    for f in range(mesh.n_faces):
        var heAB = mesh.first_halfedge(f)
        var heBC = mesh.next(heAB)
        var heCA = mesh.next(heBC)
        var pA = _pos(positions, mesh.vertex(heAB))
        var pB = _pos(positions, mesh.vertex(heBC))
        var pC = _pos(positions, mesh.vertex(heCA))

        var normal_sum = (pB - pA).cross(pC - pA)
        var l = normal_sum.norm()
        if l > 0.0:
            normal_sum = normal_sum * (1.0 / l)
        face_normals.unsafe_store(3 * f, normal_sum.x)
        face_normals.unsafe_store(3 * f + 1, normal_sum.y)
        face_normals.unsafe_store(3 * f + 2, normal_sum.z)


# Corner angles (EmbeddedGeometryInterface::computeCornerAngles), from the
# embedded positions rather than from edge lengths. Indexed by the halfedge
# sitting at the corner.
def compute_corner_angles_embedded(
    mesh: HalfedgeMesh, positions: Ptr, corner_angles: Ptr
):
    for f in range(mesh.n_faces):
        for k in range(3):
            var he = mesh.first_halfedge(f) + k
            var he_next = mesh.next(he)
            var pA = _pos(positions, mesh.vertex(he))
            var pB = _pos(positions, mesh.vertex(he_next))
            var pC = _pos(positions, mesh.vertex(mesh.next(he_next)))

            var q = (pC - pB).normalize().dot((pA - pB).normalize())
            q = _clamp(q, -1.0, 1.0)
            corner_angles.unsafe_store(he_next, acos(q))


# Vertex normals (computeVertexNormals): corner-angle-weighted face normals.
def compute_vertex_normals(
    mesh: HalfedgeMesh, face_normals: Ptr, corner_angles: Ptr, vertex_normals: Ptr
):
    for v in range(mesh.n_vertices):
        var normal_sum = Vec3(0.0, 0.0, 0.0)
        var first_he = mesh.first_outgoing(v)
        var he = first_he
        while he >= 0:
            var f = mesh.face(he)
            var weight = corner_angles.unsafe_load(he)
            normal_sum = normal_sum + Vec3(
                face_normals.unsafe_load(3 * f), face_normals.unsafe_load(3 * f + 1), face_normals.unsafe_load(3 * f + 2)
            ) * weight
            he = mesh.twin(mesh.next(mesh.next(he)))
            if he == first_he:
                break
        var l = normal_sum.norm()
        if l > 0.0:
            normal_sum = normal_sum * (1.0 / l)
        vertex_normals.unsafe_store(3 * v, normal_sum.x)
        vertex_normals.unsafe_store(3 * v + 1, normal_sum.y)
        vertex_normals.unsafe_store(3 * v + 2, normal_sum.z)


# Vertex tangent basis (computeVertexTangentBasis): the 1-ring edge vectors,
# rotated into the intrinsic frame at each vertex and averaged.
def compute_vertex_tangent_basis(
    mesh: HalfedgeMesh,
    positions: Ptr,
    vertex_normals: Ptr,
    halfedge_vec_in_vertex: Ptr,
    tangent_basis: Ptr,
):
    for v in range(mesh.n_vertices):
        var N = Vec3(vertex_normals.unsafe_load(3 * v), vertex_normals.unsafe_load(3 * v + 1), vertex_normals.unsafe_load(3 * v + 2))
        var basis_x_sum = Vec3(0.0, 0.0, 0.0)

        var first_he = mesh.first_outgoing(v)
        var he = first_he
        while he >= 0:
            var e_vec = _pos(positions, mesh.vertex(mesh.next(he))) - _pos(
                positions, mesh.vertex(he)
            )
            e_vec = e_vec.remove_component(N)

            var angle = Vec2(
                halfedge_vec_in_vertex.unsafe_load(2 * he),
                halfedge_vec_in_vertex.unsafe_load(2 * he + 1),
            ).arg()
            var e_vec_x = e_vec.rotate_around(N, -angle)

            basis_x_sum = basis_x_sum + e_vec_x

            he = mesh.twin(mesh.next(mesh.next(he)))
            if he == first_he:
                break

        var l = basis_x_sum.norm()
        if l > 0.0:
            basis_x_sum = basis_x_sum * (1.0 / l)
        var basis_y = N.cross(basis_x_sum)
        tangent_basis.unsafe_store(6 * v, basis_x_sum.x)
        tangent_basis.unsafe_store(6 * v + 1, basis_x_sum.y)
        tangent_basis.unsafe_store(6 * v + 2, basis_x_sum.z)
        tangent_basis.unsafe_store(6 * v + 3, basis_y.x)
        tangent_basis.unsafe_store(6 * v + 4, basis_y.y)
        tangent_basis.unsafe_store(6 * v + 5, basis_y.z)
