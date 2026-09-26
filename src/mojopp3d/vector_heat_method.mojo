"""VectorHeatMethodSolver, in the order of
`geometry-central/src/surface/vector_heat_method.cpp`.

The scalar heat operator is shared with the distance method; what is new here is
the complex connection Laplacian that transports tangent vectors, the affine
connection that the log map needs, and the source-term assembly for each.
"""

from std.math import cos, sin

from mojopp3d.heat_method import source_face_corner
from mojopp3d.mesh import HalfedgeMesh
from mojopp3d.vector import Vec2

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

comptime PI = 3.141592653589793


# extendScalar: scatter the source values and a unit indicator over the faces
# holding each source.
def extend_scalar_rhs(
    mesh: HalfedgeMesh,
    sources: IPtr,
    values: Ptr,
    n_sources: Int,
    data_rhs: Ptr,
    indicator_rhs: Ptr,
):
    for v in range(mesh.n_vertices):
        data_rhs.unsafe_store(v, 0.0)
        indicator_rhs.unsafe_store(v, 0.0)
    for s in range(n_sources):
        var resolved = source_face_corner(mesh, Int(sources.unsafe_load(Int64(s))))
        var he = mesh.first_halfedge(resolved[0])
        for k in range(3):
            if k == resolved[1]:
                var corner = mesh.vertex(he)
                var value = values.unsafe_load(s)
                data_rhs.unsafe_store(corner, data_rhs.unsafe_load(corner) + value)
                indicator_rhs.unsafe_store(
                    corner, indicator_rhs.unsafe_load(corner) + 1.0
                )
            he = mesh.next(he)


# transportTangentVectors: the complex right-hand side, one unit direction per
# source spread over the three corners of its face.
def transport_rhs(
    mesh: HalfedgeMesh,
    sources: IPtr,
    vectors: Ptr,
    n_sources: Int,
    rhs_re: Ptr,
    rhs_im: Ptr,
):
    for v in range(mesh.n_vertices):
        rhs_re.unsafe_store(v, 0.0)
        rhs_im.unsafe_store(v, 0.0)
    for s in range(n_sources):
        var resolved = source_face_corner(mesh, Int(sources.unsafe_load(Int64(s))))
        var vec = Vec2(vectors.unsafe_load(2 * s), vectors.unsafe_load(2 * s + 1))
        var unit = vec.normalize()
        var he = mesh.first_halfedge(resolved[0])
        for k in range(3):
            if k == resolved[1]:
                var corner = mesh.vertex(he)
                rhs_re.unsafe_store(corner, rhs_re.unsafe_load(corner) + unit.x)
                rhs_im.unsafe_store(corner, rhs_im.unsafe_load(corner) + unit.y)
            he = mesh.next(he)
