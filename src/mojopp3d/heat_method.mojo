"""HeatMethodDistanceSolver, in the order of
`geometry-central/src/surface/heat_method_distance.cpp`.

The method itself is three steps -- diffuse a unit mass for short time, read
the gradient off the result, integrate that gradient back to a distance -- and
the two linear solves around it. The solves live in `mojopp3d.linalg`; what is
here is everything else, in the order upstream writes it.
"""

from std.math import sqrt

from mojopp3d.mesh import HalfedgeMesh
from mojopp3d.vector import Vec2

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

comptime PI = 3.141592653589793


# SurfacePoint::inSomeFace, for a vertex source: the face holding the vertex's
# outgoing halfedge, and which of that face's three corners it is. Returns the
# corner index; upstream returns it implicitly through the barycentric coords.
def source_face_corner(mesh: HalfedgeMesh, v: Int) -> Tuple[Int, Int]:
    var he = mesh.first_outgoing(v)
    var target_he = mesh.first_halfedge(mesh.face(he))
    if he == target_he:
        return (mesh.face(he), 0)
    if mesh.next(he) == target_he:
        return (mesh.face(he), 2)
    return (mesh.face(he), 1)


# === Build RHS: the unit mass of a delta source, spread over its face
# (HeatMethodDistanceSolver::computeDistance).
def build_rhs(mesh: HalfedgeMesh, sources: IPtr, n_sources: Int, rhs: Ptr):
    for v in range(mesh.n_vertices):
        rhs.unsafe_store(v, 0.0)
    for s in range(n_sources):
        var resolved = source_face_corner(mesh, Int(sources.unsafe_load(Int64(s))))
        var he = mesh.first_halfedge(resolved[0])
        for k in range(3):
            if k == resolved[1]:
                var corner = mesh.vertex(he)
                rhs.unsafe_store(corner, rhs.unsafe_load(corner) + 1.0)
            he = mesh.next(he)


# === Shift the distance so that it vanishes at the source set
# (HeatMethodDistanceSolver::computeDistance, the "Shift distance" block).
def shift_distance(
    mesh: HalfedgeMesh,
    dist_vec: Ptr,
    edge_lengths: Ptr,
    sources: IPtr,
    n_sources: Int,
    shift: Ptr,
) -> Float64:
    # Shindler & Chen 2012, Barycentric Coordinates in Olympiad Geometry,
    # Section 3.2 -- the distance between two points given their barycentric
    # coordinates in a triangle with the given edge lengths.
    var dist_diff_at_source = 0.0
    var weight_sum = 0.0
    for s in range(n_sources):
        var resolved = source_face_corner(mesh, Int(sources.unsafe_load(Int64(s))))
        var he0 = mesh.first_halfedge(resolved[0])
        var l0 = edge_lengths.unsafe_load(he0)
        var l1 = edge_lengths.unsafe_load(mesh.next(he0))
        var l2 = edge_lengths.unsafe_load(mesh.next(mesh.next(he0)))

        # The source sits at one of the three corners, so its barycentric
        # coords are a unit vector; the corner index says which.
        var b = resolved[1]
        var fc0 = 0.0
        var fc1 = 0.0
        var fc2 = 0.0
        if b == 0:
            fc0 = 1.0
        elif b == 1:
            fc1 = 1.0
        else:
            fc2 = 1.0

        var he = he0
        for i in range(3):
            var tx = 0.0
            var ty = 0.0
            var tz = 0.0
            if i == 0:
                tx = 1.0
            elif i == 1:
                ty = 1.0
            else:
                tz = 1.0

            var b0 = tx - fc0
            var b1 = ty - fc1
            var b2 = tz - fc2
            var d2 = 0.0
            d2 += l0 * l0 * b0 * b1
            d2 += l1 * l1 * b1 * b2
            d2 += l2 * l2 * b2 * b0
            if not (d2 <= 0.0):
                d2 = 0.0  # ensure it's negative so the sqrt below succeeds
            var expected_dist_at_vert = sqrt(-d2)

            var act_dist_at_vert = dist_vec.unsafe_load(mesh.vertex(he))
            var w = fc0
            if i == 1:
                w = fc1
            elif i == 2:
                w = fc2
            dist_diff_at_source += (act_dist_at_vert - expected_dist_at_vert) * w
            weight_sum += w

            he = mesh.next(he)

    dist_diff_at_source = dist_diff_at_source / weight_sum
    shift.unsafe_store(0, -dist_diff_at_source)
    return -dist_diff_at_source


# === Normalize in each face and evaluate divergence
# (HeatMethodDistanceSolver::computeDistanceRHS, the middle block).
def compute_divergence(
    mesh: HalfedgeMesh,
    halfedge_vec_in_face: Ptr,
    halfedge_cotan_weights: Ptr,
    heat_vec: Ptr,
    divergence: Ptr,
):
    for v in range(mesh.n_vertices):
        divergence.unsafe_store(v, 0.0)
    for f in range(mesh.n_faces):
        # warning, wrong magnitude because we don't care
        var grad_u_dir = Vec2(0.0, 0.0)
        var he = mesh.first_halfedge(f)
        for k in range(3):
            var e_perp = Vec2(
                halfedge_vec_in_face.unsafe_load(2 * mesh.next(he)),
                halfedge_vec_in_face.unsafe_load(2 * mesh.next(he) + 1),
            ).rotate90()
            grad_u_dir = grad_u_dir + e_perp * heat_vec.unsafe_load(mesh.vertex(he))
            he = mesh.next(he)

        grad_u_dir = grad_u_dir.normalize_cutoff()

        he = mesh.first_halfedge(f)
        for k in range(3):
            var e = Vec2(
                halfedge_vec_in_face.unsafe_load(2 * he),
                halfedge_vec_in_face.unsafe_load(2 * he + 1),
            )
            var val = halfedge_cotan_weights.unsafe_load(he) * e.dot(grad_u_dir)
            var tail = mesh.vertex(he)
            var tip = mesh.vertex(mesh.next(he))
            divergence.unsafe_store(tail, divergence.unsafe_load(tail) + val)
            divergence.unsafe_store(tip, divergence.unsafe_load(tip) - val)
            he = mesh.next(he)
