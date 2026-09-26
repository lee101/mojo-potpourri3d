"""The tufted-cover passes, in the order of the three upstream files they come
from:

* `intrinsic_mollification.cpp` -- `mollifyIntrinsic`
* `tufted_laplacian.cpp`      -- `buildIntrinsicTuftedCover`
* `simple_idt.cpp`            -- `flipToDelaunay`

`mollifyIntrinsic` is not the noisy surface mollifier: it offsets every edge
length by a single scalar, the amount that makes the most degenerate triangle
in the mesh just barely non-degenerate. The cover then duplicates every face,
inverts the copy, and glues front to back around each edge so that the result
is edge-manifold -- a double cover, which is what the point cloud and the
robust Laplacian need in order to have a well-defined intrinsic neighbourhood
at a non-manifold vertex. The flip pass is a plain queue-driven edge flip to the
Euclidean Delaunay condition.
"""

from std.math import sqrt, isfinite

from mojopp3d.general_mesh import (
    GenMesh,
    duplicate_face,
    flip,
    invert_orientation,
    separate_to_new_edge,
)
from mojopp3d.vector import Vec2

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

comptime INVALID = -1


# elementary_geometry.ipp
def triangle_area(l_ab: Float64, l_bc: Float64, l_ca: Float64) -> Float64:
    var s = (l_ab + l_bc + l_ca) / 2.0
    var arg = s * (s - l_ab) * (s - l_bc) * (s - l_ca)
    if arg < 0.0:
        arg = 0.0
    return sqrt(arg)


# elementary_geometry.ipp: the point at distance l_bc from pB and l_ca from pA
def layout_triangle_vertex(
    pa: Vec2, pb: Vec2, l_bc: Float64, l_ca: Float64
) -> Vec2:
    var l_ab = (pb - pa).norm()
    var area = triangle_area(l_ab, l_bc, l_ca)
    var h = 2.0 * area / l_ab
    var w = (l_ab * l_ab - l_bc * l_bc + l_ca * l_ca) / (2.0 * l_ab)
    var v_abn = (pb - pa) / l_ab
    var v_abn_perp = Vec2(-v_abn.y, v_abn.x)
    return pa + v_abn * w + v_abn_perp * h


# mollifyIntrinsic
def mollify_intrinsic(m: GenMesh, edge_lengths: Ptr, n_edges: Int, relative_factor: Float64) -> Float64:
    var edge_sum = 0.0
    for e in range(n_edges):
        edge_sum += edge_lengths.unsafe_load(e)
    var mean_edge = edge_sum / Float64(n_edges)
    var mollify_delta = mean_edge * relative_factor

    var mollify_eps = 0.0
    # gc loops `mesh.interiorHalfedges()`, and `heIsInterior` tests the *face*
    # against the boundary loops, which a face soup does not have -- so the
    # range covers every halfedge, including one on a boundary edge. The worst
    # triangle in a mesh usually touches the boundary, and its `l_C - l_A - l_B`
    # is exactly the term that has to be lifted off zero.
    for he in range(m.n_he()):
        var l_a = edge_lengths.unsafe_load(m.edge(he))
        var l_b = edge_lengths.unsafe_load(m.edge(m.next(he)))
        var l_c = edge_lengths.unsafe_load(m.edge(m.next(m.next(he))))
        var this_eps = l_c - l_a - l_b + mollify_delta
        if this_eps > mollify_eps:
            mollify_eps = this_eps
    for e in range(n_edges):
        edge_lengths.unsafe_store(e, edge_lengths.unsafe_load(e) + mollify_eps)
    return mollify_eps


# buildIntrinsicTuftedCover, with posGeom == null (the intrinsic-only form both
# callers use). Returns the number of edges the cover created, or -1 if the
# halfedge arrays would overflow their capacity.
def build_intrinsic_tufted_cover(
    m: GenMesh,
    edge_lengths: Ptr,
    n_orig_faces: Int,
    n_orig_edges: Int,
    other_sheet: IPtr,
    is_front: IPtr,
    is_orig_edge: IPtr,
    edge_faces: IPtr,
    max_faces: Int,
    max_he: Int,
    max_edges: Int,
) -> Int:
    for f in range(n_orig_faces):
        is_front.unsafe_store(Int64(f), Int64(1))
    for e in range(n_orig_edges):
        is_orig_edge.unsafe_store(Int64(e), Int64(1))

    # Create two copies of each input face; the original copy serves as front
    for f_front in range(n_orig_faces):
        if is_front.unsafe_load(Int64(f_front)) == 0:
            continue
        if m.n_faces() >= max_faces or m.n_he() >= max_he:
            return -1
        var f_back = duplicate_face(m, f_front)

        # read off the correspondence between the halfedges, before inverting
        var he_f = 3 * f_front
        var first_f = he_f
        var he_b = m.first_halfedge(f_back)
        var first_b = he_b
        while True:
            other_sheet.unsafe_store(Int64(he_f), Int64(he_b))
            other_sheet.unsafe_store(Int64(he_b), Int64(he_f))
            he_f = m.next(he_f)
            he_b = m.next(he_b)
            if he_f == first_f:
                break
        if he_b != first_b:
            return -1

        invert_orientation(m, f_back)
        is_front.unsafe_store(Int64(f_back), Int64(0))

    # Around each edge, glue back faces to front along newly created edges
    for e in range(n_orig_edges):
        if is_orig_edge.unsafe_load(Int64(e)) == 0:
            continue

        # Gather the original faces incident on the edge
        var n_faces_on_edge = 0
        var cur = m.halfedge(e)
        while True:
            if is_front.unsafe_load(Int64(m.face(cur))) != 0:
                edge_faces.unsafe_store(Int64(n_faces_on_edge), Int64(cur))
                n_faces_on_edge += 1
            cur = m.sibling(cur)
            if cur == m.halfedge(e):
                break
        if n_faces_on_edge == 0:
            return -1

        # Sequentially connect the faces
        var curr_he = Int(edge_faces.unsafe_load(0))
        if m.orientation(curr_he) != 0:
            curr_he = Int(other_sheet.unsafe_load(Int64(curr_he)))
        for i in range(n_faces_on_edge):
            var next_he = Int(
                edge_faces.unsafe_load(Int64((i + 1) % n_faces_on_edge))
            )
            if m.orientation(curr_he) == m.orientation(next_he):
                next_he = Int(other_sheet.unsafe_load(Int64(next_he)))
            if m.n_edges() >= max_edges:
                return -1
            var new_e = separate_to_new_edge(m, curr_he, next_he)
            if new_e < 0:
                return -1
            if new_e >= n_orig_edges:
                is_orig_edge.unsafe_store(Int64(new_e), Int64(0))
                edge_lengths.unsafe_store(new_e, edge_lengths.unsafe_load(e))
            curr_he = Int(other_sheet.unsafe_load(Int64(next_he)))
    return m.n_edges() - n_orig_edges


# flippedEdgeLen, FlipType::Euclidean
def flipped_edge_len(m: GenMesh, edge_lengths: Ptr, he: Int) -> Float64:
    var he_a0 = he
    var he_a1 = m.next(he_a0)
    var he_a2 = m.next(he_a1)
    var he_b0 = m.sibling(he_a0)
    var he_b1 = m.next(he_b0)
    var he_b2 = m.next(he_b1)

    # Handle non-oriented edges
    if m.orientation(he_a0) == m.orientation(he_b0):
        var tmp = he_b1
        he_b1 = he_b2
        he_b2 = tmp

    var l01 = edge_lengths.unsafe_load(m.edge(he_a1))
    var l12 = edge_lengths.unsafe_load(m.edge(he_a2))
    var l23 = edge_lengths.unsafe_load(m.edge(he_b1))
    var l30 = edge_lengths.unsafe_load(m.edge(he_b2))
    var l02 = edge_lengths.unsafe_load(m.edge(he_a0))

    var p3 = Vec2(0.0, 0.0)
    var p0 = Vec2(l30, 0.0)
    # involves more arithmetic than strictly necessary
    var p2 = layout_triangle_vertex(p3, p0, l02, l23)
    var p1 = layout_triangle_vertex(p2, p0, l01, l12)
    return (p1 - p3).norm()


# the `area` lambda: Heron on the three edge lengths of a face
def _face_area(m: GenMesh, edge_lengths: Ptr, f: Int) -> Float64:
    var he = m.first_halfedge(f)
    var a = edge_lengths.unsafe_load(m.edge(he))
    he = m.next(he)
    var b = edge_lengths.unsafe_load(m.edge(he))
    he = m.next(he)
    var c = edge_lengths.unsafe_load(m.edge(he))
    return triangle_area(a, b, c)


# the `halfedgeCotanWeight` lambda
def _halfedge_cotan_weight(m: GenMesh, edge_lengths: Ptr, he: Int) -> Float64:
    if m.sibling(he) == he:
        return 0.0
    var l_ij = edge_lengths.unsafe_load(m.edge(he))
    var he1 = m.next(he)
    var l_jk = edge_lengths.unsafe_load(m.edge(he1))
    var l_jk2 = m.next(he1)
    var l_ki = edge_lengths.unsafe_load(m.edge(l_jk2))
    var area_v = _face_area(m, edge_lengths, m.face(he))
    var cot_value = (-l_ij * l_ij + l_jk * l_jk + l_ki * l_ki) / (4.0 * area_v)
    return cot_value / 2


# the `edgeCotanWeight` lambda
def _edge_cotan_weight(m: GenMesh, edge_lengths: Ptr, e: Int) -> Float64:
    var he = m.halfedge(e)
    var w = _halfedge_cotan_weight(m, edge_lengths, he)
    var twin = m.sibling(he)
    if twin != he:
        w += _halfedge_cotan_weight(m, edge_lengths, twin)
    return w


# the `shouldFlipEdge` lambda, FlipType::Euclidean
def _should_flip_edge(m: GenMesh, edge_lengths: Ptr, e: Int, delaunay_eps: Float64) -> Bool:
    if m.is_boundary_edge(e):
        return False
    return _edge_cotan_weight(m, edge_lengths, e) < -delaunay_eps


# flipToDelaunay. `queue` is a ring buffer of `queue_cap` entries; returns the
# number of flips, or -1 if the queue overflowed. Upstream throws on a boundary
# or non-manifold edge and on a queue that cannot grow; both become the same
# -1 here, and the caller turns it into an exception.
def flip_to_delaunay(
    m: GenMesh,
    edge_lengths: Ptr,
    n_edges: Int,
    queue: IPtr,
    in_queue: IPtr,
    queue_cap: Int,
    delaunay_eps: Float64,
) -> Int:
    for e in range(n_edges):
        queue.unsafe_store(Int64(e), Int64(e))
        in_queue.unsafe_store(Int64(e), Int64(1))
    var head = 0
    var tail = n_edges
    var live = n_edges

    var n_flips = 0
    while live > 0:
        if m.is_boundary_edge(Int(queue.unsafe_load(Int64(head)))):
            return -1
        var e = Int(queue.unsafe_load(Int64(head)))
        head = (head + 1) % queue_cap
        live -= 1
        in_queue.unsafe_store(Int64(e), Int64(0))

        if not m.is_manifold_edge(e):
            return -1
        if not _should_flip_edge(m, edge_lengths, e, delaunay_eps):
            continue

        # Get geometric data
        var he = m.halfedge(e)
        var new_length = flipped_edge_len(m, edge_lengths, he)
        # If we're going to create a non-finite edge length, abort the flip
        if not isfinite(new_length):
            continue
        if not flip(m, e):
            continue
        edge_lengths.unsafe_store(e, new_length)
        n_flips += 1

        # Add neighbors to queue, as they may need flipping now
        he = m.halfedge(e)
        var he_n = m.next(he)
        var he_t = m.sibling(he)
        var he_tn = m.next(he_t)
        var neigh = [
            m.edge(he_n),
            m.edge(m.next(he_n)),
            m.edge(he_tn),
            m.edge(m.next(he_tn)),
        ]
        for q in neigh:
            if in_queue.unsafe_load(Int64(q)) == 0:
                if live == queue_cap:
                    return -1
                queue.unsafe_store(Int64(tail), Int64(q))
                tail = (tail + 1) % queue_cap
                live += 1
                in_queue.unsafe_store(Int64(q), Int64(1))
    return n_flips
