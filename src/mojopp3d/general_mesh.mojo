"""geometry-central's general `SurfaceMesh` -- the explicit-twin form -- in the
order of `src/surface/surface_mesh.cpp`.

The heat method only ever sees a static face soup, so the port in `mesh.mojo`
stops at the constructor. The robust path does not: `buildIntrinsicTuftedCover`
duplicates every face, inverts the copies, and pulls the sibling lists apart
edge by edge, and `flipToDelaunay` rewires the mesh as it goes. Those four
mutators -- `duplicateFace`, `invertOrientation`, `separateToNewEdge` and
`flip` -- are what this module adds, on top of the same soup constructor.

Upstream keeps a halfedge-capacity vector per array and lets `getNewHalfedge`
double it; here Python owns the buffers and sizes them once, so the fill counts
live in three `Int64` cells the caller can read back after each call. The
`vHalfedge` array is not carried: nothing in the cover or the flip path reads
it, and the static mesh the heat method consumes rebuilds it from the final
face list.
"""

from mojopp3d.mesh import radix_sort_pairs

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

comptime INVALID = -1


struct GenMesh:
    var he_vertex: Int
    var he_next: Int
    var he_face: Int
    var he_edge: Int
    var he_orient: Int
    var he_sibling: Int
    var e_halfedge: Int
    var f_halfedge: Int
    var counts: Int

    def __init__(
        out self,
        he_vertex: Int,
        he_next: Int,
        he_face: Int,
        he_edge: Int,
        he_orient: Int,
        he_sibling: Int,
        e_halfedge: Int,
        f_halfedge: Int,
        counts: Int,
    ):
        self.he_vertex = he_vertex
        self.he_next = he_next
        self.he_face = he_face
        self.he_edge = he_edge
        self.he_orient = he_orient
        self.he_sibling = he_sibling
        self.e_halfedge = e_halfedge
        self.f_halfedge = f_halfedge
        self.counts = counts

    def iv(self) -> IPtr:
        return IPtr(unsafe_from_address=self.he_vertex)

    def inx(self) -> IPtr:
        return IPtr(unsafe_from_address=self.he_next)

    def ifa(self) -> IPtr:
        return IPtr(unsafe_from_address=self.he_face)

    def ied(self) -> IPtr:
        return IPtr(unsafe_from_address=self.he_edge)

    def ior(self) -> IPtr:
        return IPtr(unsafe_from_address=self.he_orient)

    def isb(self) -> IPtr:
        return IPtr(unsafe_from_address=self.he_sibling)

    def ieh(self) -> IPtr:
        return IPtr(unsafe_from_address=self.e_halfedge)

    def ifh(self) -> IPtr:
        return IPtr(unsafe_from_address=self.f_halfedge)

    def ic(self) -> IPtr:
        return IPtr(unsafe_from_address=self.counts)

    def vertex(self, he: Int) -> Int:
        return Int(self.iv().unsafe_load(Int64(he)))

    def next(self, he: Int) -> Int:
        return Int(self.inx().unsafe_load(Int64(he)))

    def face(self, he: Int) -> Int:
        return Int(self.ifa().unsafe_load(Int64(he)))

    def edge(self, he: Int) -> Int:
        return Int(self.ied().unsafe_load(Int64(he)))

    def orientation(self, he: Int) -> Int:
        return Int(self.ior().unsafe_load(Int64(he)))

    def sibling(self, he: Int) -> Int:
        return Int(self.isb().unsafe_load(Int64(he)))

    def halfedge(self, e: Int) -> Int:
        return Int(self.ieh().unsafe_load(Int64(e)))

    def first_halfedge(self, f: Int) -> Int:
        return Int(self.ifh().unsafe_load(Int64(f)))

    def n_he(self) -> Int:
        return Int(self.ic().unsafe_load(0))

    def n_faces(self) -> Int:
        return Int(self.ic().unsafe_load(1))

    def n_edges(self) -> Int:
        return Int(self.ic().unsafe_load(2))

    # Edge::degree
    def edge_degree(self, e: Int) -> Int:
        var first = self.halfedge(e)
        if self.sibling(first) == first:
            return 1
        var n = 1
        var cur = first
        while True:
            cur = self.sibling(cur)
            if cur == first:
                break
            n += 1
        return n

    # Edge::isBoundary
    def is_boundary_edge(self, e: Int) -> Bool:
        var he = self.halfedge(e)
        return self.sibling(he) == he

    # Edge::isManifold
    def is_manifold_edge(self, e: Int) -> Bool:
        var he = self.halfedge(e)
        var sib = self.sibling(he)
        if sib == he:
            return True
        return self.sibling(sib) == he


# SurfaceMesh::SurfaceMesh(polygons), generalized. gc keys edges on the
# unordered vertex pair and links each new halfedge to the previous one seen on
# that edge, then closes the sibling cycle; the radix sort reproduces the
# encounter order because it is stable in the halfedge index.
def build_general_mesh(
    F: IPtr,
    n_faces: Int,
    n_vertices: Int,
    m: GenMesh,
    keys: IPtr,
    vals: IPtr,
    tmpk: IPtr,
    tmpv: IPtr,
    bucket: IPtr,
):
    var n_he = 3 * n_faces
    m.ic().unsafe_store(0, Int64(n_he))
    m.ic().unsafe_store(1, Int64(n_faces))
    m.ic().unsafe_store(2, Int64(0))

    for f in range(n_faces):
        for k in range(3):
            var he = 3 * f + k
            var v = F.unsafe_load(Int64(3 * f + k))
            m.iv().unsafe_store(Int64(he), v)
            m.ifa().unsafe_store(Int64(he), Int64(f))
            m.inx().unsafe_store(Int64(he), Int64(3 * f + (k + 1) % 3))
        m.ifh().unsafe_store(Int64(f), Int64(3 * f))

    for he in range(n_he):
        var k = he % 3
        var tail = F.unsafe_load(Int64(3 * (he // 3) + k))
        var tip = F.unsafe_load(Int64(3 * (he // 3) + (k + 1) % 3))
        var lo = tail
        var hi = tip
        if lo > hi:
            lo = tip
            hi = tail
        keys.unsafe_store(Int64(he), lo * Int64(n_vertices) + hi)
        vals.unsafe_store(Int64(he), Int64(he))
    radix_sort_pairs(keys, vals, tmpk, tmpv, bucket, n_he)

    var n_edges = 0
    var p = 0
    while p < n_he:
        var q = p + 1
        while q < n_he and keys.unsafe_load(Int64(q)) == keys.unsafe_load(Int64(p)):
            q += 1
        var e = n_edges
        n_edges += 1
        var first = Int(vals.unsafe_load(Int64(p)))
        var prev = first
        m.ied().unsafe_store(Int64(first), Int64(e))
        m.isb().unsafe_store(Int64(first), Int64(INVALID))
        m.ior().unsafe_store(Int64(first), Int64(1))
        m.ieh().unsafe_store(Int64(e), Int64(first))
        for a in range(p + 1, q):
            var he = Int(vals.unsafe_load(Int64(a)))
            m.isb().unsafe_store(Int64(he), Int64(prev))
            m.ied().unsafe_store(Int64(he), Int64(e))
            # "best we can do is set orientation to match endpoints"
            m.ior().unsafe_store(
                Int64(he), Int64(m.vertex(he) == m.vertex(m.halfedge(e)))
            )
            prev = he
        if q - p == 1:
            m.isb().unsafe_store(Int64(first), Int64(first))
        else:
            m.isb().unsafe_store(Int64(first), Int64(prev))
        p = q
    m.ic().unsafe_store(2, Int64(n_edges))


# SurfaceMesh::invertOrientation
def invert_orientation(m: GenMesh, f: Int):
    var first_he = m.first_halfedge(f)
    var first_vert = m.vertex(first_he)
    var prev_he = INVALID
    var curr_he = first_he
    while True:
        var next_he = m.next(curr_he)
        var next_vert = first_vert
        if next_he != first_he:
            next_vert = m.vertex(next_he)
        m.iv().unsafe_store(Int64(curr_he), Int64(next_vert))
        m.ior().unsafe_store(Int64(curr_he), Int64(1 - m.orientation(curr_he)))
        if prev_he != INVALID:
            m.inx().unsafe_store(Int64(curr_he), Int64(prev_he))
        prev_he = curr_he
        curr_he = next_he
        if curr_he == first_he:
            break
    m.inx().unsafe_store(Int64(first_he), Int64(prev_he))


# SurfaceMesh::duplicateFace
def duplicate_face(m: GenMesh, f: Int) -> Int:
    var n_faces = m.n_faces()
    var new_f = n_faces
    m.ic().unsafe_store(1, Int64(n_faces + 1))
    var first_new = INVALID
    var prev_new = INVALID
    for k in range(3):
        var old_he = 3 * f + k
        var new_he = m.n_he()
        m.ic().unsafe_store(0, Int64(new_he + 1))
        if prev_new == INVALID:
            first_new = new_he
            m.ifh().unsafe_store(Int64(new_f), Int64(new_he))
        else:
            m.inx().unsafe_store(Int64(prev_new), Int64(new_he))
        m.iv().unsafe_store(Int64(new_he), Int64(m.vertex(old_he)))
        m.ied().unsafe_store(Int64(new_he), Int64(m.edge(old_he)))
        m.ior().unsafe_store(Int64(new_he), Int64(m.orientation(old_he)))
        m.ifa().unsafe_store(Int64(new_he), Int64(new_f))
        # insert into the sibling list, right after old_he
        var sib_n = m.sibling(old_he)
        m.isb().unsafe_store(Int64(old_he), Int64(new_he))
        m.isb().unsafe_store(Int64(new_he), Int64(sib_n))
        prev_new = new_he
    m.inx().unsafe_store(Int64(prev_new), Int64(first_new))
    return new_f


# SurfaceMesh::removeFromSiblingList
def remove_from_sibling_list(m: GenMesh, he: Int):
    var prev = he
    while m.sibling(prev) != he:
        prev = m.sibling(prev)
    m.isb().unsafe_store(Int64(prev), Int64(m.sibling(he)))
    m.isb().unsafe_store(Int64(he), Int64(INVALID))


# SurfaceMesh::separateToNewEdge. Returns the edge holding the pair, or -1 if
# the two halfedges are not incident on the same edge.
def separate_to_new_edge(m: GenMesh, he_a: Int, he_b: Int) -> Int:
    var e = m.edge(he_a)
    if e != m.edge(he_b):
        return -1
    if m.edge_degree(e) <= 2:
        return e
    var new_e = m.n_edges()
    m.ic().unsafe_store(2, Int64(new_e + 1))
    # find some other halfedge incident on the old edge, make it e.halfedge()
    var cur = m.halfedge(e)
    while True:
        if cur != he_a and cur != he_b:
            m.ieh().unsafe_store(Int64(e), Int64(cur))
            break
        cur = m.sibling(cur)
    remove_from_sibling_list(m, he_a)
    remove_from_sibling_list(m, he_b)
    m.ieh().unsafe_store(Int64(new_e), Int64(he_a))
    m.ied().unsafe_store(Int64(he_a), Int64(new_e))
    m.ied().unsafe_store(Int64(he_b), Int64(new_e))
    m.isb().unsafe_store(Int64(he_a), Int64(he_b))
    m.isb().unsafe_store(Int64(he_b), Int64(he_a))
    return new_e


# SurfaceMesh::flip
def flip(m: GenMesh, e_flip: Int) -> Bool:
    if m.is_boundary_edge(e_flip):
        return False
    var ha1 = m.halfedge(e_flip)
    var ha2 = m.next(ha1)
    var ha3 = m.next(ha2)
    if m.next(ha3) != ha1:
        return False
    var hb1 = m.sibling(ha1)
    if hb1 == INVALID:
        return False
    var hb2 = m.next(hb1)
    var hb3 = m.next(hb2)
    if m.next(hb3) != hb1:
        return False
    if m.sibling(hb1) != ha1:
        return False
    if ha2 == hb1 or hb2 == ha1:
        return False
    if m.orientation(ha1) == m.orientation(hb1):
        # the faces have the same orientation; flip the other one instead
        var f = m.face(ha1)
        invert_orientation(m, f)
        var res = flip(m, e_flip)
        invert_orientation(m, f)
        return res

    var fa = m.face(ha1)
    var fb = m.face(hb1)
    var vc = m.vertex(ha3)
    var vd = m.vertex(hb3)

    m.ifh().unsafe_store(Int64(fa), Int64(ha1))
    m.ifh().unsafe_store(Int64(fb), Int64(hb1))

    m.inx().unsafe_store(Int64(ha1), Int64(hb3))
    m.inx().unsafe_store(Int64(hb3), Int64(ha2))
    m.inx().unsafe_store(Int64(ha2), Int64(ha1))
    m.inx().unsafe_store(Int64(hb1), Int64(ha3))
    m.inx().unsafe_store(Int64(ha3), Int64(hb2))
    m.inx().unsafe_store(Int64(hb2), Int64(hb1))

    m.iv().unsafe_store(Int64(ha1), Int64(vc))
    m.iv().unsafe_store(Int64(hb1), Int64(vd))

    m.ifa().unsafe_store(Int64(ha3), Int64(fb))
    m.ifa().unsafe_store(Int64(hb3), Int64(fa))
    return True


# The face list of the current mesh, in halfedge order per face. The heat method
# consumes a static soup, so the mutated mesh is read out this way.
def write_faces(m: GenMesh, F: IPtr):
    for f in range(m.n_faces()):
        var he = m.first_halfedge(f)
        for k in range(3):
            F.unsafe_store(Int64(3 * f + k), Int64(m.vertex(he)))
            he = m.next(he)


# The twin map of the mutated mesh, for handing off to the static halfedge
# mesh the heat method consumes: a halfedge whose edge has two halfedges points
# at the other one, a boundary halfedge gets -1. After the cover every edge has
# exactly two, which is what `build_intrinsic_tufted_cover` leaves behind.
def write_twins(m: GenMesh, twins: IPtr, index_of: IPtr):
    # The twin is named by the general mesh's halfedge numbering; the output is
    # the static face-soup numbering, so map through `index_of`.
    for f in range(m.n_faces()):
        var he = m.first_halfedge(f)
        for k in range(3):
            index_of.unsafe_store(Int64(he), Int64(3 * f + k))
            he = m.next(he)
    for f in range(m.n_faces()):
        var he = m.first_halfedge(f)
        for k in range(3):
            var sib = m.sibling(he)
            if sib == he:
                twins.unsafe_store(Int64(3 * f + k), Int64(INVALID))
            else:
                twins.unsafe_store(
                    Int64(3 * f + k),
                    index_of.unsafe_load(Int64(sib)),
                )
            he = m.next(he)


# Per-halfedge edge lengths, in the static face-soup layout `IntrinsicGeometry`
# uses: face `f` owns slots `3f, 3f+1, 3f+2`. The general mesh's own halfedge
# numbering drifts as soon as a face is duplicated, so walk the faces.
def write_halfedge_edge_lengths(m: GenMesh, edge_lengths: Ptr, dst: Ptr):
    for f in range(m.n_faces()):
        var he = m.first_halfedge(f)
        for k in range(3):
            dst.unsafe_store(3 * f + k, edge_lengths.unsafe_load(m.edge(he)))
            he = m.next(he)
