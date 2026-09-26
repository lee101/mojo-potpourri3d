"""The halfedge mesh, mirroring `SurfaceMesh::SurfaceMesh(polygons)`.

geometry-central numbers halfedges in the order the faces arrive, so face `f`
owns halfedges `3f, 3f+1, 3f+2` and vertex indices are the caller's own. The
only thing that needs care is `v_halfedge`: the constructor writes
`vHalfedgeArr[tail] = he` for every halfedge as it walks the faces, so the
value that survives is the *last* outgoing halfedge of each vertex, and that is
the one the tangent-frame orbit starts from.

Storage: geometry-central indexes edge-keyed quantities by edge. Here they are
stored per halfedge instead, which is the same number (every edge quantity
satisfies `L(i,j) == L(twin,j)`) and saves a second array.

The connectivity travels as `Int` addresses because a struct field cannot
expose `AnyOrigin`; the pointers are rebuilt inside each accessor.
"""

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]


struct HalfedgeMesh:
    var he_vertex: Int
    var he_next: Int
    var he_twin: Int
    var he_face: Int
    var v_halfedge: Int
    var f_halfedge: Int
    var n_he: Int
    var n_vertices: Int
    var n_faces: Int

    def __init__(
        out self,
        he_vertex: Int,
        he_next: Int,
        he_twin: Int,
        he_face: Int,
        v_halfedge: Int,
        f_halfedge: Int,
        n_he: Int,
        n_vertices: Int,
        n_faces: Int,
    ):
        self.he_vertex = he_vertex
        self.he_next = he_next
        self.he_twin = he_twin
        self.he_face = he_face
        self.v_halfedge = v_halfedge
        self.f_halfedge = f_halfedge
        self.n_he = n_he
        self.n_vertices = n_vertices
        self.n_faces = n_faces

    def iv(self) -> IPtr:
        return IPtr(unsafe_from_address=self.he_vertex)

    def inx(self) -> IPtr:
        return IPtr(unsafe_from_address=self.he_next)

    def itw(self) -> IPtr:
        return IPtr(unsafe_from_address=self.he_twin)

    def ifa(self) -> IPtr:
        return IPtr(unsafe_from_address=self.he_face)

    def ivh(self) -> IPtr:
        return IPtr(unsafe_from_address=self.v_halfedge)

    def ifh(self) -> IPtr:
        return IPtr(unsafe_from_address=self.f_halfedge)

    def vertex(self, he: Int) -> Int:
        return Int(self.iv().unsafe_load(Int64(he)))

    def next(self, he: Int) -> Int:
        return Int(self.inx().unsafe_load(Int64(he)))

    def twin(self, he: Int) -> Int:
        return Int(self.itw().unsafe_load(Int64(he)))

    def face(self, he: Int) -> Int:
        return Int(self.ifa().unsafe_load(Int64(he)))

    def first_halfedge(self, f: Int) -> Int:
        return Int(self.ifh().unsafe_load(Int64(f)))

    def first_outgoing(self, v: Int) -> Int:
        return Int(self.ivh().unsafe_load(Int64(v)))

    def is_interior(self, he: Int) -> Bool:
        return Int(self.itw().unsafe_load(Int64(he))) >= 0

    def is_boundary(self, he: Int) -> Bool:
        return Int(self.itw().unsafe_load(Int64(he))) < 0


# LSD radix sort of (key, value) pairs, 8 bits at a time. Eight passes is an
# even number, so the sorted result lands back in the caller's buffers.
def radix_sort_pairs(
    keys: IPtr, vals: IPtr, tmpk: IPtr, tmpv: IPtr, count: IPtr, m: Int
):
    for shift in range(0, 64, 8):
        var sh = Int64(shift)
        for b in range(256):
            count.unsafe_store(Int64(b), Int64(0))
        for p in range(m):
            var bucket = (keys.unsafe_load(Int64(p)) >> sh) & Int64(255)
            count.unsafe_store(bucket, count.unsafe_load(bucket) + Int64(1))
        var running = 0
        for b in range(256):
            var c = Int(count.unsafe_load(Int64(b)))
            count.unsafe_store(Int64(b), Int64(running))
            running += c
        for p in range(m):
            var bucket = (keys.unsafe_load(Int64(p)) >> sh) & Int64(255)
            var slot = count.unsafe_load(bucket)
            count.unsafe_store(bucket, slot + Int64(1))
            tmpk.unsafe_store(slot, keys.unsafe_load(Int64(p)))
            tmpv.unsafe_store(slot, vals.unsafe_load(Int64(p)))
        for p in range(m):
            keys.unsafe_store(Int64(p), tmpk.unsafe_load(Int64(p)))
            vals.unsafe_store(Int64(p), tmpv.unsafe_load(Int64(p)))


def build_halfedge_mesh(
    F: IPtr,
    n_faces: Int,
    n_vertices: Int,
    he_vertex: IPtr,
    he_next: IPtr,
    he_twin: IPtr,
    he_face: IPtr,
    v_halfedge: IPtr,
    f_halfedge: IPtr,
    keys: IPtr,
    vals: IPtr,
    tmpk: IPtr,
    tmpv: IPtr,
    count: IPtr,
):
    var n_he = 3 * n_faces
    for v in range(n_vertices):
        v_halfedge.unsafe_store(Int64(v), Int64(-1))
    for he in range(n_he):
        var f = he // 3
        var k = he % 3
        var v = F.unsafe_load(Int64(3 * f + k))
        he_vertex.unsafe_store(Int64(he), v)
        he_face.unsafe_store(Int64(he), Int64(f))
        he_next.unsafe_store(Int64(he), Int64(3 * f + (k + 1) % 3))
        f_halfedge.unsafe_store(Int64(f), Int64(3 * f))
        # Written unconditionally, so the last outgoing halfedge wins.
        v_halfedge.unsafe_store(v, Int64(he))

    # Match halfedges to twins. Key on the *unordered* vertex pair, so the two
    # orientations of an edge land next to each other after the sort.
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
    radix_sort_pairs(keys, vals, tmpk, tmpv, count, n_he)

    for he in range(n_he):
        he_twin.unsafe_store(Int64(he), Int64(-1))
    var p = 0
    while p < n_he:
        var q = p + 1
        while q < n_he:
            if keys.unsafe_load(Int64(q)) != keys.unsafe_load(Int64(p)):
                break
            q += 1
        # A directed edge is unique in a valid manifold triangulation, so each
        # group is a lone boundary halfedge or a matched interior pair.
        var a = p
        while a + 1 < q:
            he_twin.unsafe_store(vals.unsafe_load(Int64(a)), vals.unsafe_load(Int64(a + 1)))
            he_twin.unsafe_store(vals.unsafe_load(Int64(a + 1)), vals.unsafe_load(Int64(a)))
            a += 2
        p = q
