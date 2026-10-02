# geometry-central surface/surface_mesh.{h,cpp}, restricted to the manifold triangulations this
# port supports. The halfedge connectivity is stored flat:
#
#   F[3f + i]   vertex of corner i of face f
#   FT[h]       twin of halfedge h (h == 3f + i); FT[h] == h marks a boundary halfedge
#   EID[h]      edge id of halfedge h
#   EL[e]       intrinsic length of edge e
#
# halfedge ids are the corner ids themselves, so next/prev are arithmetic. A flip rewrites the six
# corner->vertex entries of the two incident faces and leaves FT and EID untouched: the two halfedges
# of the flipped edge simply become the new diagonal, which is what upstream's `mesh.flip(e, false)`
# does too (it keeps the edge id and swaps its endpoints).

from std.math import sqrt, isnan, isinf, abs
from vector_util import Vector2, nan64, layout_triangle_vertex


# Bottom-up merge sort over (key, payload) pairs. Deterministic. It lives in a struct so the
# arrays stay owned by the caller instead of being passed through a move-only signature.
struct PairSorter:
    var keys: List[Int]
    var vals: List[Int]
    var scratch_k: List[Int]
    var scratch_v: List[Int]

    def __init__(out self, n: Int):
        self.keys = List[Int](length=n, fill=0)
        self.vals = List[Int](length=n, fill=0)
        self.scratch_k = List[Int](length=n, fill=0)
        self.scratch_v = List[Int](length=n, fill=0)

    def sort(mut self) -> None:
        var n = len(self.keys)
        var width = 1
        while width < n:
            var i = 0
            while i < n:
                var mid = min(i + width, n)
                var hi = min(i + 2 * width, n)
                var a = i
                var b = mid
                var o = i
                while a < mid and b < hi:
                    if self.keys[a] <= self.keys[b]:
                        self.scratch_k[o] = self.keys[a]
                        self.scratch_v[o] = self.vals[a]
                        a += 1
                    else:
                        self.scratch_k[o] = self.keys[b]
                        self.scratch_v[o] = self.vals[b]
                        b += 1
                    o += 1
                while a < mid:
                    self.scratch_k[o] = self.keys[a]
                    self.scratch_v[o] = self.vals[a]
                    a += 1
                    o += 1
                while b < hi:
                    self.scratch_k[o] = self.keys[b]
                    self.scratch_v[o] = self.vals[b]
                    b += 1
                    o += 1
                i += 2 * width
            var t = 0
            while t < n:
                self.keys[t] = self.scratch_k[t]
                self.vals[t] = self.scratch_v[t]
                t += 1
            width *= 2


struct SurfaceMesh:
    var nV: Int
    var nF: Int
    var nE: Int
    var nB: Int
    var F: List[Int]
    var FT: List[Int]
    var EID: List[Int]
    var EL: List[Float64]
    var EHA: List[Int]

    def __init__(out self, n_v: Int, n_f: Int, n_e: Int, n_b: Int):
        self.nV = n_v
        self.nF = n_f
        self.nE = n_e
        self.nB = n_b
        self.F = List[Int]()
        self.FT = List[Int]()
        self.EID = List[Int]()
        self.EL = List[Float64]()
        self.EHA = List[Int]()

    def nextHe(self, h: Int) -> Int:
        var i = h % 3
        return h - i + (i + 1) % 3

    def prevHe(self, h: Int) -> Int:
        var i = h % 3
        return h - i + (i + 2) % 3

    def isInterior(self, h: Int) -> Bool:
        return self.FT[h] // 3 != h // 3

    def edgeLength(self, h: Int) -> Float64:
        return self.EL[self.EID[h]]

    # SurfaceMesh::flip. Returns the new length of the edge, or NaN when the flip is not possible.
    # Upstream's `mesh.flip(e, false)` keeps the edge id, so the two halfedges of `e` end up carrying
    # the new diagonal. Upstream then permutes heNextArr *within* each face, which moves its halfedges
    # between the two faces; a flat corner array cannot do that, so the corner CONTENTS are rewritten
    # instead: F as below, and EID/FT/EHA rotated to match. The vertex sets per face are the mirror
    # image of upstream's (it leaves {b,c,d} in fa, this leaves it in fb), which is the same flip.
    def flipEdge(mut self, e: Int) -> Float64:
        var h = self.EHA[e]
        var t = self.FT[h]
        if h // 3 == t // 3:
            return nan64()
        var n1 = self.nextHe(h)
        var p1 = self.prevHe(h)
        var n2 = self.nextHe(t)
        var p2 = self.prevHe(t)
        var a = self.F[h]
        var b = self.F[t]
        var u = self.F[n1]
        var c = self.F[p1]
        var d = self.F[p2]
        if a == c or a == d or b == c or b == d or c == d:
            return nan64()
        # simple_idt.cpp: `if (iHeA0.orientation() == iHeB0.orientation()) std::swap(iHeB1, iHeB2)`.
        # Both faces then traverse `e` from the same vertex `a`, so the other end of the edge is `u`
        # in both, and each face's cycle runs the opposite way round to the oriented case. The
        # tufted cover of the robust Laplacian is full of such edges, so the two cases differ in the
        # layout lengths, in which vertex each new face keeps, and in the edge ids the corners carry.
        var same_orient = a == b

        # simple_idt.cpp: flippedEdgeLen, FlipType::Euclidean
        var l01 = self.EL[self.EID[n1]]
        var l12 = self.EL[self.EID[p1]]
        var l23 = self.EL[self.EID[n2]]
        var l30 = self.EL[self.EID[p2]]
        var l02 = self.EL[self.EID[h]]
        if l02 <= 0.0:
            return nan64()
        if same_orient:
            var t23 = l23
            l23 = l30
            l30 = t23
        var p3 = Vector2(0.0, 0.0)
        var p0 = Vector2(l30, 0.0)
        var p2v = layout_triangle_vertex(p3, p0, l02, l23)
        var p1v = layout_triangle_vertex(p2v, p0, l01, l12)
        var new_len = (p1v - p3).norm()
        if isnan(new_len) or isinf(new_len) or new_len <= 0.0:
            return nan64()
        # The new diagonal is c-d, and each face keeps the endpoint of the old edge that it was
        # pointing at: face A keeps a, face B keeps b when the edge is oriented and u when it is not.
        self.F[h] = d
        self.F[n1] = c
        self.F[p1] = a
        self.F[t] = c
        self.F[n2] = d
        self.F[p2] = b if not same_orient else u

        # Corner `h` keeps the edge id, which now labels the new diagonal, exactly as upstream's
        # untouched heEdgeArr[ha1] does. Each of the other four edges is carried by exactly one
        # corner inside the flipped pair both before and after, and by one corner outside it, so the
        # pair to re-link is (old carrier, new carrier) and the twin is read before it is written.
        var from_n1 = p2
        var from_p1 = n1
        var from_n2 = p1
        var from_p2 = n2
        if same_orient:
            from_n2 = n2
            from_p2 = p1
        var twin_n1 = self.FT[n1]
        var twin_p1 = self.FT[p1]
        var twin_n2 = self.FT[n2]
        var twin_p2 = self.FT[p2]
        var e_n1 = self.EID[n1]
        var e_p1 = self.EID[p1]
        var e_n2 = self.EID[n2]
        var e_p2 = self.EID[p2]
        self.EID[from_n1] = e_n1
        self.EID[from_p1] = e_p1
        self.EID[from_n2] = e_n2
        self.EID[from_p2] = e_p2
        self.FT[from_n1] = twin_n1
        self.FT[twin_n1] = from_n1
        self.FT[from_p1] = twin_p1
        self.FT[twin_p1] = from_p1
        self.FT[from_n2] = twin_n2
        self.FT[twin_n2] = from_n2
        self.FT[from_p2] = twin_p2
        self.FT[twin_p2] = from_p2
        if self.EHA[e_n1] == n1:
            self.EHA[e_n1] = from_n1
        if self.EHA[e_p1] == p1:
            self.EHA[e_p1] = from_p1
        if self.EHA[e_n2] == n2:
            self.EHA[e_n2] = from_n2
        if self.EHA[e_p2] == p2:
            self.EHA[e_p2] = from_p2
        return new_len


# Every corner must name a vertex that exists. Upstream leaves this to the caller's index check, which
# uses np.amin and so only catches negative indices; an out-of-range positive index would walk off the
# vertex array, so the kernels check it themselves and report it.
def faces_in_range(faces: Pointer[Int, AnyOrigin[mut=True]], n_v: Int, n_f: Int) -> Bool:
    var i = 0
    while i < 3 * n_f:
        var v = faces.unsafe_load(i)
        if v < 0 or v >= n_v:
            return False
        i += 1
    return True


# SurfaceMesh's constructor. Halfedges are paired on the sorted endpoint pair; a corner whose pair
# occurs once is a boundary halfedge and twins with itself.
def build_connectivity(faces: Pointer[Int, AnyOrigin[mut=True]], n_v: Int, n_f: Int) -> SurfaceMesh:
    var nnz = 3 * n_f
    var f = List[Int](length=nnz, fill=0)
    var i = 0
    while i < n_f:
        f[3 * i] = faces[3 * i]
        f[3 * i + 1] = faces[3 * i + 1]
        f[3 * i + 2] = faces[3 * i + 2]
        i += 1

    var srt = PairSorter(nnz)
    var h = 0
    while h < nnz:
        var a = f[h]
        var b = f[h - h % 3 + (h % 3 + 1) % 3]
        if a > b:
            srt.keys[h] = b * n_v + a
        else:
            srt.keys[h] = a * n_v + b
        srt.vals[h] = h
        h += 1
    srt.sort()

    var ft = List[Int](length=nnz, fill=0)
    var eid = List[Int](length=nnz, fill=0)
    var n_e = 0
    var n_b = 0
    var lo = 0
    while lo < nnz:
        var hi = lo
        while hi < nnz and srt.keys[hi] == srt.keys[lo]:
            hi += 1
        var count = hi - lo
        var j = 0
        while j + 1 < count:
            eid[srt.vals[lo + j]] = n_e
            eid[srt.vals[lo + j + 1]] = n_e
            ft[srt.vals[lo + j]] = srt.vals[lo + j + 1]
            ft[srt.vals[lo + j + 1]] = srt.vals[lo + j]
            n_e += 1
            j += 2
        if j < count:
            eid[srt.vals[lo + j]] = n_e
            ft[srt.vals[lo + j]] = srt.vals[lo + j]
            n_e += 1
            n_b += 1
        lo = hi

    var el = List[Float64](length=n_e, fill=0.0)
    var eha = List[Int](length=n_e, fill=0)
    var seen = List[Bool](length=n_e, fill=False)
    var e = 0
    while e < n_e:
        seen[e] = False
        e += 1
    h = 0
    while h < nnz:
        e = eid[h]
        if not seen[e]:
            seen[e] = True
            eha[e] = h
        h += 1

    var mesh = SurfaceMesh(n_v, n_f, n_e, n_b)
    mesh.F = f^
    mesh.FT = ft^
    mesh.EID = eid^
    mesh.EL = el^
    mesh.EHA = eha^
    return mesh^


# Intrinsic edge lengths from the embedded positions
def fill_edge_lengths(var mesh: SurfaceMesh, verts: Pointer[Float64, AnyOrigin[mut=True]]) -> SurfaceMesh:
    var h = 0
    while h < 3 * mesh.nF:
        var e = mesh.EID[h]
        var p = mesh.F[h]
        var q = mesh.F[mesh.nextHe(h)]
        var dx = verts[3 * p] - verts[3 * q]
        var dy = verts[3 * p + 1] - verts[3 * q + 1]
        var dz = verts[3 * p + 2] - verts[3 * q + 2]
        mesh.EL[e] = sqrt(dx * dx + dy * dy + dz * dz)
        h += 1
    return mesh^


def build_surface_mesh(
    verts: Pointer[Float64, AnyOrigin[mut=True]],
    faces: Pointer[Int, AnyOrigin[mut=True]],
    n_v: Int,
    n_f: Int,
) -> SurfaceMesh:
    var mesh = build_connectivity(faces, n_v, n_f)
    return fill_edge_lengths(mesh^, verts)
