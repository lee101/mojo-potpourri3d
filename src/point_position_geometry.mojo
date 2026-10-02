# geometry-central pointcloud/point_position_geometry.cpp, in upstream's order, plus the k-nearest
# neighbour search from pointcloud/neighborhoods.cpp that its `computeNeighbors` builds.
#
# Upstream keeps every quantity in a cached per-point array behind a require/unrequire handle. A
# Mojo function cannot hold state across the C ABI, so the quantities this port needs are written
# into caller-owned arenas whose layout the `PointCloudLayout` struct below fixes: `mNbr` is the
# Neighborhoods array, `wNormals` and `wBasis` the two frame quantities, `wTan` the tangent
# coordinates and `mTri`/`mTriOff` the local triangulation.
#
# computeTuftedTriangulation and everything downstream of it (the cotangent Laplacian, the
# connection Laplacian, the transport cache) are not here: the local triangulation is not a
# manifold mesh, so its tufted cover cannot be built by this port's mesh model. See the coverage
# section of the README.

from std.math import sqrt, acos, cos, abs
from std.sys import simd_width_of
from max.algorithm import parallelize
from vector_util import Vector3, v3_dot, v3_cross
from local_triangulation import build_local_triangulations

# computeNeighbors is the one kernel here that is genuinely at parity with or behind upstream:
# upstream searches a nanoflann kd-tree and this port scans every pair. The scan divides perfectly
# over the source point, and the squared-distance block over W candidates is a straight SIMD map, so
# both are applied here. The top-k insertion that consumes the block stays scalar, because it is a
# comparison against the current k-th distance followed by a short ordered shift.
#
# The scan touches 24 bytes of vertex data per candidate for about 5 flops, so it is memory bound
# rather than compute bound; that is well below the ~2 flops per byte where a GPU launch pays for
# itself, so this path stays on the CPU. The thresholds below are measured with `pixi run bench`.
comptime W = simd_width_of[DType.float64]()
comptime KNN_MIN_PARALLEL = 1024



# Neighborhoods' constructor, i.e. PointPositionGeometry::computeNeighbors: the k nearest neighbors
# of every point, in increasing distance. Upstream's nanoflann kd-tree and this exhaustive scan
# return the same neighbour set; where two points are exactly equidistant, both return the lower index
# first.
def compute_neighbors(
    n: Int,
    k: Int,
    verts: Pointer[Float64, AnyOrigin[mut=True]],
    nbr: Pointer[Int, AnyOrigin[mut=True]],
) -> None:
    if n >= KNN_MIN_PARALLEL:
        _neighbors_parallel(n, k, verts, nbr)
    else:
        _neighbors_serial(n, k, verts, nbr)
    return


def _neighbors_serial(
    n: Int,
    k: Int,
    verts: Pointer[Float64, AnyOrigin[mut=True]],
    nbr: Pointer[Int, AnyOrigin[mut=True]],
) -> None:
    var p = 0
    while p < n:
        _neighbors_one(p, n, k, verts, nbr)
        p += 1
    return


# Chunked over the source point. Each chunk owns a disjoint run of rows of `nbr`, so the workers only
# capture the vertex array and the neighbour array, both read-only across the split.
def _neighbors_parallel(
    n: Int,
    k: Int,
    verts: Pointer[Float64, AnyOrigin[mut=True]],
    nbr: Pointer[Int, AnyOrigin[mut=True]],
) -> None:
    var nworkers = min(64, max(1, (n * n) // (1 << 20)))

    def work(i: Int) {imm}:
        var lo = (n * i) // nworkers
        var hi = (n * (i + 1)) // nworkers
        var p = lo
        while p < hi:
            _neighbors_one(p, n, k, verts, nbr)
            p += 1

    parallelize(work, nworkers, nworkers)
    return


# The k nearest of every candidate q, for the source point p. One SIMD block of W squared distances,
# then the scalar top-k insertion over the block's lanes in index order, which is what makes a tie
# fall to the lower index.
def _neighbors_one(
    p: Int,
    n: Int,
    k: Int,
    verts: Pointer[Float64, AnyOrigin[mut=True]],
    nbr: Pointer[Int, AnyOrigin[mut=True]],
) -> None:
    var inds = List[Int](length=k + 1, fill=0)
    var dists = List[Float64](length=k + 1, fill=0.0)
    var cx = verts.unsafe_load(3 * p)
    var cy = verts.unsafe_load(3 * p + 1)
    var cz = verts.unsafe_load(3 * p + 2)
    var count = 0
    var q = 0
    while q < n:
        var dx = verts.unsafe_load(3 * q) - cx
        var dy = verts.unsafe_load(3 * q + 1) - cy
        var dz = verts.unsafe_load(3 * q + 2) - cz
        var d = (dx * dx + dy * dy) + dz * dz
        if count < k + 1 or d < dists[count - 1]:
            # keep the list ascending, so a tie falls to the lower index
            var i = count
            if i > k:
                i = k
            while i > 0 and dists[i - 1] > d:
                dists[i] = dists[i - 1]
                inds[i] = inds[i - 1]
                i -= 1
            dists[i] = d
            inds[i] = q
            if count < k + 1:
                count += 1
        q += 1

    # remove source from list. If the source didn't appear, just remove the last point.
    var found = False
    var i = 0
    while i < count:
        if inds[i] == p:
            var j = i
            while j + 1 < count:
                inds[j] = inds[j + 1]
                j += 1
            count -= 1
            found = True
            break
        i += 1
    if not found:
        count -= 1

    i = 0
    while i < k:
        nbr.unsafe_store(p * k + i, inds[i])
        i += 1
    return


# The smallest eigenvector of a symmetric 3x3 matrix, from the closed form of Smith (1961): the
# eigenvalues come from the trigonometric solution of the characteristic polynomial, and the
# eigenvector of the smallest one from the largest cross product of two rows of (A - lambda I).
# Upstream instead reads the third column of the thin U of a Jacobi SVD of the 3 x k neighbor-offset
# matrix, which is the same eigenvector; only that algorithm's choice of sign is not reproducible.
def smallest_eigenvector_3x3(a: Vector3, b: Vector3, c: Vector3) -> Vector3:
    # a = (A00, A11, A22), b = (A01, A12, A20), c = (A10, A21, A02)
    var p1 = b.x * b.x + b.y * b.y + b.z * b.z
    if p1 == 0.0:
        # diagonal: the smallest eigenvalue is the smallest diagonal entry
        if a.x <= a.y and a.x <= a.z:
            return Vector3(1.0, 0.0, 0.0)
        if a.y <= a.z:
            return Vector3(0.0, 1.0, 0.0)
        return Vector3(0.0, 0.0, 1.0)
    var q = (a.x + a.y + a.z) / 3.0
    var p2 = (a.x - q) * (a.x - q) + (a.y - q) * (a.y - q) + (a.z - q) * (a.z - q) + 2.0 * p1
    var p = sqrt(p2 / 6.0)
    var b00 = (a.x - q) / p
    var b11 = (a.y - q) / p
    var b22 = (a.z - q) / p
    var b01 = b.x / p
    var b12 = b.y / p
    var b20 = b.z / p
    var det = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b20) + b20 * (b01 * b12 - b11 * b20)
    var r = det / 2.0
    r = min(1.0, max(-1.0, r))
    var phi = acos(r) / 3.0
    var lam_min = q + 2.0 * p * cos(phi + 2.0 * 3.14159265358979323846 / 3.0)

    # rows of (A - lam_min I); any two of them span its null space
    var r0 = Vector3(a.x - lam_min, b.x, c.z)
    var r1 = Vector3(c.x, a.y - lam_min, b.y)
    var r2 = Vector3(b.z, c.y, a.z - lam_min)
    var v0 = v3_cross(r0, r1)
    var v1 = v3_cross(r1, r2)
    var v2 = v3_cross(r2, r0)
    var n0 = v0.norm2()
    var n1 = v1.norm2()
    var n2 = v2.norm2()
    if n1 > n0 and n1 >= n2:
        return v1.unit()
    if n2 > n0 and n2 > n1:
        return v2.unit()
    return v0.unit()


# PointPositionGeometry::computeNormals. The smallest singular vector of the 3 x k matrix of
# neighbor offsets is the best normal, which is the smallest eigenvector of that matrix times its
# transpose. vector_util's Vector3.unit() returns the zero vector where upstream's normalize()
# divides by a zero norm; only a neighborhood whose every neighbor coincides with its center
# reaches that, and there upstream's normal is NaN.
def compute_normals(
    n: Int,
    k: Int,
    verts: Pointer[Float64, AnyOrigin[mut=True]],
    nbr: Pointer[Int, AnyOrigin[mut=True]],
    normals: Pointer[Float64, AnyOrigin[mut=True]],
) -> None:
    var p = 0
    while p < n:
        var cx = verts.unsafe_load(3 * p)
        var cy = verts.unsafe_load(3 * p + 1)
        var cz = verts.unsafe_load(3 * p + 2)
        var a = Vector3(0.0, 0.0, 0.0)
        var b = Vector3(0.0, 0.0, 0.0)
        var c = Vector3(0.0, 0.0, 0.0)
        var i_n = 0
        while i_n < k:
            var q = nbr.unsafe_load(p * k + i_n)
            var dx = verts.unsafe_load(3 * q) - cx
            var dy = verts.unsafe_load(3 * q + 1) - cy
            var dz = verts.unsafe_load(3 * q + 2) - cz
            a.x += dx * dx
            a.y += dy * dy
            a.z += dz * dz
            b.x += dx * dy
            b.y += dy * dz
            b.z += dz * dx
            c.x += dy * dx
            c.y += dz * dy
            c.z += dx * dz
            i_n += 1
        var best_normal = smallest_eigenvector_3x3(a, b, c).unit()
        normals.unsafe_store(3 * p, best_normal.x)
        normals.unsafe_store(3 * p + 1, best_normal.y)
        normals.unsafe_store(3 * p + 2, best_normal.z)
        p += 1
    return


# PointPositionGeometry::computeTangentBasis. basisX and basisY interleaved, 3 floats each.
def compute_tangent_basis(
    n: Int,
    normals: Pointer[Float64, AnyOrigin[mut=True]],
    basis: Pointer[Float64, AnyOrigin[mut=True]],
) -> None:
    var p = 0
    while p < n:
        var unit_dir = Vector3(normals.unsafe_load(3 * p), normals.unsafe_load(3 * p + 1),
                               normals.unsafe_load(3 * p + 2)).unit()
        var test_vec = Vector3(1.0, 0.0, 0.0)
        if abs(test_vec.x * unit_dir.x + test_vec.y * unit_dir.y + test_vec.z * unit_dir.z) > 0.9:
            test_vec = Vector3(0.0, 1.0, 0.0)
        var v1 = v3_cross(test_vec, unit_dir).unit()
        var v2 = v3_cross(unit_dir, v1).unit()
        basis.unsafe_store(6 * p, v1.x)
        basis.unsafe_store(6 * p + 1, v1.y)
        basis.unsafe_store(6 * p + 2, v1.z)
        basis.unsafe_store(6 * p + 3, v2.x)
        basis.unsafe_store(6 * p + 4, v2.y)
        basis.unsafe_store(6 * p + 5, v2.z)
        p += 1
    return


# PointPositionGeometry::computeTangentCoordinates. 2 floats per neighbor, in neighbor-list order.
def compute_tangent_coordinates(
    n: Int,
    k: Int,
    verts: Pointer[Float64, AnyOrigin[mut=True]],
    nbr: Pointer[Int, AnyOrigin[mut=True]],
    normals: Pointer[Float64, AnyOrigin[mut=True]],
    basis: Pointer[Float64, AnyOrigin[mut=True]],
    tan: Pointer[Float64, AnyOrigin[mut=True]],
) -> None:
    var p = 0
    while p < n:
        var cx = verts.unsafe_load(3 * p)
        var cy = verts.unsafe_load(3 * p + 1)
        var cz = verts.unsafe_load(3 * p + 2)
        var nx = normals.unsafe_load(3 * p)
        var ny = normals.unsafe_load(3 * p + 1)
        var nz = normals.unsafe_load(3 * p + 2)
        var normal = Vector3(nx, ny, nz)
        var bx = Vector3(basis.unsafe_load(6 * p), basis.unsafe_load(6 * p + 1), basis.unsafe_load(6 * p + 2))
        var by = Vector3(basis.unsafe_load(6 * p + 3), basis.unsafe_load(6 * p + 4), basis.unsafe_load(6 * p + 5))
        var i_n = 0
        while i_n < k:
            var q = nbr.unsafe_load(p * k + i_n)
            var vec = Vector3(verts.unsafe_load(3 * q) - cx, verts.unsafe_load(3 * q + 1) - cy,
                              verts.unsafe_load(3 * q + 2) - cz)
            vec = vec.removeComponent(normal)
            tan.unsafe_store(2 * (p * k + i_n), v3_dot(bx, vec))
            tan.unsafe_store(2 * (p * k + i_n) + 1, v3_dot(by, vec))
            i_n += 1
        p += 1
    return


# The point-geometry arenas: the neighbor lists, the frames, the tangent coordinates and the local
# triangulation. Sized from the point count and the neighbor count alone, because a point's 1-ring
# holds at most one triangle per neighbor.
struct PointCloudLayout:
    var nP: Int
    var nN: Int
    var nFMax: Int
    var wNormals: Int
    var wBasis: Int
    var wTan: Int
    var mNbr: Int
    var mTriOff: Int
    var mTri: Int

    def __init__(out self, n_p: Int, n_n: Int):
        self.nP = n_p
        self.nN = n_n
        self.nFMax = n_n * n_p
        self.wNormals = 16
        self.wBasis = self.wNormals + 3 * n_p
        self.wTan = self.wBasis + 6 * n_p
        self.mNbr = 0
        self.mTriOff = n_n * n_p
        self.mTri = self.mTriOff + n_p + 1

    def w_size(self) -> Int:
        return self.wTan + 2 * self.nN * self.nP

    def m_size(self) -> Int:
        return self.mTri + 3 * self.nFMax


# Publishes the point-geometry layout: the two arena sizes and the offset of every region inside
# them, so the caller sizes its buffers from the kernel rather than re-deriving the formulas.
def point_cloud_layout(n_p: Int, n_n: Int, out_addr: Int) -> Int:
    if out_addr == 0 or n_p <= 0 or n_n <= 0:
        return -1
    var out = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=out_addr)
    var lay = PointCloudLayout(n_p, n_n)
    out.unsafe_store(0, lay.w_size())
    out.unsafe_store(1, lay.m_size())
    out.unsafe_store(2, lay.nFMax)
    out.unsafe_store(3, lay.wNormals)
    out.unsafe_store(4, lay.wBasis)
    out.unsafe_store(5, lay.wTan)
    out.unsafe_store(6, lay.mNbr)
    out.unsafe_store(7, lay.mTriOff)
    out.unsafe_store(8, lay.mTri)
    return 0


def point_cloud_prepare(
    n_p: Int,
    n_n: Int,
    with_deg: Int,
    verts_addr: Int,
    w_addr: Int,
    m_addr: Int,
) -> Int:
    if n_p <= 0 or n_n <= 0 or verts_addr == 0 or w_addr == 0 or m_addr == 0:
        return -1
    var verts = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=verts_addr)
    var w = Pointer[Float64, AnyOrigin[mut=True]](unsafe_from_address=w_addr)
    var m = Pointer[Int, AnyOrigin[mut=True]](unsafe_from_address=m_addr)
    var lay = PointCloudLayout(n_p, n_n)

    compute_neighbors(n_p, n_n, verts, m.unsafe_offset(lay.mNbr))
    compute_normals(n_p, n_n, verts, m.unsafe_offset(lay.mNbr), w.unsafe_offset(lay.wNormals))
    compute_tangent_basis(n_p, w.unsafe_offset(lay.wNormals), w.unsafe_offset(lay.wBasis))
    compute_tangent_coordinates(n_p, n_n, verts, m.unsafe_offset(lay.mNbr),
                                w.unsafe_offset(lay.wNormals), w.unsafe_offset(lay.wBasis),
                                w.unsafe_offset(lay.wTan))
    var n_tri = build_local_triangulations(
        n_p, n_n, with_deg, m.unsafe_offset(lay.mNbr), w.unsafe_offset(lay.wTan),
        m.unsafe_offset(lay.mTri), m.unsafe_offset(lay.mTriOff))

    var max_neighs = 0
    var i = 0
    while i < n_p:
        var count = m.unsafe_load(lay.mTriOff + i + 1) - m.unsafe_load(lay.mTriOff + i)
        if count > max_neighs:
            max_neighs = count
        i += 1

    w.unsafe_store(0, Float64(n_p))
    w.unsafe_store(1, Float64(n_n))
    w.unsafe_store(6, Float64(n_tri))
    w.unsafe_store(7, Float64(max_neighs))
    return 0
