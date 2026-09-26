"""The point-cloud local triangulation, in the order of
`geometry-central/src/pointcloud/local_triangulation.cpp`, together with the
three quantities it consumes from `point_position_geometry.cpp`:
`computeNeighbors`, `computeNormals`, `computeTangentCoordinates`.

For each point this builds the local Delaunay triangulation of its
neighbourhood projected onto its tangent plane: angularly sort the neighbours
around the centre, then drop the ones whose two adjacent triangles are not
Delaunay, until nothing changes. The survivors are emitted as
`{p, neighbour i, neighbour j}` in angular order.

`NearestNeighborFinder` upstream is a KD-tree; here it is a partial selection
scan, which gives the same neighbour list (ties broken by index) at O(n^2 k)
rather than O(n log n). The normal is the smallest singular vector of the
neighbourhood, taken as the smallest eigenvector of the 3x3 Gram matrix by
cyclic Jacobi rotations rather than by a Jacobi SVD.
"""

from std.math import abs as fabs
from std.math import sqrt, atan2, cos, sin, isfinite

from mojopp3d.vector import Vec2, Vec3

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

comptime PI = 3.141592653589793

# geometry-central's `INVALID_IND`
comptime INVALID_IND = 9223372036854775807

# "An innocent numerical parameter used for the degeneracy heuristic",
# in units of relative length
comptime DEGENERATE_THRESH = 1e-7


struct Mat4:
    var a: SIMD[DType.float64, 16]

    def __init__(
        out self,
        a00: Float64, a01: Float64, a02: Float64, a03: Float64,
        a10: Float64, a11: Float64, a12: Float64, a13: Float64,
        a20: Float64, a21: Float64, a22: Float64, a23: Float64,
        a30: Float64, a31: Float64, a32: Float64, a33: Float64,
    ):
        self.a = SIMD[DType.float64, 16](
            a00, a01, a02, a03, a10, a11, a12, a13,
            a20, a21, a22, a23, a30, a31, a32, a33,
        )

    def at(self, r: Int, c: Int) -> Float64:
        return self.a[4 * r + c]


# inCircleTest: `A.determinant() > 0` on the 4x4 whose rows are
# (x, y, norm2, 1). Eigen computes a fixed-size 4x4 determinant as a six-term
# cofactor expansion; the order of the terms is what decides the sign of a
# determinant that is mathematically zero, so it is reproduced term by term.
def in_circle_test(p_a: Vec2, p_b: Vec2, p_c: Vec2, p_test: Vec2) -> Bool:
    var m = Mat4(
        p_a.x, p_a.y, p_a.x * p_a.x + p_a.y * p_a.y, 1.0,
        p_b.x, p_b.y, p_b.x * p_b.x + p_b.y * p_b.y, 1.0,
        p_c.x, p_c.y, p_c.x * p_c.x + p_c.y * p_c.y, 1.0,
        p_test.x, p_test.y, p_test.x * p_test.x + p_test.y * p_test.y, 1.0,
    )
    var d = _det4_term(m, 0, 1, 2, 3) - _det4_term(m, 0, 2, 1, 3)
    d += _det4_term(m, 0, 3, 1, 2) + _det4_term(m, 1, 2, 0, 3)
    d -= _det4_term(m, 1, 3, 0, 2)
    d += _det4_term(m, 2, 3, 0, 1)
    return d > 0.0


# bruteforce_det4_helper
def _det4_term(m: Mat4, j: Int, k: Int, mm: Int, n: Int) -> Float64:
    return (m.at(j, 0) * m.at(k, 1) - m.at(k, 0) * m.at(j, 1)) * (
        m.at(mm, 2) * m.at(n, 3) - m.at(n, 2) * m.at(mm, 3)
    )


# Neighborhoods: the k nearest other points to each point.
def compute_neighbors(
    points: Ptr, n: Int, k: Int, neighbors: IPtr, keys: Ptr, vals: IPtr
):
    for i in range(n):
        var x = points.unsafe_load(3 * i)
        var y = points.unsafe_load(3 * i + 1)
        var z = points.unsafe_load(3 * i + 2)
        var count = 0
        for j in range(n):
            if j == i:
                continue
            var dx = points.unsafe_load(3 * j) - x
            var dy = points.unsafe_load(3 * j + 1) - y
            var dz = points.unsafe_load(3 * j + 2) - z
            var d2 = dx * dx + dy * dy + dz * dz
            if count == k and d2 >= keys.unsafe_load(k - 1):
                continue
            # insert into the sorted prefix, shifting the rest down
            var pos = count
            if pos > k - 1:
                pos = k - 1
            while pos > 0:
                var pk = keys.unsafe_load(pos - 1)
                if pk < d2 or (pk == d2 and vals.unsafe_load(pos - 1) < Int64(j)):
                    break
                keys.unsafe_store(pos, pk)
                vals.unsafe_store(pos, vals.unsafe_load(pos - 1))
                pos -= 1
            keys.unsafe_store(pos, d2)
            vals.unsafe_store(pos, Int64(j))
            if count < k:
                count += 1
        for a in range(count):
            neighbors.unsafe_store(Int64(k * i + a), vals.unsafe_load(a))
        for a in range(count, k):
            neighbors.unsafe_store(Int64(k * i + a), vals.unsafe_load(0))


# computeNormals: the smallest singular vector of the 3 x k neighbour matrix is
# the eigenvector of its 3 x 3 Gram matrix for the smallest eigenvalue.
def compute_normals(
    points: Ptr, neighbors: IPtr, n: Int, k: Int, normals: Ptr, a: Ptr, v: Ptr
):
    for p in range(n):
        var a00 = 0.0
        var a01 = 0.0
        var a02 = 0.0
        var a11 = 0.0
        var a12 = 0.0
        var a22 = 0.0
        var cx = points.unsafe_load(3 * p)
        var cy = points.unsafe_load(3 * p + 1)
        var cz = points.unsafe_load(3 * p + 2)
        for j in range(k):
            var nb = Int(neighbors.unsafe_load(Int64(k * p + j)))
            var vx = points.unsafe_load(3 * nb) - cx
            var vy = points.unsafe_load(3 * nb + 1) - cy
            var vz = points.unsafe_load(3 * nb + 2) - cz
            a00 += vx * vx
            a01 += vx * vy
            a02 += vx * vz
            a11 += vy * vy
            a12 += vy * vz
            a22 += vz * vz

        a.unsafe_store(0, a00)
        a.unsafe_store(1, a01)
        a.unsafe_store(2, a02)
        a.unsafe_store(3, a01)
        a.unsafe_store(4, a11)
        a.unsafe_store(5, a12)
        a.unsafe_store(6, a02)
        a.unsafe_store(7, a12)
        a.unsafe_store(8, a22)
        for r in range(9):
            v.unsafe_store(r, 0.0)
        v.unsafe_store(0, 1.0)
        v.unsafe_store(4, 1.0)
        v.unsafe_store(8, 1.0)

        for _sweep in range(30):
            var off = a.unsafe_load(1) * a.unsafe_load(1)
            off += a.unsafe_load(2) * a.unsafe_load(2)
            off += a.unsafe_load(5) * a.unsafe_load(5)
            if off <= 1e-300:
                break
            for rot in range(3):
                var pi = rot
                var qi = rot + 1
                if rot == 2:
                    pi = 0
                    qi = 2
                var apq = a.unsafe_load(3 * pi + qi)
                if fabs(apq) <= 1e-300:
                    continue
                var app = a.unsafe_load(3 * pi + pi)
                var aqq = a.unsafe_load(3 * qi + qi)
                var theta = (aqq - app) / (2.0 * apq)
                var t = 1.0 / (fabs(theta) + sqrt(theta * theta + 1.0))
                if theta < 0.0:
                    t = -t
                var c = 1.0 / sqrt(t * t + 1.0)
                var s = t * c
                for r in range(3):
                    var arp = a.unsafe_load(3 * r + pi)
                    var arq = a.unsafe_load(3 * r + qi)
                    a.unsafe_store(3 * r + pi, c * arp - s * arq)
                    a.unsafe_store(3 * r + qi, s * arp + c * arq)
                for r in range(3):
                    var apr = a.unsafe_load(3 * pi + r)
                    var aqr = a.unsafe_load(3 * qi + r)
                    a.unsafe_store(3 * pi + r, c * apr - s * aqr)
                    a.unsafe_store(3 * qi + r, s * apr + c * aqr)
                for r in range(3):
                    var vrp = v.unsafe_load(3 * r + pi)
                    var vrq = v.unsafe_load(3 * r + qi)
                    v.unsafe_store(3 * r + pi, c * vrp - s * vrq)
                    v.unsafe_store(3 * r + qi, s * vrp + c * vrq)

        # the smallest diagonal is the smallest eigenvalue, and its eigenvector
        # is the matching column of the accumulated rotations
        var smallest = a.unsafe_load(0)
        var col = 0
        if a.unsafe_load(4) < smallest:
            smallest = a.unsafe_load(4)
            col = 1
        if a.unsafe_load(8) < smallest:
            col = 2
        var nx = v.unsafe_load(col)
        var ny = v.unsafe_load(3 + col)
        var nz = v.unsafe_load(6 + col)
        var length = sqrt(nx * nx + ny * ny + nz * nz)
        if length > 0.0:
            nx = nx / length
            ny = ny / length
            nz = nz / length
        normals.unsafe_store(3 * p, nx)
        normals.unsafe_store(3 * p + 1, ny)
        normals.unsafe_store(3 * p + 2, nz)


# computeTangentBasis: `Vector3::buildTangentBasis`, the orthonormal pair
# perpendicular to the normal. Then the neighbourhood projected onto it.
def compute_tangent_coordinates(
    points: Ptr,
    normals: Ptr,
    neighbors: IPtr,
    n: Int,
    k: Int,
    coords: Ptr,
):
    for p in range(n):
        var normal = Vec3(
            normals.unsafe_load(3 * p),
            normals.unsafe_load(3 * p + 1),
            normals.unsafe_load(3 * p + 2),
        )
        var basis_x = normal.build_tangent_basis_x()
        var ax = basis_x.x
        var ay = basis_x.y
        var az = basis_x.z
        var basis_y = normal.normalize().cross(basis_x).normalize()
        var bx = basis_y.x
        var by = basis_y.y
        var bz = basis_y.z
        var nx = normal.x
        var ny = normal.y
        var nz = normal.z
        var cx = points.unsafe_load(3 * p)
        var cy = points.unsafe_load(3 * p + 1)
        var cz = points.unsafe_load(3 * p + 2)
        for j in range(k):
            var nb = Int(neighbors.unsafe_load(Int64(k * p + j)))
            var vx = points.unsafe_load(3 * nb) - cx
            var vy = points.unsafe_load(3 * nb + 1) - cy
            var vz = points.unsafe_load(3 * nb + 2) - cz
            # removeComponent(normal)
            var d = vx * nx + vy * ny + vz * nz
            vx = vx - d * nx
            vy = vy - d * ny
            vz = vz - d * nz
            coords.unsafe_store(2 * (k * p + j), vx * ax + vy * ay + vz * az)
            coords.unsafe_store(2 * (k * p + j) + 1, vx * bx + vy * by + vz * bz)


# buildLocalTriangulations. `tri` receives the flat triangle list, `offsets`
# the per-point start of each point's run; scratch is 2*k doubles for the
# perturbed points, k for the angles, k for the sorted indices.
def build_local_triangulations(
    coords: Ptr,
    neighbors: IPtr,
    n: Int,
    k: Int,
    with_degeneracy_heuristic: Int,
    tri: IPtr,
    offsets: IPtr,
    pts: Ptr,
    angles: Ptr,
    sort_inds: IPtr,
) -> Int:
    var total = 0
    for p in range(n):
        offsets.unsafe_store(Int64(p), Int64(total))
        var n_neigh = k

        # Compute a lengthscale for the neighborhood as the radius of the most
        # distant point
        var len_scale2 = 0.0
        for i in range(n_neigh):
            var qx = coords.unsafe_load(2 * (k * p + i))
            var qy = coords.unsafe_load(2 * (k * p + i) + 1)
            len_scale2 = max(len_scale2, qx * qx + qy * qy)
        var len_scale = sqrt(len_scale2)

        # Something is hopelessly degenerate, don't even bother trying
        if not isfinite(len_scale) or len_scale <= 0.0:
            continue

        for i in range(n_neigh):
            pts.unsafe_store(2 * i, coords.unsafe_load(2 * (k * p + i)))
            pts.unsafe_store(2 * i + 1, coords.unsafe_load(2 * (k * p + i) + 1))

        if with_degeneracy_heuristic != 0:
            # Perturb points which are extremely close to the source
            for i in range(n_neigh):
                var qx = pts.unsafe_load(2 * i)
                var qy = pts.unsafe_load(2 * i + 1)
                var dist = sqrt(qx * qx + qy * qy)
                if dist < len_scale * DEGENERATE_THRESH:
                    var dx: Float64
                    var dy: Float64
                    if dist > 0.0:
                        dx = qx / dist
                        dy = qy / dist
                    else:
                        # even the direction is degenerate: pick one by index
                        var theta_dir = 2.0 * PI * Float64(i) / Float64(n_neigh)
                        dx = cos(theta_dir)
                        dy = sin(theta_dir)
                    # Including the index avoids creating many co-circular
                    # points; no need to stress the Delaunay triangulation
                    # unnecessarily.
                    var length = (
                        1.0 + Float64(i) / Float64(n_neigh)
                    ) * len_scale * DEGENERATE_THRESH * 10.0
                    pts.unsafe_store(2 * i, length * dx)
                    pts.unsafe_store(2 * i + 1, length * dy)

        # Angularly sort the points CCW
        for i in range(n_neigh):
            var qx = pts.unsafe_load(2 * i)
            var qy = pts.unsafe_load(2 * i + 1)
            var angle = atan2(qy, qx)
            if not isfinite(angle):
                # sentinel value for below
                angle = -777.0
            angles.unsafe_store(i, angle)
            sort_inds.unsafe_store(Int64(i), Int64(i))
        for a in range(1, n_neigh):
            var av = angles.unsafe_load(a)
            var ai = Int(sort_inds.unsafe_load(Int64(a)))
            var b = a - 1
            while b >= 0 and angles.unsafe_load(b) > av:
                angles.unsafe_store(b + 1, angles.unsafe_load(b))
                sort_inds.unsafe_store(Int64(b + 1), sort_inds.unsafe_load(Int64(b)))
                b -= 1
            angles.unsafe_store(b + 1, av)
            sort_inds.unsafe_store(Int64(b + 1), Int64(ai))

        # Immediately skip any invalid indices in the search below, by detecting
        # sentinels from above
        for i in range(n_neigh):
            if angles.unsafe_load(Int(sort_inds.unsafe_load(Int64(i)))) == -777.0:
                sort_inds.unsafe_store(Int64(i), Int64(INVALID_IND))

        # == Find the local Delaunay triangulation. The output is a subset of
        # the sorted list, which corresponds to the 1-ring of the centre in the
        # local Delaunay triangulation: repeatedly remove a point if the
        # diamond it forms with its two angular neighbours is not Delaunay.
        var N = n_neigh
        var any_changed = True
        while any_changed:
            any_changed = False
            for i_middle in range(N):
                if sort_inds.unsafe_load(Int64(i_middle)) == Int64(INVALID_IND):
                    continue

                var i_prev = i_middle
                while True:
                    i_prev = (i_prev + N - 1) % N
                    if sort_inds.unsafe_load(Int64(i_prev)) != Int64(INVALID_IND):
                        break
                var i_next = i_middle
                while True:
                    i_next = (i_next + 1) % N
                    if sort_inds.unsafe_load(Int64(i_next)) != Int64(INVALID_IND):
                        break

                var prev = Int(sort_inds.unsafe_load(Int64(i_prev)))
                var curr = Int(sort_inds.unsafe_load(Int64(i_middle)))
                var nxt = Int(sort_inds.unsafe_load(Int64(i_next)))

                # Degenerate cases
                if curr == prev or curr == nxt or prev == nxt:
                    continue

                # For any collinear points, keep only the closest
                if with_degeneracy_heuristic != 0:
                    var len_prev = _norm(pts, prev)
                    var len_curr = _norm(pts, curr)
                    var len_next = _norm(pts, nxt)
                    var collinear_prev = (
                        fabs(_cross(pts, curr, prev)) < len_prev * len_curr * DEGENERATE_THRESH
                    ) and _dot(pts, curr, prev) > 0.0
                    var collinear_next = (
                        fabs(_cross(pts, curr, nxt)) < len_next * len_curr * DEGENERATE_THRESH
                    ) and _dot(pts, curr, nxt) > 0.0
                    if (collinear_next and len_curr > len_next) or (
                        collinear_prev and len_curr > len_prev
                    ):
                        sort_inds.unsafe_store(Int64(i_middle), Int64(INVALID_IND))
                        any_changed = True
                        continue

                # If either of the triangles is empty (aka actually the
                # boundary), skip this
                if _is_boundary(pts, prev, curr) or _is_boundary(pts, curr, nxt):
                    continue

                # Test if the triangles should be merged
                var origin = Vec2(0.0, 0.0)
                if not in_circle_test(
                    origin, _pt(pts, prev), _pt(pts, nxt), _pt(pts, curr)
                ):
                    sort_inds.unsafe_store(Int64(i_middle), Int64(INVALID_IND))
                    any_changed = True

        # Emit the actual triangles
        for i_prev in range(N):
            if sort_inds.unsafe_load(Int64(i_prev)) == Int64(INVALID_IND):
                continue
            var i_next = i_prev
            while True:
                i_next = (i_next + 1) % N
                if sort_inds.unsafe_load(Int64(i_next)) != Int64(INVALID_IND):
                    break
            if i_prev == i_next:
                continue
            var prev = Int(sort_inds.unsafe_load(Int64(i_prev)))
            var nxt = Int(sort_inds.unsafe_load(Int64(i_next)))
            if not _is_boundary(pts, prev, nxt):
                tri.unsafe_store(Int64(total), Int64(p))
                tri.unsafe_store(Int64(total + 1), neighbors.unsafe_load(Int64(k * p + prev)))
                tri.unsafe_store(Int64(total + 2), neighbors.unsafe_load(Int64(k * p + nxt)))
                total += 3
    offsets.unsafe_store(Int64(n), Int64(total))
    return total


def _pt(pts: Ptr, i: Int) -> Vec2:
    return Vec2(pts.unsafe_load(2 * i), pts.unsafe_load(2 * i + 1))


def _norm(pts: Ptr, i: Int) -> Float64:
    var x = pts.unsafe_load(2 * i)
    var y = pts.unsafe_load(2 * i + 1)
    return sqrt(x * x + y * y)


def _cross(pts: Ptr, i: Int, j: Int) -> Float64:
    return (
        pts.unsafe_load(2 * i) * pts.unsafe_load(2 * j + 1)
        - pts.unsafe_load(2 * i + 1) * pts.unsafe_load(2 * j)
    )


def _dot(pts: Ptr, i: Int, j: Int) -> Float64:
    return (
        pts.unsafe_load(2 * i) * pts.unsafe_load(2 * j)
        + pts.unsafe_load(2 * i + 1) * pts.unsafe_load(2 * j + 1)
    )


# the `isBoundary` lambda: is the pair more than half a turn apart?
def _is_boundary(pts: Ptr, i: Int, j: Int) -> Bool:
    return _cross(pts, i, j) <= 0.0
