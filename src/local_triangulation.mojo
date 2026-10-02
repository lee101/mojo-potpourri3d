# geometry-central pointcloud/local_triangulation.cpp: buildLocalTriangulations, handleToInds and
# handleToFlatInds.
#
# Upstream returns a `PointData<std::vector<std::array<Point, 3>>>`, i.e. a ragged per-point list.
# A Mojo function cannot return a move-only List and also read caller-owned geometry, so the same
# ragged list is written into a caller-owned int arena: the triangles of point p occupy
# `handles[3 * tri_off[p] : 3 * tri_off[p + 1]]`. On a compressed cloud a Point handle IS its
# index, so the numpy read-back in python/mojo_potpourri3d/point_cloud.py is the handleToInds /
# handleToFlatInds pair inlined as one slice of the same arrays.
from std.math import sqrt, isfinite
from vector_util import Vector2, v2_from_angle, v2_cross, v2_dot, nan64

# Upstream's INVALID_IND is SIZE_MAX, so the removed neighbour slots are -1 here because the port's
# index arrays are signed, and DEGENERATE_THRESH (1e-7, in units of relative length) is inlined
# where upstream names it.


def _det3(
    a: Float64, b: Float64, c: Float64, d: Float64, e: Float64, f: Float64, g: Float64,
    h: Float64, i: Float64,
) -> Float64:
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


# elementary_geometry.cpp: inCircleTest, the determinant of the 4x4 matrix
# [[x, y, |p|^2, 1]] over the four points, tested > 0. vector_util's copy of this returns its
# negative, so it is transcribed here rather than imported.
def in_circle_test(p_a: Vector2, p_b: Vector2, p_c: Vector2, p_test: Vector2) -> Bool:
    var r = p_a.x
    var s = p_a.y
    var t = p_a.norm2()
    var u = p_b.x
    var v = p_b.y
    var w = p_b.norm2()
    var x = p_c.x
    var y = p_c.y
    var z = p_c.norm2()
    var p = p_test.x
    var q = p_test.y
    var o = p_test.norm2()
    # cofactor expansion along the last row
    var m1 = _det3(s, t, 1.0, v, w, 1.0, y, z, 1.0)
    var m2 = _det3(r, t, 1.0, u, w, 1.0, x, z, 1.0)
    var m3 = _det3(r, s, 1.0, u, v, 1.0, x, y, 1.0)
    var d = _det3(r, s, t, u, v, w, x, y, z)
    return -p * m1 + q * m2 - o * m3 + d > 0.0


# Vector2::normalize divides by sqrt(norm2) with no zero guard, so the zero vector becomes
# (nan, nan). vector_util's unit() returns the zero vector instead, which would hide the
# degenerate neighbourhood the sentinel below exists to catch.
def unit_nan(v: Vector2) -> Vector2:
    if v.norm2() == 0.0:
        return Vector2(nan64(), nan64())
    return v / sqrt(v.norm2())


# The isBoundary lambda in buildLocalTriangulations
def is_boundary(p_a: Vector2, p_b: Vector2) -> Bool:
    return v2_cross(p_a, p_b) <= 0.0


# buildLocalTriangulations. The 1-ring of the center point in the local planar Delaunay
# triangulation, found by repeatedly removing points from an angularly sorted list whose diamond is
# not Delaunay.
def build_local_triangulations(
    n_p: Int,
    n_n: Int,
    with_deg: Int,
    nbr: Pointer[Int, AnyOrigin[mut=True]],
    tan: Pointer[Float64, AnyOrigin[mut=True]],
    handles: Pointer[Int, AnyOrigin[mut=True]],
    tri_off: Pointer[Int, AnyOrigin[mut=True]],
) -> Int:
    # Perturb points / sorted indices / angles, all of length n_n
    var perturb_x = List[Float64](length=n_n, fill=0.0)
    var perturb_y = List[Float64](length=n_n, fill=0.0)
    var sort_inds = List[Int](length=n_n, fill=0)
    var point_angles = List[Float64](length=n_n, fill=0.0)

    var count = 0
    var p = 0
    while p < n_p:
        tri_off.unsafe_store(p, count)
        var n_neigh = n_n

        # Compute a lengthscale for the neighborhood as the radius of the most distant point
        var len_scale = 0.0
        var len_scale2 = 0.0
        var i_neigh = 0
        while i_neigh < n_neigh:
            var neigh_pt = Vector2(tan.unsafe_load(2 * (p * n_n + i_neigh)),
                                   tan.unsafe_load(2 * (p * n_n + i_neigh) + 1))
            var dist2 = neigh_pt.norm2()
            len_scale2 = max(len_scale2, dist2)
            i_neigh += 1
        len_scale = sqrt(len_scale2)

        # Something is hopelessly degenerate, don't even bother trying. No triangles for this point.
        if not isfinite(len_scale) or len_scale <= 0.0:
            p += 1
            continue

        # Local copies of points
        i_neigh = 0
        while i_neigh < n_neigh:
            perturb_x[i_neigh] = tan.unsafe_load(2 * (p * n_n + i_neigh))
            perturb_y[i_neigh] = tan.unsafe_load(2 * (p * n_n + i_neigh) + 1)
            i_neigh += 1

        if with_deg:
            # Perturb points which are extremely close to the source
            i_neigh = 0
            while i_neigh < n_neigh:
                var neigh_pt = Vector2(perturb_x[i_neigh], perturb_y[i_neigh])
                var dist = neigh_pt.norm()
                if dist < len_scale * 1e-7:  # need to perturb
                    var dir = unit_nan(neigh_pt)
                    if not isfinite(dir.x) or not isfinite(dir.y):
                        # even direction is degenerate :(
                        # pick a direction from index
                        var theta_dir = (2.0 * 3.14159265358979323846 * Float64(i_neigh)) / Float64(n_neigh)
                        dir = v2_from_angle(theta_dir)

                    # Set the distance from the origin for the perturbed point. Including the index
                    # avoids creating many co-circular points; no need to stress the Delaunay
                    # triangulation unnecessarily.
                    var len = (1.0 + Float64(i_neigh) / Float64(n_neigh)) * len_scale * 1e-7 * 10

                    perturb_x[i_neigh] = len * dir.x  # update the point
                    perturb_y[i_neigh] = len * dir.y
                i_neigh += 1

        # = Angularly sort the points CCW, such that the closest point comes first
        # sentinel value for below
        var bad_angle = -777.0  # sentinel value for below
        var i = 0
        while i < n_neigh:
            var angle = unit_nan(Vector2(perturb_x[i], perturb_y[i])).arg()
            if not isfinite(angle):
                angle = bad_angle
            sort_inds[i] = i
            point_angles[i] = angle
            i += 1

        # Angular sort. Upstream's std::sort is unstable; the angles are distinct for a generic
        # neighborhood, and where they are not this stable insertion sort is deterministic.
        i = 1
        while i < n_neigh:
            var key_ind = sort_inds[i]
            var key_angle = point_angles[key_ind]
            var j = i - 1
            while j >= 0 and point_angles[sort_inds[j]] > key_angle:
                sort_inds[j + 1] = sort_inds[j]
                j -= 1
            sort_inds[j + 1] = key_ind
            i += 1

        # Immediately skip any invalid indices in the search below, by detecting sentinels from
        # above. NOTE: a non-finite angle was already replaced by the sentinel, so this does fire.
        i = 0
        while i < n_neigh:
            if point_angles[sort_inds[i]] == bad_angle:
                sort_inds[i] = -1
            i += 1

        # == Find the local Delaunay triangulation. The output we seek is a subset of the sorted
        # list, which corresponds to the 1-ring of the center vertex; points are repeatedly removed
        # (marked -1) if the diamond they form is not Delaunay.
        var n = n_neigh
        var any_changed = True
        while any_changed:
            any_changed = False
            var i_middle = 0
            while i_middle < n:
                if sort_inds[i_middle] == -1:
                    i_middle += 1  # skip unused indices
                    continue

                # Find the previous and next indices
                var i_prev = i_middle
                var found = False
                while not found:
                    i_prev = (i_prev + n - 1) % n
                    if sort_inds[i_prev] != -1:
                        found = True
                var i_next = i_middle
                found = False
                while not found:
                    i_next = (i_next + 1) % n
                    if sort_inds[i_next] != -1:
                        found = True

                # Indices in to the local neighbor list
                var prev = sort_inds[i_prev]
                var curr = sort_inds[i_middle]
                var next = sort_inds[i_next]

                # Degenerate cases
                if curr == prev or curr == next or prev == next:
                    i_middle += 1
                    continue

                # For any collinear points, keep only the closest
                if with_deg:
                    var len_prev = Vector2(perturb_x[prev], perturb_y[prev]).norm()
                    var len_curr = Vector2(perturb_x[curr], perturb_y[curr]).norm()
                    var len_next = Vector2(perturb_x[next], perturb_y[next]).norm()

                    var collinear_prev = (
                        abs(v2_cross(Vector2(perturb_x[curr], perturb_y[curr]),
                                     Vector2(perturb_x[prev], perturb_y[prev])))
                        < (len_prev * len_curr) * 1e-7
                    ) and (v2_dot(Vector2(perturb_x[curr], perturb_y[curr]),
                                  Vector2(perturb_x[prev], perturb_y[prev])) > 0.0)
                    var collinear_next = (
                        abs(v2_cross(Vector2(perturb_x[curr], perturb_y[curr]),
                                     Vector2(perturb_x[next], perturb_y[next])))
                        < (len_next * len_curr) * 1e-7
                    ) and (v2_dot(Vector2(perturb_x[curr], perturb_y[curr]),
                                  Vector2(perturb_x[next], perturb_y[next])) > 0.0)

                    if (collinear_next and len_curr > len_next) or (collinear_prev and len_curr > len_prev):
                        sort_inds[i_middle] = -1
                        any_changed = True
                        i_middle += 1
                        continue

                # If either of the triangles is empty (aka actually the boundary), skip this
                if is_boundary(Vector2(perturb_x[prev], perturb_y[prev]),
                               Vector2(perturb_x[curr], perturb_y[curr])) or \
                    is_boundary(Vector2(perturb_x[curr], perturb_y[curr]),
                                Vector2(perturb_x[next], perturb_y[next])):
                    i_middle += 1
                    continue

                # Test if the triangles should be merged
                if not in_circle_test(Vector2(0.0, 0.0), Vector2(perturb_x[prev], perturb_y[prev]),
                                      Vector2(perturb_x[next], perturb_y[next]),
                                      Vector2(perturb_x[curr], perturb_y[curr])):
                    sort_inds[i_middle] = -1
                    any_changed = True
                i_middle += 1

        # Emit the actual triangles
        var i_prev2 = 0
        while i_prev2 < n:
            if sort_inds[i_prev2] == -1:
                i_prev2 += 1  # skip unused indices
                continue

            var i_next2 = i_prev2
            var found2 = False
            while not found2:
                i_next2 = (i_next2 + 1) % n
                if sort_inds[i_next2] != -1:
                    found2 = True

            if i_prev2 == i_next2:
                i_prev2 += 1
                continue

            # Indices in to the local neighbor list
            var prev2 = sort_inds[i_prev2]
            var next2 = sort_inds[i_next2]

            if not is_boundary(Vector2(perturb_x[prev2], perturb_y[prev2]),
                               Vector2(perturb_x[next2], perturb_y[next2])):
                handles.unsafe_store(3 * count, p)
                handles.unsafe_store(3 * count + 1, nbr.unsafe_load(p * n_n + prev2))
                handles.unsafe_store(3 * count + 2, nbr.unsafe_load(p * n_n + next2))
                count += 1
            i_prev2 += 1
        p += 1
    tri_off.unsafe_store(n_p, count)
    return count
