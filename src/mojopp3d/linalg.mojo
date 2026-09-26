"""Sparse Cholesky (LDL^T) factorization and solve, real and complex.

The heat method, the vector heat method and the point-cloud heat solver all
reduce to the same shape of problem: factor a symmetric matrix once, then apply
the factor to many right-hand sides. geometry-central hands this to SuiteSparse
CHOLMOD; this is a self-contained replacement.

`A = L D L^T` with `L` unit lower triangular. Everything is driven off the
sparsity pattern of `A`, which is symmetric:

- `S` (rows per column of `L`) is the structure of the factor.
- `R` is, for each row `k` of `L`, the ascending list of columns `j < k` with
  `L(k,j) != 0`, each paired with the offset of that entry in `S`'s values.

Both come out of `ldl_symbolic` in one pass, along with the `U` lists the
sweep threads while it runs. `ldl_numeric` then runs the outer-product
recurrence

    D(k) = A(k,k) - sum_{j<k} L(k,j)^2 D(j)
    L(i,k) = ( A(i,k) - sum_{j<k} L(i,j) L(k,j) D(j) ) / D(k)

restricted to the pattern, and `ldl_solve` applies `L^-1` and `L^-T`.

`A` is supplied as its upper triangle in column-compressed form, in the vertex
order the caller chose. Index arrays are Int64 to match what crosses the FFI,
so every offset below is Int64 as well.
"""

from std.sys import simd_width_of

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

# The vector gather/scatter intrinsics are microcoded on this target and lose
# to four independent scalar accesses issued back to back, so the sparse inner
# loops batch by hand. The elementwise passes do go through the registers.
comptime W = simd_width_of[DType.float64]()

# A single complex-symmetric factor entry. The real kernels work on bare
# Float64 so their hot loops stay vectorizable.
struct Cplx:
    var re: Float64
    var im: Float64

    def __init__(out self, re: Float64, im: Float64 = 0.0):
        self.re = re
        self.im = im

    def __add__(self, other: Self) -> Self:
        return Self(self.re + other.re, self.im + other.im)

    def __sub__(self, other: Self) -> Self:
        return Self(self.re - other.re, self.im - other.im)

    def __neg__(self) -> Self:
        return Self(-self.re, -self.im)

    def __mul__(self, other: Self) -> Self:
        return Self(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    def __truediv__(self, other: Self) -> Self:
        var d = other.re * other.re + other.im * other.im
        return Self(
            (self.re * other.re + self.im * other.im) / d,
            (self.im * other.re - self.re * other.im) / d,
        )


# ============================================================ ordering

# Reverse Cuthill-McKee over the graph induced by the symmetric pattern. Not as
# good as AMD in general, but short, deterministic, and close to optimal on the
# quasi-2D graphs a mesh triangulation produces.
def rcm(
    n: Int, Ap: IPtr, Ai: IPtr, degree: IPtr, visited: IPtr, queue: IPtr, order: IPtr
):
    var m = Int64(n)
    for i in range(m):
        degree.unsafe_store(i, Int64(0))
        visited.unsafe_store(i, Int64(0))
    for j in range(m):
        var acc = Int64(0)
        for p in range(Ap.unsafe_load(j), Ap.unsafe_load(j + 1)):
            var i = Ai.unsafe_load(p)
            if i != j:
                acc += Int64(1)
        degree.unsafe_store(j, acc)

    # BFS each connected component, then reverse the concatenation.
    var written = Int64(0)
    for start in range(m):
        if visited.unsafe_load(start) != Int64(0):
            continue
        var head = Int64(0)
        var tail = Int64(0)
        queue.unsafe_store(tail, start)
        tail += Int64(1)
        visited.unsafe_store(start, Int64(1))
        while head < tail:
            var j = queue.unsafe_load(head)
            head += Int64(1)
            order.unsafe_store(written, j)
            written += Int64(1)
            for p in range(Ap.unsafe_load(j), Ap.unsafe_load(j + 1)):
                var i = Ai.unsafe_load(p)
                if i != j:
                    if visited.unsafe_load(i) == Int64(0):
                        visited.unsafe_store(i, Int64(1))
                        queue.unsafe_store(tail, i)
                        tail += Int64(1)
    var half = written // Int64(2)
    var i = Int64(0)
    while i < half:
        var a = order.unsafe_load(i)
        var b = order.unsafe_load(written - Int64(1) - i)
        order.unsafe_store(i, b)
        order.unsafe_store(written - Int64(1) - i, a)
        i += Int64(1)


# ============================================================ symbolic

# Column k of the factor is, exactly,
#
#     L(:,k) = A(:,k) below diag  union  union_{j in U_k} L(:,j) above k
#
# where `U_k` is row k of the factor restricted to `j < k`: the columns k is
# eliminated against. The union over `U_k` is the expensive term, because
# |U_k| grows like the column width, so a plain sweep costs
# O(nnz(L) * width) -- where the old symbolic spent 1.2 s on a 10242-vertex
# icosphere.
#
# `cov` cuts that down without changing the pattern. Visiting `j` and then
# `m < j` is redundant whenever `m` is in `U_j`, i.e. `L(j,m) != 0`: the
# elimination of column j subtracts `L(i,m) L(j,m) D(m)` into every row `i` of
# `L(:,m)` above j, so everything `L(:,m)` has to offer above `k` is already
# inside what `L(:,j)` contributed. `U_j` is complete by the time column k is
# reached (j < k), and `U_k` is threaded in descending order, so the first
# child visited marks the rest and the sweep drops to O(nnz(L)): one column
# read and one list walk per factor column.
#
# Emitted alongside the column-compressed pattern is the row-compressed one,
# `R_p/R_i/R_pos`: row k of L as the ascending list of its columns `j < k`,
# each paired with the offset of `L(k,j)` in the column-compressed values. The
# numeric pass needs exactly that, so it never binary-searches.
#
# `limit` bounds the entry buffers. If the pattern does not fit, `info` reports
# how far the sweep got so the caller can extrapolate nnz(L) from the fill rate
# rather than doubling blindly.


def _sort_ascending(gather: IPtr, aux: IPtr, count: Int64):
    """Natural merge sort of gather[0:count].

    The gather is `A(:,k)` below the diagonal followed by one ascending tail
    per column the sweep visited, so it arrives as a handful of sorted runs --
    two, once the coverage test has pruned the sweep to a single column.
    """
    var lo = Int64(0)
    while lo < count:
        var mid = lo + Int64(1)
        while mid < count and gather.unsafe_load(mid - Int64(1)) <= gather.unsafe_load(mid):
            mid += Int64(1)
        if mid >= count:
            break
        var end = mid + Int64(1)
        while end < count and gather.unsafe_load(end - Int64(1)) <= gather.unsafe_load(end):
            end += Int64(1)
        var a = lo
        var b = mid
        var o = lo
        while a < mid and b < end:
            if gather.unsafe_load(a) <= gather.unsafe_load(b):
                aux.unsafe_store(o, gather.unsafe_load(a))
                a += Int64(1)
            else:
                aux.unsafe_store(o, gather.unsafe_load(b))
                b += Int64(1)
            o += Int64(1)
        while a < mid:
            aux.unsafe_store(o, gather.unsafe_load(a))
            a += Int64(1)
            o += Int64(1)
        while b < end:
            aux.unsafe_store(o, gather.unsafe_load(b))
            b += Int64(1)
            o += Int64(1)
        var t = lo
        while t < end:
            gather.unsafe_store(t, aux.unsafe_load(t))
            t += Int64(1)
        lo = end


def ldl_symbolic(
    n: Int,
    Ap: IPtr,
    Ai: IPtr,
    S_p: IPtr,
    S_i: IPtr,
    R_p: IPtr,
    R_i: IPtr,
    R_pos: IPtr,
    U_head: IPtr,
    U_i: IPtr,
    U_next: IPtr,
    mark: IPtr,
    cov: IPtr,
    gather: IPtr,
    aux: IPtr,
    cnt: IPtr,
    cursor: IPtr,
    info: IPtr,
    limit: Int,
) -> Int:
    """Structure of the factor: nnz(L), or -1 if it does not fit `limit`.

    On -1, `info` holds the entries written and the number of columns finished.
    """
    var m = Int64(n)
    var cap = Int64(limit)
    for i in range(m):
        U_head.unsafe_store(i, Int64(-1))
        mark.unsafe_store(i, Int64(0))
        cov.unsafe_store(i, Int64(0))
        cnt.unsafe_store(i, Int64(0))
    S_p.unsafe_store(Int64(0), Int64(0))
    var write = Int64(0)

    for k in range(m):
        # Published before any child column's end offset is read.
        S_p.unsafe_store(k, write)
        var stamp = k + Int64(1)
        var nGather = Int64(0)
        for p in range(Ap.unsafe_load(k), Ap.unsafe_load(k + Int64(1))):
            var i = Ai.unsafe_load(p)
            if i > k:
                mark.unsafe_store(i, stamp)
                gather.unsafe_store(nGather, i)
                nGather += Int64(1)
        # `U_k` is threaded in descending order, which is what the coverage
        # test needs: the first child visited marks the rest.
        var q = U_head.unsafe_load(k)
        while q >= Int64(0):
            var j = U_i.unsafe_load(q)
            if cov.unsafe_load(j) != stamp:
                for r in range(S_p.unsafe_load(j), S_p.unsafe_load(j + Int64(1))):
                    var i = S_i.unsafe_load(r)
                    if i > k and mark.unsafe_load(i) != stamp:
                        mark.unsafe_store(i, stamp)
                        gather.unsafe_store(nGather, i)
                        nGather += Int64(1)
                # Every m in U_j has already had its column folded into this
                # one, so m itself needs no visit.
                var s = U_head.unsafe_load(j)
                while s >= Int64(0):
                    cov.unsafe_store(U_i.unsafe_load(s), stamp)
                    s = U_next.unsafe_load(s)
            q = U_next.unsafe_load(q)
        if write + nGather > cap:
            # Entries written and columns finished, so the caller can
            # extrapolate nnz(L) instead of doubling blindly.
            info.unsafe_store(Int64(0), write)
            info.unsafe_store(Int64(1), k)
            return -Int(1)
        _sort_ascending(gather, aux, nGather)

        for r in range(nGather):
            var i = gather.unsafe_load(r)
            S_i.unsafe_store(write, i)
            cnt.unsafe_store(i, cnt.unsafe_load(i) + Int64(1))
            # L(i,k) != 0, so k becomes a child of i.
            U_i.unsafe_store(write, k)
            U_next.unsafe_store(write, U_head.unsafe_load(i))
            U_head.unsafe_store(i, write)
            write += Int64(1)
    S_p.unsafe_store(m, write)

    # The rows, by counting sort over the columns. `cnt` already holds the row
    # counts, so this is one pass with a cursor per row.
    R_p.unsafe_store(Int64(0), Int64(0))
    for k in range(m):
        R_p.unsafe_store(k + Int64(1), R_p.unsafe_load(k) + cnt.unsafe_load(k))
        cursor.unsafe_store(k, R_p.unsafe_load(k))
    for k in range(m):
        var pos = S_p.unsafe_load(k)
        var end = S_p.unsafe_load(k + Int64(1))
        while pos < end:
            var i = S_i.unsafe_load(pos)
            var slot = cursor.unsafe_load(i)
            R_i.unsafe_store(slot, k)
            R_pos.unsafe_store(slot, pos)
            cursor.unsafe_store(i, slot + Int64(1))
            pos += Int64(1)
    return Int(write)


# ============================================================ numeric

# nnz(L) is passed in so the factor can be zeroed before the numeric pass; a
# vanished pivot is nudged to this relative floor so the (possibly indefinite)
# affine-connection system stays solvable. It is never reached for the
# positive definite heat and Poisson operators.
comptime PIVOT_FLOOR = 1.0e-14

def ldl_numeric(
    n: Int,
    nnzL: Int,
    Ap: IPtr,
    Ai: IPtr,
    Ax: Ptr,
    S_p: IPtr,
    S_i: IPtr,
    R_p: IPtr,
    R_i: IPtr,
    R_pos: IPtr,
    upto: IPtr,
    Lx: Ptr,
    D: Ptr,
    Y: Ptr,
    W: Ptr,
):
    var m = Int64(n)
    # Lx needs no pre-zeroing: column k is written in full before any later
    # column reads it, so every entry is live before it is read.
    for i in range(m):
        Y.unsafe_store(i, 0.0)
        upto.unsafe_store(i, S_p.unsafe_load(i))

    for k in range(m):
        # Row k of L is the list of `j < k` this column is built from, and it
        # carries the offset of each L(k,j) in the column-compressed values.
        var rp = R_p.unsafe_load(k)
        var re = R_p.unsafe_load(k + Int64(1))

        # D(k) = A(k,k) - sum_j L(k,j)^2 D(j)
        var dsum = 0.0
        var q = rp
        while q + Int64(4) <= re:
            var k0 = R_pos.unsafe_load(q)
            var k1 = R_pos.unsafe_load(q + Int64(1))
            var k2 = R_pos.unsafe_load(q + Int64(2))
            var k3 = R_pos.unsafe_load(q + Int64(3))
            var l0 = Lx.unsafe_load(k0)
            var l1 = Lx.unsafe_load(k1)
            var l2 = Lx.unsafe_load(k2)
            var l3 = Lx.unsafe_load(k3)
            var w0 = l0 * D.unsafe_load(R_i.unsafe_load(q))
            var w1 = l1 * D.unsafe_load(R_i.unsafe_load(q + Int64(1)))
            var w2 = l2 * D.unsafe_load(R_i.unsafe_load(q + Int64(2)))
            var w3 = l3 * D.unsafe_load(R_i.unsafe_load(q + Int64(3)))
            W.unsafe_store(q - rp, w0)
            W.unsafe_store(q - rp + Int64(1), w1)
            W.unsafe_store(q - rp + Int64(2), w2)
            W.unsafe_store(q - rp + Int64(3), w3)
            dsum += (w0 * l0 + w1 * l1) + (w2 * l2 + w3 * l3)
            q += Int64(4)
        while q < re:
            var lkj = Lx.unsafe_load(R_pos.unsafe_load(q))
            var w = lkj * D.unsafe_load(R_i.unsafe_load(q))
            W.unsafe_store(q - rp, w)
            dsum += w * lkj
            q += Int64(1)

        # Y(i) = A(i,k) - sum_j L(i,j) L(k,j) D(j). Every row this touches is
        # in column k of L and Y is zero everywhere else, so one sweep of
        # A(:,k) seeds it -- no search of A per row of the factor.
        var akk = 0.0
        var ap = Ap.unsafe_load(k)
        var aq = Ap.unsafe_load(k + Int64(1))
        while ap < aq:
            var i = Ai.unsafe_load(ap)
            if i > k:
                Y.unsafe_store(i, Ax.unsafe_load(ap))
            elif i == k:
                akk = Ax.unsafe_load(ap)
            ap += Int64(1)
        q = rp
        while q < re:
            var j = R_i.unsafe_load(q)
            var w = W.unsafe_load(q - rp)
            # `upto` is the first entry of column j above the diagonal at k.
            # k only ever increases, so it is kept between columns instead of
            # re-tested for every entry: one advance per entry, once.
            var r = upto.unsafe_load(j)
            var je = S_p.unsafe_load(j + Int64(1))
            while r < je and S_i.unsafe_load(r) <= k:
                r += Int64(1)
            upto.unsafe_store(j, r)
            while r + Int64(4) <= je:
                var i0 = S_i.unsafe_load(r)
                var i1 = S_i.unsafe_load(r + Int64(1))
                var i2 = S_i.unsafe_load(r + Int64(2))
                var i3 = S_i.unsafe_load(r + Int64(3))
                var v0 = Y.unsafe_load(i0)
                var v1 = Y.unsafe_load(i1)
                var v2 = Y.unsafe_load(i2)
                var v3 = Y.unsafe_load(i3)
                Y.unsafe_store(i0, v0 - Lx.unsafe_load(r) * w)
                Y.unsafe_store(i1, v1 - Lx.unsafe_load(r + Int64(1)) * w)
                Y.unsafe_store(i2, v2 - Lx.unsafe_load(r + Int64(2)) * w)
                Y.unsafe_store(i3, v3 - Lx.unsafe_load(r + Int64(3)) * w)
                r += Int64(4)
            while r < je:
                var i = S_i.unsafe_load(r)
                Y.unsafe_store(i, Y.unsafe_load(i) - Lx.unsafe_load(r) * w)
                r += Int64(1)
            q += Int64(1)

        var d = akk - dsum
        if d > 0.0:
            if d < PIVOT_FLOOR:
                d = PIVOT_FLOOR
        else:
            if d > -PIVOT_FLOOR:
                d = -PIVOT_FLOOR
        D.unsafe_store(k, d)

        # L is strictly lower triangular, so every entry of column k is a row.
        var pos = S_p.unsafe_load(k)
        var end = S_p.unsafe_load(k + Int64(1))
        while pos < end:
            var i = S_i.unsafe_load(pos)
            Lx.unsafe_store(pos, Y.unsafe_load(i) / d)
            Y.unsafe_store(i, 0.0)
            pos += Int64(1)


def ldl_solve(n: Int, S_p: IPtr, S_i: IPtr, Lx: Ptr, D: Ptr, B: Ptr, X: Ptr, nrhs: Int):
    """In-place solve of A X = B, `nrhs` right-hand sides, one per column."""
    # B and X are `n x nrhs` in C order, i.e. right-hand side r starts at r.
    # The two sparse passes are batches of four rather than one-at-a-time: a
    # vector gather/scatter is microcoded on this target and slower, but four
    # independent scalar accesses issued back to back do overlap.
    var m = Int64(n)
    var stride = Int64(nrhs)
    for r in range(Int64(nrhs)):
        # X = B, and w = D^-1 y below, are elementwise in k. They are only
        # contiguous in memory for a single right-hand side.
        var kk = Int64(0)
        if stride == Int64(1):
            while kk + Int64(W) <= m:
                X.unsafe_store[width=W](kk, B.unsafe_load[width=W](kk))
                kk += Int64(W)
        while kk < m:
            X.unsafe_store(kk * stride + r, B.unsafe_load(kk * stride + r))
            kk += Int64(1)
        for k in range(m):  # L y = b
            var xk = X.unsafe_load(k * stride + r)
            var p = S_p.unsafe_load(k)
            var pe = S_p.unsafe_load(k + Int64(1))
            while p + Int64(4) <= pe:
                var i0 = S_i.unsafe_load(p) * stride + r
                var i1 = S_i.unsafe_load(p + Int64(1)) * stride + r
                var i2 = S_i.unsafe_load(p + Int64(2)) * stride + r
                var i3 = S_i.unsafe_load(p + Int64(3)) * stride + r
                var v0 = X.unsafe_load(i0)
                var v1 = X.unsafe_load(i1)
                var v2 = X.unsafe_load(i2)
                var v3 = X.unsafe_load(i3)
                X.unsafe_store(i0, v0 - Lx.unsafe_load(p) * xk)
                X.unsafe_store(i1, v1 - Lx.unsafe_load(p + Int64(1)) * xk)
                X.unsafe_store(i2, v2 - Lx.unsafe_load(p + Int64(2)) * xk)
                X.unsafe_store(i3, v3 - Lx.unsafe_load(p + Int64(3)) * xk)
                p += Int64(4)
            while p < pe:
                var idx = S_i.unsafe_load(p) * stride + r
                X.unsafe_store(idx, X.unsafe_load(idx) - Lx.unsafe_load(p) * xk)
                p += Int64(1)
        kk = Int64(0)
        if stride == Int64(1):
            while kk + Int64(W) <= m:  # w = D^-1 y
                X.unsafe_store[width=W](
                    kk, X.unsafe_load[width=W](kk) / D.unsafe_load[width=W](kk)
                )
                kk += Int64(W)
        while kk < m:
            var idx = kk * stride + r
            X.unsafe_store(idx, X.unsafe_load(idx) / D.unsafe_load(kk))
            kk += Int64(1)
        for k in range(m - Int64(1), Int64(-1), Int64(-1)):  # L^T z = w
            var p = S_p.unsafe_load(k)
            var pe = S_p.unsafe_load(k + Int64(1))
            var a0 = 0.0
            var a1 = 0.0
            var a2 = 0.0
            var a3 = 0.0
            while p + Int64(4) <= pe:
                a0 += Lx.unsafe_load(p) * X.unsafe_load(S_i.unsafe_load(p) * stride + r)
                a1 += Lx.unsafe_load(p + Int64(1)) * X.unsafe_load(
                    S_i.unsafe_load(p + Int64(1)) * stride + r
                )
                a2 += Lx.unsafe_load(p + Int64(2)) * X.unsafe_load(
                    S_i.unsafe_load(p + Int64(2)) * stride + r
                )
                a3 += Lx.unsafe_load(p + Int64(3)) * X.unsafe_load(
                    S_i.unsafe_load(p + Int64(3)) * stride + r
                )
                p += Int64(4)
            var acc = X.unsafe_load(k * stride + r) - ((a0 + a1) + (a2 + a3))
            while p < pe:
                acc -= Lx.unsafe_load(p) * X.unsafe_load(S_i.unsafe_load(p) * stride + r)
                p += Int64(1)
            X.unsafe_store(k * stride + r, acc)


# ============================================================ complex Hermitian

# The vertex connection Laplacian comes out of geometry-central with
# A[i,j] = conj(A[j,i]) and a real diagonal, i.e. Hermitian rather than
# complex-symmetric, so the factorization is A = L D L^H with D real. The
# structure is the same one the real path uses; only the inner products
# conjugate.

def ldl_numeric_c(
    n: Int,
    nnzL: Int,
    Ap: IPtr,
    Ai: IPtr,
    Axr: Ptr,
    Axi: Ptr,
    S_p: IPtr,
    S_i: IPtr,
    R_p: IPtr,
    R_i: IPtr,
    R_pos: IPtr,
    upto: IPtr,
    Lxr: Ptr,
    Lxi: Ptr,
    Dr: Ptr,
    Di: Ptr,
    Yr: Ptr,
    Yi: Ptr,
    Wr: Ptr,
    Wi: Ptr,
):
    var m = Int64(n)
    for p in range(Int64(nnzL)):
        Lxr.unsafe_store(p, 0.0)
        Lxi.unsafe_store(p, 0.0)
    for i in range(m):
        Yr.unsafe_store(i, 0.0)
        Yi.unsafe_store(i, 0.0)
        Dr.unsafe_store(i, 0.0)
        Di.unsafe_store(i, 0.0)
        upto.unsafe_store(i, S_p.unsafe_load(i))
    for k in range(m):
        var rp = R_p.unsafe_load(k)
        var re = R_p.unsafe_load(k + Int64(1))

        # D(k) = A(k,k) - sum_j |L(k,j)|^2 D(j)
        var dsum = 0.0
        var q = rp
        while q < re:
            var j = R_i.unsafe_load(q)
            var slot = R_pos.unsafe_load(q)
            var lr = Lxr.unsafe_load(slot)
            var li = Lxi.unsafe_load(slot)
            var dj = Dr.unsafe_load(j)
            Wr.unsafe_store(q - rp, lr * dj)
            Wi.unsafe_store(q - rp, -li * dj)
            dsum += (lr * lr + li * li) * dj
            q += Int64(1)

        # Y(i) = A(i,k) - sum_j L(i,j) L(k,j) D(j)
        var akk = 0.0
        var ap = Ap.unsafe_load(k)
        var aq = Ap.unsafe_load(k + Int64(1))
        while ap < aq:
            var i = Ai.unsafe_load(ap)
            if i > k:
                Yr.unsafe_store(i, Axr.unsafe_load(ap))
                Yi.unsafe_store(i, Axi.unsafe_load(ap))
            elif i == k:
                akk = Axr.unsafe_load(ap)
            ap += Int64(1)
        q = rp
        while q < re:
            var j = R_i.unsafe_load(q)
            var wr = Wr.unsafe_load(q - rp)
            var wi = Wi.unsafe_load(q - rp)
            var r = upto.unsafe_load(j)
            var je = S_p.unsafe_load(j + Int64(1))
            while r < je and S_i.unsafe_load(r) <= k:
                r += Int64(1)
            upto.unsafe_store(j, r)
            while r < je:
                var i = S_i.unsafe_load(r)
                var lr = Lxr.unsafe_load(r)
                var li = Lxi.unsafe_load(r)
                Yr.unsafe_store(i, Yr.unsafe_load(i) - (lr * wr - li * wi))
                Yi.unsafe_store(i, Yi.unsafe_load(i) - (li * wr + lr * wi))
                r += Int64(1)
            q += Int64(1)

        var d = akk - dsum
        if d > 0.0:
            if d < PIVOT_FLOOR:
                d = PIVOT_FLOOR
        else:
            if d > -PIVOT_FLOOR:
                d = -PIVOT_FLOOR
        Dr.unsafe_store(k, d)
        Di.unsafe_store(k, 0.0)

        # L is strictly lower triangular, so every entry of column k is a row.
        var pos = S_p.unsafe_load(k)
        var end = S_p.unsafe_load(k + Int64(1))
        while pos < end:
            var i = S_i.unsafe_load(pos)
            Lxr.unsafe_store(pos, Yr.unsafe_load(i) / d)
            Lxi.unsafe_store(pos, Yi.unsafe_load(i) / d)
            Yr.unsafe_store(i, 0.0)
            Yi.unsafe_store(i, 0.0)
            pos += Int64(1)


def ldl_solve_c(
    n: Int,
    S_p: IPtr,
    S_i: IPtr,
    Lxr: Ptr,
    Lxi: Ptr,
    Dr: Ptr,
    Di: Ptr,
    Br: Ptr,
    Bi: Ptr,
    Xr: Ptr,
    Xi: Ptr,
    nrhs: Int,
):
    var m = Int64(n)
    for r in range(Int64(nrhs)):
        var off = r * m
        var t = Int64(0)
        while t + Int64(W) <= m:  # X = B
            Xr.unsafe_store[width=W](off + t, Br.unsafe_load[width=W](off + t))
            Xi.unsafe_store[width=W](off + t, Bi.unsafe_load[width=W](off + t))
            t += Int64(W)
        while t < m:
            Xr.unsafe_store(off + t, Br.unsafe_load(off + t))
            Xi.unsafe_store(off + t, Bi.unsafe_load(off + t))
            t += Int64(1)
        for k in range(m):  # L y = b
            var yk = Cplx(Xr.unsafe_load(off + k), Xi.unsafe_load(off + k))
            var p = S_p.unsafe_load(k)
            var pe = S_p.unsafe_load(k + Int64(1))
            while p + Int64(4) <= pe:
                var i0 = off + S_i.unsafe_load(p)
                var i1 = off + S_i.unsafe_load(p + Int64(1))
                var i2 = off + S_i.unsafe_load(p + Int64(2))
                var i3 = off + S_i.unsafe_load(p + Int64(3))
                var v0 = Cplx(Xr.unsafe_load(i0), Xi.unsafe_load(i0))
                var v1 = Cplx(Xr.unsafe_load(i1), Xi.unsafe_load(i1))
                var v2 = Cplx(Xr.unsafe_load(i2), Xi.unsafe_load(i2))
                var v3 = Cplx(Xr.unsafe_load(i3), Xi.unsafe_load(i3))
                v0 = v0 - Cplx(Lxr.unsafe_load(p), Lxi.unsafe_load(p)) * yk
                v1 = v1 - Cplx(Lxr.unsafe_load(p + Int64(1)), Lxi.unsafe_load(p + Int64(1))) * yk
                v2 = v2 - Cplx(Lxr.unsafe_load(p + Int64(2)), Lxi.unsafe_load(p + Int64(2))) * yk
                v3 = v3 - Cplx(Lxr.unsafe_load(p + Int64(3)), Lxi.unsafe_load(p + Int64(3))) * yk
                Xr.unsafe_store(i0, v0.re)
                Xi.unsafe_store(i0, v0.im)
                Xr.unsafe_store(i1, v1.re)
                Xi.unsafe_store(i1, v1.im)
                Xr.unsafe_store(i2, v2.re)
                Xi.unsafe_store(i2, v2.im)
                Xr.unsafe_store(i3, v3.re)
                Xi.unsafe_store(i3, v3.im)
                p += Int64(4)
            while p < pe:
                var tv = Cplx(Lxr.unsafe_load(p), Lxi.unsafe_load(p))
                var idx = off + S_i.unsafe_load(p)
                var xv = Cplx(Xr.unsafe_load(idx), Xi.unsafe_load(idx))
                var upd = xv - tv * yk
                Xr.unsafe_store(idx, upd.re)
                Xi.unsafe_store(idx, upd.im)
                p += Int64(1)
            Xr.unsafe_store(off + k, yk.re)
            Xi.unsafe_store(off + k, yk.im)
        t = Int64(0)
        while t + Int64(W) <= m:  # w = D^-1 y, D is real
            Xr.unsafe_store[width=W](off + t, Xr.unsafe_load[width=W](off + t) / Dr.unsafe_load[width=W](t))
            Xi.unsafe_store[width=W](off + t, Xi.unsafe_load[width=W](off + t) / Dr.unsafe_load[width=W](t))
            t += Int64(W)
        while t < m:
            var d = Dr.unsafe_load(t)
            Xr.unsafe_store(off + t, Xr.unsafe_load(off + t) / d)
            Xi.unsafe_store(off + t, Xi.unsafe_load(off + t) / d)
            t += Int64(1)
        for k in range(m - Int64(1), Int64(-1), Int64(-1)):  # L^H z = w
            var p = S_p.unsafe_load(k)
            var pe = S_p.unsafe_load(k + Int64(1))
            var ar0 = 0.0
            var ai0 = 0.0
            var ar1 = 0.0
            var ai1 = 0.0
            while p + Int64(4) <= pe:
                var q0 = off + S_i.unsafe_load(p)
                var q1 = off + S_i.unsafe_load(p + Int64(1))
                var q2 = off + S_i.unsafe_load(p + Int64(2))
                var q3 = off + S_i.unsafe_load(p + Int64(3))
                var t0 = Cplx(Lxr.unsafe_load(p), Lxi.unsafe_load(p))
                var t1 = Cplx(Lxr.unsafe_load(p + Int64(1)), Lxi.unsafe_load(p + Int64(1)))
                var t2 = Cplx(Lxr.unsafe_load(p + Int64(2)), Lxi.unsafe_load(p + Int64(2)))
                var t3 = Cplx(Lxr.unsafe_load(p + Int64(3)), Lxi.unsafe_load(p + Int64(3)))
                var u0 = Cplx(Xr.unsafe_load(q0), Xi.unsafe_load(q0))
                var u1 = Cplx(Xr.unsafe_load(q1), Xi.unsafe_load(q1))
                var u2 = Cplx(Xr.unsafe_load(q2), Xi.unsafe_load(q2))
                var u3 = Cplx(Xr.unsafe_load(q3), Xi.unsafe_load(q3))
                ar0 += t0.re * u0.re + t0.im * u0.im
                ai0 += t0.re * u0.im - t0.im * u0.re
                ar1 += t1.re * u1.re + t1.im * u1.im
                ai1 += t1.re * u1.im - t1.im * u1.re
                ar0 += t2.re * u2.re + t2.im * u2.im
                ai0 += t2.re * u2.im - t2.im * u2.re
                ar1 += t3.re * u3.re + t3.im * u3.im
                ai1 += t3.re * u3.im - t3.im * u3.re
                p += Int64(4)
            var acc = Cplx(Xr.unsafe_load(off + k), Xi.unsafe_load(off + k)) - Cplx(
                (ar0 + ar1), (ai0 + ai1)
            )
            while p < pe:
                var tv = Cplx(Lxr.unsafe_load(p), Lxi.unsafe_load(p))
                var idx = off + S_i.unsafe_load(p)
                var xv = Cplx(Xr.unsafe_load(idx), Xi.unsafe_load(idx))
                acc = acc - Cplx(tv.re * xv.re + tv.im * xv.im, tv.re * xv.im - tv.im * xv.re)
                p += Int64(1)
            Xr.unsafe_store(off + k, acc.re)
            Xi.unsafe_store(off + k, acc.im)
