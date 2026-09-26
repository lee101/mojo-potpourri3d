"""COO -> symmetric, ordered, column-compressed form.

Eigen builds its sparse matrices with `setFromTriplets`, which sums duplicate
triplets, and geometry-central hands the result to a sparse Cholesky that wants
a symmetric matrix. This module does the same job: sum duplicates, mirror the
upper triangle into a full symmetric pattern, order the vertices with RCM, and
emit the result in the permuted order that `mojopp3d.linalg` factorizes.

Both triangles are stored, because the outer-product factorization reads
`A(i,k)` for `i > k` even though it only ever writes the lower factor.
"""

from mojopp3d.linalg import rcm

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]


# Column offsets and row indices of the full symmetric pattern of the triplet
# list, for the RCM traversal. Both orientations are emitted, because a BFS only
# ever looks inside one column and the upper triangle alone would hide half of
# every vertex's neighbours. Rows come out in triplet order, which the BFS does
# not care about.
def adjacency_pattern(
    n: Int, ti: IPtr, tj: IPtr, nnzT: Int, Ap: IPtr, Ai: IPtr
) -> Int:
    var m = Int64(n)
    var t = Int64(nnzT)
    for j in range(m):
        Ap.unsafe_store(j, Int64(0))
    for p in range(t):
        var i = ti.unsafe_load(p)
        var j = tj.unsafe_load(p)
        Ap.unsafe_store(j + Int64(1), Ap.unsafe_load(j + Int64(1)) + Int64(1))
        if i != j:
            Ap.unsafe_store(i + Int64(1), Ap.unsafe_load(i + Int64(1)) + Int64(1))
    for j in range(m):
        Ap.unsafe_store(j + Int64(1), Ap.unsafe_load(j + Int64(1)) + Ap.unsafe_load(j))
    for p in range(t):
        var i = ti.unsafe_load(p)
        var j = tj.unsafe_load(p)
        var slot = Ap.unsafe_load(j)
        Ap.unsafe_store(j, slot + Int64(1))
        Ai.unsafe_store(slot, i)
        if i != j:
            slot = Ap.unsafe_load(i)
            Ap.unsafe_store(i, slot + Int64(1))
            Ai.unsafe_store(slot, j)
    # The scatter left Ap[j] holding the end of column j, i.e. the start of
    # column j+1. Shift left by one so the traversal sees proper offsets;
    # descending order keeps each source slot intact until it is read.
    for j in range(m, Int64(0), Int64(-1)):
        Ap.unsafe_store(j, Ap.unsafe_load(j - Int64(1)))
    Ap.unsafe_store(Int64(0), Int64(0))
    return Int(Ap.unsafe_load(m))


def permute_upper(
    n: Int,
    ti: IPtr,
    tj: IPtr,
    txr: Ptr,
    txi: Ptr,
    nnzT: Int,
    degree: IPtr,
    visited: IPtr,
    queue: IPtr,
    perm: IPtr,
    iperm: IPtr,
    count: IPtr,
    Ap: IPtr,
    Ai: IPtr,
    Axr: Ptr,
    Axi: Ptr,
    ApNew: IPtr,
) -> Int:
    var m = Int64(n)
    var t = Int64(nnzT)

    # Pass 1: the symmetric pattern, for the ordering.
    _ = adjacency_pattern(n, ti, tj, nnzT, Ap, Ai)

    # Pass 2: order the vertices.
    rcm(n, Ap, Ai, degree, visited, queue, perm)
    for k in range(m):
        iperm.unsafe_store(perm.unsafe_load(k), k)

    # Pass 3: the pattern in the permuted order. The caller already emits both
    # orientations of every entry -- geometry-central's own triplet lists are
    # symmetric -- so nothing is mirrored here; duplicates are summed, which is
    # what Eigen's setFromTriplets does.
    for j in range(m):
        Ap.unsafe_store(j, Int64(0))
    for p in range(t):
        var j = iperm.unsafe_load(tj.unsafe_load(p))
        Ap.unsafe_store(j + Int64(1), Ap.unsafe_load(j + Int64(1)) + Int64(1))
    for j in range(m):
        Ap.unsafe_store(j + Int64(1), Ap.unsafe_load(j + Int64(1)) + Ap.unsafe_load(j))
    for p in range(t):
        var i = iperm.unsafe_load(ti.unsafe_load(p))
        var j = iperm.unsafe_load(tj.unsafe_load(p))
        var vr = txr.unsafe_load(p)
        var vi = txi.unsafe_load(p)
        var slot = Ap.unsafe_load(j)
        Ap.unsafe_store(j, slot + Int64(1))
        Ai.unsafe_store(slot, i)
        Axr.unsafe_store(slot, vr)
        Axi.unsafe_store(slot, vi)

    # Pass 4: sort each column by row index and sum duplicates, compacting
    # leftwards. Offsets move, so they land in ApNew.
    var write = Int64(0)
    for j in range(m):
        # Pass 3 left Ap[j] holding the *end* of column j, so the start is
        # the previous entry.
        var lo = Int64(0)
        if j > Int64(0):
            lo = Ap.unsafe_load(j - Int64(1))
        var hi = Ap.unsafe_load(j)
        ApNew.unsafe_store(j, write)
        var a = lo
        while a < hi:
            var b = a + Int64(1)
            while b < hi:
                if Ai.unsafe_load(b) < Ai.unsafe_load(a):
                    var si = Ai.unsafe_load(a)
                    Ai.unsafe_store(a, Ai.unsafe_load(b))
                    Ai.unsafe_store(b, si)
                    var sv = Axr.unsafe_load(a)
                    Axr.unsafe_store(a, Axr.unsafe_load(b))
                    Axr.unsafe_store(b, sv)
                    var sw = Axi.unsafe_load(a)
                    Axi.unsafe_store(a, Axi.unsafe_load(b))
                    Axi.unsafe_store(b, sw)
                b += Int64(1)
            a += Int64(1)
        var read = lo
        while read < hi:
            var r = Ai.unsafe_load(read)
            var accr = Axr.unsafe_load(read)
            var acci = Axi.unsafe_load(read)
            read += Int64(1)
            while read < hi:
                if Ai.unsafe_load(read) != r:
                    break
                accr += Axr.unsafe_load(read)
                acci += Axi.unsafe_load(read)
                read += Int64(1)
            Ai.unsafe_store(write, r)
            Axr.unsafe_store(write, accr)
            Axi.unsafe_store(write, acci)
            write += Int64(1)
    ApNew.unsafe_store(m, write)
    return Int(write)
