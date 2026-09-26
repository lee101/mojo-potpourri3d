"""COO -> symmetric, ordered, column-compressed form.

Eigen builds its sparse matrices with `setFromTriplets`, which sums duplicate
triplets, and geometry-central hands the result to a sparse Cholesky that wants
a symmetric matrix. This module does the same job: sum duplicates, mirror the
upper triangle into a full symmetric pattern, order the vertices with RCM, and
emit the result in the permuted order that `mojopp3d.linalg` factorizes.

Both triangles are stored, because the outer-product factorization reads
`A(i,k)` for `i > k` even though it only ever writes the lower factor.
"""

from std.utils import StaticTuple

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
    Ap: IPtr,
    Ai: IPtr,
    Axr: Ptr,
    Axi: Ptr,
    ApNew: IPtr,
    keys: IPtr,
    aux: IPtr,
    keybuf: IPtr,
    auxbuf: IPtr,
) -> Int:
    var m = Int64(n)
    var t = Int64(nnzT)

    # Pass 1: the symmetric pattern, for the ordering.
    _ = adjacency_pattern(n, ti, tj, nnzT, Ap, Ai)

    # Pass 2: order the vertices.
    rcm(n, Ap, Ai, degree, visited, queue, perm)
    for k in range(m):
        iperm.unsafe_store(perm.unsafe_load(k), k)

    # Pass 3: order the whole triplet list by (permuted column, permuted row)
    # in one stable radix sort, then emit it straight into the column-compressed
    # form. The caller already emits both orientations of every entry --
    # geometry-central's own triplet lists are symmetric -- so nothing is
    # mirrored here; duplicates are summed, which is what Eigen's
    # setFromTriplets does, in the order the sort leaves them in.
    #
    # Sorting the list as a whole rather than a column at a time is what makes
    # this linear: a point cloud's columns hold tens of entries each, where a
    # per-column comparison sort is quadratic and was the bulk of the build.
    for p in range(t):
        # Column in the high half, row in the low half, so both are recovered
        # by a shift and a mask rather than a division.
        var col = iperm.unsafe_load(tj.unsafe_load(p))
        var row = iperm.unsafe_load(ti.unsafe_load(p))
        keys.unsafe_store(p, (col << Int64(32)) + row)
        aux.unsafe_store(p, p)
    # Least significant digit first over the row index, which leaves the list
    # sorted by row; the counting sort by column below is stable, so the two
    # together order by (column, row) without disturbing the triplet order
    # within a duplicate run.
    var shift = Int64(0)
    var span = Int64(1)  # the row range one pass so far has covered
    var in_keys = keys
    var in_aux = aux
    var out_keys = keybuf
    var out_aux = auxbuf
    var passes = Int64(0)
    while True:  # at least one pass, so the row order is right even for m < 256
        _radix_pass(in_keys, in_aux, out_keys, out_aux, t, shift)
        var ktmp = in_keys
        in_keys = out_keys
        out_keys = ktmp
        var atmp = in_aux
        in_aux = out_aux
        out_aux = atmp
        passes += Int64(1)
        shift += Int64(8)
        span *= Int64(256)
        if span >= m:
            break
    if passes % Int64(2) == Int64(1):  # an odd count left the result in scratch
        for p in range(t):
            keys.unsafe_store(p, in_keys.unsafe_load(p))
            aux.unsafe_store(p, in_aux.unsafe_load(p))

    for c in range(m):
        degree.unsafe_store(c, Int64(0))
    for p in range(t):
        var col = keys.unsafe_load(p) >> Int64(32)
        degree.unsafe_store(col, degree.unsafe_load(col) + Int64(1))
    var acc = Int64(0)
    for c in range(m):
        var d = degree.unsafe_load(c)
        degree.unsafe_store(c, acc)
        visited.unsafe_store(c, acc)
        acc += d
    for p in range(t):
        var key = keys.unsafe_load(p)
        var col = key >> Int64(32)
        var slot = visited.unsafe_load(col)
        visited.unsafe_store(col, slot + Int64(1))
        keybuf.unsafe_store(slot, key)
        auxbuf.unsafe_store(slot, aux.unsafe_load(p))

    # Emit column by column, summing each run of equal keys.
    var read = Int64(0)
    var write = Int64(0)
    for j in range(m):
        ApNew.unsafe_store(j, write)
        var hi = visited.unsafe_load(j)  # the scatter left the column end here
        while read < hi:
            var key = keybuf.unsafe_load(read)
            var accr = 0.0
            var acci = 0.0
            while read < hi and keybuf.unsafe_load(read) == key:
                var src_idx = auxbuf.unsafe_load(read)
                accr += txr.unsafe_load(src_idx)
                acci += txi.unsafe_load(src_idx)
                read += Int64(1)
            Ai.unsafe_store(write, key & Int64(4294967295))
            Axr.unsafe_store(write, accr)
            Axi.unsafe_store(write, acci)
            write += Int64(1)
    ApNew.unsafe_store(m, write)
    return Int(write)


# One stable least-significant-digit pass of eight bits, swapping the key and
# its payload together.
def _radix_pass(src: IPtr, srcv: IPtr, dst: IPtr, dstv: IPtr, t: Int64, shift: Int64):
    # The histogram is a local rather than a buffer behind a pointer: the
    # compiler cannot otherwise prove that a store to `dst` does not alias it,
    # and the counter's read-modify-write then serialises the whole pass --
    # measured at 11.3 ms against 5.7 ms for the mesh operator.
    var hist = StaticTuple[Int64, 256]()
    for b in range(256):
        hist[b] = Int64(0)
    for p in range(t):
        var d = (src.unsafe_load(p) >> shift) & Int64(255)
        hist[d] += Int64(1)
    var acc = Int64(0)
    for b in range(256):
        var c = hist[b]
        hist[b] = acc
        acc += c
    for p in range(t):
        var key = src.unsafe_load(p)
        var d = (key >> shift) & Int64(255)
        var slot = hist[d]
        hist[d] = slot + Int64(1)
        dst.unsafe_store(slot, key)
        dstv.unsafe_store(slot, srcv.unsafe_load(p))
