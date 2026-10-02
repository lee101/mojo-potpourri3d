# A sparse symmetric LDL^T factorization and triangular solve, standing in for
# geometry-central's PositiveDefiniteSolver (Eigen::SimplicialLDLT, or CHOLMOD's simplicial LDLt).
#
# The matrix is supplied as the CSC storage of its UPPER TRIANGLE INCLUDING THE DIAGONAL, which is
# what the upstream solver asks SuiteSparse/Eigen for when it is handed a symmetric matrix.
# `w` is the number of float64 scalars per entry: 1 for a real symmetric matrix, 2 for a COMPLEX
# SYMMETRIC one stored interleaved (re, im). A complex symmetric (not Hermitian) matrix factors the
# same way with complex arithmetic, which is what the vector heat method's connection Laplacian
# needs; upstream reaches for a general SquareSolver (LU) there, which solves the same system.
#
# Three entry points, all taking buffers as Int addresses so they are callable from C:
#
#   chol_analyze    writes an RCM ordering and returns nnz(L)
#   chol_factorize  computes A = L D L^T in the permuted ordering
#   chol_solve      solves A x = b, both vectors in the caller's own ordering
#
# The numeric phase is the standard left-looking column-oriented sparse Cholesky. The permutation is
# applied on the way in, so callers never see it except through `perm`. The L values go straight
# into the caller's `lx` buffer and the pattern into `lp`/`li`, and the scatter accumulator is the
# caller's `dx` region (which this function already writes w*n doubles to), so the factorization
# allocates one array of n for the diagonal and nothing else.
#
# `chol_factorize` takes the pattern of L from `reuse_lp_addr`/`reuse_li_addr` when they are nonzero.
# That is how the heat method factorizes its two operators with one symbolic pass between them: they
# have the same index pattern and differ only in their values, so L has the same pattern for both.
# A borrowed pattern is checked as it is read and anything malformed falls back to computing it.

from std.math import isfinite

comptime IPtr = Pointer[Int, AnyOrigin[mut=True]]
comptime Dptr = Pointer[Float64, AnyOrigin[mut=True]]


# The upper triangle must be sorted by row within a column, and contain no index past the diagonal.
# Checked once, on the way in, and reported rather than trusted.
def _pattern_ok(n: Int, ap: IPtr, ai: IPtr) -> Bool:
    return _pattern_code(n, ap, ai) == 0


def _pattern_code(n: Int, ap: IPtr, ai: IPtr) -> Int:
    if ap.unsafe_load(0) != 0:
        return 1
    for k in range(n + 1):
        if ap.unsafe_load(k) < 0:
            return 2
    for j in range(n):
        var lo = ap.unsafe_load(j)
        var hi = ap.unsafe_load(j + 1)
        if lo > hi:
            return 3
        var prev = -1
        for p in range(lo, hi):
            var i = ai.unsafe_load(p)
            if i > j or i <= prev:
                return 4 + j
            prev = i
    return 0


# The permuted matrix. The numeric phase sweeps whole columns of A, so the full symmetric pattern
# is built here: the input is the upper triangle, and each off-diagonal entry of it stands for both
# (r,c) and (c,r). Diagonals are emitted once.
struct Permuted:
    var ap: List[Int]
    var ai: List[Int]
    var ax: List[Float64]
    var cursor: List[Int]

    def __init__(out self, n: Int, cap: Int, w: Int):
        self.ap = List[Int](length=n + 1, fill=0)
        self.ai = List[Int](length=cap, fill=0)
        self.ax = List[Float64](length=w * cap, fill=0.0)
        self.cursor = List[Int](length=n, fill=0)

    def build(
        mut self, n: Int, ap: IPtr, ai: IPtr, ax: Dptr, w: Int, pos: List[Int], with_values: Bool
    ) -> None:
        # The input is the upper triangle of a symmetric matrix, so an entry (r,c) of it stands for
        # both (r,c) and (c,r). After the permutation exactly one of the two lands in the upper
        # triangle, which is the one to emit.
        var c = 0
        while c < n:
            var p = ap.unsafe_load(c)
            var hi = ap.unsafe_load(c + 1)
            while p < hi:
                var r = ai.unsafe_load(p)
                self.cursor[pos[r]] += 1
                if r != c:
                    self.cursor[pos[c]] += 1
                p += 1
            c += 1
        var total = 0
        c = 0
        while c < n:
            var v = self.cursor[c]
            self.ap[c] = total
            self.cursor[c] = total
            total += v
            c += 1
        self.ap[n] = total

        c = 0
        while c < n:
            var p = ap.unsafe_load(c)
            var hi = ap.unsafe_load(c + 1)
            var pc = pos[c]
            while p < hi:
                var r = ai.unsafe_load(p)
                var pr = pos[r]
                var pairs = 1
                if r != c:
                    pairs = 2
                var q = 0
                while q < pairs:
                    var row = pr
                    var col = pc
                    if q == 1:
                        row = pc
                        col = pr
                    var slot = self.cursor[col]
                    self.cursor[col] += 1
                    self.ai[slot] = row
                    if with_values:
                        var s = 0
                        while s < w:
                            # The input is the upper triangle of a Hermitian matrix, so the mirrored
                            # position holds the CONJUGATE, not the same number. For w == 1 that is
                            # the identity and nothing changes.
                            var v = ax.unsafe_load(p * w + s)
                            if q == 1 and w == 2 and s == 1:
                                v = -v
                            self.ax[slot * w + s] = v
                            s += 1
                    q += 1
                p += 1
            c += 1

        # The scatter left every column holding both entries of each off-diagonal adjacency, in the
        # order the input happened to visit them, so each column has to be put into row order with its
        # duplicates summed. The rows present in a column are bounded by that column's length, so the
        # work is O(entries per column) rather than O(n) per column: the two length-n arrays the
        # per-column counting sort used to clear and re-prefix on every column are replaced by a
        # timestamp mark (never cleared, only compared) and a cursor that is initialised lazily from
        # the column's own start. Summing the duplicates keeps the scatter order, which is the order
        # the per-column sort summed them in.
        var snap_r = List[Int](length=total, fill=0)
        var snap_v = List[Float64](length=w * total, fill=0.0)
        var src_start = List[Int](length=n + 1, fill=0)
        c = 0
        while c <= n:
            src_start[c] = self.ap[c]
            c += 1
        c = 0
        while c < total:
            snap_r[c] = self.ai[c]
            snap_v[c * w] = self.ax[c * w]
            if w == 2:
                snap_v[c * w + 1] = self.ax[c * w + 1]
            c += 1

        # `mark[r] == c` means row r is present in column c; the mark array is never cleared, so the
        # sweep below costs one write per distinct row rather than one per row of the matrix.
        var mark = List[Int](length=n, fill=-1)
        # `place[r]` is the destination slot of row r within its column; valid only when marked.
        var place = List[Int](length=n, fill=0)
        # The distinct rows of a column never exceed that column's entry count, and a permuted column
        # is no wider than the input's widest column doubled. One scratch buffer, reused per column.
        var scratch = List[Int](length=n, fill=0)
        var out_ptr = 0
        c = 0
        while c < n:
            var lo = src_start[c]
            var hi = src_start[c + 1]
            # Distinct rows of this column, ascending, occupy [out_ptr, out_ptr + ndistinct). Collect
            # them into the shared scratch, then place them.
            var s = 0
            var p2 = lo
            while p2 < hi:
                var row = snap_r[p2]
                if mark[row] != c:
                    mark[row] = c
                    scratch[s] = row
                    s += 1
                p2 += 1
            # Insertion sort the distinct rows: a column holds only its own adjacencies, so this is
            # O(entries^2) per column in the worst case but tiny in practice, and it avoids the
            # length-n prefix sum entirely.
            var a = 1
            while a < s:
                var rv = scratch[a]
                var b = a - 1
                while b >= 0 and scratch[b] > rv:
                    scratch[b + 1] = scratch[b]
                    b -= 1
                scratch[b + 1] = rv
                a += 1
            # Give every distinct row its slot and zero its value.
            var t = 0
            while t < s:
                var row = scratch[t]
                place[row] = out_ptr + t
                self.ai[out_ptr + t] = row
                self.ax[(out_ptr + t) * w] = 0.0
                if w == 2:
                    self.ax[(out_ptr + t) * w + 1] = 0.0
                t += 1
            # Accumulate the scattered values into those slots, in scatter order.
            p2 = lo
            while p2 < hi:
                var slot = place[snap_r[p2]]
                self.ax[slot * w] += snap_v[p2 * w]
                if w == 2:
                    self.ax[slot * w + 1] += snap_v[p2 * w + 1]
                p2 += 1
            out_ptr += s
            self.ap[c] = out_ptr - s
            c += 1
        self.ap[n] = out_ptr


# The nonzero pattern of L, column by column, by a symbolic left-looking factorization: column k
# starts from the rows of A(:,k) ABOVE the diagonal (the input is the upper triangle, and L lives
# below it), then every column j already in the pattern contributes its own rows. That is the
# closure L(i,k) != 0 iff A(i,k) != 0 or there is a j with A(j,k) != 0 and L(i,j) != 0, which is
# exactly the pattern the numeric pass then fills.
#
# Rows within a column are kept ascending, and every entry is also threaded onto the list of its
# row, so row k of L (the columns j < k with L(j,k) != 0) can be walked while column k is being
# built. `lnext` chains those lists in ascending column order, which is the order the left-looking
# solve needs: v[j] is only final once every column before j has been swept.
struct LPattern:
    var lp: List[Int]
    var lrows: List[Int]
    var lnext: List[Int]
    var lcol: List[Int]
    var rfirst: List[Int]
    var rlast: List[Int]
    # The seeds of each column of L: the rows i > k with A(i,k) != 0.
    var lap: List[Int]
    var lai: List[Int]

    def __init__(out self, n: Int):
        self.lp = List[Int](length=n + 1, fill=0)
        self.lrows = List[Int]()
        self.lnext = List[Int]()
        self.lcol = List[Int]()
        self.rfirst = List[Int](length=n, fill=-1)
        self.rlast = List[Int](length=n, fill=-1)
        self.lap = List[Int](length=n + 1, fill=0)
        self.lai = List[Int]()

    # `permuted` holds the full symmetric matrix, so column k carries both the rows above the diagonal
    # and the rows below it. The seeds of column k of L are the rows i > k with A(i,k) != 0, which is
    # the strictly lower part of that column, so bucket k lists those rows in ascending order.
    def transpose(mut self, n: Int, permuted: Permuted) -> None:
        var c = 0
        while c < n:
            var p = permuted.ap[c]
            var hi = permuted.ap[c + 1]
            while p < hi:
                var i = permuted.ai[p]
                if i > c:
                    self.lap[c] += 1
                p += 1
            c += 1
        var total = 0
        c = 0
        while c < n:
            var v = self.lap[c]
            self.lap[c] = total
            total += v
            c += 1
        self.lap[n] = total
        self.lai = List[Int](length=total, fill=0)
        var cursor = List[Int](length=n, fill=0)
        c = 0
        while c < n:
            cursor[c] = self.lap[c]
            c += 1
        c = 0
        while c < n:
            var p = permuted.ap[c]
            var hi = permuted.ap[c + 1]
            while p < hi:
                var i = permuted.ai[p]
                if i > c:
                    self.lai[cursor[c]] = i
                    cursor[c] += 1
                p += 1
            c += 1
        # `permuted` holds each column in ascending row order, so bucket c of `lai` comes out in
        # ascending order too and needs no sort.

    # Column k of L holds the rows i > k with L(i,k) != 0. L(i,k) is nonzero when A(i,k) is, or
    # when some column j with L(j,k) != 0 has L(i,j) != 0; the first case is the seed set, the
    # second is the union of the patterns of the columns listed in row k of L. Processing k in
    # ascending order is what makes this well defined: every L(j,k) with j < k is already placed,
    # and so is the whole of column j, which the union walks. `mark` is a timestamp rather than a
    # flag, so the marks are never cleared. `rowcol` threads the columns of each row together in
    # `lnext`/`lcol` order, which `append_column` already maintains as ascending columns.
    def compute(mut self, n: Int, permuted: Permuted) -> None:
        self.transpose(n, permuted)
        var mark = List[Int](length=n, fill=-1)
        # Column k of L holds at most the rows k+1 .. n-1, so one length-n scratch serves every
        # column. It used to be allocated per column, which is n heap allocations and the growth
        # reallocations that come with append.
        var cur = List[Int](length=n, fill=0)
        var k = 0
        while k < n:
            var cnt = 0
            var p = self.lap[k]
            var lap_hi = self.lap[k + 1]
            while p < lap_hi:
                var i = self.lai[p]
                if i > k and mark[i] != k:
                    mark[i] = k
                    cur[cnt] = i
                    cnt += 1
                p += 1
            var t = self.rfirst[k]
            while t >= 0:
                var j = self.lcol[t]
                var q = self.lp[j]
                var hi = self.lp[j + 1]
                while q < hi:
                    var i2 = self.lrows[q]
                    if i2 > k and mark[i2] != k:
                        mark[i2] = k
                        cur[cnt] = i2
                        cnt += 1
                    q += 1
                t = self.lnext[t]
            var a = 1
            while a < cnt:
                var rv = cur[a]
                var b = a - 1
                while b >= 0 and cur[b] > rv:
                    cur[b + 1] = cur[b]
                    b -= 1
                cur[b + 1] = rv
                a += 1
            self.append_column(k, cur, cnt)
            # Column k is closed, so lp[k+1] is known. Without it the union above would read
            # lp[j+1] as 0 for j == k - 1 and walk an empty column k-1, dropping its rows.
            self.lp[k + 1] = len(self.lrows)
            k += 1
        self.lp[n] = len(self.lrows)

    # Appends the sorted column `cur[0:cnt]` to `lrows` and threads every entry onto the list of its
    # row. The lists are walked by prepending nothing and appending at the tail, so they come out in
    # ascending column order.
    def append_column(mut self, k: Int, cur: List[Int], cnt: Int) -> None:
        self.lp[k] = len(self.lrows)
        var t = 0
        while t < cnt:
            var row = cur[t]
            self.lrows.append(row)
            self.lnext.append(-1)
            self.lcol.append(k)
            if self.rfirst[row] < 0:
                self.rfirst[row] = len(self.lrows) - 1
            else:
                self.lnext[self.rlast[row]] = len(self.lrows) - 1
            self.rlast[row] = len(self.lrows) - 1
            t += 1

    # Rebuilds the per-row chains of L from `lp`/`lrows`. Used when the pattern was read back from a
    # caller, where only the column pointers and the row indices came across: `lcol`/`lnext` have to
    # be rebuilt before the numeric phase can walk row k of L.
    def link_rows(mut self, n: Int) -> None:
        var i = 0
        while i < n:
            self.rfirst[i] = -1
            self.rlast[i] = -1
            i += 1
        self.lnext = List[Int](length=len(self.lrows), fill=-1)
        self.lcol = List[Int](length=len(self.lrows), fill=0)
        var k = 0
        while k < n:
            var q = self.lp[k]
            var hi = self.lp[k + 1]
            while q < hi:
                var row = self.lrows[q]
                self.lcol[q] = k
                if self.rfirst[row] < 0:
                    self.rfirst[row] = q
                else:
                    self.lnext[self.rlast[row]] = q
                self.rlast[row] = q
                q += 1
            k += 1

    def nnz(self) -> Int:
        return len(self.lrows)


# Reverse Cuthill-McKee. Deterministic: each BFS starts at the lowest unvisited vertex and visits
# neighbours in ascending order; the visited order is then reversed.
def _rcm(n: Int, ap: IPtr, ai: IPtr, perm: IPtr) -> None:
    var count = List[Int](length=n, fill=0)
    var j = 0
    while j < n:
        var p = ap.unsafe_load(j)
        var hi = ap.unsafe_load(j + 1)
        while p < hi:
            var i = ai.unsafe_load(p)
            if i != j:
                count[i] += 1
                count[j] += 1
            p += 1
        j += 1
    var start = List[Int](length=n + 1, fill=0)
    j = 0
    while j < n:
        start[j + 1] = start[j] + count[j]
        j += 1
    var adj = List[Int](length=2 * start[n], fill=0)
    var cursor = List[Int](length=n, fill=0)
    j = 0
    while j < n:
        cursor[j] = start[j]
        j += 1
    j = 0
    while j < n:
        var p = ap.unsafe_load(j)
        var hi = ap.unsafe_load(j + 1)
        while p < hi:
            var i = ai.unsafe_load(p)
            if i != j:
                adj[cursor[i]] = j
                cursor[i] += 1
                adj[cursor[j]] = i
                cursor[j] += 1
            p += 1
        j += 1

    var visited = List[Bool](length=n, fill=False)
    var order = List[Int]()
    var queue = List[Int]()
    var root = 0
    while root < n:
        if not visited[root]:
            queue.append(root)
            visited[root] = True
            var head = 0
            while head < len(queue):
                var cur = queue[head]
                head += 1
                order.append(cur)
                var p = start[cur]
                var hi = start[cur + 1]
                while p < hi:
                    var nb = adj[p]
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
                    p += 1
            queue = List[Int]()
        root += 1
    var k = 0
    while k < len(order):
        perm.unsafe_store(k, order[len(order) - 1 - k])
        k += 1


# pos[original] = position after the permutation, and the permutation is checked for the bijection
# it has to be: a repeated or out-of-range index would silently factor a different matrix.
def _positions(n: Int, perm: IPtr) -> List[Int]:
    var pos = List[Int](length=n, fill=0)
    var seen = List[Bool](length=n, fill=False)
    var k = 0
    while k < n:
        var p = perm.unsafe_load(k)
        if p < 0 or p >= n or seen[p]:
            return List[Int]()
        seen[p] = True
        pos[p] = k
        k += 1
    return pos^


# Reads a pattern of L back out of a caller's lp/li pair, checking the parts the numeric phase
# relies on for memory safety as it goes. A pattern that does not check out comes back empty
# (lp[n] == 0), which is the caller's signal to compute it instead.
def _read_pattern(n: Int, rlp: IPtr, rli: IPtr) -> LPattern:
    var pat = LPattern(n)
    if rlp.unsafe_load(0) != 0:
        return pat^
    var nnz = rlp.unsafe_load(n)
    if nnz <= 0:
        return pat^
    var c = 0
    while c <= n:
        var v = rlp.unsafe_load(c)
        if v < 0 or v > nnz or (c > 0 and v < rlp.unsafe_load(c - 1)):
            return pat^
        c += 1
    var q = 0
    while q < nnz:
        var row = rli.unsafe_load(q)
        if row <= 0 or row >= n:
            return pat^
        pat.lrows.append(row)
        q += 1
    c = 0
    while c <= n:
        pat.lp[c] = rlp.unsafe_load(c)
        c += 1
    pat.link_rows(n)
    return pat^


def chol_analyze(
    n: Int, w: Int, ap_addr: Int, ai_addr: Int, ax_addr: Int, perm_addr: Int
) -> Int:
    if n <= 0 or (w != 1 and w != 2):
        return -1
    if perm_addr == 0 or ax_addr == 0:
        return -1
    var ap = IPtr(unsafe_from_address=ap_addr)
    var ai = IPtr(unsafe_from_address=ai_addr)
    var perm = IPtr(unsafe_from_address=perm_addr)
    if not _pattern_ok(n, ap, ai):
        return -1
    _rcm(n, ap, ai, perm)
    var pos = _positions(n, perm)
    if len(pos) == 0:
        return -1
    var permuted = Permuted(n, 2 * ap.unsafe_load(n) + n, w)
    permuted.build(n, ap, ai, Dptr(unsafe_from_address=ax_addr), w, pos, True)
    var pat = LPattern(n)
    pat.compute(n, permuted)
    return pat.nnz()

# The diagonal comes back with the status rather than through a parameter: a List is move-only, so a
# function cannot both fill one it was handed and hand it on.
struct Factor:
    var code: Int
    var d: List[Float64]

    def __init__(out self, code: Int, n: Int, w: Int):
        self.code = code
        self.d = List[Float64](length=w * n, fill=0.0)


# The numeric phase, real symmetric. The L values land in the caller's `lx`, the diagonal in the
# returned Factor, and `wbuf` (the caller's dx region) is the scatter accumulator.
def _factorize_real(
    n: Int,
    pat: LPattern,
    permuted: Permuted,
    li: IPtr,
    lx: Dptr,
    wbuf: Dptr,
) -> Factor:
    var out = Factor(0, n, 1)
    var k = 0
    while k < n:
        # Left-looking Cholesky. Column k of A satisfies
        #   A(:,k) = L(:,0:k) v,   v[j] = D(j) L(j,k) for j < k,  v[k] = D(k)
        # so zero the scatter buffer on everything the solve reads, accumulate the column, then
        # sweep the columns j of L below the diagonal in ascending order.
        var t = pat.lp[k]
        var hi = pat.lp[k + 1]
        while t < hi:
            wbuf.unsafe_store(li.unsafe_load(t), 0.0)
            t += 1
        wbuf.unsafe_store(k, 0.0)
        var r = pat.rfirst[k]
        while r >= 0:
            wbuf.unsafe_store(pat.lcol[r], 0.0)
            r = pat.lnext[r]

        var p = permuted.ap[k]
        var phi = permuted.ap[k + 1]
        while p < phi:
            var row = permuted.ai[p]
            wbuf.unsafe_store(row, wbuf.unsafe_load(row) + permuted.ax[p])
            p += 1

        r = pat.rfirst[k]
        while r >= 0:
            var col = pat.lcol[r]
            # v[col] = D(col) L(col,k) = wbuf[col] once the earlier columns are off it
            var zr = wbuf.unsafe_load(col)
            wbuf.unsafe_store(col, 0.0)
            var q = pat.lp[col]
            var chi = pat.lp[col + 1]
            while q < chi:
                var i2 = li.unsafe_load(q)
                wbuf.unsafe_store(i2, wbuf.unsafe_load(i2) - lx.unsafe_load(q) * zr)
                q += 1
            r = pat.lnext[r]

        var dk = wbuf.unsafe_load(k)
        if not isfinite(dk) or dk <= 0.0:
            return Factor(1, n, 1)
        out.d[k] = dk

        t = pat.lp[k]
        hi = pat.lp[k + 1]
        while t < hi:
            lx.unsafe_store(t, wbuf.unsafe_load(li.unsafe_load(t)) / dk)
            t += 1
        k += 1
    return out^


# The numeric phase, complex symmetric (see the header).
def _factorize_complex(
    n: Int,
    pat: LPattern,
    permuted: Permuted,
    li: IPtr,
    lx: Dptr,
    wbuf: Dptr,
) -> Factor:
    var out = Factor(0, n, 2)
    var k = 0
    while k < n:
        var t = pat.lp[k]
        var hi = pat.lp[k + 1]
        while t < hi:
            var row = li.unsafe_load(t)
            wbuf.unsafe_store(2 * row, 0.0)
            wbuf.unsafe_store(2 * row + 1, 0.0)
            t += 1
        wbuf.unsafe_store(2 * k, 0.0)
        wbuf.unsafe_store(2 * k + 1, 0.0)
        var r = pat.rfirst[k]
        while r >= 0:
            var c2 = 2 * pat.lcol[r]
            wbuf.unsafe_store(c2, 0.0)
            wbuf.unsafe_store(c2 + 1, 0.0)
            r = pat.lnext[r]

        var p = permuted.ap[k]
        var phi = permuted.ap[k + 1]
        while p < phi:
            var row2 = 2 * permuted.ai[p]
            wbuf.unsafe_store(row2, wbuf.unsafe_load(row2) + permuted.ax[2 * p])
            wbuf.unsafe_store(row2 + 1, wbuf.unsafe_load(row2 + 1) + permuted.ax[2 * p + 1])
            p += 1

        r = pat.rfirst[k]
        while r >= 0:
            var col = pat.lcol[r]
            var zr = wbuf.unsafe_load(2 * col)
            var zi = wbuf.unsafe_load(2 * col + 1)
            wbuf.unsafe_store(2 * col, 0.0)
            wbuf.unsafe_store(2 * col + 1, 0.0)
            var q = pat.lp[col]
            var chi = pat.lp[col + 1]
            while q < chi:
                var i2 = 2 * li.unsafe_load(q)
                var lr = lx.unsafe_load(2 * q)
                var lim = lx.unsafe_load(2 * q + 1)
                wbuf.unsafe_store(i2, wbuf.unsafe_load(i2) - lr * zr + lim * zi)
                wbuf.unsafe_store(i2 + 1, wbuf.unsafe_load(i2 + 1) - lr * zi - lim * zr)
                q += 1
            r = pat.lnext[r]

        var dk = wbuf.unsafe_load(2 * k)
        var dki = wbuf.unsafe_load(2 * k + 1)
        if not isfinite(dk) or dk <= 0.0:
            return Factor(1, n, 2)
        # A complex-symmetric matrix with a real positive diagonal factors to a real diagonal, up to
        # roundoff, so only reject a genuinely complex pivot.
        if abs(dki) > 1e-8 * abs(dk):
            return Factor(1, n, 2)
        out.d[2 * k] = dk
        out.d[2 * k + 1] = dki

        t = pat.lp[k]
        hi = pat.lp[k + 1]
        while t < hi:
            var row = li.unsafe_load(t)
            # Hermitian: L(k,i) = conj(wbuf[i]) / D(k)
            lx.unsafe_store(2 * t, wbuf.unsafe_load(2 * row) / dk)
            lx.unsafe_store(2 * t + 1, -wbuf.unsafe_load(2 * row + 1) / dk)
            t += 1
        k += 1
    return out^


def chol_factorize(
    n: Int,
    w: Int,
    ap_addr: Int,
    ai_addr: Int,
    ax_addr: Int,
    perm_addr: Int,
    lp_addr: Int,
    li_addr: Int,
    lx_addr: Int,
    dx_addr: Int,
    lx_cap: Int = -1,
    reuse_lp_addr: Int = 0,
    reuse_li_addr: Int = 0,
) -> Int:
    if n <= 0 or (w != 1 and w != 2):
        return 3
    var ap = IPtr(unsafe_from_address=ap_addr)
    var ai = IPtr(unsafe_from_address=ai_addr)
    var ax = Dptr(unsafe_from_address=ax_addr)
    var perm = IPtr(unsafe_from_address=perm_addr)
    var lp = IPtr(unsafe_from_address=lp_addr)
    var li = IPtr(unsafe_from_address=li_addr)
    var lx = Dptr(unsafe_from_address=lx_addr)
    var dx = Dptr(unsafe_from_address=dx_addr)
    var pcode = _pattern_code(n, ap, ai)
    if pcode != 0:
        return 300 + pcode

    var pos = _positions(n, perm)
    if len(pos) == 0:
        return 32

    var permuted = Permuted(n, 2 * ap.unsafe_load(n) + n, w)
    permuted.build(n, ap, ai, ax, w, pos, True)

    var pat = LPattern(n)
    if reuse_lp_addr != 0 and reuse_li_addr != 0:
        pat = _read_pattern(
            n, IPtr(unsafe_from_address=reuse_lp_addr), IPtr(unsafe_from_address=reuse_li_addr)
        )
    if pat.lp[n] == 0:
        pat = LPattern(n)
        pat.compute(n, permuted)
    var nnzl = pat.nnz()
    if nnzl == 0:
        return 5000 + pcode
    # Guard against a factor buffer that is too small for the factor, rather than overrunning it.
    if lx_cap >= 0 and nnzl > lx_cap:
        return 2
    var k = 0
    while k <= n:
        lp.unsafe_store(k, pat.lp[k])
        k += 1
    k = 0
    while k < nnzl:
        li.unsafe_store(k, pat.lrows[k])
        k += 1

    var res: Factor
    if w == 1:
        res = _factorize_real(n, pat, permuted, li, lx, dx)
    else:
        res = _factorize_complex(n, pat, permuted, li, lx, dx)
    if res.code != 0:
        return res.code
    k = 0
    while k < w * n:
        dx.unsafe_store(k, res.d[k])
        k += 1
    return 0


def chol_solve(
    n: Int,
    w: Int,
    lp_addr: Int,
    li_addr: Int,
    lx_addr: Int,
    dx_addr: Int,
    perm_addr: Int,
    b_addr: Int,
    x_addr: Int,
) -> Int:
    if n <= 0 or (w != 1 and w != 2):
        return 1
    var lp = IPtr(unsafe_from_address=lp_addr)
    var li = IPtr(unsafe_from_address=li_addr)
    var lx = Dptr(unsafe_from_address=lx_addr)
    var dx = Dptr(unsafe_from_address=dx_addr)
    var perm = IPtr(unsafe_from_address=perm_addr)
    var b = Dptr(unsafe_from_address=b_addr)
    var x = Dptr(unsafe_from_address=x_addr)
    if lp.unsafe_load(0) != 0:
        return 1

    # Forward substitution through L, then the diagonal, then back through L^T.
    var y = List[Float64](length=w * n, fill=0.0)
    var k = 0
    while k < n:
        var p = perm.unsafe_load(k)
        var s0 = 0
        while s0 < w:
            y[k * w + s0] = b.unsafe_load(p * w + s0)
            s0 += 1
        k += 1
    k = 0
    while k < n:
        var t = lp.unsafe_load(k)
        var hi = lp.unsafe_load(k + 1)
        while t < hi:
            var i = li.unsafe_load(t)
            var lr = lx.unsafe_load(t * w)
            y[i * w] -= lr * y[k * w]
            if w == 2:
                y[i * w + 1] -= lr * y[k * w + 1] + lx.unsafe_load(t * w + 1) * y[k * w]
            t += 1
        k += 1
    k = 0
    while k < n:
        y[k * w] /= dx.unsafe_load(k * w)
        if w == 2:
            y[k * w + 1] /= dx.unsafe_load(k * w)
        k += 1
    # Backward substitution through L^H (L^T when real). Walking the column pointers of L and
    # gathering z[i] for each stored L(i,k) is exactly sum_j L(j,k) z[j] for j > k, which is the
    # L^H solve; because the sweep is descending, every z[i] it reads is already final.
    k = n - 1
    while k >= 0:
        var sr = y[k * w]
        var si = 0.0
        var t3 = lp.unsafe_load(k)
        var hi3 = lp.unsafe_load(k + 1)
        while t3 < hi3:
            var i3 = li.unsafe_load(t3)
            var lr = lx.unsafe_load(t3 * w)
            sr -= lr * y[i3 * w]
            if w == 2:
                var lim = lx.unsafe_load(t3 * w + 1)
                sr += lim * y[i3 * w + 1]
                si -= lr * y[i3 * w + 1] - lim * y[i3 * w]
            t3 += 1
        y[k * w] = sr
        if w == 2:
            y[k * w + 1] = si
        k -= 1

    k = 0
    while k < n:
        var p2 = perm.unsafe_load(k)
        var s2 = 0
        while s2 < w:
            x.unsafe_store(p2 * w + s2, y[k * w + s2])
            s2 += 1
        k += 1
    return 0
