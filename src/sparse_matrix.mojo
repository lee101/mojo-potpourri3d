# Eigen::SparseMatrix::setFromTriplets, for the two shapes this port builds: a real symmetric
# matrix and a complex symmetric one whose entries are stored interleaved (re, im). Upstream keeps
# the full matrix but hands the solver a SYMMETRIC view, so only the upper triangle including the
# diagonal is stored here as well. Duplicate triplets are summed, as setFromTriplets does.

from surface_mesh import PairSorter


# The triplet list upstream accumulates in a std::vector<Eigen::Triplet<T>>.
struct Triplets:
    var rows: List[Int]
    var cols: List[Int]
    var vr: List[Float64]
    var vi: List[Float64]

    def __init__(out self):
        self.rows = List[Int]()
        self.cols = List[Int]()
        self.vr = List[Float64]()
        self.vi = List[Float64]()

    def add(mut self, r: Int, c: Int, a: Float64) -> None:
        self.rows.append(r)
        self.cols.append(c)
        self.vr.append(a)
        self.vi.append(0.0)

    def add_complex(mut self, r: Int, c: Int, a: Float64, b: Float64) -> None:
        self.rows.append(r)
        self.cols.append(c)
        self.vr.append(a)
        self.vi.append(b)


def triplets_to_csc(
    n: Int,
    w: Int,
    t: Triplets,
    ap_base: Pointer[Int, AnyOrigin[mut=True]],
    ap_off: Int,
    ai_base: Pointer[Int, AnyOrigin[mut=True]],
    ai_off: Int,
    ax_base: Pointer[Float64, AnyOrigin[mut=True]],
    ax_off: Int,
) -> Int:
    var n_triplet = len(t.rows)
    var srt = PairSorter(n_triplet)
    var i = 0
    while i < n_triplet:
        srt.keys[i] = t.cols[i] * n + t.rows[i]
        srt.vals[i] = i
        i += 1
    srt.sort()

    var count = 0
    var last_col = -1

    var k = 0
    while k < n_triplet:
        var key = srt.keys[k]
        var col_i = key // n
        var row_i = key % n
        # Column offsets are only known once the previous columns are closed out, so a column with
        # no stored entries still has to get its offset written.
        if col_i != last_col:
            var c = last_col + 1
            if c < 0:
                c = 0
            while c <= col_i:
                ap_base.unsafe_store(ap_off + c, count)
                c += 1
            last_col = col_i
        var acc_r = 0.0
        var acc_i = 0.0
        var j = k
        while j < n_triplet and srt.keys[j] == key:
            var src = srt.vals[j]
            acc_r += t.vr[src]
            if w == 2:
                acc_i += t.vi[src]
            j += 1
        # The solver reads the upper triangle, matching the SYMMETRIC storage the upstream
        # PositiveDefiniteSolver asks SuiteSparse/Eigen for.
        if row_i <= col_i:
            ai_base.unsafe_store(ai_off + count, row_i)
            ax_base.unsafe_store(ax_off + count * w, acc_r)
            if w == 2:
                ax_base.unsafe_store(ax_off + count * w + 1, acc_i)
            count += 1
        k = j
    var tail = last_col + 1
    while tail <= n:
        ap_base.unsafe_store(ap_off + tail, count)
        tail += 1
    return count
