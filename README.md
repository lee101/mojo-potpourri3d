# mojo-potpourri3d

A Mojo port of the compute-bound parts of [potpourri3d](https://github.com/nmwsharp/potpourri3d):
geodesic distance by the heat method, on meshes and on point clouds; tangent-vector
transport; and the mesh Laplacians and triangulations they are built on.

The Python API is upstream's -- same module layout, same class and function names,
same argument order and defaults -- so it is a drop-in for the covered subset.
`tests/test_parity.py` asserts the signatures match upstream's exactly, and
asserts the numbers do too.

## Usage

    import numpy as np
    import mojo_potpourri3d as pp3d

    V, F = pp3d.read_mesh("bunny.ply")   # not covered; bring your own loader
    d = pp3d.compute_distance(V, F, 0)   # heat-method distance to vertex 0

This block runs as written on the icosahedron built inline below
(`tests/test_readme_example.py` executes it, so it cannot rot):

    import numpy as np
    import mojo_potpourri3d as pp3d

    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                  [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    F = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    V /= np.linalg.norm(V, axis=1, keepdims=True)      # unit icosahedron

    d = pp3d.compute_distance(V, F, 0)                 # (12,) distances
    s = pp3d.MeshHeatMethodDistanceSolver(V, F, use_robust=False)
    dm = s.compute_distance_multisource([0, 3])        # (12,) multi-source
    L = pp3d.cotan_laplacian(V, F)                     # (12, 12) csr_matrix

    vs = pp3d.MeshVectorHeatSolver(V, F, use_intrinsic_delaunay=False)
    bX, bY, n = vs.get_tangent_frames()                # (12, 3) each
    C = vs.get_connection_laplacian()                  # (12, 12) complex system
    u = vs.transport_tangent_vector(0, [1.0, 0.0])      # (12, 2) transported
    h = vs.extend_scalar([0, 3], [1.0, 1.0])           # (12,) extended

    P = np.random.default_rng(0).normal(size=(300, 3))  # a point cloud
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    pc = pp3d.PointCloudHeatSolver(P)
    pd = pc.compute_distance(0)                        # (300,) distances
    tri = pp3d.PointCloudLocalTriangulation(P).get_local_triangulation()

`d[0]` is 0, the unit-icosahedron's antipodal distance comes out at ~2.732 (the
true geodesic is pi = 3.1416, so this is the usual heat-method underestimate on a
mesh this coarse), `abs(L @ ones).max()` is ~1e-15, and every row of `u` has
unit norm to ~1e-16. `use_intrinsic_delaunay` is set to `False` only because this
port does not cover the intrinsic-Delaunay preprocessing; see the coverage table.
`compute_distance` above is the module-level one, which takes no flags and so runs
the robust Laplacian, as upstream does.

## Install

    pixi install
    pixi run build     # mojo build --emit shared-lib -> dist/libmojo-potpourri3d.so
    pixi run test
    pixi run bench

`pixi install` pulls the pinned Mojo toolchain, NumPy, SciPy and upstream
`potpourri3d` itself (from PyPI, which is the only place a CPython 3.13 wheel
exists; conda-forge only builds it for 3.11).

## What is covered

| upstream | status |
| --- | --- |
| `cotan_laplacian(V, F, denom_eps=0.)` | exact parity |
| `face_areas(V, F)` | exact parity |
| `vertex_areas(V, F)` | exact parity |
| `MeshHeatMethodDistanceSolver.compute_distance` | exact parity, robust and plain |
| `MeshHeatMethodDistanceSolver.compute_distance_multisource` | exact parity |
| `compute_distance`, `compute_distance_multisource` | exact parity |
| `MeshVectorHeatSolver.extend_scalar` | exact parity |
| `MeshVectorHeatSolver.get_tangent_frames` | exact parity |
| `MeshVectorHeatSolver.get_connection_laplacian` | exact parity |
| `MeshVectorHeatSolver.transport_tangent_vector(s)` | exact parity except one vertex (see below) |
| `PointCloudLocalTriangulation.get_local_triangulation` | same triangles, rotated order (see below) |
| `PointCloudHeatSolver.compute_distance` | same pipeline, agrees to a fraction of a percent |
| `PointCloudHeatSolver.compute_distance_multisource` | same |
| `validate_mesh`, `validate_points` | verbatim from upstream |

The robust Laplacian (`use_robust=True`, which is upstream's *default* for the
distance solver) is the real thing: `buildIntrinsicTuftedCover`,
`mollifyIntrinsic` and `flipToDelaunay` are ported, and the distances agree with
upstream's robust distances to 1e-15 relative. `test_robust_path_runs_the_delaunay_flips`
perturbs an icosphere so the flip pass has 1169 flips to do and pins the parity
through them.

### What is not covered, and why

* **The emitted order of a point cloud's local triangulation.** The triangles
  themselves are upstream's, exactly, for every point of every non-degenerate
  cloud tested (sphere, scaled sphere, torus; see
  `test_local_triangulation_matches_upstream`). The *order* is not, and it is
  not reproducible. Upstream emits a point's triangles in angular order around
  the centre, in a tangent frame built from a normal that comes from Eigen's
  `JacobiSVD` on the 3 x k neighbourhood matrix. The normal this port computes
  -- the smallest eigenvector of the 3x3 Gram matrix, by cyclic Jacobi
  rotations -- agrees with LAPACK's `eigh` on the same matrix to 2e-6 degrees,
  but not with the SVD, and the angular order only has to be off by a
  thousandth of a radian for the `-pi` cut to fall between different
  neighbours. Reproducing the order would mean reproducing Eigen's SVD
  operation for operation, which is not a port, it is a copy.

  This is not cosmetic downstream: `buildIntrinsicTuftedCover` glues the cover
  in the order the faces arrive, so a rotated emission order gives a different
  (equally valid) Delaunay triangulation of the cover, and hence a slightly
  different heat operator. Measured on 200- and 600-point clouds, the distance
  field agrees with upstream's to 0.25% median and 2.4% worst case, with
  correlation above 0.9999. `test_point_cloud_distance_agrees_with_upstream`
  pins that bound rather than pretending to roundoff parity.

* **Exactly cocircular neighbourhoods.** With one, the in-circle determinant is
  mathematically zero and its computed sign is rounding noise, so the
  degeneracy heuristic decides nothing and the fan is kept or dropped by
  luck. Upstream's own test cloud, the "cartwheel" (a centre point on a ring of
  30), is exactly this: upstream keeps the full 30-triangle fan, this port
  keeps none of it. `test_cartwheel_ring_is_cocircular` pins the divergence
  rather than hiding it. `inCircleTest` itself is a faithful transcription of
  Eigen's six-term 4x4 cofactor expansion, so this is the input perturbation
  and not a different predicate.

* **The point cloud vector-heat methods** -- `extend_scalar`,
  `get_tangent_frames`, `transport_tangent_vector(s)`, `compute_log_map`,
  `compute_signed_distance`. They need the point cloud's *extrinsic* tangent
  frames and the vertex connection Laplacian built from the parallel transport
  between them (`computeConnectionLaplacian`, `transportBetweenOriented`), which
  is a different set of kernels from the mesh ones. They are present and raise
  `NotImplementedError`.

* **`MeshVectorHeatSolver.compute_log_map`.** Raises `NotImplementedError`. All
  three strategies (`VectorHeat`, `AffineLocal`, `AffineAdaptive`) end in a
  factorization of a singular or indefinite operator -- the affine connection
  heat operator for the two affine strategies, the unshifted cotan Laplacian for
  `VectorHeat` -- which upstream hands to Eigen's `SparseLU` with partial
  pivoting. This port has a serial LDL^T with a pivot floor, and on those
  operators it returns a visibly different answer. Rather than ship a wrong
  number, the method raises and says so.

* **`use_intrinsic_delaunay=True`** on the vector heat solver. That flag runs a
  Delaunay flip pass on the *mesh* to build an intrinsic triangulation for the
  connection Laplacian, upstream via `IntrinsicTriangulation` and
  `IntegerCoordinatesIntrinsicTriangulation`; the simple-IDT flipper here is
  the one the heat method uses, which is not the same operator. The flag is
  accepted and stored and the extrinsic path is run, so passing `True` or
  `False` gives bit-identical results here. Parity is asserted with
  `use_intrinsic_delaunay=False`, the configuration that isolates the solver
  from that preprocessing.

* **One vertex of `transport_tangent_vector`.** On a sphere, the vertex directly
  opposite the source is the antipode, where the transported direction is
  numerically degenerate; normalizing there amplifies rounding until the sign
  flips. Every other vertex agrees to 1e-9. `test_transport_tangent_vector_matches_except_the_antipode`
  asserts exactly that: at most one vertex may differ, and all others must agree.

* **The brute-force nearest-neighbour search.** `NearestNeighborFinder` is a
  nanoflann KD-tree; here it is a partial selection scan, O(n^2 k) instead of
  O(n log n). The neighbour lists agree (ties broken by index), which
  `test_local_triangulation_matches_upstream` checks indirectly, and it is why
  the local triangulation row below is a loss.

* Everything else in `potpourri3d` -- signed heat, fast marching, marching
  triangles, edge-flip geodesics, the geodesic tracer, the polygon-mesh solvers,
  `read_mesh` / `write_mesh` -- is out of scope for this port and absent.

## Fidelity

`src/mojopp3d/` is laid out to be read next to the geometry-central sources it
comes from, in the same order, with the same function names:

| file | upstream file |
| --- | --- |
| `vector.mojo` | `include/geometrycentral/utilities/vector{2,3}.{h,ipp}` |
| `mesh.mojo` | `src/surface/surface_mesh.cpp` (the face-soup constructor) |
| `general_mesh.mojo` | `src/surface/surface_mesh.cpp` (the general mesh: `duplicateFace`, `invertOrientation`, `separateToNewEdge`, `flip`) |
| `tufted.mojo` | `src/surface/tufted_laplacian.cpp`, `src/surface/intrinsic_mollification.cpp`, `src/surface/simple_idt.cpp` |
| `local_triangulation.mojo` | `src/pointcloud/local_triangulation.cpp`, `src/pointcloud/neighborhoods.cpp`, `src/pointcloud/point_position_geometry.cpp`, `src/utilities/elementary_geometry.cpp` |
| `intrinsic_geometry.mojo` | `src/surface/intrinsic_geometry_interface.cpp` |
| `embedded_geometry.mojo` | `src/surface/embedded_geometry_interface.cpp` |
| `heat_method.mojo` | `src/surface/heat_method_distance.cpp` |
| `vector_heat_method.mojo` | `src/surface/vector_heat_method.cpp` |
| `linalg.mojo`, `sparse.mojo` | no upstream counterpart; replaces CHOLMOD |
| `api.mojo` | `potpourri3d/mesh.py` (the NumPy free functions) |
| `capi.mojo` | the `abi("C")` boundary, nothing else |

The branch structure and the order of the arithmetic follow upstream statement by
statement. Where a faithful port is impossible the divergence is listed above.

## How it works

**FFI.** `build/build.sh` runs `mojo build --emit shared-lib -I src src/capi.mojo`,
producing `dist/libmojo-potpourri3d.so`. `capi.mojo` is the only compilation
unit; it imports the kernel modules and re-exports them under `mpp3d_` names
with an explicit `abi("C")` effect, which Mojo requires for `@export`. Because
`@export` rejects parametric functions and a pointer with an inferred origin is
parametric, every buffer crosses as an `Int` address and the pointers are
rebuilt inside the wrapper. `python/mojo_potpourri3d/_lib.py` holds the ctypes
signatures and builds the library on first import if it is missing or stale;
`tests/test_parity.py::test_ffi_signatures_match_the_exports` pins every ctypes
arity against the Mojo source, because an arity mismatch is a silent
memory-corrupting bug.

**Memory layout.** NumPy owns every array. The mesh is a halfedge structure
mirroring `SurfaceMesh`: face `f` owns halfedges `3f, 3f+1, 3f+2` and the vertex
indices are the caller's own, so `vertexIndices[v] == v` exactly as upstream.
Per-vertex quantities are `(n,)` float64; per-halfedge quantities are
`(3 * n_faces,)` float64; per-edge quantities are stored per halfedge, which is
the same number of entries because every edge quantity satisfies
`L(i,j) == L(twin,j)`. The `HalfedgeMesh` struct carries the addresses as `Int`
rather than pointers, because a Mojo struct field cannot expose `AnyOrigin`.

**The tufted cover.** The robust Laplacian and the point cloud both need a mesh
in which every edge is manifold but a vertex may not be -- a "tufted" cover.
`buildIntrinsicTuftedCover` gets there by duplicating every face, inverting the
copy, and pulling each edge's sibling list apart so that each front halfedge is
paired with a back one. That needs the general form of `SurfaceMesh`, with
mutable connectivity, so `general_mesh.mojo` carries it: eight parallel arrays,
element counts in three `Int64` cells, and the four mutators
(`duplicate_face`, `invert_orientation`, `separate_to_new_edge`, `flip`) in
upstream's order. Python sizes the element arrays once, from the fact that the
cover at most doubles the faces and triples the halfedges. Afterwards the mesh
is edge-manifold again, so it is handed to the heat method as an ordinary static
mesh -- with one difference: several edges can share a vertex pair, so the
twin map cannot be recovered from the face list and is written out explicitly.
`mollifyIntrinsic` is not a noisy mollifier: it offsets every edge length by the
single scalar that makes the most degenerate triangle in the mesh just barely
non-degenerate. `flipToDelaunay` is upstream's queue-driven edge flip to the
Euclidean Delaunay condition, with the new diagonal length laid out from the
quad's four sides the way `layoutTriangleVertex` does it.

**Linear algebra.** geometry-central hands every system to SuiteSparse CHOLMOD.
That is not available here, so `linalg.mojo` is a self-contained sparse Cholesky:
Reverse Cuthill-McKee for a fill-reducing ordering, a symbolic pass that derives
the elimination tree, and a numeric pass in the same recurrence. The heat and
Poisson operators are symmetric, so the factor is `A = L D L^T` with real `D`.
The vertex connection Laplacian, which the vector heat method needs, comes out of
geometry-central *Hermitian* (`A[j,i] = conj(A[i,j])`, real diagonal), so the
complex path factors `A = L D L^H` with real `D` as well. `W` for a zero pivot is
clamped to 1e-14; that never triggers for the positive definite heat and Poisson
operators and is the reason the log map is not covered.

Everything is serial. The pinned toolchain has no `parallelize` and no threading
primitives in its standard library (see `MOJO_NOTES.md`), so the only places this
port can beat CHOLMOD are the ones where upstream is doing per-element work in
C++ rather than calling a multithreaded BLAS. What that rules out, and what a
GPU path would and would not buy, is spelled out under the benchmark table.

## Benchmarks

Machine: Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 threads, 251 GiB RAM,
Linux 6.8.0-139-generic, glibc 2.39, CPython 3.13.14, NumPy 2.5.1,
Mojo 1.2.0.dev2026092605. Produced by `pixi run bench`, which holds a
machine-wide `flock` so a concurrent factory job cannot distort the numbers.
Each cell is the best of 3 runs (1 for the largest cases); "speedup" is
potpourri3d / mojo-potpourri3d, so above 1.0 means the Mojo port is faster.

| case | mojo-potpourri3d | potpourri3d (geometry-central) | speedup |
| --- | ---: | ---: | ---: |
| face_areas (icosphere, 2562 v) | 0.05 ms | 0.82 ms | 15.47x |
| vertex_areas (icosphere, 2562 v) | 0.07 ms | 0.93 ms | 13.40x |
| cotan_laplacian (icosphere, 2562 v) | 5.05 ms | 10.23 ms | 2.03x |
| Heat method: construct (icosphere, 2562 v) | 67.46 ms | 30.56 ms | 0.45x |
| Heat method: 1 distance query (icosphere, 2562 v) | 1.68 ms | 1.22 ms | 0.73x |
| Heat method: 32 distance queries (icosphere, 2562 v) | 46.52 ms | 34.87 ms | 0.75x |
| Heat method: construct, robust Laplacian (icosphere, 2562 v) | 65.27 ms | 40.68 ms | 0.62x |
| Vector heat: construct (icosphere, 2562 v) | 36.61 ms | 35.76 ms | 0.98x |
| Vector heat: extend_scalar (icosphere, 2562 v) | 1.36 ms | 0.46 ms | 0.34x |
| Vector heat: transport_tangent_vectors (icosphere, 2562 v) | 2.95 ms | 1.14 ms | 0.39x |
| get_connection_laplacian (icosphere, 2562 v) | 0.00 ms | 0.20 ms | 407.26x |
| Heat method: construct (icosphere, 10242 v) | 644.58 ms | 217.78 ms | 0.34x |
| Heat method: 1 distance query (icosphere, 10242 v) | 24.65 ms | 9.75 ms | 0.40x |
| Vector heat: 1 transport query (icosphere, 10242 v) | 16.71 ms | 248.70 ms | 14.88x |
| Point cloud: local triangulation (2000 points) | 69.74 ms | 40.14 ms | 0.58x |
| Point cloud: construct (2000 points) | 166.09 ms | 92.58 ms | 0.56x |
| Point cloud: 1 distance query (2000 points) | 2.33 ms | 31.47 ms | 13.50x |

This is one run of a shared machine, and the variance is not small: the
10242-vertex rows are measured once each rather than three times, and across
four runs of this same code the construct came out at 600, 629, 645 and
712 ms and the query at 15.6, 19.3, 24.7 and 19.5 ms. `cotan_laplacian`, which
this pass does not touch, moved by 28% across the same runs. Read the kernel
figures below, which were taken back to back in one process, as the reliable
ones and the table as +/-20%.

### What the last optimization pass changed

Profiling the two paths that dominate every `construct` row -- the COO
assembly and the sparse LDL^T -- gave these before/after figures, each the
best of several runs in the same process on the same build:

| kernel | before | after | note |
| --- | ---: | ---: | --- |
| `ldl_numeric`, icosphere 10242 | 204.6 ms | 153.5 ms | both sparse inner loops batched four entries at a time |
| `ldl_solve`, icosphere 10242 | 9.1 ms | 6.2 ms | same batching on both triangular solves |
| `ldl_solve`, icosphere 2562 | 0.65 ms | 0.34 ms | same |
| `permute_upper`, point cloud operator (146 k triplets) | 46.0 ms | 26.9 ms | per-column bubble sort replaced by one stable radix sort of the whole triplet list |
| `permute_upper`, mesh operator (133 k triplets) | 9.6 ms | 10.0 ms | a wash: mesh columns hold ~13 entries, where the old sort was already cheap |
| `compute_neighbors`, 2000 points | 30.6 ms | 30.6 ms | unchanged, see below |
| `ldl_symbolic`, icosphere 10242 | 90 ms | 90 ms | the same batching was applied here and made no difference that survived the noise, so it was reverted |

Two things did not pay and were reverted rather than kept:

* **Vector gather/scatter in the sparse kernels.** `Pointer.unsafe_gather`
  and `unsafe_scatter` compile and run on this target, but `vgatherqpd` is
  microcoded without AVX-512: the four-wide sparse solve measured 10.4 ms
  against 9.1 ms for the scalar loop it replaced. Four independent scalar
  accesses issued back to back beat it, and that is what the kernels do.
* **A SIMD block filter for the point cloud neighbour search.** The scan is
  O(n^2) and looked like the obvious SIMD target, so it was rewritten to
  test sixteen points at a time against the running threshold over a
  struct-of-arrays copy of the cloud, with a max-heap replacing the
  O(k) shift-insertion. Every variant was slower than the 30.6 ms it
  replaced (33-50 ms): the array-of-structs layout the scan already uses is
  three doubles per cache line in one sequential stream, and the insertion
  was not the cost the profile implied. The original is still what ships.

One more change did pay, and is in the radix sort: the 256-entry digit
counter moved from a buffer behind a pointer to a `StaticTuple` local,
because the compiler could not prove a store to the destination did not
alias it, and the counter's read-modify-write then serialised the whole pass
(11.3 ms -> 5.7 ms for the mesh operator's sort).

### Parallelism and the GPU: both declined, honestly

**No threads.** The pinned toolchain has no `parallelize` and no threading
primitives in its standard library (`MOJO_NOTES.md` section 4), and no
replacement was found in the `max` package that compiles. The factorization
is the one place where threads would pay, and it stays serial.

**No GPU path.** `DeviceContext` is not in this toolchain -- not in
`std.gpu.host`, not in `std.gpu`, and the compiler has no replacement to
suggest -- so `enqueue_create_buffer`, `enqueue_copy` and
`enqueue_function` are unavailable too (`MOJO_NOTES.md` section 5). Writing
a device path against that API would not compile, so none was written. The
GPU on this machine (an RTX 5090, compute 12.0, 18.3 GiB free of 32.6 GiB)
is idle as far as this port is concerned. It would not be the right target
anyway: the hot loops are a sparse triangular solve at well under one flop
per byte, which is memory bound on a pattern the GPU cannot address without
gathering through a dense supernode representation this port does not have.

### Reading the table honestly

The elementwise kernels win, and they win big: `face_areas` and `vertex_areas`
are 13-15x faster because upstream runs a Python loop over faces in NumPy while
this port does the same arithmetic in a Mojo loop. Neither was touched by this
pass; they were already far enough ahead.

`get_connection_laplacian` at 407x is not a speed claim, it is a caching
artifact. Both sides cache the operator after the first call (upstream via
geometry-central's `ensureHaveVertexConnectionLaplacian`); upstream still
rebuilds the returned object on every call and this port returns the cached
matrix, so the benchmark measures a memcpy against a pointer store. Read it as
"both are O(1) after the first call", not as a 407x kernel win.

The losses are real and they are mostly the same reason: **the sparse
factorization**. Every `construct` row and the 0.34x at 10242 vertices is the
one-time Cholesky, not the geometry. Measured directly at 10242 vertices
(20480 faces, 71682 nonzeros in the operator): building the halfedge structure
takes 10 ms, the intrinsic geometry 30 ms, the cotan triplets 3 ms -- 43 ms
of real work, all of it competitive, against 200 ms for the COO assembly, the
symbolic pass and the numeric factorization. This pass cut the numeric pass
from 205 ms to 154 ms and made the COO assembly 1.7x faster on the point
cloud's much denser operator, which is where most of the movement in the
query rows comes from. The factorization is still the wall, and the reason
is fill and the absence of supernodes: this port's Reverse Cuthill-McKee
ordering is much weaker than CHOLMOD's AMD, and at this size the factor has
1.37 M nonzeros against the matrix's 71.7 k -- 19x fill -- which a serial
left-looking pass then walks entry by entry. CHOLMOD's multithreaded
supernodal BLAS kernels do the same algebra against a much better-ordered
factor. Beating that needs a different ordering, not a faster loop.

The 0.34x-0.75x query rows are the same effect amortised: a query is two
back-solves, so once the factorization is slower the per-query number is
slower too, even though the RHS assembly and the back-solve themselves are
fast. The back-solve is where this pass paid: `ldl_solve` is 1.4x faster on
the 10242-vertex factor and 1.9x on the 2562-vertex one, and the complex
version behind `extend_scalar` and `transport_tangent_vector` got the same
treatment (1.61 ms -> 1.36 ms and 3.84 ms -> 2.95 ms on the table). The two
query rows that invert this are the ones where upstream pays for something
this port does not: complex vector heat transport at 10242 vertices
(14.88x), and a point cloud distance query (13.50x), where upstream's KD-tree
neighbour search and its own cover dominate a query that this port has
already paid for at construction.

The point cloud rows are the least flattering and the most honest. The
0.58x on the local triangulation is the brute-force neighbour search against
nanoflann's KD-tree, and 0.56x on construction is that plus the same
factorization story as the mesh rows. The 13.50x on a query is not a claim
about the kernel: it is the same amortization as everywhere else in this
table, read the other way. The neighbour search is the one kernel where a
SIMD rewrite was tried and lost, and the table above says so with numbers.

The robust-Laplacian construct row (0.62x) is the plain construct row plus the
cover and the flips, and the cover costs about what the mesh construction does,
so there is no surprise in it.

This port is correct and complete over its subset first; accelerating the
factorization without changing the structure it exposes is a later pass. In its
current state it is the right choice for many queries on a small-to-medium mesh
and the wrong choice for a one-shot solve on a large one.

## License

MIT, as upstream. See `LICENSE`.
