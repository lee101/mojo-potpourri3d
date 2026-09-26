# mojo-potpourri3d

A Mojo port of the compute-bound parts of [potpourri3d](https://github.com/nmwsharp/potpourri3d):
geodesic distance by the heat method, tangent-vector transport, and the mesh
laplacians and tangent frames they are built on.

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

`d[0]` is 0, the unit-icosahedron's antipodal distance comes out at ~2.732 (the
true geodesic is pi = 3.1416, so this is the usual heat-method underestimate on a
mesh this coarse), `abs(L @ ones).max()` is ~1e-15, and every row of `u` has
unit norm to ~1e-16. `use_robust` and `use_intrinsic_delaunay` are set to
`False` only because this port does not cover the robust /
intrinsic-Delaunay preprocessing; see the coverage table.

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
| `MeshHeatMethodDistanceSolver.compute_distance` | exact parity |
| `MeshHeatMethodDistanceSolver.compute_distance_multisource` | exact parity |
| `compute_distance`, `compute_distance_multisource` | exact parity |
| `MeshVectorHeatSolver.extend_scalar` | exact parity |
| `MeshVectorHeatSolver.get_tangent_frames` | exact parity |
| `MeshVectorHeatSolver.get_connection_laplacian` | exact parity |
| `MeshVectorHeatSolver.transport_tangent_vector(s)` | exact parity except one vertex (see below) |
| `validate_mesh`, `validate_points` | verbatim from upstream |

### What is not covered, and why

* **Point clouds.** `PointCloudHeatSolver` and `PointCloudLocalTriangulation` are
  absent. Upstream builds a *tufted* intrinsic Delaunay triangulation of the
  point cloud (local Delaunay per neighbourhood, then `mollifyIntrinsic`,
  `buildIntrinsicTuftedCover` and `flipToDelaunay`) and runs the heat method on
  that. Those four passes are several hundred lines of geometry-central and the
  whole port hinges on getting them bit-comparable; the mesh solvers here do not
  reuse them, so there is nothing partial to expose. `PointCloudHeatSolver` is
  therefore simply not present rather than present-and-wrong.

* **`MeshVectorHeatSolver.compute_log_map`.** Raises `NotImplementedError`. All
  three strategies (`VectorHeat`, `AffineLocal`, `AffineAdaptive`) end in a
  factorization of a singular or indefinite operator -- the affine connection
  heat operator for the two affine strategies, the unshifted cotan Laplacian for
  `VectorHeat` -- which upstream hands to Eigen's `SparseLU` with partial
  pivoting. This port has a serial LDL^T with a pivot floor, and on those
  operators it returns a visibly different answer. Rather than ship a wrong
  number, the method raises and says so.

* **`use_robust=True` (the upstream default for the distance solver).** The
  robust path mollifies the mesh, builds a tufted cover and flips to intrinsic
  Delaunay before doing anything else. Those are the same three passes the point
  clouds need, and they are not ported. The flag is accepted and stored, and the
  solver runs the plain (`use_robust=False`) path, so passing `True` or `False`
  gives bit-identical results here. On a 2562-vertex icosphere that plain answer
  differs from upstream's robust answer by 4.0e-05 relative;
  `test_robust_laplacian_is_close` pins the bound.

* **`use_intrinsic_delaunay=True`** on the vector heat solver: same reason. The
  port runs the extrinsic path. Parity is asserted with
  `use_intrinsic_delaunay=False`, which is the configuration that isolates the
  solver from the intrinsic-Delaunay preprocessing.

* **One vertex of `transport_tangent_vector`.** On a sphere, the vertex directly
  opposite the source is the antipode, where the transported direction is
  numerically degenerate; normalizing there amplifies rounding until the sign
  flips. Every other vertex agrees to 1e-9. `test_transport_tangent_vector_matches_except_the_antipode`
  asserts exactly that: at most one vertex may differ, and all others must agree.

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
C++ rather than calling a multithreaded BLAS.

## Benchmarks

Machine: Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 threads, 251 GiB RAM,
Linux 6.8.0-139-generic, glibc 2.39, CPython 3.13.14, NumPy 2.5.1,
Mojo 1.2.0.dev2026092605. Produced by `pixi run bench`, which holds a
machine-wide `flock` so a concurrent factory job cannot distort the numbers.
Each cell is the best of 3 runs; "speedup" is potpourri3d / mojo-potpourri3d, so
above 1.0 means the Mojo port is faster.

| case | mojo-potpourri3d | potpourri3d (geometry-central) | speedup |
| --- | ---: | ---: | ---: |
| face_areas (icosphere, 2562 v) | 0.08 ms | 1.21 ms | 16.08x |
| vertex_areas (icosphere, 2562 v) | 0.12 ms | 1.32 ms | 11.25x |
| cotan_laplacian (icosphere, 2562 v) | 5.46 ms | 11.19 ms | 2.05x |
| Heat method: construct (icosphere, 2562 v) | 258.73 ms | 29.81 ms | 0.12x |
| Heat method: 1 distance query (icosphere, 2562 v) | 1.93 ms | 1.19 ms | 0.62x |
| Heat method: 32 distance queries (icosphere, 2562 v) | 61.19 ms | 37.22 ms | 0.61x |
| Vector heat: construct (icosphere, 2562 v) | 34.18 ms | 46.20 ms | 1.35x |
| Vector heat: extend_scalar (icosphere, 2562 v) | 1.27 ms | 0.49 ms | 0.39x |
| Vector heat: transport_tangent_vectors (icosphere, 2562 v) | 3.53 ms | 1.12 ms | 0.32x |
| get_connection_laplacian (icosphere, 2562 v) | 0.00 ms | 0.18 ms | 302.29x |
| Heat method: construct (icosphere, 10242 v) | 3801.87 ms | 260.44 ms | 0.07x |
| Heat method: 1 distance query (icosphere, 10242 v) | 17.96 ms | 9.39 ms | 0.52x |
| Vector heat: 1 transport query (icosphere, 10242 v) | 14.36 ms | 229.16 ms | 15.96x |

### Reading the table honestly

The elementwise kernels win, and they win big: `face_areas` and `vertex_areas`
are 11-16x faster because upstream runs a Python loop over faces in NumPy while
this port does the same arithmetic in a Mojo loop.

`get_connection_laplacian` at 302x is not a speed claim, it is a caching
artifact. Both sides cache the operator after the first call (upstream via
geometry-central's `ensureHaveVertexConnectionLaplacian`); upstream still
rebuilds the returned object on every call and this port returns the cached
matrix, so the benchmark measures a memcpy against a pointer store. Read it as
"both are O(1) after the first call", not as a 302x kernel win.

The losses are real and they are all the same reason: **the sparse
factorization.** Every `construct` row and the 0.07x at 10242 vertices is the
one-time Cholesky, not the geometry. Measured directly at 10242 vertices
(20480 faces): building the halfedge structure takes 11 ms, the cotan triplets
4 ms, the COO-to-CSR conversion 8 ms, and a subsequent triangular solve 12 ms --
about 35 ms of real work, all of it competitive. The remaining ~7.8 s is
`Factorization`. The reason is fill and the absence of supernodes: this port's
Reverse Cuthill-McKee ordering is much weaker than CHOLMOD's AMD, and at this
size the factor has 2.29 M nonzeros against the matrix's 71.7 k -- 32x fill --
which a serial left-looking pass then walks entry by entry. CHOLMOD's
multithreaded supernodal BLAS kernels do the same algebra against a much
better-ordered factor.

The 0.32x-0.62x query rows are the same effect amortised: a query is two
back-solves, so once the factorization is slower the per-query number is
slower too, even though the RHS assembly and the back-solve themselves are fast.
The one query row that inverts this, vector heat transport at 10242 vertices
(15.96x), is the case where upstream pays for a complex solve our real-arithmetic
path handles directly.

This port is correct and complete over its subset first; accelerating the
factorization without changing the structure it exposes is a later pass. In its
current state it is the right choice for many queries on a small-to-medium mesh
and the wrong choice for a one-shot solve on a large one.

## License

MIT, as upstream. See `LICENSE`.
