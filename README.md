# mojo-potpourri3d

A Mojo port of the compute-bound parts of [potpourri3d](https://github.com/nmwsharp/potpourri3d):
the heat method for geodesic distance, the intrinsic tufted Laplacian behind its robust default,
the cotangent Laplacian and mesh areas, the point cloud's local triangulation and tangent frames,
and the sparse symmetric solver that backs them.

The Python API mirrors upstream's names, argument order and defaults, so the covered subset is a
drop-in replacement. The kernels are transcribed from the C++ the installed upstream wheel is built
from (geometry-central at commit `30fd34d`, the submodule pin of `potpourri3d` v1.4.0), and the
results are asserted against that wheel in the test suite.

## What this is

potpourri3d is a pybind11 wrapper over a fork of geometry-central. The algorithms live in C++
(`heat_method_distance.cpp`, `vector_heat_method.cpp`, `tufted_laplacian.cpp`, `simple_idt.cpp`,
`local_triangulation.cpp`, `positive_definite_solvers.cpp`, …) and the linear algebra is CHOLMOD or
Eigen's `SimplicialLDLT`. This repo reimplements that layer in Mojo and exposes it over a fixed-ABI
shared library driven by ctypes. The source files mirror the upstream files one for one and in
upstream order, so the two can be read side by side.

## Install

The toolchain is pinned in `pixi.toml`: Mojo `1.2.0.dev2026092905` with the matching `max`
`26.7.0.dev2026092905`, on Linux x86-64.

```
pixi install
pixi run build     # mojo build --emit shared-lib -I src src/ported.mojo -o dist/libmojo-potpourri3d.so
pixi run test
pixi run bench     # takes a machine-wide lock; always benchmark through this
```

`pixi run build` is one compilation unit over all of `src/`, and it takes a few seconds. The Python
package is picked up from `python/` via the `PYTHONPATH` in `pixi.toml`; there is nothing to install
beyond `pixi install`. Upstream `potpourri3d` 1.4.0 comes from PyPI as a dependency and is what the
test suite compares against — it is needed to run the tests and the benchmarks, not to use the
library.

## Usage

```python
import numpy as np
from mojo_potpourri3d.mesh import (
    MeshHeatMethodDistanceSolver,
    cotan_laplacian,
    face_areas,
    vertex_areas,
)

# a unit icosphere: 12 vertices, 20 faces
t = (1.0 + 5.0**0.5) / 2.0
V = np.array([[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],
              [0,-1,-t],[0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]], dtype=np.float64)
V /= np.linalg.norm(V, axis=1, keepdims=True)
F = np.array([[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],
              [11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],
              [3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]], dtype=np.int64)

solver = MeshHeatMethodDistanceSolver(V, F)          # use_robust=True is upstream's default
d = solver.compute_distance(0)                       # shape (12,), d[0] == 0
d_multi = solver.compute_distance_multisource([0, 1])  # distance to the nearer source

L = cotan_laplacian(V, F)                            # scipy CSR, same as upstream
a = face_areas(V, F)
m = vertex_areas(V, F)                               # lumped mass
```

The point cloud side:

```python
import numpy as np
import potpourri3d as pp3d
from mojo_potpourri3d.point_cloud import PointCloudLocalTriangulation, PointCloudHeatSolver

P = np.random.default_rng(0).standard_normal((300, 3))
P /= np.linalg.norm(P, axis=1, keepdims=True)

tri = PointCloudLocalTriangulation(P).get_local_triangulation()   # (300, max_neighs, 3), -1 padded
X, Y, N = PointCloudHeatSolver(P).get_tangent_frames()           # three (300, 3) frames
```

For the covered subset, `import potpourri3d as pp3d` and the call sites transfer unchanged. Read the
coverage section before you rely on a name being here: the methods that are not covered raise rather
than return numbers that would not match upstream.

## Coverage

### Covered, and asserted equal to upstream `potpourri3d` 1.4.0 in `pixi run test`

Every "agreement" below is the worst relative error `max|a - b| / max|b|` over the meshes and clouds
the named test covers, measured against the installed upstream wheel. Each row names the test that
proves it.

| API | measured agreement | test |
|---|---|---|
| `mojo_potpourri3d.mesh.cotan_laplacian(V, F, denom_eps=0.)` | 4e-16 | `test_cotan_laplacian_matches_upstream` |
| `cotan_laplacian(V, F, denom_eps=...)` at `0.`, `1e-6`, `1e-2` | < 1e-12 | `test_cotan_laplacian_denom_eps` |
| `mojo_potpourri3d.mesh.face_areas(V, F)` | 3e-16 (bit-exact on flat grids) | `test_face_areas_match_upstream` |
| `mojo_potpourri3d.mesh.vertex_areas(V, F)` | 3e-16 | `test_vertex_areas_match_upstream` |
| `mojo_potpourri3d.mesh.edges(V, F)` | identical edge set | `test_edges_match_upstream` |
| `MeshHeatMethodDistanceSolver(..., use_robust=False)` | 4e-15, worst over **every** vertex of five meshes | `test_heat_distance_matches_upstream` |
| `MeshHeatMethodDistanceSolver(..., use_robust=True)` | 3e-15 on closed meshes, 1.4e-09 on open ones | `test_heat_distance_robust_matches_upstream` |
| `….compute_distance_multisource(v_inds)` | 2e-15 | `test_heat_distance_multisource_matches_upstream` |
| `compute_distance(V, F, v_ind)` / `compute_distance_multisource(V, F, v_inds)` | < 1e-10 | `test_the_module_level_helpers_match_the_class` |
| `PointCloudLocalTriangulation(P).get_local_triangulation()` | identical as a set of neighbour triples on every point of all five clouds | `test_local_triangulation_matches_upstream_as_a_set` |
| `PointCloudHeatSolver(P).get_tangent_frames()` | 1e-14, up to the per-point normal sign | `test_tangent_frames_match_upstream_up_to_sign` |
| `PointCloudLocalTriangulation(P, with_degeneracy_heuristic=False)` | identical set on every point sampled | `test_degeneracy_heuristic_off_is_accepted` |
| `validate_mesh` / `validate_points` | upstream's source, compared as an AST | `test_the_core_checkers_are_upstreams_verbatim` |
| the sparse symmetric `LDL^T` and its solve | 3e-15 against `scipy.sparse.linalg.spsolve` | `tests/test_sparse.py` |

`use_robust=True` is the upstream **default**: the intrinsic tufted Laplacian of [Sharp & Crane
2020], where the mesh is doubled into its two-sheeted cover, mollified, and flipped to intrinsic
Delaunay before the operators are built. `src/tufted_laplacian.mojo` and `src/simple_idt.mojo` port
that path, and `HeatMethodDistanceSolver` calls it exactly as upstream does.

Two behaviours are pinned because they are the kind that drift silently:

- A query whose source index is outside `[0, nV)` raises `IndexError`, and an empty source list raises
  `ValueError`. Upstream indexes its vertex array with the caller's list unchecked, so on those inputs upstream returns
  whatever its allocator left there, which is not a field worth matching (`test_out_of_range_source_index_is_reported`,
  `test_an_empty_source_set_is_reported`).
- A zero-area face makes the cotangent weights non-finite, and both `use_robust` settings report it as
  a `ValueError` rather than solving with a NaN operator. Upstream returns a NaN-filled operator for
  `cotan_laplacian` on the same mesh (`test_solver_setup_failures_are_not_swallowed`).

### Not covered

Each entry names the function and the reason. Nothing here returns a plausible-but-wrong number: the
method raises `NotImplementedError` with the reason.

| API | why |
|---|---|
| `PointCloudHeatSolver.compute_distance`, `.compute_distance_multisource`, `.extend_scalar`, `.compute_log_map` | These go through `PointPositionGeometry::computeTuftedTriangulation`, which makes the local triangulation into a geometry-central **general** `SurfaceMesh` whose edges carry more than two halfedges — on a 300-point sphere, 794 of its 927 edges carry six. The intrinsic tufted cover the robust heat method needs is then built with `separateToNewEdge`, which that mesh type has and this port's mesh model does not: the flat corner array holds one twin pair per edge and a self-twin for a boundary edge, so the sheet assignment around a six-fold edge cannot be expressed. A port of that half of the solver was written and measured at 1-20% from upstream; it was removed rather than shipped, and the local triangulation it produced (which is exact) is what this port keeps. |
| `PointCloudHeatSolver.transport_tangent_vector`, `.transport_tangent_vectors` | the same operator, plus the complex-Hermitian connection Laplacian. |
| `PointCloudHeatSolver.compute_signed_distance` | a level-set solve on the sign function; out of scope. |
| `MeshVectorHeatSolver` — all of `extend_scalar`, `get_tangent_frames`, `get_connection_laplacian`, `transport_tangent_vector`, `transport_tangent_vectors`, `compute_log_map` | Two separate blockers. (a) `compute_log_map` needs `ensureHaveAffineHeatSolver`'s 3N x 3N affine connection Laplacian, which is not symmetric even on a Delaunay mesh, so it needs a general sparse LU; `src/sparse.mojo` is a symmetric `LDL^T` and no LU is shipped. (b) The other five were specified and verified against upstream (to 1.5e-15, as a NumPy transcription of the same C++) but not ported before the budget ran out. Two shared-file bugs that blocked them are fixed and are worth knowing about either way: `src/sparse.mojo`'s complex path mirrored the off-diagonal value instead of its conjugate, so `w=2` factored a complex-**symmetric** matrix where the connection Laplacian is Hermitian; and the boundary "ghost" halfedges that upstream's `for (Halfedge he : mesh.halfedges())` iterates have no counterpart in this port's mesh model, so the connection Laplacian and the affine operator are each missing a mirrored term per boundary edge on an open mesh. |
| `MeshSignedHeatSolver`, `PolygonMeshHeatSolver` | signed distance needs a level-set solve on the sign function; the polygon solver needs the tufted cover of a polygon mesh, which the manifold-only mesh model cannot build. |
| `MeshFastMarchingDistanceSolver`, `marching_triangles`, `EdgeFlipGeodesicSolver`, `GeodesicTracer` | not heat method; out of scope. |
| `MeshMarchingTrianglesSolver` | a polygon-mesh contouring solve; out of scope. |
| `read_mesh`, `write_mesh`, `read_point_cloud`, `write_point_cloud`, `read_polygon_mesh`, `potpourri3d.io` | file I/O. Nothing here is compute-bound: this port has no reason to own them, and `point_cloud.py` and `mesh.py` re-export the names their own modules use. |

### Documented divergences inside the covered set

| where | what |
|---|---|
| `compute_distance_multisource` with sources at the three corners of one face | That face's heat values are equal, so its gradient is rounding noise, and upstream calls `normalizeCutoff()` with its default cutoff of zero — it divides by the noise and normalizes it to a unit vector. Which noise it gets depends on the last bit of its solver's output, which differs from this port's, so the two fields are a couple of percent apart on that input. This port keeps upstream's arithmetic exactly rather than papering over it; a cutoff was tried and rejected because the gradient genuinely passes through zero at critical points of the heat field, and cutting those changes well-resolved results by 3%. `tests/test_parity.py::test_multisource_at_the_corners_of_one_face` pins the size of the disagreement. |
| `heat_method_distance.mojo` | The RHS region and the solution region of the query arena are kept apart, so a solve can never read and write the same vector. Upstream's `Eigen` solve copies first; ours does not have to. |
| `surface_mesh.mojo`, `simple_idt.mojo` | A boundary halfedge is a self-twinned corner here and is also a corner of a real face, so it receives that face's cotangent weight (upstream gives the ghost 0 and the real halfedge the weight). The fan orbit stops when the step leaves the fan, because there is no separate ghost to stop at. |
| `mesh.py` | `validate_mesh` keeps upstream's `np.amin` index check, which only catches negative indices. The kernels therefore check every face index themselves and report it as `ValueError` with upstream's message; without that an out-of-range positive index walks off the vertex array and kills the process. |
| `compute_distance`, `compute_distance_multisource` | An out-of-range or empty source set raises rather than returning a field. Upstream indexes its vertex array with the caller's list unchecked; every alternative here (drop the index, or clamp) yields a finite field that is wrong by construction. |
| `cotan_laplacian`, `face_areas`, `vertex_areas` on a zero-area face | Match upstream, NaNs included. Only the solver refuses that input, because only the solver cannot carry the NaN onward. |

## How it works

**FFI strategy.** A Mojo function cannot hold state across the C ABI and this dialect has no
module-level globals, so every solver is driven as three calls over caller-owned arenas:

1. `pp3d_heat_layout` publishes the arena layout, so Python sizes its buffers from the kernel rather
   than repeating the formulas.
2. `pp3d_heat_setup` builds the geometry (with the robust laplacian that means the tufted cover) and
   the two operators, runs the symbolic analysis of each, and reports `nnz(L)` for both.
3. Python sizes the factor buffers and calls `pp3d_heat_factor`, which factorises both operators.
4. Each `pp3d_heat_compute_distance` call is two triangular solves plus the divergence loop.

`@export` symbols are only emitted from the file handed to `mojo build`, so every kernel module ends
in plain `def`s that take `Int` addresses and `src/ported.mojo` holds the `@export` wrappers. That,
and the fact that buffers cross the ABI as addresses, are the only places the dialect diverges from
upstream's structure; the logic itself stays in the kernels.

**Memory layout.** Each kernel module fixes its own arena layout in a `*Layout` struct and publishes
it through a `*_layout` export, so `python/mojo_potpourri3d/_lib*.py` mirrors the numbers from the
kernel. Inside an arena the regions are packed in order, each sized by the element count the kernel
actually writes, so two solvers' arrays cannot overlap however big the operands get. The snapshot of
the *original* mesh goes first, because the distance shift reads it while the solver is working on
the tufted cover.

**The solver.** `src/sparse.mojo` is a sparse symmetric `LDL^T` with an RCM ordering, standing in for
geometry-central's `PositiveDefiniteSolver` (CHOLMOD's simplicial `LDLt`, or Eigen's
`SimplicialLDLT`). The matrix arrives as the CSC of its upper triangle including the diagonal, which
is what upstream hands SuiteSparse as `SType::SYMMETRIC`. The real path (`w == 1`) is what every
covered method uses and is verified against `scipy.sparse.linalg.spsolve` to 3e-15 in
`tests/test_sparse.py`.

The complex path (`w == 2`, entries interleaved `(re, im)`) is implemented, and the mirrored entry is
conjugated so that it factors a Hermitian matrix rather than a complex-symmetric one. It is **not
usable and not tested**: its numeric phase still rejects a Hermitian positive-definite matrix, which is
the only kind the vector heat method would hand it, so nothing that depends on it ships.
`MeshVectorHeatSolver` is the only consumer and it is not covered. The path is carried because the
connection Laplacian it exists for is not.

The symbolic phase is a left-looking column factorization: the pattern
of column *k* of *L* is the rows *i > k* reachable from the rows of the matrix below the diagonal,
closed under the columns already in the pattern.

**Parallelism.** One kernel is parallel: the point cloud's k-nearest-neighbour scan, which divides
perfectly over the source point and forks to `parallelize` above 1024 points. Everything else, the
solver included, is serial. There is no GPU path — the kNN scan reads 24 bytes per candidate for about
5 flops, which is well under the ratio where a launch pays for itself, and the heat method is bound by
the fill in its factor rather than by throughput.

**Source layout.** One Mojo file per upstream file, in upstream order:

| `src/` | upstream |
|---|---|
| `vector_util.mojo` | `utilities/vector2.h`, `vector3.h`, `elementary_geometry.h` |
| `surface_mesh.mojo` | `surface/surface_mesh.{h,cpp}` (manifold triangulations only) |
| `intrinsic_geometry_interface.mojo` | `surface/intrinsic_geometry_interface.cpp` |
| `tufted_laplacian.mojo` | `surface/tufted_laplacian.cpp`, `surface/intrinsic_mollification.cpp` |
| `simple_idt.mojo` | `surface/simple_idt.cpp` |
| `heat_method_distance.mojo` | `surface/heat_method_distance.cpp` |
| `local_triangulation.mojo` | `pointcloud/local_triangulation.cpp` |
| `point_position_geometry.mojo` | `pointcloud/point_position_geometry.cpp`, `pointcloud/neighborhoods.cpp` |
| `sparse.mojo` | `numerical/positive_definite_solvers.cpp` |
| `mesh.mojo` | `potpourri3d/mesh.py` (the module-level helpers) |
| `ported.mojo` | the `@export` ABI |

## Benchmarks

Every table below is the verbatim output of `pixi run bench` (`bench/bench.py`, best of 5 per row) on
this machine. Reproduce them with `pixi run bench`, which takes a machine-wide lock; the script prints
the environment it measured on in its first section.

The box is shared and heavily loaded, so the wall-clock figures move 10-20% between runs while the
ratios do not. Read the `ours / upstream` column, not the milliseconds.

### Environment

```
Linux 6.8.0-142-generic x86_64, Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz, 72 cores
```

### Heat method distance (use_robust=False)

| mesh | vertices | mojo (ms) | upstream (ms) | ours / upstream |
|---|---|---|---|---|
| icosphere(1) | 42 | 0.965 | 0.221 | 4.37x |
| icosphere(2) | 162 | 4.036 | 0.778 | 5.19x |
| icosphere(3) | 642 | 24.945 | 3.675 | 6.79x |
| open grid 32 | 1024 | 45.246 | 4.270 | 10.60x |
| open grid 64 | 4096 | 444.473 | 20.101 | 22.11x |
### Heat method distance (use_robust=True, upstream's default)

| mesh | vertices | mojo (ms) | upstream (ms) | ours / upstream |
|---|---|---|---|---|
| icosphere(1) | 42 | 1.293 | 0.317 | 4.08x |
| icosphere(2) | 162 | 5.653 | 1.073 | 5.27x |
| icosphere(3) | 642 | 31.462 | 4.352 | 7.23x |
| open grid 32 | 1024 | 57.185 | 5.830 | 9.81x |
| open grid 64 | 4096 | 497.840 | 27.112 | 18.36x |

### Multi-source distance (three sources, use_robust=False)

| mesh | vertices | mojo (ms) | upstream (ms) | ours / upstream |
|---|---|---|---|---|
| icosphere(1) | 42 | 0.919 | 0.224 | 4.10x |
| icosphere(2) | 162 | 3.962 | 0.770 | 5.14x |
| icosphere(3) | 642 | 24.218 | 3.208 | 7.55x |
| open grid 32 | 1024 | 44.620 | 4.074 | 10.95x |
| open grid 64 | 4096 | 449.432 | 20.419 | 22.01x |

### Setup only (operator build + factorisation, no query)

| mesh | vertices | mojo (ms) | upstream (ms) | ours / upstream |
|---|---|---|---|---|
| icosphere(1) | 42 | 0.901 | 0.196 | 4.58x |
| icosphere(2) | 162 | 3.848 | 0.723 | 5.32x |
| icosphere(3) | 642 | 24.360 | 3.001 | 8.12x |

### Mesh operators

| op | mesh | mojo (ms) | upstream (ms) | ours / upstream |
|---|---|---|---|---|
| cotan_laplacian / icosphere(1) | 42 | 0.197 | 0.507 | 0.39x |
| face_areas / icosphere(1) | 42 | 0.010 | 0.062 | 0.16x |
| vertex_areas / icosphere(1) | 42 | 0.010 | 0.077 | 0.13x |
| cotan_laplacian / icosphere(2) | 162 | 0.291 | 0.722 | 0.40x |
| face_areas / icosphere(2) | 162 | 0.012 | 0.096 | 0.12x |
| vertex_areas / icosphere(2) | 162 | 0.012 | 0.113 | 0.11x |
| cotan_laplacian / icosphere(3) | 642 | 0.699 | 1.548 | 0.45x |
| face_areas / icosphere(3) | 642 | 0.020 | 0.192 | 0.10x |
| vertex_areas / icosphere(3) | 642 | 0.023 | 0.217 | 0.11x |

### Point cloud local triangulation and tangent frames

| cloud | points | local triangulation (ms) | upstream (ms) | ours / upstream |
|---|---|---|---|---|
| random sphere | 300 | 5.473 | 4.083 | 1.34x |
| random sphere | 1000 | 23.898 | 14.232 | 1.68x |
| random sphere | 3000 | 51.352 | 43.320 | 1.19x |

### Sparse Cholesky (5-point Laplacian, heat operator shape)

| grid | vertices | nnz(L) | factorise (ms) | solve (ms) |
|---|---|---|---|---|
| 20x20 | 400 | 10907 | 1.783 | 0.017 |
| 40x40 | 1600 | 86607 | 23.415 | 0.032 |
| 60x60 | 3600 | 291107 | 109.307 | 0.069 |

**This port is slower than upstream on the heat method distance: 4-8x on the closed icospheres,
10x on a 1024-vertex grid, 22x on a 4096-vertex grid.** The reason is the solver, not the geometry.
Upstream links a tuned sparse Cholesky (SuiteSparse/CHOLMOD, or Eigen's supernodal `SimplicialLDLT`)
that reorders with AMD and exploits supernodes, while `src/sparse.mojo` is a straightforward serial
left-looking `LDL^T` with RCM ordering and scalar arithmetic. On a 64x64 grid RCM leaves a factor with
far more fill than AMD would, which is where the largest ratio comes from. The setup-only table shows
the same shape, which locates the cost in factorisation rather than in the geometry or the divergence
loop. A later pass can change the ordering, add supernodal blocking and vectorise the kernels; this
pass deliberately left the structure alone so the port could be read against the upstream source.

The elementwise operators are 2-6x faster than upstream, and `cotan_laplacian` 2.2-2.6x,
because upstream runs the same vectorised numpy expression and then assembles a `scipy.sparse` matrix, while this port emits the triplets
straight out of Mojo. The point cloud's local triangulation is 1.2-1.7x slower, because upstream's
nanoflann k-nearest-neighbour search is sub-quadratic and this port's exhaustive scan is not. That
scan does fork to `parallelize` above 1024 points, which is where the 3000-point row closes most of its
gap; below the threshold it runs serially, so small clouds are its worst case.

The sparse Cholesky tables have no upstream counterpart: they time the same 5-point Laplacian
`src/sparse.mojo` factorises inside the heat method, straight through the C ABI, and they are what the
ratios above are made of. They are verified against `scipy.sparse.linalg.spsolve` to a relative error
below 1e-13 on random SPD matrices in `tests/test_sparse.py`. The complex path the same file also
carries is not verified at all; see the coverage section.

## Licence

MIT, matching upstream. See `LICENSE`.
