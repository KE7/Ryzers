# Numerical and performance validation

Tested 2026-10-21–23 on Radeon 8060S (`gfx1151`) and Radeon AI PRO R9700
(`gfx1201`), ROCm 7.14 / PyTorch 2.12. SAM 3D source is pinned to
`f91db411c50efee93d8db7aeb323885650f6f722`; model weights are
`facebook/sam-3d-objects` revision `2e73555018d2741ccd486e56c24fac41155a1dc6`.

The checks answer different questions: reconstruction/export success, shape
quality against Meta’s artist meshes, numerical agreement between implementations,
and inference latency. Numerical agreement counts below are regression diagnostics;
they are not percentages of cases that pass Meta’s benchmark. The historical
49/100 iGPU comparison contains 100 completed reconstructions, with 49 meeting
all four local output-agreement limits. See [benchmark coverage](#benchmark-coverage)
for what the artist-reference evaluation establishes.

## Reproduce the checks

`ryzers run` performs a headless installation smoke test without weights or
credentials. The following explicit checks require the model access described
in the README only when they reconstruct images:

```sh
ryzers run "python /ryzers/check_sam3d_numerics.py"
ryzers run "python /ryzers/test_sam3d_conditioning_cache.py"
ryzers run "python /ryzers/test_sam3d.py --check-repeatability"
ryzers run "python /ryzers/test_sam3d.py --mask-index 14 3"
ryzers run "python /ryzers/test_sam3d.py --all-masks"
```

For frozen CUDA sparse-convolution references, use a trusted pinned checkout:

```sh
ryzers run bash
# Inside the container:
git clone https://github.com/L-Reichardt/spconv-triton.git /tmp/spconv-reference
git -C /tmp/spconv-reference checkout 63361695e891de7aff32841ebb86f2661a41b059
python /ryzers/check_sam3d_numerics.py --golden-root /tmp/spconv-reference
```

The checker verifies fixture SHA-256 hashes before loading their pickled
metadata. Reports persist in `workspace/sam3d/numerics.json`. Run each check on
both devices using `HIP_VISIBLE_DEVICES` as shown in the README.

## Numerical references and maximum errors

The rebuilt default image passed the operator checks below on both devices. The 39 sparse
fixtures come from the pinned spconv-triton checkout, whose reference generators
use unchanged CUDA spconv. Coordinates and spatial shapes match exactly;
features use the fixtures' original absolute tolerances with relative tolerance
zero. The seven tight FP16 cases compare against CUDA FP32 values with bounds
derived from CUDA's own FP16 rounding; older loose FP16 fixtures are excluded.
Coverage includes tails, up to 256 channels, empty/multiple batches, submanifold,
strided, dilated, transposed and inverse convolution.

| Check and reference | R9700 maximum absolute error | 8060S maximum absolute error |
|---|---:|---:|
| 32 FP32 sparse CUDA fixtures | 8.5831e-6 | 8.5831e-6 |
| 7 FP16 sparse CUDA fixtures | 9.7609e-4 | 9.7943e-4 |
| FP32 attention, CPU float64 | 4.9863e-7 | 4.9863e-7 |
| FP16 attention, CPU float64 | 6.2741e-4 | 6.2741e-4 |
| BF16 attention, CPU float64 | 4.4183e-3 | 4.4183e-3 |
| FP16 Triton attention, sampled CPU float64 | 6.2402e-5 | 1.0694e-4 |
| Six synthetic SLAT normalization cases, CPU float64 | 1.9505e-3 | 1.9456e-3 |
| Native FP16 GEMM/bias/tanh-GELU, sampled CPU float64 | 9.7065e-4 | 9.5104e-4 |
| Compact zero context, PyTorch reference | 0 | 0 |
| Twenty pointwise cases, PyTorch reference | 0 | 0 |
| Cached linear casts, PyTorch reference | 0 | 0 |

Attention checks cover unequal lengths, independent blocks, empty query blocks,
packed strides and tails. Absolute tolerances are 2e-5 / 2e-3 / 2e-2 for
FP32 / FP16 / BF16, with relative tolerance zero. Triton checks sample 32 query
rows against every key/value at lengths 2,049/2,071 and 4,096/4,096.
Normalization, layout and GEMM/GELU float64 checks use mixed
`atol=rtol=0.002`; this is not a universal 0.002 absolute-error bound.

Additional real-checkpoint operator replays sampled 32 rows per shape. FP16
attention at 5,252 queries and 5,252/5,496 keys reached maximum absolute error
0.001043 / 0.001054 on R9700 / 8060S against CPU float64. Sparse RMS reached
0.003441 / 0.003852 and SLAT layer normalization 0.003829 / 0.003614. A
2,048-channel decoder-shaped normalization replay reached 0.02963 / 0.02992
on larger-magnitude outputs and passed the same mixed tolerance; actual decoder
normalization retains PyTorch. The iGPU MLP layout check differed from row-major
PyTorch by at most 0.000244140625, and from float64 by 0.000122829.

Sixteen dense RMS cases match PyTorch numerically exactly on R9700, including
zero/subnormal inputs and packed strides; 8060S retains the PyTorch path.
Eight compact-context cases cover parameter/input mutation and nonfinite
fallbacks. Twelve CPU tests cover request/stage isolation, cache bounds,
mutation, hooks, frozen-backbone equality including nonpersistent buffers,
and cleanup after exceptions. Image conversion and PLY field checks match
their references exactly.

These CUDA comparisons are at operator level. Full-model comparisons below
use ROCm reference paths. The SA-3DAO evaluation separately compares with
artist-created meshes. No full-model NVIDIA CUDA comparison or official
leaderboard score is claimed.

## Full-pipeline agreement

The timed mask-14 captures compare the compatibility-only port with final
defaults in separate image processes, using identical seed, weights, default
steps and both geometry decoders. Occupancy and pose match exactly on both
GPUs. These are the same runs used for the performance comparison.

| Original port to final defaults | R9700 | 8060S |
|---|---:|---:|
| Sparse occupancy IoU | 1.0 | 1.0 |
| Pose tensor absolute difference | 0 | 0 |
| SLAT feature RMS difference / reference RMS | 0.02666% | 0.02633% |
| SLAT maximum absolute difference | 0.01890 | 0.05177 |
| Gaussian-center symmetric NN p99 / bounding-box diagonal | 0.04460% | 0.04274% |
| Mesh symmetric NN p99 / bounding-box diagonal | 0.005706% | 0.006021% |

Preselected limits are exact occupancy/pose, feature normalized RMS <=1%,
and geometry p99 <=0.5% of the reference bounding-box diagonal. Geometry uses
10,000 sampled vertices or Gaussian centers in each direction against each
complete point set, with a fixed random seed. These sampled nearest-neighbor
percentiles are not Hausdorff bounds.

The broader comparison uses all 100 SA-3DAO objects on each GPU, with the
same original-port and optimized images. Both versions generated valid meshes
and finite Gaussian centers for every object. Raw RGB-only results are:

| Original port to final defaults, 100 objects | R9700 | 8060S |
|---|---:|---:|
| Pairs meeting all strict numerical limits | 100/100 | 49/100 |
| Identical sparse coordinates | 100/100 | 58/100 |
| Exact occupancy and pose together | 100/100 | 49/100 |
| Maximum SLAT normalized RMS difference | 0.13564% | 0.09167%* |
| Maximum SLAT absolute difference | 0.14934 | 0.09552* |
| Maximum rotation tensor component difference | 0 | 0.008141 |
| Maximum translation tensor component difference | 0 | 0.02416 |
| Maximum scale tensor component difference | 0 | 0.003907 |
| Maximum mesh symmetric NN p99 / bbox diagonal | 0.02644% | 6.67722% |

*8060S feature comparisons cover the 58 pairs with identical coordinates;
features cannot be compared row-for-row in the other 42. None of these
comparable feature tensors exceeds the 1% normalized RMS limit. All 51 raw
8060S failures include a structure/pose difference; eight also exceed the
mesh p99 limit. These failures remain part of the reported results.

Recorded-depth replays on 8060S isolate the two SAM 3D stages from variations
in the preceding MoGe estimate. All 52 controls retain exact occupancy and
pose: 47 use original-batch depth captures, including a passing control; five
use fresh original-port references for inputs without saved depth. These
controls cover all 51 raw mismatch cases. For the largest raw outlier, the
telephone booth, a full
optimized reconstruction using the original run's recorded depth also passes
the original limits: exact coordinates and pose, 0.03498% SLAT normalized RMS
difference, and 0.009728% mesh p99, compared with 6.67722% in the independent
RGB-only runs. Input RGB tensors match exactly while estimated depth and
intrinsics differ. An unchanged original-port counter rerun also differs
from its earlier result: mesh p99 0.53416%, above the same 0.5% limit.
These controls check the SAM 3D stages under identical depth conditioning.
They do not replace the raw comparisons or relax their limits.

Subsequent cold/warm controls identified two runtime stability problems on
8060S. Timing-based MIOpen solver selection changes FP16 depth estimates across
fresh processes. Expandable allocator segments can also produce nonfinite
depth tensors: four fresh processes using the native allocator passed all
three reconstructions, while all four expandable-segment processes produced
nonfinite depth on their first reconstruction. The image therefore uses the
native allocator and defaults to MIOpen `FAST` on `gfx1151`. Fresh paired
comparisons must use the same stability settings for the compatibility-only
port and optimized implementation. Such comparisons are distinct from the
historical raw results above.

Both devices also exported PLY/GLB for all 27 example masks and assembled a
scene of about 13.6 million Gaussians. Checks verify finite geometry and PLY
attributes, read back GLB meshes, and require the assembled count to equal
the sum of individual objects. Cross-device output counts need not be equal.

### Current default comparison

With the native allocator, the corrected optimized image passes the same four
limits on all 100 objects on each GPU. R9700 is compared with its original
saved compatibility-port outputs. On 8060S, both the unoptimized compatibility
port and optimized image were rerun with the native allocator and MIOpen
`FAST`. This fresh common-runtime comparison does not replace the historical
49/100 result above or claim to reproduce its unstable depth estimates.

| Compatibility-only port to corrected defaults, 100 objects | R9700 | 8060S |
|---|---:|---:|
| Pairs meeting all four strict limits | **100/100** | **100/100** |
| Exact SS and SLAT sparse coordinates | 100/100 | 100/100 |
| Exact rotation, translation and scale tensors | 100/100 | 100/100 |
| Maximum SLAT normalized RMS difference | 0.10055% | 0.10848% |
| Maximum SLAT absolute difference | 0.12493 | 0.07251 |
| Maximum mesh symmetric NN p99 / bbox diagonal | 0.02426% | 0.02571% |

Each run reconstructs from RGB and mask, independently estimating depth with
the unmodified public MoGe path. No recorded or reference depth is supplied.
All 100 8060S pairs also have bitwise-identical public pointmap and pointmap-color
return tensors, captured after inference completes. The numerical gates remain
exact coordinates and pose, SLAT normalized RMS at most 1%, and mesh p99 at most
0.5% of the reference bounding-box diagonal.

The public `--check-repeatability` check also passes on both GPUs: one cold
reconstruction and three warm reconstructions per device, with all three
cold-to-warm comparisons meeting those same four limits.

Both corrected images also export all 27 masks and assemble the scene. The
8060S scene contains 13,593,632 Gaussians; its peak allocated memory across
inference and assembly is 14.80 GiB, excluding checkpoint loading. Process
peak RSS, including loading, is 11.54 GiB; temporary export files reside in
shared-memory storage outside process RSS. These measures are not additive
on unified-memory hardware. R9700 produces 13,611,264 Gaussians; its measured
maximum per-mask allocation is 14.80 GiB, with final assembly allocation not
captured by that run.

Compared instead with the original saved 8060S `DEFAULT`-mode outputs, the
corrected image meets all four limits on **0/100** cases: coordinates match
on 11/100, pose on 0/100, and mesh p99 is within its limit on 84/100. Maximum
mesh p99 is 6.69693%. Changing the solver policy therefore does not preserve
those historical outputs. The common-runtime 100/100 result establishes
optimization agreement under the stable policy, not exact restoration of the
original outputs.

Fresh-process R9700 mask-14 benchmarks also expose variation in the unchanged
compatibility port under its default solver policy: two processes produce
26,346 versus 26,344 sparse coordinates and different pose tensors. One of
the two port-to-corrected benchmark pairs meets all four gates; the other
fails coordinates and pose. The two corrected processes agree with each
other under the same gates. The 100-case result above therefore establishes
agreement for that cohort, not repeatability across every fresh default-mode
process. This variation remains relevant to repeatability; it is not a failure
against Meta’s artist-reference benchmark.

Explicit FAST/native controls on the same R9700 example pass 12 cold-to-warm
and 16 cross-process comparisons across four fresh processes, including
compatibility-port versus optimized comparisons. All cross-process pointmaps
and colors match bitwise. This covers one mask, not the 100-object dataset;
it does not establish full-cohort FAST agreement or change the R9700 default.

## Performance and memory

### Current 8060S defaults

An isolated A/B/C/C/B/A matrix compares the original compatibility-only port,
merged defaults and corrected defaults. Each version has two fresh processes,
each with one cold warmup and three timed requests: six timed samples per
version. The port and corrected image use the same native allocator and FAST
solver policy. Both captured port-to-corrected pairs meet all four numerical
gates: exact coordinates and pose, maximum SLAT normalized RMS 0.02649%, and
maximum mesh p99 0.005635% of the reference bounding-box diagonal.

| Warm mask-14 inference, 8060S | Port only | Corrected defaults |
|---|---:|---:|
| Median of six | 134.48 s | **51.89 s** |
| Range | 133.77–134.61 s | 51.42–52.59 s |
| Peak allocated memory | 18.88 GiB | 13.75 GiB |
| Peak reserved memory | 22.41 GiB | 19.51 GiB |

The total saving is **82.59 s per object, 61.4% less latency, or 2.59× faster**.
Against the former merged defaults (51.07 s median), the stability changes
cost **0.82 s, or 1.6%**. This is a measured cost, not a zero-penalty claim.
Allocator statistics exclude checkpoint loading; allocations use shared RAM
on this iGPU and are not an additional physical-memory pool.

The corresponding R9700 matrix measures **24.73 to 13.13 s (46.9% less
latency)**. Its fresh-process numerical variation is documented above. All
planned inference calls completed; output agreement and artist-reference
quality are separate checks. Neither device’s timing results establish
parity with Meta’s complete four-metric benchmark.

### Recorded comparison before the stability changes

The recorded comparison uses the original compatibility-only port from
`54bde2c` against final defaults from `d3a563b`, including all subsequent
optimizations. The original image's patch SHA-256 matches the first commit;
only the sparse-convolution import substitutions and Kaolin compatibility
changes are present. Both images share their first 27 filesystem layers,
including the pinned upstream checkout and dependencies, and use the pinned
model revision above. These measurements include the merged optimized image's
expandable allocator setting; the current image uses the native allocator
for the stability reasons above.

Both images use the same harness: one loaded model, one warmup followed by
three timed mask-14 runs, seed 42, default steps, both geometry decoders and
`compile=False`. Devices run serially with no competing GPU workload. Timings
synchronize before/after inference and include preprocessing; loading, export
and diagnostic transfers/serialization are excluded. Intermediate captures
retain detached references identically on both paths and are released before
the next inference. The baseline runs its original source and defaults directly.

| Warm mask-14 inference | R9700 | 8060S |
|---|---:|---:|
| First working AMD port, median of three | 24.97 s | 130.60 s |
| All optimizations, median of three | **13.22 s** | **51.15 s** |
| Final range | 13.21–13.27 s | 51.04–51.17 s |
| Time saved per object | **11.75 s** | **79.45 s** |
| Latency reduction | **47.1%** | **60.8%** |
| Speedup | **1.89×** | **2.55×** |
| Port-only range | 24.94–25.25 s | 130.31–130.75 s |
| Port-only peak allocated GPU memory | 18.88 GiB | 18.88 GiB |
| Final peak allocated GPU memory | 13.70 GiB | 13.70 GiB |
| Port-only peak reserved GPU memory | 22.41 GiB | 22.41 GiB |
| Final peak reserved GPU memory | 14.89 GiB | 14.89 GiB |
| Maximum per-mask inference allocation, 27-object run | 14.74 GiB | 14.74 GiB |
| Maximum per-mask inference reservation, 27-object run | 15.80 GiB | 16.19 GiB |
| Full-run CPU process peak RSS, including loading | 11.67 GiB | 11.64 GiB |

The earlier same-process comparison measured the final group of changes only:
15.66 to 12.99 s on R9700 and 63.14 to 50.59 s on 8060S. Those incremental
results are distinct from the full port-to-final comparison above. CPU FP32
masters enabled the paired storage-format toggles for that experiment and
were released before its final-default runs. Savings from separate attribution
experiments are not added together.

Full-scene runs retain individual Gaussian outputs until assembly. The recorded
per-mask GPU peaks exclude checkpoint loading and final scene assembly.
On Strix Halo GPU allocations use shared physical
RAM; do not add CPU RSS and GPU allocator statistics as an exact physical-memory
total.

DINO deduplication removes 3,652,472,832 bytes (3.40 GiB) of identical frozen
weights. Four stage-specific preprocessing/feature-selection wrappers remain;
byte comparison follows checkpoint loading, so the initial load-time peak is
not eliminated. Matching inputs produce four backbone computations and four
cross-stage cache hits per object. Frozen SS linear storage removes another
1,914,175,488 bytes (1.78 GiB) of FP32 weight storage. Normalization stays FP32.

PLY attribute packing was separately measured on 13,611,264 Gaussians:
17.98 s to 0.062 ms, peak CPU process RSS 11.90 to 1.75 GiB, identical
925,565,952-byte attribute payload SHA-256. This isolates packing; it excludes
attribute preparation, device transfer and disk writing, and is not a full-run
RAM or export-time measurement.

The selected FP16/default-step configuration is the measured stopping point
for this tuning sweep. The final GELU fusion saved about 0.31–0.35 s in the
iGPU latent stage; tested HIP attention and persistent-graph alternatives did
not provide a worthwhile full-pipeline improvement at the required memory and
numerical behavior. This establishes diminishing returns among the tested
candidates, not an absolute hardware-optimality proof.

## Defaults and overrides

Set these variables inside the container before constructing the pipeline.
A value of `0` disables the respective optimization. Guards retain reference
paths for unsupported shapes, modes and devices.

| Optimization | Variable |
|---|---|
| Conditioning projections/casts | `SAM3D_CACHE_CROSS_ATTN` |
| Compact zero CFG context | `SAM3D_COMPACT_CFG` |
| SLAT attention | `SAM3D_TRITON_ATTN` |
| SLAT normalization / R9700 dense RMS tail | `SAM3D_TRITON_NORM` / `SAM3D_TRITON_DENSE_NORM` |
| Separate sparse Q/K/V | `SAM3D_SEPARATE_QKV` |
| Frozen linear FP16 storage / cast cache | `SAM3D_FP16_LINEAR_WEIGHTS` / `SAM3D_CACHE_LINEAR` |
| iGPU MLP layout / native GELU epilogue | `SAM3D_CACHE_LAYOUT` / `SAM3D_FUSED_GELU` |
| Frozen DINO sharing / raw feature reuse | `SAM3D_SHARE_BACKBONES` / `SAM3D_CACHE_IMAGE_FEATURES` |
| Exact elementwise fusion | `SAM3D_TRITON_POINTWISE` |

`PYTORCH_ALLOC_CONF=expandable_segments:False` is the image default. Expandable
segments are excluded from the validated configuration because of the cold
depth failures above. On `gfx1151`, the pipeline sets `MIOPEN_FIND_MODE=FAST`
unless it was explicitly set before construction. Other architectures do not
set this variable. These are process-wide runtime settings.
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` enables fused SDPA on 8060S.
Optional rasterization, texture baking and layout optimization remain outside
the validated reconstruction/export paths, as described in the README.

## SA-3DAO reference evaluation

Download `objects_0001-0100.zip` from Meta's
[SA-3DAO download page](https://ai.meta.com/datasets/sa-3dao-sam-3d-artist-objects-downloads/)
and extract it under `workspace/sam3d/sa3dao`. The checker selects folders
containing reference `mesh.glb` files; input-only folders are excluded. The
evaluated archive contains 100 reference objects from 81 distinct RGB files.
Its SHA-256 is:

```text
e594558ee2dcddcfc43f802b49902114912124347ca182d348569b4161af47a4
```

Reconstruction uses only the RGB image and JPEG mask, thresholded at 127,
with seed 42, default steps, both decoders and `compile=False`. It estimates
its own MoGe pointmap and does not use the artist mesh, released pointmap or
reference pose as model inputs.

```sh
# CPU metric checks without model weights or a dataset.
ryzers run "python /ryzers/check_sam3d_dataset.py --self-test"

# Requires normal model access and an empty prediction directory.
ryzers run "python /ryzers/check_sam3d_dataset.py \
  --dataset-root /ryzers/output/sa3dao \
  --predictions /ryzers/output/sa3dao-predictions \
  --reconstruct"
```

Use `HIP_VISIBLE_DEVICES` as shown in the README and a distinct prediction
folder for each device/version. Omit `--reconstruct` to score existing meshes
on CPU without downloading weights. The report defaults to
`workspace/sam3d/sa3dao-scores.json`; select `--output` for separate reports.
Missing or invalid predictions produce error entries and a nonzero exit,
rather than silently reducing the evaluated set.

The CPU checker follows the core shape definitions in
[paper v2, Appendix D.3.1](https://arxiv.org/html/2511.16624v2#A4.SS3.SSS1):
independent bbox-centering and uniform scaling into `[-1,1]`, rigid ICP,
one million uniform surface samples per mesh, F-score at 0.01, and surface
voxel occupancy IoU at resolution 64. It preserves aspect ratios and permits
proper rotations/translations during alignment, without reflections or fitted
scale. Deterministic PCA-axis initializations reduce ICP local-minimum errors.
Both aligned point sets share an isotropic 64-cubed voxel grid enclosing them.
Chamfer reports distinguish the mean/sum of unsquared Euclidean directional
distances and the sum of directional mean squared distances. Identity, known rigid motion and
shape-deformation controls check the evaluator independently of model outputs.

The historical compatibility-only port and optimized image each reconstructed all
100 objects on both GPUs: 400 successful reconstructions and CPU scores,
without inference/scoring errors. Means over the same 100 references are:

| GPU / version | F-score @0.01 | Surface voxel IoU, 64 | Mean unsquared Chamfer |
|---|---:|---:|---:|
| R9700, original port | 0.21437 | 0.22265 | 0.060131 |
| R9700, optimized | 0.21393 | 0.22115 | 0.060139 |
| 8060S, original port | 0.21671 | 0.22573 | 0.060391 |
| 8060S, optimized | 0.21647 | 0.22481 | 0.060456 |

Before scoring, the quality-preservation criterion was a paired mean
final-minus-port difference whose 95% bootstrap interval lies within
`[-0.005, +0.005]` for both F-score and IoU. Both GPUs meet this criterion
using 10,000 object-level bootstrap resamples, seed 42. The following values
are **percentage points**, with an equivalence margin of +/-0.5 points:

| GPU / metric | Mean change | 95% paired bootstrap interval |
|---|---:|---:|
| R9700, F-score | -0.0438 | [-0.1564, +0.0802] |
| R9700, IoU | -0.1497 | [-0.3928, +0.0045] |
| 8060S, F-score | -0.0239 | [-0.1921, +0.1443] |
| 8060S, IoU | -0.0914 | [-0.4041, +0.1569] |

Independent ICP fitting and voxel-grid placement can amplify small mesh
changes. An additional diagnostic fits alignment on the original port and
uses that transform and one common voxel grid for both predictions. Its mean
F-score changes are +0.0033 / -0.0419 percentage points for R9700 / 8060S;
IoU changes are +0.0037 / -0.0296 points. This shared-frame diagnostic does not
replace the primary independently registered scores or predeclared criterion.
The concurrent dataset runs test quality; their elapsed times are not the
isolated performance measurements above.

The current-default evaluation adds 300 successful reconstructions and scores:
the corrected R9700 image, and both the compatibility-only port and corrected
image under the common 8060S stability settings. All use the same 100 artist
references and unchanged scoring protocol.

| Current evaluation | F-score @0.01 | Surface voxel IoU, 64 | Mean unsquared Chamfer |
|---|---:|---:|---:|
| R9700, corrected defaults | 0.21465 | 0.22300 | 0.060088 |
| 8060S, compatibility port with stable runtime | 0.21568 | 0.22553 | 0.060282 |
| 8060S, corrected defaults | 0.21602 | 0.22546 | 0.060281 |

All current comparisons meet the same paired-mean equivalence criterion.
Changes and intervals below are percentage points:

| Comparison / metric | Mean change | 95% paired bootstrap interval |
|---|---:|---:|
| R9700, original port to corrected, F-score | +0.0276 | [-0.0815, +0.1799] |
| R9700, original port to corrected, IoU | +0.0352 | [-0.0708, +0.1632] |
| 8060S, common stable runtime, F-score | +0.0341 | [-0.0882, +0.1588] |
| 8060S, common stable runtime, IoU | -0.0071 | [-0.1002, +0.0816] |
| 8060S, historical raw port to corrected, F-score | -0.0691 | [-0.3266, +0.1539] |
| 8060S, historical raw port to corrected, IoU | -0.0270 | [-0.2510, +0.1835] |

These aggregate results measure preservation of mean shape quality. They do not
promise unchanged quality on each object. The numerical comparisons above are
separate implementation diagnostics, not Meta benchmark acceptance criteria.

These are local metrics with explicit conventions, rather than official
leaderboard scores. The public 100-object split differs from the paper's
aggregate evaluation; published aggregate scores are not a direct reference
for this subset. Artist meshes are untextured: these checks measure shape
after alignment, not texture quality or camera-pose accuracy. Complete model
agreement with an NVIDIA CUDA run is a separate, unmeasured property.


### Benchmark coverage

Meta releases [100 reference image/mesh pairs and another 900 held-out benchmark
inputs](https://ai.meta.com/datasets/sa-3dao-sam-3d-artist-objects-downloads/).
The public 100 references are fully covered here. Meta’s [paper, Table 2 and
Appendix D.3.1](https://arxiv.org/html/2511.16624v2#A4.SS3.SSS1), reports four
shape metrics on its evaluation set:

| Metric | Local coverage | Comparison limit |
|---|---|---|
| F-score @0.01 | 1M samples per mesh after normalization/ICP | Same definition; different evaluation set and local registration implementation |
| Voxel IoU | 64³ surface occupancy | Voxel origin/bounds and alignment conventions are explicit local choices |
| Chamfer distance | Mean/sum unsquared and sum squared directional distances | Paper does not specify all reduction conventions |
| Earth Mover’s Distance / W2 | Exact weighted-centroid transport using all 1M sample weights | Spatial compression remains approximate; Meta’s Sinkhorn settings are unpublished |

The published SAM 3D aggregates are F-score **0.2344**, voxel IoU **0.2311**,
Chamfer **0.0400** and EMD **0.1211**. They are not direct targets for our
100-object subset: matching numbers on different inputs would not establish
benchmark parity. No official evaluator or compatible per-case baseline was
found in the released model repository or the leaderboard’s static frontend.
The hosted [official leaderboard](https://huggingface.co/spaces/facebook/sa3dao-leaderboard)
evaluation API returned HTTP 503 during the 2026-10-22 check. Official benchmark
parity remains unmeasured; it requires the hosted evaluation service or a
released offline evaluator and a compatible reference.

As a sensitivity check, the quality confidence intervals were also recomputed
with 10,000 paired bootstrap resamples grouped by the **81 distinct decoded RGB
images**, retaining all objects from each sampled image. This accounts for
objects sharing an input image. Both F-score and IoU intervals remain inside
the same +/-0.5 percentage-point equivalence margin. Current-default minus
original-port intervals below are percentage points:

| Device | F-score interval | Voxel-IoU interval |
|---|---:|---:|
| R9700 | [-0.0786, +0.1809] | [-0.0728, +0.1613] |
| 8060S | [-0.3349, +0.1580] | [-0.2531, +0.1819] |

This supports preservation of average quality on these references, not a claim
that all objects are accurate or that all four official benchmark metrics pass.


### Four-metric local evaluation

All four 100-object cohorts now have all four local shape metrics. The EMD
column uses **Wasserstein-2**, consistent with the official frontend’s metric
label: the square root of minimum squared Euclidean transport cost. Lower is
better for Chamfer and EMD; higher is better for F-score and IoU.

| Device / version | F-score @0.01 | Voxel IoU | Mean unsquared Chamfer | Weighted EMD / W2 |
|---|---:|---:|---:|---:|
| R9700, compatibility port | 0.21437 | 0.22265 | 0.060131 | 0.218852 |
| R9700, corrected defaults | 0.21465 | 0.22300 | 0.060088 | 0.218561 |
| 8060S, historical compatibility port | 0.21671 | 0.22573 | 0.060391 | 0.221372 |
| 8060S, corrected defaults | 0.21602 | 0.22546 | 0.060281 | 0.219528 |

EMD reuses the existing normalization, rigid registration and one million
surface samples per mesh. Every sample contributes its mass: occupied grid
cells are represented by their weighted centroids, rather than discarding
points through random subsampling. A common per-object pitch was selected
across all four cohorts before EMD scoring, using the finest of 0.02, 0.03,
0.04 and 0.06 that kept every mesh below 16,384 centroids. The included
`emd_grid_map.json` records those pitches. The default cap of 24,576 leaves
headroom for rerun variation; exceeding it fails explicitly.

POT 0.9.7.post1 solves the weighted transport problem in float64 with network
simplex. All **400 primary solves and 48 finer-grid solves** reached certified
optima. The maximum absolute primal/dual squared-cost discrepancy was
**3.45e-10**. Thirty-three initial solves exceeded a one-million-iteration
budget; those attempts were retained and all completed with ten million
iterations. The public checker uses that larger budget. Integer empirical
masses and a fixed support permutation improve solver conditioning without
changing the transport objective; a difficult full-size case matched the
original optimum within 3e-16.

Independent GPU Sinkhorn checks comprise 400 primary scores, 48 finer-grid
scores and 48 smaller-regularization scores. All **496** primal/dual intervals
contained their corresponding exact weighted optimum within 1e-7. However,
71 exceeded the chosen numerical-gap target; smaller regularization alone did
not reliably improve finite-iteration accuracy. Therefore the table uses the
certified CPU optima, and the default `--emd` solver is exact. Sinkhorn remains
an explicit option, with mass-correction and numerical-bound diagnostics.
These evaluator choices do not affect model inference or its latency.

The finer-grid check covers 12 preselected objects on all four cohorts. Its
maximum absolute EMD change was **0.000702**. Exactness here applies to the
weighted centroid distributions, not the uncompressed million-point problem.
The checker also reports conservative bounds on that full empirical W2:
subtract the two RMS quantization displacements from the compressed lower
bound, and add their squared values to the compressed upper-bound squared
cost before taking its square root. Across the 100 cases, mean lower/upper
bounds are 0.192201/0.219991 and 0.191916/0.219714 for R9700 port/final, and
0.194712/0.222511 and 0.192879/0.220675 for 8060S port/final.

| EMD, corrected minus port | Mean change | Relative mean change | 95% paired image-cluster bootstrap interval |
|---|---:|---:|---:|
| R9700 | -0.000291 | -0.13% | [-0.001269, +0.000496] |
| 8060S | -0.001844 | -0.83% | [-0.007133, +0.002128] |

Intervals use 10,000 paired resamples, seed 42, grouped by the 81 distinct RGB
images. Both contain zero: the slightly lower means do not establish a
statistically significant quality improvement. The predeclared +/-0.005 mean
equivalence margin applies to F-score and IoU only. No Chamfer or EMD acceptance
margin was predeclared, so neither is assigned a retrospective pass threshold.
All four metrics are available locally; official benchmark parity remains
unmeasured because the evaluation set and unpublished evaluator conventions
differ. This does not certify every reconstruction as accurate.

Reproduce the controls and score existing meshes without weights or a token:

```sh
ryzers run "python /ryzers/check_sam3d_dataset.py --self-test --emd"
ryzers run "python /ryzers/check_sam3d_dataset.py --dataset-root /ryzers/output/sa3dao --predictions /ryzers/output/sa3dao-predictions --output /ryzers/output/sa3dao-four-metrics.json --emd --emd-grid-map /ryzers/sam3d_emd_grids.json"
```

The controls include analytic identity/translation and unequal-mass cases,
independent tiny assignment enumeration, 12 Sinkhorn comparisons with exact
transport, mass/moment preservation and full-sample compression bounds.
Final image smoke tests passed on both GPUs. Adding EMD left the original
F-score, IoU and Chamfer fields unchanged in a full-million-sample CLI check;
a missing prediction returned a nonzero exit status. Dense transport requires
quadratic memory in centroid count; run fewer concurrent evaluations on UMA
systems to bound total RAM use.


### Fresh-process cold-start coverage

With the proposed defaults, **24/24 cold starts passed on each GPU**: two fresh
processes per input across 12 preselected inputs. These include the original
high-resolution kidsroom mask-14 reproduction, 11 SA-3DAO objects, portrait and
landscape images, and the counter, telephone booth, perfume, eagle statue and
printer depth-outlier cases. Every planned attempt is retained; none was
retried or omitted. The first request after model construction must complete,
export a nonempty finite nondegenerate mesh, produce finite pose and Gaussian
attributes (position, features, scale, rotation and opacity), and contain finite
depth points. Invalid depth outside the valid scene region is not classified
as an inference failure. No depth hooks, recorded depth or ground-truth model
inputs are used. Model/kernel caches persist as in normal use. Both GPUs run
concurrently for this reliability check; its elapsed times are not performance
benchmarks.

In the earlier targeted allocator reproduction on one input under load, all
**4/4** fresh processes with expandable segments produced nonfinite first-pass
depth, versus **0/4** with native allocation. The newer 48-process sweep reports
zero reconstruction/finite-output failures under the corrected settings. These
are counts for the tested conditions, not a population failure-rate estimate,
a proof that no bugs remain, or a four-metric benchmark score.
