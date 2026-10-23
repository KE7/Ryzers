# SAM 3D Objects

Reconstruct an object's mesh, vertex colors, pose and Gaussian splat from an image
and segmentation mask using Meta's [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects).
Runs on Ryzen AI integrated graphics and discrete Radeon GPUs using ROCm PyTorch.

## Build and run

```sh
ryzers build sam3d
ryzers run
```

The default runs a headless smoke test: imports SAM 3D, checks the ROCm GPU and
compares 1x1 sparse convolution with a dense CPU PyTorch reference. It exits on
completion and requires no model download or Hugging Face token. First
reconstruction compiles Triton kernels; the kernel cache persists in
`workspace/.cache/triton`.

## Reconstruct the example

Accept the [model license](https://huggingface.co/facebook/sam-3d-objects) and
export a Hugging Face token with access to that repository:

```sh
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
ryzers run "python /ryzers/test_sam3d.py"
```

This runs the upstream kidsroom example, mask 14, seed 42, default inference
steps and both geometry decoders, without `torch.compile`. Upstream recommends
at least 32 GB GPU memory. Weights download on first reconstruction and persist
in `workspace/.cache/huggingface` and `workspace/.cache/torch`. Exported PLY splats,
vertex-colored GLB meshes and validation JSON persist in `workspace/sam3d`.
The token is forwarded at runtime; keep it out of `config.yaml` and build arguments.

Select a GPU on machines with multiple devices. List devices first; their
indices can change after a reboot:

```sh
ryzers run "python -c 'import torch; print([(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())])'"
HIP_VISIBLE_DEVICES=0 ryzers run "python /ryzers/test_sam3d.py"  # replace 0 with the chosen index
```

Additional masks and multi-object scene assembly are explicit checks:

```sh
ryzers run "python /ryzers/test_sam3d.py --mask-index 14 3"
ryzers run "python /ryzers/test_sam3d.py --all-masks"
ryzers run "python /ryzers/test_sam3d.py --check-repeatability"
```

The repeatability check compares one cold reconstruction with three warm runs,
requiring exact sparse coordinates and pose, SLAT normalized RMS error at most
1%, and sampled mesh p99 distance at most 0.5% of the reference bounding-box
diagonal. It requires the model weights and runs one mask.

## ROCm support and validation

The image pins upstream sources and retains the base ROCm PyTorch installation.
Both validated GPUs use Triton sparse convolutions, independent unmasked SDPA,
bounded conditioning caches and architecture-tuned FP16 sparse-latent attention.
The default also compacts constant zero-conditioning attention, avoids Q/K/V
repacking and uses the native GEMM/bias/GELU epilogue for eligible latent MLPs.
Structure sampling preserves intermediate rounding; its RMS normalization tail
is fused on R9700 and retains PyTorch on 8060S. The iGPU reuses a faster MLP
weight layout.

Four DINO preprocessing wrappers share one byte-identical frozen backbone;
matching backbone features are reused across stages. Frozen structure-sampler
linear weights use the FP16 storage already used by autocast, while normalization
parameters remain FP32. A shared NumPy buffer for PLY export reduces temporary
allocations, including RAM shared with the iGPU.
These optimizations are enabled by default, with guards for supported shapes,
precision and inference mode. See [VALIDATION.md](VALIDATION.md) for overrides.

The image uses the native PyTorch allocator because expandable segments caused
nonfinite depth tensors on 8060S. On `gfx1151`, convolution solver selection
defaults to `MIOPEN_FIND_MODE=FAST` to avoid depth variation between fresh
processes; an explicit runtime setting takes precedence. Other GPUs retain
their existing convolution solver policy.

Tested on Radeon 8060S (`gfx1151`) and Radeon AI PRO R9700 (`gfx1201`),
ROCm 7.14 / PyTorch 2.12. Under the same native-allocator/FAST runtime,
8060S warm mask-14 medians fell **134.48 → 51.89 s (61.4% less latency)**
from the first working port to corrected defaults, using default steps and both
geometry decoders. Peak allocated memory fell **18.88 → 13.75 GiB**.
The stability settings cost 0.82 s (1.6%) versus the former merged defaults.
Both captured benchmark pairs and all 100 dataset pairs meet the four numerical
limits under that common runtime; historical DEFAULT-mode outputs differ.

Both devices exported all 27 objects and assembled the complete scene. The
8060S peak allocated memory including assembly is **14.80 GiB**, excluding
checkpoint loading. The R9700 100-case cohort passes, but a separate fresh-process
benchmark pair fails exact coordinates and pose; the unchanged port also varies
between fresh default-mode processes. This remains unresolved. See
[VALIDATION.md](VALIDATION.md) for all reference comparisons, maximum errors,
historical results and measurement scopes.

Optional mesh cleanup, texture baking, layout optimization and Gaussian video
rendering need additional GPU rasterizers and are not supported here. The
upstream example disables these paths; PyTorch3D/gsplat are installed without
their CUDA rasterizers.

## Optional reference-dataset check

The [SA-3DAO public data](https://ai.meta.com/datasets/sa-3dao-sam-3d-artist-objects/)
provides artist-created reference meshes for independent shape evaluation.
The optional dataset checker reconstructs those inputs and computes CPU mesh
metrics; it is separate from the default smoke test. See
[dataset validation](VALIDATION.md#sa-3dao-reference-evaluation) for the data
layout, commands, scoring conventions and coverage. The 400 historical and
300 current reconstructions and scores completed; paired mean F-score and IoU
confidence intervals met the predeclared margin of +/-0.5 percentage points.
These aggregate quality checks do not replace per-case numerical agreement.

## References

- [Upstream setup](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)
- [Single-object notebook](https://github.com/facebookresearch/sam-3d-objects/blob/main/notebook/demo_single_object.ipynb)
- [Multi-object notebook](https://github.com/facebookresearch/sam-3d-objects/blob/main/notebook/demo_multi_object.ipynb)

Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
