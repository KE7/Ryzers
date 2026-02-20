# Running Ryzers on WSL2 with AMD Ryzen AI

This guide documents the changes and setup required to run Ryzers on Windows Subsystem for Linux 2 (WSL2) with AMD Ryzen AI hardware (e.g., Ryzen AI MAX+ 395 with Radeon 8060S iGPU).

## Overview

Ryzers is designed for native Linux with `/dev/kfd` and `/dev/dri` GPU device access. On WSL2, GPU compute is accessed through `/dev/dxg` via Microsoft's DXG kernel interface. The [librocdxg](https://github.com/ROCm/librocdxg) library bridges ROCm's HSA runtime to this DXG interface, enabling GPU-accelerated compute workloads inside Docker containers.

## Prerequisites

- Windows 11 with WSL2 enabled
- AMD Ryzen AI hardware (Ryzen AI MAX/HX series or Radeon RX 7000/9000 series)
- AMD GPU driver installed on Windows ([download](https://www.amd.com/support/download/drivers.html))
- Docker installed inside WSL2
- ROCm 7.2+ installed on the WSL2 host (for librocdxg)

## Step 1: Install librocdxg

[librocdxg](https://github.com/ROCm/librocdxg) enables ROCm on WSL2 by translating HSA runtime calls to the DXG kernel interface. Follow the installation instructions in the librocdxg repository:

```bash
cd ~/Documents/open-source
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg
# Follow the build/install instructions in the README
```

After installation, verify it works:

```bash
HSA_ENABLE_DXG_DETECTION=1 rocminfo
```

You should see your AMD GPU listed (e.g., `gfx1151` for Ryzen AI MAX+ 395).

## Step 2: Code Changes

Two modifications are required to the upstream Ryzers repository:

### 2a. ROS Distribution (jazzy)

RAI requires the `jazzy` ROS 2 distribution. Edit `packages/ros/ros/config.yaml`:

```yaml
build_arguments:
  - "ROS_DISTRO=jazzy"
```

### 2b. OpenCV Conflict Fix

RAI's `poetry install` fails due to a conflict between `opencv-python` (installed by ROS) and `opencv-python-headless` (required by RAI). Add the following line to `packages/robotics/rai/Dockerfile` before the `git clone` step:

```dockerfile
# Remove opencv-python to avoid conflict with opencv-python-headless from RAI
RUN pip uninstall -y opencv-python opencv-python-headless || true

# Clone and install RAI framework
WORKDIR /ryzers
RUN git clone https://github.com/RobotecAI/rai.git
```

## Step 3: Build

Build the full stack as normal:

```bash
pip install Ryzers/
ryzers build ros o3de rai ollama
```

This chains: `rocm/pytorch:rocm7.0` -> `ryzer_env` -> `ros` -> `o3de` -> `rai` -> `ollama` (`ryzerdocker:latest`).

## Step 4: Running on WSL2

The auto-generated `ryzers run` script uses native Linux GPU flags (`--device=/dev/kfd --device=/dev/dri`) which don't exist on WSL2. Instead, run Docker manually with librocdxg flags:

```bash
docker run -it --rm \
    -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so \
    -v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so \
    -v /opt/rocm/lib/libhsa-runtime64.so.1.18.70200:/opt/rocm/lib/libhsa-runtime64.so.1 \
    -e HSA_ENABLE_DXG_DETECTION=1 \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    --device=/dev/dxg \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --shm-size 16G \
    --network=host \
    -v $PWD/workspace/.ollama:/root/.ollama/ \
    -v $PWD/experiments:/ryzers/rai/src/rai_bench/rai_bench/experiments \
    ryzerdocker:latest bash
```

### Key flags explained

| Flag | Purpose |
|------|---------|
| `-v .../libdxcore.so:/usr/lib/libdxcore.so` | WSL2 DirectX core library for GPU access |
| `-v .../librocdxg.so:/usr/lib/librocdxg.so` | ROCm-DXG bridge library |
| `-v .../libhsa-runtime64.so.1.18.70200:/opt/rocm/lib/libhsa-runtime64.so.1` | Host ROCm 7.2 HSA runtime (overrides container's ROCm 7.0 which doesn't support librocdxg) |
| `-e HSA_ENABLE_DXG_DETECTION=1` | Tells HSA runtime to use DXG path |
| `-e HSA_OVERRIDE_GFX_VERSION=11.0.0` | GPU architecture override for gfx1151 |
| `--device=/dev/dxg` | Expose WSL2 DXG device (replaces `/dev/kfd` + `/dev/dri`) |

### Adjusting HSA_OVERRIDE_GFX_VERSION

The `HSA_OVERRIDE_GFX_VERSION` value depends on your GPU. Check your device ID:

```bash
cat /sys/bus/pci/devices/*/device  # Look for your GPU's PCI device
HSA_ENABLE_DXG_DETECTION=1 rocminfo  # Shows gfx version
```

Common values:
- `gfx1151` (Ryzen AI MAX+ 395): `11.0.0`
- `gfx1100` (Radeon RX 7900): `11.0.0`
- `gfx1036` (Radeon 680M): `10.3.6`

## Step 5: Verify GPU Access

Inside the container:

```bash
# ROCm detection
rocminfo 2>&1 | grep -i "name\|gfx"

# PyTorch GPU (torch.cuda is the generic GPU API on ROCm)
python3 -c 'import torch; print(torch.cuda.is_available())'
```

Note: `torch.cuda` is the generic way to access the GPU in PyTorch. On AMD hardware with ROCm, `torch.cuda.is_available()` returns `True` and all `torch.cuda.*` APIs work through ROCm's HIP compatibility layer.

## Known Limitations

### O3DE / Vulkan Rendering

The O3DE manipulation benchmark requires Vulkan GPU rendering. On WSL2, O3DE currently **does not work** due to Vulkan driver limitations. O3DE maintainers have stated that WSL-G support is ["not planned"](https://github.com/o3de/o3de/issues/13439).

#### Background: DRI devices on WSL2

`/dev/dri/renderD128` is needed for Vulkan. On WSL2 kernel 6.6+, the `vgem` module changed from built-in to loadable:

```bash
sudo modprobe vgem
ls /dev/dri/  # Should show card0 and renderD128
```

#### Vulkan drivers tested

**Mesa dzn (D3D12-backed Vulkan)**: Available via `ppa:kisak/kisak-mesa`. Detects the GPU correctly ("Microsoft Direct3D12 (AMD Radeon 8060S Graphics)") but O3DE crashes during shader resource allocation with "D3D12: Removing Device." dzn is officially "not a conformant Vulkan implementation" and cannot handle O3DE's Vulkan requirements.

```bash
# Install dzn inside container:
add-apt-repository -y ppa:kisak/kisak-mesa
apt-get install -y mesa-vulkan-drivers
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.json vulkaninfo --summary
```

**llvmpipe (software Vulkan)**: O3DE initializes further (RHI device creation succeeds, begins texture upload) but crashes with a SIGSEGV in O3DE's custom memory allocator (`HphaAllocator::tree_free`) due to a race condition when llvmpipe calls the Vulkan free callback from a graphics thread. The crash is timing-dependent (does not reproduce under gdb).

**`--rhi=null` (null renderer)**: O3DE starts successfully but produces no rendered output. The manipulation benchmark requires camera image topics (`/color_image5`, `/depth_image5`) which need actual rendering.

#### Docker flags for WSLg display + Vulkan (for testing)

```bash
docker run -it --rm \
    --device=/dev/dxg \
    --device=/dev/dri/card0 \
    --device=/dev/dri/renderD128 \
    -v /usr/lib/wsl:/usr/lib/wsl \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/wslg/.X11-unix:/mnt/wslg/.X11-unix \
    -v /mnt/wslg/runtime-dir:/mnt/wslg/runtime-dir \
    -e DISPLAY=:0 \
    -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir \
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    -e MESA_D3D12_DEFAULT_ADAPTER_NAME=AMD \
    -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/dzn_icd.json \
    ...
```

#### What would fix this

- Microsoft improving dzn to full Vulkan 1.2+ conformance ([wslg#1340](https://github.com/microsoft/wslg/issues/1340))
- O3DE fixing the allocator race condition with software Vulkan drivers
- A native RADV or AMDVLK driver path through WSL2's DXG interface

### PyTorch ROCm vs CUDA

RAI's `poetry install` replaces the ROCm-enabled PyTorch (`torch 2.6.0+rocm7.0.0`) with a CUDA build (`torch 2.3.1+cu121`). This does not affect the benchmarks since they use Ollama for inference, not PyTorch directly. Ollama has its own ROCm-based inference engine.

## Troubleshooting

### "No HSA devices found"
- Ensure `HSA_ENABLE_DXG_DETECTION=1` is set
- Verify librocdxg is installed and the `.so` is bind-mounted
- Check that the host HSA runtime (ROCm 7.2+) is mounted over the container's version

### Docker build cache corruption
If you see `parent snapshot does not exist` during builds:
```bash
docker builder prune -f
```

### Ollama not detecting GPU
Check Ollama logs for `total_vram="0 B"`. This means the GPU isn't detected. Ensure all librocdxg bind mounts and environment variables are correct.
