# RAI Benchmarks Guide

This guide covers how to build, run, and interpret the three RAI benchmarks included in the Ryzers project: Tool Calling, VLM (Vision Language Model), and O3DE Manipulation.

## Architecture

All three benchmarks use [Ollama](https://ollama.com/) as the local LLM inference server. Ollama runs models on the AMD iGPU via ROCm and exposes an HTTP API at `http://127.0.0.1:11434`. The benchmark scripts communicate with Ollama through LangChain's `ChatOllama` integration.

```
Benchmark Script --> LangChain --> Ollama API (:11434) --> LLM on GPU
                                                           (ROCm/HIP)
```

## Building the Docker Images

### Full stack build (all benchmarks)

```bash
pip install Ryzers/
ryzers build ros o3de rai ollama
```

This builds the following chain of Docker images:

| Step | Image | Description |
|------|-------|-------------|
| 1 | `ryzer_env` | Base environment from `rocm/pytorch:rocm7.0_ubuntu24.04_py3.12_pytorch_release_2.6.0` |
| 2 | `ros` | ROS 2 (jazzy) with core packages |
| 3 | `o3de` | Open 3D Engine for simulation (needed for manipulation benchmark) |
| 4 | `rai` | RAI framework with benchmarks, MoveIt, Gazebo messages |
| 5 | `ollama` (tagged `ryzerdocker:latest`) | Ollama LLM inference server |

### Minimal build (tool calling + VLM only)

If you don't need the manipulation benchmark (which requires O3DE and a display):

```bash
ryzers build ros rai ollama
```

## Running the Container

### Native Linux

```bash
ryzers run bash
```

### WSL2 (with librocdxg)

See [WSL2_SETUP.md](WSL2_SETUP.md) for the full docker run command with GPU passthrough flags.

## Starting Ollama

Inside the container, start the Ollama server and pull the required models:

```bash
# Start Ollama server in the background
ollama serve &

# Wait for server to be ready
sleep 5

# Pull models for benchmarks
ollama pull qwen2.5:7b    # Used by tool calling and manipulation benchmarks
ollama pull gemma3:4b      # Used by VLM benchmark
```

### How Ollama works with --network=host

The container runs with `--network=host`, meaning Ollama binds to `127.0.0.1:11434` on the host's network namespace. This means:

- Only one Ollama server can run at a time across all containers using `--network=host`
- Multiple containers can share the same Ollama server (they all see `127.0.0.1:11434`)
- The model cache is persisted via the volume mount `-v $PWD/workspace/.ollama:/root/.ollama/` so models don't need to be re-downloaded between runs

### Using Ollama with a custom host

To run Ollama on a different address or port:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve &
```

The `OLLAMA_HOST` environment variable controls the bind address. Setting it to `0.0.0.0` makes it accessible from other machines on the network.

## Setup ROS Environment

Before running any benchmark, source the ROS 2 environment:

```bash
cd /ryzers/rai
source install/setup.bash
source /opt/ros/jazzy/setup.sh
```

## Benchmark 1: Tool Calling

Tests an LLM agent's ability to correctly invoke ROS 2 tools (service calls, topic queries, image capture) to accomplish tasks.

### Description

The agent receives a task description and has access to ROS 2 tools like:
- `get_ros2_topics_names_and_types` - List available topics
- `get_ros2_services_names_and_types` - List available services
- `get_ros2_message_interface` - Get message type definitions
- `call_ros2_service` - Call a ROS 2 service
- `get_ros2_image` - Capture an image from a camera topic
- `receive_ros2_message` - Read a message from a topic

The benchmark evaluates whether the agent calls the correct tools with the correct arguments in the correct order.

### Run command

```bash
python src/rai_bench/rai_bench/examples/tool_calling_agent.py \
    --model-name qwen2.5:7b \
    --vendor ollama \
    --extra-tool-calls 5 \
    --task-types basic \
    --n-shots 5 \
    --prompt-detail descriptive \
    --complexities easy
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--model-name` | Ollama model name (e.g., `qwen2.5:7b`, `llama3.1:8b`) |
| `--vendor` | LLM vendor (`ollama`, `openai`, `aws`) |
| `--extra-tool-calls` | Additional tool calls allowed beyond the minimum required |
| `--task-types` | Task categories to run (`basic`, `advanced`, or both) |
| `--n-shots` | Number of few-shot examples in the system prompt |
| `--prompt-detail` | Detail level of task descriptions (`descriptive`, `concise`) |
| `--complexities` | Task difficulty levels (`easy`, `medium`, `hard`) |

### What the benchmark measures

Each task has a set of expected tool calls with expected arguments. The benchmark checks:
- **Tool name correctness**: Did the agent call the right tool?
- **Argument correctness**: Were service names, types, and field values correct?
- **Ordering**: For ordered tasks, were tools called in the right sequence?
- **Efficiency**: How many extra tool calls were used beyond the minimum?

## Benchmark 2: VLM (Vision Language Model)

Tests a vision-language model's ability to answer questions about images of a simulated room environment.

### Description

The model receives an image of a room and a yes/no question about its contents (e.g., "Is there a plant behind the rack?", "Are there 3 pictures on the wall?"). The benchmark tests spatial reasoning, object detection, counting, and attribute recognition.

### Run command

```bash
python src/rai_bench/rai_bench/examples/vlm_benchmark.py \
    --model-name gemma3:4b \
    --vendor ollama
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--model-name` | Ollama model with vision capabilities (e.g., `gemma3:4b`, `llava:7b`) |
| `--vendor` | LLM vendor (`ollama`, `openai`, `aws`) |

### What the benchmark measures

Each task is a boolean (True/False) question about an image. The model must output a structured response. The benchmark measures:
- **Accuracy**: Percentage of correct True/False answers
- **Response time**: How long each inference takes
- Tasks cover positive identification ("Do you see X?"), negatives ("Is someone in the room?" when empty), counting ("Are there 4 pictures?"), and spatial relations ("Is X on the left of Y?")

## Benchmark 3: O3DE Manipulation

Tests an LLM agent's ability to control a robotic arm in a 3D simulation to manipulate objects on a table.

### Description

Uses the Open 3D Engine (O3DE) to render a simulation with a Panda robotic arm and objects on a table. The agent must use ROS 2 MoveIt services to pick, place, and arrange objects. This benchmark requires a display for O3DE rendering.

### Prerequisites

- O3DE must be included in the build (`ryzers build ros o3de rai ollama`)
- A display must be available (X11 forwarding on native Linux, WSLg on WSL2)
- Vulkan GPU rendering support (O3DE requires hardware Vulkan, software renderers like lavapipe are insufficient)

### Run command

```bash
python src/rai_bench/rai_bench/examples/manipulation_o3de.py \
    --model-name qwen2.5:7b \
    --vendor ollama \
    --levels trivial
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--model-name` | Ollama model name |
| `--vendor` | LLM vendor |
| `--levels` | Difficulty levels (`trivial`, `easy`, `medium`, `hard`) |

### What the benchmark measures

The agent must use ROS 2 services (`/spawn_entity`, `/delete_entity`, MoveIt planning) to manipulate objects. Tasks range from simple entity management to complex pick-and-place sequences.

## Results

### Output location

Results are saved to the `experiments/` directory (mounted from the host via `-v $PWD/experiments:/ryzers/rai/src/rai_bench/rai_bench/experiments`):

```
experiments/
  tool_calling/
    2026-02-20_01-14-34/
      results_summary.csv
      results.csv
      benchmark.log
  vlm_benchmark/
    2026-02-20_01-34-14/
      results_summary.csv
      results.csv
  manipulation_o3de/
    <timestamp>/
      results_summary.csv
      results.csv
```

### Reading results_summary.csv

**Tool calling:**
```csv
model_name,success_rate,avg_time,total_tasks,total_extra_tool_calls_used
qwen2.5:7b,53.33,114.426,15,45
```

- `success_rate`: Percentage of tasks completed correctly (0-100)
- `avg_time`: Average seconds per task
- `total_tasks`: Number of tasks attempted
- `total_extra_tool_calls_used`: Total additional tool calls beyond minimum required

**VLM:**
```csv
model_name,success_rate,avg_time,total_tasks
gemma3:4b,50.0,26.483,14
```

- `success_rate`: Percentage of correct True/False answers (0-100)
- `avg_time`: Average seconds per task
- `total_tasks`: Number of questions answered

### Reading results.csv

The detailed `results.csv` contains per-task results with columns:
- `task_prompt`: The task description given to the agent
- `complexity`: Task difficulty level
- `score`: 1.0 for success, 0.0 for failure
- `total_time`: Seconds taken for this task
- `validation_info`: Detailed breakdown of expected vs. actual tool calls (tool calling benchmark)
- `extra_tool_calls_used`: How many extra calls the agent needed

### Current Results (AMD Ryzen AI MAX+ 395 / Radeon 8060S, WSL2)

| Benchmark | Model | Success Rate | Avg Time/Task | Tasks |
|-----------|-------|-------------|---------------|-------|
| Tool Calling | qwen2.5:7b | 53.33% | 114.4s | 15 |
| VLM | gemma3:4b | 50.0% | 26.5s | 14 |
| Manipulation | qwen2.5:7b | N/A | N/A | Blocked (O3DE Vulkan crash on WSL2) |

#### Tool Calling Breakdown (qwen2.5:7b)

| Task | Complexity | Score | Time (s) |
|------|-----------|-------|----------|
| Get all camera images | medium | 1.0 | 36.8 |
| Configure AI vision pipeline (3 params) | hard | 0.0 | 245.1 |
| Check available spawnable entities | easy | 0.0 | 192.2 |
| Set publish frequency to 25.0 Hz | medium | 0.0 | 48.3 |
| Reconfigure simulation (delete + spawn) | hard | 0.0 | 426.7 |
| Set publish frequency to 30.0 Hz | medium | 0.0 | 85.0 |
| Get all topics | easy | 1.0 | 31.4 |
| List robot state publisher parameters | easy | 0.0 | 41.7 |
| Get robot description | easy | 1.0 | 87.0 |
| Get joint states | easy | 1.0 | 19.5 |
| Get depth camera image | easy | 1.0 | 35.1 |
| Get RGB camera image | easy | 1.0 | 24.9 |
| Get all services | easy | 1.0 | 14.3 |
| Set grounding_dino confidence to 0.6 | medium | 0.0 | 94.1 |
| Get current robot joint positions | easy | 1.0 | 34.3 |

#### VLM Breakdown (gemma3:4b)

| Task | Expected | Score | Time (s) |
|------|----------|-------|----------|
| Is the door on the left from the desk? | True | 1.0 | 34.5 |
| Is the light on in the room? | True | 1.0 | 31.9 |
| Do you see the plant? | True | 1.0 | 6.1 |
| Are there any pictures on the wall? | True | 1.0 | 33.1 |
| Are there 3 pictures on the wall? | True | 1.0 | 33.9 |
| Is there a plant behind the rack? | True | 1.0 | 31.0 |
| Is there a pillow on the armchair? | True | 1.0 | 30.2 |
| Is the door open? | False | 0.0 | 35.2 |
| Is someone in the room? | False | 0.0 | 5.5 |
| Do you see the plant? | False | 0.0 | 31.2 |
| Are there 4 pictures on the wall? | False | 0.0 | 30.6 |
| Is there a rack on the left from the sofa? | False | 0.0 | 4.6 |
| Is there a plant on the right from the window? | False | 0.0 | 31.8 |
| Is there a red pillow on the armchair? | False | 0.0 | 31.1 |

The model scored 7/7 on "True" answers but 0/7 on "False" answers, indicating a strong bias toward positive responses.

#### O3DE Manipulation - WSL2 Status

The O3DE manipulation benchmark is blocked on WSL2 because O3DE crashes during Vulkan rendering initialization. O3DE's Atom renderer requires Vulkan 1.2+ with specific features that are not available through WSL2's Vulkan drivers. See [WSL2_SETUP.md](WSL2_SETUP.md#o3de--vulkan-rendering) for full details.

**gdb backtrace with dzn driver:**
```
#0  AZ::RHI::DeviceObject::GetDevice() const
#1  AZ::Vulkan::DescriptorPool::Allocate(DescriptorSetLayout const&)
...
#21 AZ::Render::Bootstrap::BootstrapSystemComponent::CreateViewportContext()
```
Device creates successfully but D3D12 removes it during first shader operation.

**gdb backtrace with llvmpipe:**
```
Thread "Graphic.t Queue" received signal SIGSEGV
#0  AZStd::intrusive_multiset::erase()
#1  AZ::HphaSchemaBase::HpAllocator::tree_free()
#2  ?? from libvulkan_lvp.so
...
#6  AZ::Vulkan::AsyncUploadQueue::QueueUpload
```
Race condition in O3DE's custom allocator when llvmpipe calls the Vulkan free callback.

This benchmark works on native Linux with AMD GPUs using the RADV Vulkan driver.

## Documentation

- RAI official docs: https://robotecai.github.io/rai/
- RAI benchmarks documentation: https://robotecai.github.io/rai/tutorials/benchmarking/
- RAI GitHub: https://github.com/RobotecAI/rai
- Ollama: https://ollama.com/
- Ryzers: https://github.com/AMDResearch/Ryzers
