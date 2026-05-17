---
title: "Pushing the Limits: Running a 9B Coding Agent on a 6GB Laptop"
date: 2026-05-18
author: Mukund K. Sharma
tags: [LLM, llama.cpp, NVIDIA, Ubuntu, AI-Agents, Claude-Code]
---

# Pushing the Limits: Running a 9B Coding Agent on a 6GB Laptop

Local AI development often feels like a privilege reserved for massive desktop rigs with 24GB of VRAM. But what happens when you want to run deep project planning using a state-of-the-art model like **Qwen 3.5 9B**, and all you have is a laptop with an Intel i7, 16GB of system RAM, and a rigid 6GB RTX 2060?

You play a brutal game of **VRAM Tetris**.

This is a detailed breakdown of how I configured my system, compiled a custom `llama.cpp` fork for extreme cache compression, and heavily optimized my server flags to successfully run **Qwen 3.5 9B** with a massive **48,000-token context window** — without triggering a single `cudaMalloc` failure.

---

## TL;DR: The Speed Run

By combining a community **TurboQuant** fork, **mixed-precision KV caching**, **single-sequence queue enforcement**, and **OS graphics optimization**, I squeezed a **48K context window** and a dense **9B parameter model** into a **6GB laptop GPU**. The final setup delivers a blazing **530.44 tok/s** prompt ingestion rate and a highly stable **7.75 tok/s** generation speed, acting as a completely local backend for **Claude Code**.

---

## Hardware Profile vs. Target Performance

Before diving into the steps, here is the hardware profile of my Lenovo Legion laptop used for this implementation alongside the final benchmark performance.

### System Specs

| Component | Specification |
| :--- | :--- |
| **Hardware Model** | Lenovo Legion Y540-15IRH |
| **Processor** | Intel Core i7-9750H (6 Physical Cores, 12 Hyper-threads) |
| **Memory** | 16.0 GB RAM (2 × 8 GB DDR4 Dual-Channel) |
| **Graphics** | NVIDIA GeForce RTX 2060 (6.00 GiB Dedicated VRAM) |
| **Disk** | 1.0 TB NVMe SSD |
| **OS** | Ubuntu 24.04 LTS (Kernel: Linux 6.17) |
| **Windowing System** | Wayland (GNOME 46) |

### Production Benchmarks

The metrics below reflect the real-world performance of this optimized setup under a massive 30,147-token context payload:

| Metric | Result |
| :--- | :--- |
| **Prompt Processing (pp)** | **530.44 tok/s** — chewing through 2,936 new context tokens in just 5.53 seconds |
| **Text Generation (tg)** | **7.75 tok/s** |
| **Context Success Limit** | **48,000 tokens** — fully tested and verified up to the maximum ceiling without failure |
| **Qualitative Stability** | **turbo4** V-cache compression yielded no observable degradation in system architecture reasoning, component mapping, or context recall |

---

**Failed Configurations:**

* `64,000 ctx` + `-ngl 26`: Survived initial launch but routinely encountered **OOM** crashes during active agent multi-tool calls.
* `48,000 ctx` + `-ngl 29` (with hardware-accelerated Chrome open): Allocation hit `5.95 / 6.00 GiB`, leaving under 50 MB free. Desktop interactions caused immediate `cudaMalloc` failures. Stable only after evicting Chrome from the GPU.

---

## Why Qwen 3.5 9B for a Local Agent?

When choosing a local model for autonomous coding agents, the default choice for restricted hardware is usually a **7B model**. However, architectural planning tasks are highly fragile. Small quantization failures can break an entire system design.

**Qwen 3.5 9B** acts as a dense powerhouse that drastically outperforms legacy architectures in deep multi-turn logical tracking, context compliance, and structural layout. While it is a dense model — meaning all **9.65 billion parameters** are fully calculated for every single token — its **Grouped Query Attention (GQA)** implementation makes its memory footprint uniquely malleable compared to its peers. It is the smallest model that genuinely behaves with the intelligence of an enterprise assistant, making it the perfect target for a hyper-optimized laptop setup.

Here is a brief comparison of why I eliminated the alternatives:

| Model | Params | Architecture | Why not? |
|-------|--------|-------------|----------|
| CodeLlama 7B | **7B** | Dense | Weaker multi-turn tracking, outdated training data |
| DeepSeek Coder V2 Lite | 16B (3B active) | MoE | Sparse MoE inference not well-supported in `llama.cpp` forks |
| Qwen 2.5 Coder 7B | **7B** | Dense | Strong, but the **9B jump to Qwen 3.5** provides measurably better architectural planning and system design |
| **Qwen 3.5 9B** | **9.65B** | **Dense + GQA** | **Best intelligence-to-VRAM ratio with compressible V-cache** |

For instance, where a **7B model** often lost track of component interactions and API contracts while architecting small projects like a **chat service** or a **blob store**, **Qwen 3.5 9B** perfectly maintained the system design constraints across long contexts.

---

## Anchoring the Workflow: The Coding Agent

This local **llama.cpp** server configuration wasn't built for a simple chat interface. It was specifically optimized to act as the primary API backend for **Claude Code**, an autonomous, agentic command-line tool.

When Claude Code executes a task, it doesn't just chat; it reads whole directory trees, runs terminal commands, inspects files, and attempts to resolve complex software architectures synchronously. This creates massive, rapid spikes in context data, meaning the underlying server must be rock-solid when tracking context shifts over thousands of tokens.

**Claude Code** connects to the local server via its OpenAI-compatible API endpoint at `http://localhost:8080/v1`. I configured it as my backend by setting the following environment variable before launching Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080/v1
```

With this, every architectural plan, file analysis, and terminal command routes through the local Qwen 3.5 9B instance.

---

## Phase 1: Getting the Source Code and Model

A standard pre-compiled **llama.cpp** binary lacks the experimental, bleeding-edge KV cache compression techniques required to survive this workflow. I needed a specific community implementation.

### 1. Cloning TheTom's TurboQuant Fork

Standard **llama.cpp** releases do not yet support the granular mix-and-match cache compression types I needed. I utilized the community implementation fork by **TheTom**, which introduces highly optimized 4-bit TurboQuant caches.

I cloned the specific feature branch:

```bash
cd ~
git clone https://github.com/TheTom/llama-cpp-turboquant.git
cd llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
```

### 2. Compiling with Native CUDA Support

To unlock the RTX 2060's Tensor Cores, I compiled the binary natively using the NVIDIA CUDA Toolkit (installed via `sudo apt install nvidia-cuda-toolkit`):

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)
```

This outputs a highly optimized `llama-server` binary at `build/bin/llama-server`.

### 3. Model Weight Extraction

I downloaded the **Q4_K_M (4-bit medium quantization)** `.gguf` weight files for **Qwen 3.5 9B** from Hugging Face repositories (via the **bartowski** collection) and placed it inside a clean target directory:

```bash
mkdir -p ~/models
mv ~/Downloads/Qwen3.5-9B-Q4_K_M.gguf ~/models/
```

---

## Phase 2: OS & Hardware Optimizations

Before launching the server, I needed to claw back every single available megabyte of VRAM and prevent the Linux kernel from introducing processing bottlenecks.

### 1. Evicting Chrome from VRAM

By default, Google Chrome, Brave, and Electron apps (like VS Code) hoard between 100 MB and 300 MB of VRAM to smoothly render UI elements. When navigating tight 6GB margins, this is a death sentence.

* **The Optimization:** Navigate to Chrome **Settings → System** → Disable *"Use graphics acceleration when available"*.
* This strips the browser's hardware access and forces all web rendering onto the CPU and system RAM, returning hundreds of megabytes directly back to the GPU's pool.

### 2. Overriding the CPU Governor

Because my hardware configuration requires a portion of the model's layers to overflow onto the CPU, the processor is subjected to sudden, intense bursts of computational load. Ubuntu's default `powersave` governor downclocks the CPU during millisecond idle gaps, introducing immediate latency when processing complex prompt inputs.

Before beginning an agent workflow, I forced the CPU to maintain its maximum `performance` clock state:

```bash
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

---

## Phase 3: Architecting the VRAM Tetris Board

To understand why this system remains perfectly stable, I mapped out the exact breakdown of how every megabyte of the **6.00 GiB VRAM** pool is allocated. By completely removing Chrome from the GPU, I was able to safely push the boundaries to 29 layers.

### The VRAM Budget Allocation Table

When running at a **48,000 token** limit with my chosen optimizations, the allocation maps out like a precise grid:

| Component | VRAM Allocation | Location |
| :--- | :--- | :--- |
| **Model Weights** (29 / 32 Layers, Q4_K_M) | ~4.38 GiB | GPU VRAM |
| **KV Cache Keys** (48K Context, `q8_0`) | ~398.44 MiB | GPU VRAM |
| **KV Cache Values** (48K Context, `turbo4`) | ~199.22 MiB | GPU VRAM |
| **CUDA Compute Buffers** (Graph Splits) | ~493.00 MiB | GPU VRAM |
| **Desktop Compositor Reserve** (OS Shell UI) | ~407.00 MiB | GPU VRAM |
| **Remaining Safety Overhead** | ~34.00 MiB | GPU VRAM |
| **Total Allocation Profile** | **5.91 / 6.00 GiB** | **GPU Confirmed** |

Remaining layers (only **3 layers** left behind!) and overflow context buffers are pushed safely into the system RAM pool, which actively draws roughly `7.36 GiB / 15.5 GiB` total during runtime.

---

## Phase 4: Slicing the Runtime Parameters

Here is the technical reasoning behind the specific parameter configuration choices I used to manage the active VRAM grid:

### 1. Surgical KV Cache Splits (`--cache-type-k` vs `--cache-type-v`)

Because Qwen 3.5 utilizes **Grouped Query Attention**, the Key and Value layers play different roles.

* The **Key Cache** acts as the model's navigational coordinates. Compressing it below 8-bit causes immediate mathematical degradation, leading to hallucinations. I preserved it strictly at `--cache-type-k q8_0`.
* The **Value Cache** holds the literal token payload data. It is highly resilient to intense compression. By deploying `--cache-type-v turbo4`, the Value cache footprint dropped perfectly by 50% without corrupting the model's underlying architectural logic.

### 2. Context Defragmentation (`--defrag-thold 0.1`)

Coding agents constantly mutate historical logs, causing fragmented spaces to build up across the memory layers. Setting the threshold to `0.1` ensures that once **memory fragmentation** hits 10%, `llama.cpp` instantly reorganizes the active memory segments, stopping slow leaks over extended sessions.

### 3. Thread Slicing (`-t` vs `-tb`)

* **`-t 6`**: Limits text generation strictly to my i7's 6 physical cores, bypassing hyper-threading overhead to speed up token execution.
* **`-tb 12`**: Instructs prompt ingestion to exploit all 12 hyper-threads simultaneously, accelerating data parsing across the motherboard bus.

### 4. Memory-Mapped Math and Processing Overhead

* **`-b 4096`**: The global **batch size**. Setting this to **4096** maximizes the tensor operations executed in parallel by the GPU during prompt processing, lowering the total time spent moving weights over the PCIe lane.
* **`-ub 512`**: The physical **mini-batch size (micro-batch)**. Slicing the execution down to 512-token chunks prevents large individual prompts from overflowing the allocation limits of the CUDA execution pool.
* **`-fa on`**: Enables **Flash Attention**. This bypasses standard quadratic memory scaling algorithms, radically lowering memory consumption across massive token spans.

---

## The Production Execution Command

Bringing all phases together results in the definitive, maxed-out command for local development environments with strict hardware parameters:

```bash
~/llama-cpp-turboquant/build/bin/llama-server \
  --model "/home/user/models/Qwen3.5-9B-Q4_K_M.gguf" \
  -ngl 29 \
  -c 48000 \
  -b 4096 \
  -ub 512 \
  -fa on \
  -t 6 \
  -tb 12 \
  --cache-type-k q8_0 \
  --cache-type-v turbo4 \
  -np 1 \
  --defrag-thold 0.1 \
  --port 8080 \
  --alias "qwen3.5-9b"
```

---

## The Takeaway

The bottleneck in local AI development is no longer intelligence — it's memory management. As cache compression techniques like **TurboQuant** mature and merge into mainline `llama.cpp`, the hardware floor for running production-quality coding agents will only keep dropping. The **6GB barrier** I navigated here will eventually look generous.

---

## A Structured Troubleshooting Reference Guide

If your local server throws errors, track the root cause using this systematic mapping built on my exact trial runs:

### 1. `cudaMalloc failed` or Sudden Server Crashes

* **Root Cause**: Your allocation requested more space than the hardware can physically handle. Pushing to `-ngl 29` without evicting Chrome leaves under **50 MB** of breathing room, causing the desktop compositor to trigger a crash during window changes.
* **Remedy**: If you cannot disable browser hardware acceleration, drop your layer count to `-ngl 28`. If crashes persist during massive repository reads, step your context down from `-c 48000` to `-c 40000` to expand your safety floor.

### 2. `Segmentation fault (core dumped)` During Active Planning Tasks

* **Root Cause**: Your coding agent attempted to trigger multiple parallel queries or inspect two code folders simultaneously. The system split the compute graph, causing memory boundaries to intersect.
* **Remedy**: Ensure `-np 1` is explicitly defined in your startup command. This safely forces sequential query evaluation.

### 3. Model Output Rebounds into Unreadable Gibberish or Broken Logic

* **Root Cause**: Attempting to force extreme compression like `turbo3` or `turbo2` onto the context matrix, or running a low quantization on the Key cache layer.
* **Remedy**: Restore your configuration cleanly to `--cache-type-k q8_0` and restrict high-compression exclusively to the Value cache layer (`--cache-type-v turbo4`).

---
