# CUTracer

A comprehensive dynamic binary instrumentation tool for tracing and analyzing CUDA kernel instructions. Built on top of NVIDIA's NVBit framework.

## Overview

CUTracer is a powerful debugging and analysis tool that monitors CUDA kernel execution at the instruction level. It provides detailed tracing of GPU operations, enabling developers to:

- Capture and analyze instruction execution patterns
- Monitor register values and memory operations
- Detect execution anomalies including deadlocks and hangs
- Gain deeper insights into kernel behavior for performance optimization

## Features

- Dynamic binary instrumentation with no need to modify or recompile the target application
- Comprehensive instruction-level tracing for all CUDA kernels
- Detailed execution information for debugging and analysis:
  - Register values and changes
  - Instruction program counters
  - Warp and thread block execution patterns
  - Memory access patterns
- Deadlock detection with configurable timeouts
- Ability to filter instrumentation by function name patterns
- Flexible logging options for customizing output
- Minimal impact on application behavior

## Requirements

All requirements are aligned with NVBit.

- NVIDIA GPU with compute capability >= 3.5 and <= 9.2
- Linux operating system
- CUDA version >= 12.0
- CUDA driver version <= 555.xx
- GCC version >= 5.3.0 for x86_64 or >= 7.4.0 for aarch64

## Installation

1. Clone the repository
2. Download third-party dependencies:
```
./install_third_party.sh
```
3. Build the tool:
   ```
   make
   ```

## Usage

To use CUTracer, simply preload it before launching your CUDA application:

```bash
CUDA_INJECTION64_PATH=/path/to/lib/cutracer.so ./your_cuda_application
```

## Configuration

CUTracer can be configured using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| DEADLOCK_TIMEOUT | 10 | Timeout in seconds to detect potential deadlocks |
| ENABLE_LOGGING | 1 | Enable/disable logging (1=enabled, 0=disabled) |
| LOG_LAST_TRACES_ONLY | 0 | Only log the last trace for each warp (1=enabled, 0=disabled) |
| LOG_TO_STDOUT | 0 | Log to stdout instead of files (1=enabled, 0=disabled) |
| STORE_LAST_TRACES_ONLY | 0 | Only store the last trace for each warp in memory (1=enabled, 0=disabled) |
| DUMP_INTERMEDIA_TRACE | 0 | Dump intermediate trace data to stdout (1=enabled, 0=disabled) |
| DUMP_INTERMEDIA_TRACE_TIMEOUT | 0 | Timeout in seconds for intermediate trace dumping (0=unlimited) |
| FUNC_NAME_FILTER | - | Comma-separated list of function name patterns to instrument |
| INSTR_BEGIN | 0 | Beginning of the instruction interval to instrument |
| INSTR_END | UINT32_MAX | End of the instruction interval to instrument |
| TOOL_VERBOSE | 0 | Enable verbosity inside the tool |

## Examples

### Basic Example

```bash
DEADLOCK_TIMEOUT=30 FUNC_NAME_FILTER=kernel_a,kernel_b CUDA_INJECTION64_PATH=./lib/cutracer.so ./my_cuda_app
```

This will:
- Set the deadlock detection timeout to 30 seconds
- Only instrument functions with names containing "kernel_a" or "kernel_b"
- Run the application with instruction tracing enabled

### Advanced Example: PyTorch with Flash Attention

For debugging complex applications like PyTorch with Flash Attention operators in [TritonBench](https://github.com/pytorch-labs/tritonbench/):

```bash
TOOL_VERBOSE=1 \
DUMP_INTERMEDIA_TRACE=1 \
DUMP_INTERMEDIA_TRACE_TIMEOUT=10 \
FUNC_NAME_FILTER=attn \
CUDA_INJECTION64_PATH=~/path/to/deadlock_tracker/lib/cutracer.so \
python run.py --op flash_attention --batch 8 --seq-len 8192 --n-heads 16 --d-head 128 > deadlock_trace.log 2>&1
```

This configuration:
- Enables verbose output from the tool
- Dumps intermediate trace data during execution
- Sets a 10-second timeout for intermediate trace dumping
- Only instruments functions containing "attn" in their name (targeting attention-related kernels)
- Redirects both stdout and stderr to a log file for later analysis

### Example Output:

```bash
------------- NVBit (NVidia Binary Instrumentation Tool v1.7.4) Loaded --------------
NVBit core environment variables (mostly for nvbit-devs):
ACK_CTX_INIT_LIMITATION = 0 - if set, no warning will be printed for nvbit_at_ctx_init()
            NVDISASM = nvdisasm - override default nvdisasm found in PATH
            NOBANNER = 0 - if set, does not print this banner
       NO_EAGER_LOAD = 0 - eager module loading is turned on by NVBit to prevent potential NVBit tool deadlock, turn it off if you want to use the lazy module loading feature
---------------------------------------------------------------------------------
         INSTR_BEGIN = 0 - Beginning of the instruction interval where to apply instrumentation
           INSTR_END = 4294967295 - End of the instruction interval where to apply instrumentation
        TOOL_VERBOSE = 1 - Enable verbosity inside the tool
    DEADLOCK_TIMEOUT = 10 - Timeout in seconds to detect potential deadlocks
      ENABLE_LOGGING = 1 - Enable/disable logging (1=enabled, 0=disabled)
LOG_LAST_TRACES_ONLY = 0 - Only log the last trace for each warp (1=enabled, 0=disabled)
       LOG_TO_STDOUT = 0 - Log to stdout instead of files (1=enabled, 0=disabled)
STORE_LAST_TRACES_ONLY = 0 - Only store the last trace for each warp in memory (1=enabled, 0=disabled)
DUMP_INTERMEDIA_TRACE = 1 - Dump intermediate trace data to stdout (1=enabled, 0=disabled)
DUMP_INTERMEDIA_TRACE_TIMEOUT = 10 - Timeout in seconds for intermediate trace dumping (0=unlimited)
WARNING: Do not call CUDA memory allocation in nvbit_at_ctx_init(). It will cause deadlocks. Do them in nvbit_tool_init(). If you encounter deadlocks, remove CUDA API calls to debug.
No function name filters specified. Instrumenting all functions.
----------------------------------------------------------------------------------------------------
Instrumenting function: vecAdd(double*, double*, double*, int) (mangled: _Z6vecAddPdS_S_i)
Inspecting function vecAdd(double*, double*, double*, int) at address 0x7f34177aef00
Instr 0 @ 0x0 (0) - LDC R1, c[0x0][0x28] ;
  has_guard_pred = 0
  opcode = LDC/LDC
  memop = CONSTANT
  format = INT32
  load/store = 1/0
  size = 4
  is_extended = 0
--op[0].type = REG
  is_neg/is_not/abs = 0/0/0
  size = 4
  num = 1
  prop = 
--op[1].type = CBANK
  is_neg/is_not/abs = 0/0/0
  size = 4
  id = 0
  has_imm_offset = 1
  imm_offset = 40
  has_reg_offset = 0
  reg_offset = 0

Kernel vecAdd(double*, double*, double*, int) - PC 0x7f34177aef00 - grid size 1,1,1 - block size 1024,1,1 - nregs 14 - shmem 0 - cuda stream id 0
INTERMEDIATE TRACE - CTA 0,0,0 - warp 24:
  LDC R1, c[0x0][0x28] ; - PC Offset 1 (0x0)
  * Reg0_T0: 0x00000000 Reg0_T1: 0x00000000 Reg0_T2: 0x00000000 Reg0_T3: 0x00000000 Reg0_T4: 0x00000000 Reg0_T5: 0x00000000 Reg0_T6: 0x00000000 Reg0_T7: 0x00000000 Reg0_T8: 0x00000000 Reg0_T9: 0x00000000 Reg0_T10: 0x00000000 Reg0_T11: 0x00000000 Reg0_T12: 0x00000000 Reg0_T13: 0x00000000 Reg0_T14: 0x00000000 Reg0_T15: 0x00000000 Reg0_T16: 0x00000000 Reg0_T17: 0x00000000 Reg0_T18: 0x00000000 Reg0_T19: 0x00000000 Reg0_T20: 0x00000000 Reg0_T21: 0x00000000 Reg0_T22: 0x00000000 Reg0_T23: 0x00000000 Reg0_T24: 0x00000000 Reg0_T25: 0x00000000 Reg0_T26: 0x00000000 Reg0_T27: 0x00000000 Reg0_T28: 0x00000000 Reg0_T29: 0x00000000 Reg0_T30: 0x00000000 Reg0_T31: 0x00000000 

```
In the above trace snippet, we can see that the kernel vecAdd(double*, double*, double*,
int) is instrumented and the instructions executed by each warp are captured. `  LDC R1,
c[0x0][0x28] ; - PC Offset 1 (0x0)` is the first instruction executed by the warp and
`RegX_TY` means the X-th register value in the instruction of the Y-th thread in the warp.

## How It Works

CUTracer uses NVIDIA's NVBit (NVIDIA Binary Instrumentation Tool) to dynamically instrument CUDA applications. It:

1. Intercepts CUDA kernel launches
2. Instruments kernel code with data collection functions
3. Captures instruction execution data in real-time
4. Analyzes execution patterns and reports detailed information

For each instrumented instruction, the tool collects register values, program counters, and execution context information, which are transmitted from the GPU to the CPU for analysis. This approach provides comprehensive insights into kernel behavior without requiring source code modifications.

## Use Cases

- **Performance optimization**: Identify bottlenecks and inefficient execution patterns
- **Debugging complex issues**: Trace through problematic kernels to pinpoint errors
- **Deadlock detection**: Identify when kernels stop making progress
- **Algorithm analysis**: Understand the execution flow of complex GPU algorithms
- **Memory access pattern analysis**: Track how threads access memory for optimization

## Limitations

- Adding instrumentation adds overhead to kernel execution
- May produce false positives for long-running but non-deadlocked kernels
- Limited to NVIDIA GPUs and CUDA applications

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project uses NVBit (protected by NVIDIA CUDA Toolkit EULA) as a dependency.

## Acknowledgements

- Based on NVIDIA's NVBit framework for dynamic binary instrumentation
- Inspired by various debugging techniques for GPU applications