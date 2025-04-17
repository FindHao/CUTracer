/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include <map>
#include <string>
#include <unordered_set>
#include <vector>

// Add CUDA runtime header
#include <cuda_runtime.h>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the reg_info_t structure */
#include "common.h"

/* include environment configuration */
#include "env_config.h"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;

enum class RecvThreadState {
  WORKING,
  STOP,
  FINISHED,
};
volatile RecvThreadState recv_thread_done = RecvThreadState::STOP;

/* lock */
pthread_mutex_t cuda_event_mutex;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* opcode to id map and reverse map  */
std::map<std::string, int> sass_to_id_map;
std::map<int, std::string> id_to_sass_map;

/* Yueming: will fix this part later. grid launch id, incremented at every launch */
uint64_t global_grid_launch_id = 0;

/* Structure to represent a single trace record */
struct TraceRecord {
  int opcode_id;
  uint64_t pc;
  std::vector<std::vector<uint32_t>> reg_values;  // [reg_idx][thread_idx]
  std::vector<uint32_t> ureg_values;              // [ureg_idx]
  std::vector<std::vector<uint64_t>> addrs;       // [thread_idx][addr_idx]
};

/* Structure to identify a warp */
struct WarpKey {
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  int warp_id;

  // Operator for map comparison
  bool operator<(const WarpKey &other) const {
    if (cta_id_x != other.cta_id_x) return cta_id_x < other.cta_id_x;
    if (cta_id_y != other.cta_id_y) return cta_id_y < other.cta_id_y;
    if (cta_id_z != other.cta_id_z) return cta_id_z < other.cta_id_z;
    return warp_id < other.warp_id;
  }
};

/* Map to store traces for each warp */
std::map<WarpKey, std::vector<TraceRecord>> warp_traces;

/* Map to count executions for each warp to implement sampling */
std::map<WarpKey, uint64_t> warp_exec_counters;

/* Global counter for message-based sampling */
uint64_t global_message_counter = 0;

/**
 * Determines whether a trace should be dumped based on sampling rate configuration
 * and updates the relevant counters
 *
 * @param key The warp key for warp-based sampling
 * @return true if the trace should be dumped, false otherwise
 */
bool shouldDumpTrace(const WarpKey &key) {
  // Increment the global message counter for message-based sampling
  global_message_counter++;

  // Always count the number of instructions executed for each warp
  if (warp_exec_counters.find(key) == warp_exec_counters.end()) {
    warp_exec_counters[key] = 0;
  }
  warp_exec_counters[key]++;

  // According to the sampling rate, decide whether to dump the trace
  if (sampling_rate_warp > 1) {
    // Use warp-based sampling
    return (warp_exec_counters[key] % sampling_rate_warp == 0);
  } else if (sampling_rate > 1) {
    // Use global message-based sampling
    return (global_message_counter % sampling_rate == 0);
  } else {
    // Both sampling rates are 1, dump all traces
    return true;
  }
}

/* Store the name of the currently executing kernel */
std::string current_kernel_name;

/* File handle for intermediate trace output. not thread safe. */
FILE *log_handle = NULL;
// log_handle can be changed by create_kernel_log_file(). log_handle_main_trace is used to store the original
// log_handle.
FILE *log_handle_main_trace = NULL;

/**
 * Base template function for formatted output to different destinations
 * @param file_output if true, output to log file
 * @param stdout_output if true, output to stdout
 * @param format format string
 * @param args variable argument list
 */
template <typename... Args>
void base_fprintf(bool file_output, bool stdout_output, const char *format, Args... args) {
  // if no output, return
  if (!file_output && !stdout_output) return;

  char output_buffer[2048];  // use a large enough buffer
  snprintf(output_buffer, sizeof(output_buffer), format, args...);

  // output to stdout
  if (stdout_output) {
    fprintf(stdout, "%s", output_buffer);
  }

  // output to log file (if not stdout)
  if (file_output && log_handle != NULL && log_handle != stdout) {
    fprintf(log_handle, "%s", output_buffer);
  }
}

/**
 * lprintf - print to log file only (log print)
 */
template <typename... Args>
void lprintf(const char *format, Args... args) {
  base_fprintf(true, false, format, args...);
}

/**
 * oprintf - print to stdout only (output print)
 */
template <typename... Args>
void oprintf(const char *format, Args... args) {
  base_fprintf(false, true, format, args...);
}
/**
 * loprintf - print to log file and stdout (log and output print)
 */
template <typename... Args>
void loprintf(const char *format, Args... args) {
  base_fprintf(true, true, format, args...);
}

/**
 * Creates the intermediate trace file if needed
 * @param custom_filename Optional custom filename for the log file
 */
void create_trace_file(const char *custom_filename = nullptr, bool create_new_file = false) {
  if (log_to_stdout) {
    // If there's already a file and it's not stdout, close it and use stdout
    if (log_handle != NULL && log_handle != stdout) {
      fclose(log_handle);
    }
    log_handle = stdout;
    return;
  }

  if (log_handle_main_trace) {
    fflush(log_handle_main_trace);
  }
  // If there's already a file and it's not stdout, use it directly
  if (log_handle != NULL && log_handle != stdout && !create_new_file) {
    return;
  }
  if (create_new_file && log_handle != NULL && log_handle != stdout && log_handle != log_handle_main_trace) {
    fclose(log_handle);
  }
  // Need to create a new file
  char filename[256];

  // Use custom filename if provided, otherwise generate based on timestamp
  if (custom_filename != nullptr) {
    strncpy(filename, custom_filename, sizeof(filename) - 1);
    filename[sizeof(filename) - 1] = '\0';  // Ensure null termination
  } else {
    // Generate filename based on timestamp
    time_t now = time(0);
    struct tm *timeinfo = localtime(&now);
    char timestamp[40];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", timeinfo);
    snprintf(filename, sizeof(filename), "trace_%s.log", timestamp);
  }

  log_handle = fopen(filename, "w");
  if (!log_handle) {
    fprintf(stderr, "Error opening trace file '%s'. Falling back to stdout.\n", filename);
    log_handle = stdout;
  } else {
    loprintf("Writing traces to %s\n", filename);
  }
}

/**
 * Truncates a mangled function name to make it suitable for use as a filename
 * @param mangled_name The original mangled name
 * @param truncated_buffer The buffer to store the truncated name in
 * @param buffer_size Size of the provided buffer
 */
void truncate_mangled_name(const char *mangled_name, char *truncated_buffer, size_t buffer_size) {
  if (!truncated_buffer || buffer_size == 0) {
    return;
  }

  // Default to unknown if no name provided
  if (!mangled_name) {
    snprintf(truncated_buffer, buffer_size, "unknown_kernel");
    return;
  }

  // Truncate the name if it's longer than buffer_size - 1 (leave room for null terminator)
  size_t max_length = buffer_size - 1;
  size_t name_len = strlen(mangled_name);

  if (name_len > max_length) {
    strncpy(truncated_buffer, mangled_name, max_length);
    truncated_buffer[max_length] = '\0';  // Ensure null termination
  } else {
    strcpy(truncated_buffer, mangled_name);
  }
}

/**
 * Creates a log file specifically for a kernel based on its mangled name and iteration count
 * @param ctx CUDA context
 * @param func CUfunction representing the kernel
 * @param iteration Current iteration of the kernel execution
 */
void create_kernel_log_file(CUcontext ctx, CUfunction func, uint32_t iteration) {
  // Get mangled function name for file naming
  const char *mangled_name = nvbit_get_func_name(ctx, func, true);

  // Create a buffer for the truncated name
  char truncated_name[201];  // 200 chars + null terminator

  // Truncate the name
  truncate_mangled_name(mangled_name, truncated_name, sizeof(truncated_name));

  // Create a filename with the truncated name
  char filename[256];
  if (allow_reinstrument && single_kernel_trace) {
    // If SINGLE_KERNEL_TRACE and ALLOW_REINSTRUMENT are enabled, use only kernel name without iteration
    snprintf(filename, sizeof(filename), "%s.log", truncated_name);
  } else {
    // Otherwise include iteration number in filename
    snprintf(filename, sizeof(filename), "%s_iter%u.log", truncated_name, iteration);
  }

  // Create trace file with the custom filename
  create_trace_file(filename, true);
}

/* Counters to track the number of times each kernel has been executed */
std::map<CUfunction, uint32_t> kernel_execution_count;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
  /* Get related functions of the kernel (device function that can be
   * called by the kernel) */
  std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);
  /* iterate on function */
  for (auto f : related_functions) {
    if (kernel_execution_count.find(f) == kernel_execution_count.end()) {
      kernel_execution_count[f] = 0;
    } else {
      kernel_execution_count[f]++;
    }
    uint32_t current_iter = kernel_execution_count[f];
    if (current_iter < kernel_iter_begin) {
      continue;
    }
    if (!allow_reinstrument && current_iter > kernel_iter_begin) {
      continue;
    }
    // Get function name (both mangled and unmangled versions)
    const char *unmangled_name = nvbit_get_func_name(ctx, f, false);
    const char *mangled_name = nvbit_get_func_name(ctx, f, true);

    // Check if function name contains any of the patterns
    bool should_instrument = true;  // Default to true if no filters specified

    if (!function_patterns.empty()) {
      should_instrument = false;  // Start with false when we have filters
      for (const auto &pattern : function_patterns) {
        if ((unmangled_name && strstr(unmangled_name, pattern.c_str()) != NULL) ||
            (mangled_name && strstr(mangled_name, pattern.c_str()) != NULL)) {
          should_instrument = true;
          any_function_matched = true;  // Mark that at least one function matched
          if (verbose) {
            oprintf("Found matching function for filter '%s': %s (mangled: %s)\n", pattern.c_str(),
                    unmangled_name ? unmangled_name : "unknown", mangled_name ? mangled_name : "unknown");
          }
          break;
        }
      }
    } else if (verbose) {
      oprintf("Instrumenting function: %s (mangled: %s)\n", unmangled_name ? unmangled_name : "unknown",
              mangled_name ? mangled_name : "unknown");
    }

    // Skip this function if it doesn't match any pattern
    if (!should_instrument) {
      continue;
    }

    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    if (verbose) {
      oprintf("Inspecting function %s at address 0x%lx\n", nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx, f));
    }

    // Dump all SASS instructions to a file
    if (mangled_name) {
      // Create a filename with the mangled name
      char truncated_name[201];  // 200 chars + null terminator
      truncate_mangled_name(mangled_name, truncated_name, sizeof(truncated_name));

      char sass_filename[256];
      snprintf(sass_filename, sizeof(sass_filename), "%s.sass", truncated_name);

      // Open the file
      FILE *sass_file = fopen(sass_filename, "w");
      if (sass_file) {
        fprintf(sass_file, "// SASS instructions for kernel: %s\n", mangled_name);
        // Iterate through all instructions and write them to the file
        for (uint32_t i = 0; i < instrs.size(); i++) {
          auto instr = instrs[i];
          uint32_t offset = instr->getOffset();
          fprintf(sass_file, "%d /*%04x*/ %s\n", instr->getIdx(), offset, instr->getSass());
        }

        fclose(sass_file);
        if (verbose) {
          oprintf("Saved SASS instructions to %s\n", sass_filename);
        }
      } else {
        oprintf("Error: Could not create SASS file %s\n", sass_filename);
      }
    }

    /* iterate on all the static instructions in the function */
    for (uint32_t cnt = 0; cnt < instrs.size(); cnt++) {
      auto instr = instrs[cnt];
      if (!is_instruction_in_ranges(cnt)) {
        continue;
      }
      if (verbose) {
        oprintf("Instrumenting instruction %u: ", cnt);
        instr->printDecoded();
      }

      if (sass_to_id_map.find(instr->getSass()) == sass_to_id_map.end()) {
        int opcode_id = sass_to_id_map.size();
        sass_to_id_map[instr->getSass()] = opcode_id;
        id_to_sass_map[opcode_id] = std::string(instr->getSass());
      }

      int opcode_id = sass_to_id_map[instr->getSass()];
      std::vector<int> reg_num_list;
      std::vector<int> ureg_num_list;
      int mref_idx = 0;

      /* iterate on the operands */
      for (int i = 0; i < instr->getNumOperands(); i++) {
        /* get the operand "i" */
        const InstrType::operand_t *op = instr->getOperand(i);
        if (op->type == InstrType::OperandType::REG) {
          for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++) {
            reg_num_list.push_back(op->u.reg.num + reg_idx);
          }
        } else if (op->type == InstrType::OperandType::UREG) {
          for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++) {
            ureg_num_list.push_back(op->u.reg.num + reg_idx);
          }
        } else if (op->type == InstrType::OperandType::MREF) {
          // Not sure if this is correct
          if (op->u.mref.has_desc) {
            ureg_num_list.push_back(op->u.mref.desc_ureg_num);
            ureg_num_list.push_back(op->u.mref.desc_ureg_num + 1);
          }
          /* insert call to the instrumentation function with its
           * arguments */
          nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
          /* predicate value */
          nvbit_add_call_arg_guard_pred_val(instr);
          /* opcode id */
          nvbit_add_call_arg_const_val32(instr, opcode_id);
          /* memory reference 64 bit address */
          nvbit_add_call_arg_mref_addr64(instr, mref_idx);
          /* add instruction PC */
          nvbit_add_call_arg_const_val64(instr, instr->getOffset());
          /* add pointer to channel_dev*/
          nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);
          mref_idx++;
        }
      }
      /* insert call to the instrumentation function with its
       * arguments */
      nvbit_insert_call(instr, "record_reg_val", IPOINT_BEFORE);
      /* guard predicate value */
      nvbit_add_call_arg_guard_pred_val(instr);
      /* opcode id */
      nvbit_add_call_arg_const_val32(instr, opcode_id);
      /* add pointer to channel_dev*/
      nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);
      /* add instruction PC */
      nvbit_add_call_arg_const_val64(instr, instr->getOffset());
      /* how many register values are passed next */
      nvbit_add_call_arg_const_val32(instr, reg_num_list.size());
      nvbit_add_call_arg_const_val32(instr, ureg_num_list.size());
      for (int num : reg_num_list) {
        /* last parameter tells it is a variadic parameter passed to
         * the instrument function record_reg_val() */
        nvbit_add_call_arg_reg_val(instr, num, true);
      }
      for (int num : ureg_num_list) {
        nvbit_add_call_arg_ureg_val(instr, num, true);
      }
    }
  }
}

/* flush channel */
__global__ void flush_channel(ChannelDev *ch_dev = NULL) {
  // Get the channel to use
  ChannelDev *channel = (ch_dev == NULL) ? &channel_dev : ch_dev;

  /* push memory access with negative cta id to communicate the kernel is
   * completed */
  reg_info_t ri;
  ri.header.type = MSG_TYPE_REG_INFO;  // Set message type
  ri.cta_id_x = -1;
  ri.pc = 0;  // Set PC to 0 for completion marker
  channel->push(&ri, sizeof(reg_info_t));

  /* flush channel */
  channel->flush();
}

// Added function to handle pre-kernel launch work
static void enter_kernel_launch(CUcontext ctx, CUfunction func, uint64_t &grid_launch_id, nvbit_api_cuda_t cbid,
                                void *params, bool stream_capture = false, bool build_graph = false) {
  // Reset the function match flag for this kernel launch
  any_function_matched = false;

  // If not stream capturing or graph building, ensure GPU is idle
  if (!stream_capture && !build_graph) {
    /* Make sure GPU is idle */
    cudaDeviceSynchronize();
    assert(cudaGetLastError() == cudaSuccess);
  }

  instrument_function_if_needed(ctx, func);

  int nregs = 0;
  CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

  int shmem_static_nbytes = 0;
  CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

  /* get function name and pc */
  const char *func_name = nvbit_get_func_name(ctx, func);
  uint64_t pc = nvbit_get_func_addr(ctx, func);

  // Only enable instrumentation if:
  // 1. No function patterns were specified (instrument everything), or
  // 2. At least one function matched the specified patterns
  bool should_enable_instrumentation = function_patterns.empty() || any_function_matched;

  // Print a warning if function patterns were specified but no functions matched
  if (!function_patterns.empty() && !any_function_matched) {
    loprintf("\nINFO: No functions matched the specified FUNC_NAME_FILTER patterns\n");
    loprintf("Specified patterns:\n");
    for (const auto &pattern : function_patterns) {
      loprintf("  - %s\n", pattern.c_str());
    }
    // Get both demangled and mangled function names
    const char *mangled_name = nvbit_get_func_name(ctx, func, true);
    loprintf("Skipping instrumentation for kernel: %s\n", func_name);
    if (mangled_name && strcmp(func_name, mangled_name) != 0) {
      loprintf("Mangled name: %s\n", mangled_name);
    }
    loprintf("\n");
  }

  // During stream capture or graph building, the kernel doesn't actually launch, so don't set launch parameters
  if (!stream_capture && !build_graph) {
    /* set grid launch id at launch time */
    CUstream stream = 0;

    // Get the current kernel's stream
    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
      cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
      stream = p->config->hStream;
    } else if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel ||
               cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz || cbid == API_CUDA_cuLaunchCooperativeKernel) {
      cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
      stream = p->hStream;
    } else if (cbid == API_CUDA_cuLaunchGridAsync) {
      cuLaunchGridAsync_params *p = (cuLaunchGridAsync_params *)params;
      stream = p->hStream;
    }

    nvbit_set_at_launch(ctx, func, (uint64_t)grid_launch_id, stream);

    uint32_t current_iter = kernel_execution_count[func];
    if (should_enable_instrumentation) {
      create_kernel_log_file(ctx, func, current_iter);
    }

    if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
      cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
      loprintf(
          "Kernel %s - PC 0x%lx - grid launch id %ld - iteration %u - grid size %d,%d,%d - block size %d,%d,%d - nregs "
          "%d - shmem %d - cuda stream id %ld%s\n",
          func_name, pc, grid_launch_id, current_iter, p->config->gridDimX, p->config->gridDimY, p->config->gridDimZ,
          p->config->blockDimX, p->config->blockDimY, p->config->blockDimZ, nregs,
          shmem_static_nbytes + p->config->sharedMemBytes, (uint64_t)p->config->hStream,
          should_enable_instrumentation ? "" : " (NOT INSTRUMENTED)");
    } else {
      cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
      loprintf(
          "Kernel %s - PC 0x%lx - grid launch id %ld - iteration %u - grid size %d,%d,%d - block size %d,%d,%d - nregs "
          "%d - shmem %d - cuda stream id %ld%s\n",
          func_name, pc, grid_launch_id, current_iter, p->gridDimX, p->gridDimY, p->gridDimZ, p->blockDimX,
          p->blockDimY, p->blockDimZ, nregs, shmem_static_nbytes + p->sharedMemBytes, (uint64_t)p->hStream,
          should_enable_instrumentation ? "" : " (NOT INSTRUMENTED)");
    }

    // Increment the grid launch ID for the next launch
    grid_launch_id++;
  }

  /* enable instrumented code to run only if we should enable instrumentation */
  nvbit_enable_instrumented(ctx, func, should_enable_instrumentation);
}

// Added function to handle post-kernel launch work
static void leave_kernel_launch() {
  // Ensure user kernel completion to avoid deadlocks
  cudaDeviceSynchronize();
  /* issue flush of channel so we are sure all the memory accesses
   * have been pushed */
  flush_channel<<<1, 1>>>(&channel_dev);

  /* Make sure GPU is idle */
  cudaDeviceSynchronize();
  assert(cudaGetLastError() == cudaSuccess);
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char *name, void *params,
                         CUresult *pStatus) {
  pthread_mutex_lock(&cuda_event_mutex);

  /* we prevent re-entry on this callback when issuing CUDA functions inside
   * this function */
  if (skip_callback_flag) {
    pthread_mutex_unlock(&cuda_event_mutex);
    return;
  }
  skip_callback_flag = true;

  switch (cbid) {
    // Handle CUDA launch events without stream parameters, they won't involve CUDA graphs
    case API_CUDA_cuLaunch:
    case API_CUDA_cuLaunchGrid: {
      cuLaunch_params *p = (cuLaunch_params *)params;
      CUfunction func = p->f;
      if (!is_exit) {
        enter_kernel_launch(ctx, func, global_grid_launch_id, cbid, params);
      } else {
        leave_kernel_launch();
      }
    } break;
    // Handle kernel launches with stream parameters, which can be used for CUDA graphs
    case API_CUDA_cuLaunchKernel_ptsz:
    case API_CUDA_cuLaunchKernel:
    case API_CUDA_cuLaunchCooperativeKernel:
    case API_CUDA_cuLaunchCooperativeKernel_ptsz:
    case API_CUDA_cuLaunchKernelEx:
    case API_CUDA_cuLaunchKernelEx_ptsz:
    case API_CUDA_cuLaunchGridAsync: {
      CUfunction func;
      CUstream hStream;

      if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
        cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
        func = p->f;
        hStream = p->config->hStream;
      } else if (cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchKernel ||
                 cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz || cbid == API_CUDA_cuLaunchCooperativeKernel) {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
        func = p->f;
        hStream = p->hStream;
      } else {
        cuLaunchGridAsync_params *p = (cuLaunchGridAsync_params *)params;
        func = p->f;
        hStream = p->hStream;
      }

      cudaStreamCaptureStatus streamStatus;
      /* Check if the stream is currently capturing */
      CUDA_SAFECALL(cudaStreamIsCapturing(hStream, &streamStatus));
      if (!is_exit) {
        bool stream_capture = (streamStatus == cudaStreamCaptureStatusActive);
        enter_kernel_launch(ctx, func, global_grid_launch_id, cbid, params, stream_capture);
      } else {
        if (streamStatus != cudaStreamCaptureStatusActive) {
          if (verbose >= 1) {
            fprintf(log_handle, "kernel %s not captured by cuda graph\n", nvbit_get_func_name(ctx, func));
          }
          leave_kernel_launch();
        } else {
          if (verbose >= 1) {
            fprintf(log_handle, "kernel %s captured by cuda graph\n", nvbit_get_func_name(ctx, func));
          }
        }
      }
    } break;
    // Support for CUDA graph node additions
    case API_CUDA_cuGraphAddKernelNode: {
      cuGraphAddKernelNode_params *p = (cuGraphAddKernelNode_params *)params;
      CUfunction func = p->nodeParams->func;

      if (!is_exit) {
        // nodeParams and cuLaunchKernel_params are identical up to sharedMemBytes
        enter_kernel_launch(ctx, func, global_grid_launch_id, cbid, (void *)p->nodeParams, false, true);
      }
    } break;
    // Support for CUDA graph launches
    case API_CUDA_cuGraphLaunch: {
      // If we're exiting a CUDA graph launch, wait for the graph to complete
      if (is_exit) {
        cuGraphLaunch_params *p = (cuGraphLaunch_params *)params;

        CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
        assert(cudaGetLastError() == cudaSuccess);
        /* push a flush channel kernel */
        flush_channel<<<1, 1, 0, p->hStream>>>();
        CUDA_SAFECALL(cudaStreamSynchronize(p->hStream));
        assert(cudaGetLastError() == cudaSuccess);
      }
    } break;
    default:
      // Handle original CUDA launch events
      if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz || cbid == API_CUDA_cuLaunchGrid ||
          cbid == API_CUDA_cuLaunchGridAsync || cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernelEx ||
          cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        /* cast params to launch parameter based on cbid since if we are here
         * we know these are the right parameters types */
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
          cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
          func = p->f;
        } else {
          cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
          func = p->f;
        }

        if (!is_exit) {
          /* Make sure GPU is idle */
          cudaDeviceSynchronize();
          assert(cudaGetLastError() == cudaSuccess);

          int nregs = 0;
          CUDA_SAFECALL(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

          int shmem_static_nbytes = 0;
          CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func));

          instrument_function_if_needed(ctx, func);

          // Only enable instrumentation if:
          // 1. No function patterns were specified (instrument everything), or
          // 2. At least one function matched the specified patterns
          bool should_enable_instrumentation = function_patterns.empty() || any_function_matched;

          // Print a warning if function patterns were specified but no functions matched
          if (!function_patterns.empty() && !any_function_matched) {
            loprintf("\n!!! WARNING: No functions matched the specified FUNC_NAME_FILTER patterns !!!\n");
            loprintf("Specified patterns:\n");
            for (const auto &pattern : function_patterns) {
              loprintf("  - %s\n", pattern.c_str());
            }
            const char *kernel_name = nvbit_get_func_name(ctx, func);
            loprintf("Skipping instrumentation for kernel: %s\n\n", kernel_name);
          }

          nvbit_enable_instrumented(ctx, func, should_enable_instrumentation);

          // Get kernel name for use in logs
          const char *kernel_name = nvbit_get_func_name(ctx, func);

          // Store current kernel name for use when kernel completes
          current_kernel_name = kernel_name;

          // Create a log file for this kernel execution
          uint32_t current_iter = kernel_execution_count[func];
          if (should_enable_instrumentation) {
            create_kernel_log_file(ctx, func, current_iter);
          }

          if (cbid == API_CUDA_cuLaunchKernelEx_ptsz || cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
            loprintf(
                "Kernel %s - PC 0x%lx - grid size %d,%d,%d - block size %d,%d,%d - nregs "
                "%d - shmem %d - cuda stream id %ld\n",
                kernel_name, nvbit_get_func_addr(ctx, func), p->config->gridDimX, p->config->gridDimY,
                p->config->gridDimZ, p->config->blockDimX, p->config->blockDimY, p->config->blockDimZ, nregs,
                shmem_static_nbytes + p->config->sharedMemBytes, (uint64_t)p->config->hStream);
          } else {
            cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
            loprintf(
                "Kernel %s - PC 0x%lx - grid size %d,%d,%d - block size %d,%d,%d - nregs "
                "%d - shmem %d - cuda stream id %ld\n",
                kernel_name, nvbit_get_func_addr(ctx, func), p->gridDimX, p->gridDimY, p->gridDimZ, p->blockDimX,
                p->blockDimY, p->blockDimZ, nregs, shmem_static_nbytes + p->sharedMemBytes, (uint64_t)p->hStream);
          }

        } else {
          /* make sure current kernel is completed */
          cudaDeviceSynchronize();
          cudaError_t kernelError = cudaGetLastError();
          if (kernelError != cudaSuccess) {
            fprintf(log_handle, "Kernel launch error: %s\n", cudaGetErrorString(kernelError));
            assert(0);
          }

          /* issue flush of channel so we are sure all the memory accesses
           * have been pushed */
          flush_channel<<<1, 1>>>();
          cudaDeviceSynchronize();
          assert(cudaGetLastError() == cudaSuccess);
        }
      }
      break;
  }
  skip_callback_flag = false;
  pthread_mutex_unlock(&cuda_event_mutex);
}

// Add support for CUDA graph node launches
void nvbit_at_graph_node_launch(CUcontext ctx, CUfunction func, CUstream stream, uint64_t launch_handle) {
  func_config_t config = {0};
  const char *func_name = nvbit_get_func_name(ctx, func);
  uint64_t pc = nvbit_get_func_addr(ctx, func);

  pthread_mutex_lock(&cuda_event_mutex);
  nvbit_set_at_launch(ctx, func, (uint64_t)global_grid_launch_id, stream, launch_handle);
  nvbit_get_func_config(ctx, func, &config);

  uint32_t current_iter = kernel_execution_count[func];
  bool should_enable_instrumentation = function_patterns.empty() || any_function_matched;
  // Create a log file for this graph node kernel
  if (should_enable_instrumentation) {
    create_kernel_log_file(ctx, func, current_iter);
  }

  loprintf(
      "Graph Node Launch - Kernel %s - PC 0x%lx - grid launch id %ld - iteration %u - grid size %d,%d,%d "
      "- block size %d,%d,%d - nregs %d - shmem %d - cuda stream id %ld\n",
      func_name, pc, global_grid_launch_id, current_iter, config.gridDimX, config.gridDimY, config.gridDimZ,
      config.blockDimX, config.blockDimY, config.blockDimZ, config.num_registers,
      config.shmem_static_nbytes + config.shmem_dynamic_nbytes, (uint64_t)stream);

  // Increment the grid launch ID for the next launch
  global_grid_launch_id++;
  pthread_mutex_unlock(&cuda_event_mutex);
}

void *recv_thread_fun(void *) {
  char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

  // Variables for timeout detection
  time_t last_recv_time = time(0);

  // Variables for intermediate trace timeout
  time_t dump_start_time = time(0);
  bool dump_timeout_reached = false;

  while (recv_thread_done == RecvThreadState::WORKING) {
    // Check for timeout
    time_t current_time = time(0);
    if (difftime(current_time, last_recv_time) > deadlock_timeout) {
      loprintf("\n!!! POTENTIAL DEADLOCK DETECTED: No data received for %d seconds !!!\n", deadlock_timeout);
      break;  // Exit the loop to proceed with logging
    }

    // Use the original recv function without timeout parameter
    uint32_t num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE);

    if (num_recv_bytes > 0) {
      // Reset the timeout timer when data is received
      last_recv_time = time(0);

      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes) {
        // First read the message header to determine the message type
        message_header_t *header = (message_header_t *)&recv_buffer[num_processed_bytes];
        // printf("Received message type: %d\n", header->type);
        // Process message based on its type
        if (header->type == MSG_TYPE_REG_INFO) {
          // Process register info message
          reg_info_t *ri = (reg_info_t *)&recv_buffer[num_processed_bytes];

          /* when we get this cta_id_x it means the kernel has completed */
          if (ri->cta_id_x == -1) {
            // Clear traces to prepare for the next kernel
            warp_traces.clear();
            // Clear execution counters when a kernel completes
            warp_exec_counters.clear();
            // Reset the global message counter
            global_message_counter = 0;
            // Reset the dump timeout when a new kernel starts
            dump_start_time = time(0);
            dump_timeout_reached = false;
            if (log_handle_main_trace && log_handle != log_handle_main_trace) {
              // printf("==============debug log_handle: %p\n", log_handle);
              log_handle = log_handle_main_trace;
              // printf("==============debug log_handle: %p\n", log_handle);
            }
            break;
          }

          // Create key for this warp
          WarpKey key;
          key.cta_id_x = ri->cta_id_x;
          key.cta_id_y = ri->cta_id_y;
          key.cta_id_z = ri->cta_id_z;
          key.warp_id = ri->warp_id;

          // Create trace record
          TraceRecord trace;
          trace.opcode_id = ri->opcode_id;
          trace.pc = ri->pc;

          // Store register values
          trace.reg_values.resize(ri->num_regs);
          for (int reg_idx = 0; reg_idx < ri->num_regs; reg_idx++) {
            trace.reg_values[reg_idx].resize(32);
            for (int i = 0; i < 32; i++) {
              trace.reg_values[reg_idx][i] = ri->reg_vals[i][reg_idx];
            }
          }

          // Store unified register values
          trace.ureg_values.resize(ri->num_uregs);
          for (int i = 0; i < ri->num_uregs; i++) {
            trace.ureg_values[i] = ri->ureg_vals[i];
          }

          // Add trace to the warp's trace vector
          if (store_last_traces_only) {
            // If we're only keeping the last trace in memory, clear any existing traces
            // and just add this one
            if (warp_traces.find(key) != warp_traces.end()) {
              warp_traces[key].clear();  // Clear existing traces
            }
            warp_traces[key].push_back(trace);  // Add the new trace
          } else {
            // Normal mode: keep all traces in memory
            warp_traces[key].push_back(trace);
          }

          // Check if we should dump intermediate trace for register info
          if (dump_intermedia_trace && !dump_timeout_reached) {
            // Check if dump timeout has been reached
            if (dump_intermedia_trace_timeout > 0) {
              current_time = time(0);
              if (difftime(current_time, dump_start_time) > dump_intermedia_trace_timeout) {
                if (!dump_timeout_reached) {
                  lprintf("\n!!! INTERMEDIATE TRACE DUMPING STOPPED: Timeout of %d seconds reached !!!\n\n",
                          dump_intermedia_trace_timeout);
                  dump_timeout_reached = true;
                }
              }
            }

            bool should_dump = false;

            // Determine if we should dump this trace based on sampling configuration
            should_dump = shouldDumpTrace(key);

            // Only dump if timeout not reached and sampling condition is met
            if (!dump_timeout_reached && should_dump) {
              lprintf("INTERMEDIATE REG TRACE - CTA %d,%d,%d - warp %d:\n", key.cta_id_x, key.cta_id_y, key.cta_id_z,
                      key.warp_id);
              // To match with the PC offset in ncu reports
              lprintf("  %s - PC Offset %ld (0x%lx)\n", id_to_sass_map[trace.opcode_id].c_str(), trace.pc / 16,
                      trace.pc);

              for (size_t reg_idx = 0; reg_idx < trace.reg_values.size(); reg_idx++) {
                lprintf("  * ");
                for (int i = 0; i < 32; i++) {
                  lprintf("Reg%zu_T%d: 0x%08x ", reg_idx, i, trace.reg_values[reg_idx][i]);
                }
                lprintf("\n");
              }
              for (size_t i = 0; i < trace.ureg_values.size(); i++) {
                lprintf("  * UREG%zu: 0x%08x\n", i, trace.ureg_values[i]);
              }
              lprintf("\n");
            }
          }

          num_processed_bytes += sizeof(reg_info_t);
        } else if (header->type == MSG_TYPE_MEM_ACCESS) {
          // Process memory access message
          mem_access_t *mem = (mem_access_t *)&recv_buffer[num_processed_bytes];

          // Create key for this warp
          WarpKey key;
          key.cta_id_x = mem->cta_id_x;
          key.cta_id_y = mem->cta_id_y;
          key.cta_id_z = mem->cta_id_z;
          key.warp_id = mem->warp_id;

          // Print memory access information if intermediate trace is enabled
          if (dump_intermedia_trace && !dump_timeout_reached) {
            if (dump_intermedia_trace_timeout > 0) {
              current_time = time(0);
              if (difftime(current_time, dump_start_time) > dump_intermedia_trace_timeout) {
                if (!dump_timeout_reached) {
                  lprintf("\n!!! INTERMEDIATE TRACE DUMPING STOPPED: Timeout of %d seconds reached !!!\n\n",
                          dump_intermedia_trace_timeout);
                  dump_timeout_reached = true;
                }
              }
            }

            bool should_dump = false;

            // Determine if we should dump this trace based on sampling configuration
            should_dump = shouldDumpTrace(key);

            // Only dump if timeout not reached and sampling condition is met
            if (!dump_timeout_reached && should_dump) {
              lprintf("INTERMEDIATE MEM TRACE - CTA %d,%d,%d - warp %d:\n", key.cta_id_x, key.cta_id_y, key.cta_id_z,
                      key.warp_id);
              lprintf("  %s - PC Offset %ld (0x%lx)\n", id_to_sass_map[mem->opcode_id].c_str(), mem->pc / 16, mem->pc);

              // Print memory addresses
              lprintf("  Memory Addresses:\n  * ");
              int printed = 0;
              for (int i = 0; i < 32; i++) {
                if (mem->addrs[i] != 0) {  // Only print non-zero addresses
                  lprintf("T%d: 0x%016lx ", i, mem->addrs[i]);
                  printed++;
                  // Add a newline every 4 addresses for readability
                  if (printed % 4 == 0 && i < 31) {
                    lprintf("\n    ");
                  }
                }
              }
              lprintf("\n\n");
            }
          }

          // Here you could add code to store memory access information in a data structure
          // similar to how register traces are stored

          num_processed_bytes += sizeof(mem_access_t);
        } else {
          // Unknown message type, skip minimum amount of bytes
          lprintf("ERROR: Unknown message type %d received\n", header->type);
          num_processed_bytes += sizeof(message_header_t);
        }
      }
    } else {
      // If no data received, sleep for a short time to avoid busy waiting
      usleep(100000);  // Sleep for 100ms
    }
  }

  // Clear the map after printing
  warp_traces.clear();
  // Clear the execution counters
  warp_exec_counters.clear();
  // Reset the global message counter
  global_message_counter = 0;

  free(recv_buffer);
  if (log_handle_main_trace && log_handle_main_trace != stdout) {
    if (log_handle != log_handle_main_trace) {
      fclose(log_handle);
      log_handle = NULL;
    }
    fclose(log_handle_main_trace);
    log_handle_main_trace = NULL;
  }
  recv_thread_done = RecvThreadState::FINISHED;
  return NULL;
}

void nvbit_tool_init(CUcontext ctx) {
  /* set mutex as recursive */
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&cuda_event_mutex, &attr);

  recv_thread_done = RecvThreadState::WORKING;
  channel_host.init(0, CHANNEL_SIZE, &channel_dev, recv_thread_fun, NULL);
  nvbit_set_tool_pthread(channel_host.get_thread());
}

void nvbit_at_ctx_term(CUcontext ctx) {
  skip_callback_flag = true;
  /* Notify receiver thread and wait for receiver thread to
   * notify back */
  recv_thread_done = RecvThreadState::STOP;
  while (recv_thread_done != RecvThreadState::FINISHED);
  channel_host.destroy(false);
  skip_callback_flag = false;
}

void nvbit_at_init() {
  // Initialize configuration from environment variables
  init_config_from_env();
  // Create intermediate trace file if needed
  create_trace_file();
  log_handle_main_trace = log_handle;
}
