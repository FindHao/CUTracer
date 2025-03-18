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
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <map>
#include <vector>
#include <unordered_set>
#include <time.h>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the reg_info_t structure */
#include "common.h"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;

enum class RecvThreadState
{
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

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> sass_to_id_map;
std::map<int, std::string> id_to_sass_map;

/* Structure to represent a single trace record */
struct TraceRecord
{
    int opcode_id;
    uint64_t pc;
    std::vector<std::vector<uint32_t>> reg_values; // [reg_idx][thread_idx]
};

/* Structure to identify a warp */
struct WarpKey
{
    int cta_id_x;
    int cta_id_y;
    int cta_id_z;
    int warp_id;

    // Operator for map comparison
    bool operator<(const WarpKey &other) const
    {
        if (cta_id_x != other.cta_id_x)
            return cta_id_x < other.cta_id_x;
        if (cta_id_y != other.cta_id_y)
            return cta_id_y < other.cta_id_y;
        if (cta_id_z != other.cta_id_z)
            return cta_id_z < other.cta_id_z;
        return warp_id < other.warp_id;
    }
};

/* Map to store traces for each warp */
std::map<WarpKey, std::vector<TraceRecord>> warp_traces;

// Add timeout configuration (in seconds)
int deadlock_timeout = 10; // Default 10 seconds, can be configured via environment variable

// Add logging configuration variables
int enable_logging = 1;                // Default: enabled
int log_last_traces_only = 0;          // Default: log everything
int log_to_stdout = 0;                 // Default: log to files
int store_last_traces_only = 0;        // Default: store all traces in memory
int dump_intermedia_trace = 0;         // Default: don't dump intermediate traces
int dump_intermedia_trace_timeout = 0; // Default: no timeout (0 means unlimited)

/* Store the name of the currently executing kernel */
char *current_kernel_name = NULL;

// Add a variable to store function name patterns to instrument
std::vector<std::string> function_patterns;

// Function to handle logging of trace data
void dump_trace_logs(const std::map<WarpKey, std::vector<TraceRecord>> &traces,
                     bool timeout_occurred, const char *kernel_name = nullptr);

void nvbit_at_init()
{
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    GET_VAR_INT(deadlock_timeout, "DEADLOCK_TIMEOUT", 10,
                "Timeout in seconds to detect potential deadlocks");
    // Add new environment variable controls for logging
    GET_VAR_INT(enable_logging, "ENABLE_LOGGING", 1,
                "Enable/disable logging (1=enabled, 0=disabled)");
    GET_VAR_INT(log_last_traces_only, "LOG_LAST_TRACES_ONLY", 0,
                "Only log the last trace for each warp (1=enabled, 0=disabled)");
    GET_VAR_INT(log_to_stdout, "LOG_TO_STDOUT", 0,
                "Log to stdout instead of files (1=enabled, 0=disabled)");
    GET_VAR_INT(store_last_traces_only, "STORE_LAST_TRACES_ONLY", 0,
                "Only store the last trace for each warp in memory (1=enabled, 0=disabled)");
    GET_VAR_INT(dump_intermedia_trace, "DUMP_INTERMEDIA_TRACE", 0,
                "Dump intermediate trace data to stdout (1=enabled, 0=disabled)");
    GET_VAR_INT(dump_intermedia_trace_timeout, "DUMP_INTERMEDIA_TRACE_TIMEOUT", 0,
                "Timeout in seconds for intermediate trace dumping (0=unlimited)");

    // Get function patterns from environment variable
    const char *patterns_env = getenv("FUNC_NAME_FILTER");
    if (patterns_env)
    {
        std::string patterns_str(patterns_env);
        size_t pos = 0;
        std::string token;
        // Split the string by commas
        while ((pos = patterns_str.find(',')) != std::string::npos)
        {
            token = patterns_str.substr(0, pos);
            if (!token.empty())
            {
                function_patterns.push_back(token);
            }
            patterns_str.erase(0, pos + 1);
        }
        // Add the last token if it exists
        if (!patterns_str.empty())
        {
            function_patterns.push_back(patterns_str);
        }

        if (verbose)
        {
            printf("Function name filters to instrument:\n");
            for (const auto &pattern : function_patterns)
            {
                printf("  - %s\n", pattern.c_str());
            }
        }
    }
    else if (verbose)
    {
        printf("No function name filters specified. Instrumenting all functions.\n");
    }

    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}
/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func)
{
    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions)
    {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second)
        {
            continue;
        }
        // Get function name (both mangled and unmangled versions)
        const char *unmangled_name = nvbit_get_func_name(ctx, f, false);
        const char *mangled_name = nvbit_get_func_name(ctx, f, true);

        // Check if function name contains any of the patterns
        bool should_instrument = true; // Default to true if no filters specified

        if (!function_patterns.empty())
        {
            should_instrument = false; // Start with false when we have filters
            for (const auto &pattern : function_patterns)
            {
                if ((unmangled_name && strstr(unmangled_name, pattern.c_str()) != NULL) ||
                    (mangled_name && strstr(mangled_name, pattern.c_str()) != NULL))
                {
                    should_instrument = true;
                    if (verbose)
                    {
                        printf("Found matching function for filter '%s': %s (mangled: %s)\n",
                               pattern.c_str(),
                               unmangled_name ? unmangled_name : "unknown",
                               mangled_name ? mangled_name : "unknown");
                    }
                    break;
                }
            }
        }
        else if (verbose)
        {
            printf("Instrumenting function: %s (mangled: %s)\n",
                   unmangled_name ? unmangled_name : "unknown",
                   mangled_name ? mangled_name : "unknown");
        }

        // Skip this function if it doesn't match any pattern
        if (!should_instrument)
        {
            continue;
        }

        const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
        if (verbose)
        {
            printf("Inspecting function %s at address 0x%lx\n",
                   nvbit_get_func_name(ctx, f), nvbit_get_func_addr(ctx, f));
        }

        uint32_t cnt = 0;
        /* iterate on all the static instructions in the function */
        for (auto instr : instrs)
        {
            if (cnt < instr_begin_interval || cnt >= instr_end_interval)
            {
                cnt++;
                continue;
            }
            if (verbose)
            {
                instr->printDecoded();
            }

            if (sass_to_id_map.find(instr->getSass()) ==
                sass_to_id_map.end())
            {
                int opcode_id = sass_to_id_map.size();
                sass_to_id_map[instr->getSass()] = opcode_id;
                id_to_sass_map[opcode_id] = std::string(instr->getSass());
            }

            int opcode_id = sass_to_id_map[instr->getSass()];
            std::vector<int> reg_num_list;
            std::vector<int> unified_reg_num_list;
            /* iterate on the operands */
            for (int i = 0; i < instr->getNumOperands(); i++)
            {
                /* get the operand "i" */
                const InstrType::operand_t *op = instr->getOperand(i);
                if (op->type == InstrType::OperandType::REG)
                {
                    for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++)
                    {
                        reg_num_list.push_back(op->u.reg.num + reg_idx);
                    }
                }
                else if (op->type == InstrType::OperandType::UREG)
                {
                    for (int reg_idx = 0; reg_idx < instr->getSize() / 4; reg_idx++)
                    {
                        unified_reg_num_list.push_back(op->u.reg.num + reg_idx);
                    }
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
            nvbit_add_call_arg_const_val64(instr,
                                           (uint64_t)&channel_dev);
            /* add instruction PC */
            nvbit_add_call_arg_const_val64(instr, instr->getOffset());
            /* how many register values are passed next */
            nvbit_add_call_arg_const_val32(instr, reg_num_list.size());
            nvbit_add_call_arg_const_val32(instr, unified_reg_num_list.size());
            for (int num : reg_num_list)
            {
                /* last parameter tells it is a variadic parameter passed to
                 * the instrument function record_reg_val() */
                nvbit_add_call_arg_reg_val(instr, num, true);
            }
            for (int num : unified_reg_num_list)
            {
                nvbit_add_call_arg_ureg_val(instr, num, true);
            }
            cnt++;
        }
    }
}

__global__ void flush_channel()
{
    /* push memory access with negative cta id to communicate the kernel is
     * completed */
    reg_info_t ri;
    ri.cta_id_x = -1;
    ri.pc = 0; // Set PC to 0 for completion marker
    channel_dev.push(&ri, sizeof(reg_info_t));

    /* flush channel */
    channel_dev.flush();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus)
{
    pthread_mutex_lock(&cuda_event_mutex);

    /* we prevent re-entry on this callback when issuing CUDA functions inside
     * this function */
    if (skip_callback_flag)
    {
        pthread_mutex_unlock(&cuda_event_mutex);
        return;
    }
    skip_callback_flag = true;

    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz)
    {
        /* cast params to launch parameter based on cbid since if we are here
         * we know these are the right parameters types */
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx)
        {
            cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
            func = p->f;
        }
        else
        {
            cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
            func = p->f;
        }

        if (!is_exit)
        {
            /* Make sure GPU is idle */
            cudaDeviceSynchronize();
            assert(cudaGetLastError() == cudaSuccess);

            int nregs = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, func));

            int shmem_static_nbytes = 0;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&shmem_static_nbytes,
                                   CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                   func));

            instrument_function_if_needed(ctx, func);

            nvbit_enable_instrumented(ctx, func, true);

            // Get kernel name for use in logs
            const char *kernel_name = nvbit_get_func_name(ctx, func);

            if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                cbid == API_CUDA_cuLaunchKernelEx)
            {
                cuLaunchKernelEx_params *p = (cuLaunchKernelEx_params *)params;
                printf(
                    "Kernel %s - PC 0x%lx - grid size %d,%d,%d - block size %d,%d,%d - nregs "
                    "%d - shmem %d - cuda stream id %ld\n",
                    kernel_name, nvbit_get_func_addr(ctx, func),
                    p->config->gridDimX, p->config->gridDimY,
                    p->config->gridDimZ, p->config->blockDimX,
                    p->config->blockDimY, p->config->blockDimZ, nregs,
                    shmem_static_nbytes + p->config->sharedMemBytes,
                    (uint64_t)p->config->hStream);
            }
            else
            {
                cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
                printf(
                    "Kernel %s - PC 0x%lx - grid size %d,%d,%d - block size %d,%d,%d - nregs "
                    "%d - shmem %d - cuda stream id %ld\n",
                    kernel_name, nvbit_get_func_addr(ctx, func),
                    p->gridDimX, p->gridDimY,
                    p->gridDimZ, p->blockDimX, p->blockDimY, p->blockDimZ, nregs,
                    shmem_static_nbytes + p->sharedMemBytes, (uint64_t)p->hStream);
            }

            // Store current kernel name for use when kernel completes
            if (current_kernel_name)
            {
                free(current_kernel_name);
            }
            current_kernel_name = strdup(kernel_name);
        }
        else
        {
            /* make sure current kernel is completed */
            cudaDeviceSynchronize();
            cudaError_t kernelError = cudaGetLastError();
            if (kernelError != cudaSuccess)
            {
                printf("Kernel launch error: %s\n", cudaGetErrorString(kernelError));
                assert(0);
            }

            /* issue flush of channel so we are sure all the memory accesses
             * have been pushed */
            flush_channel<<<1, 1>>>();
            cudaDeviceSynchronize();
            assert(cudaGetLastError() == cudaSuccess);
        }
    }
    skip_callback_flag = false;
    pthread_mutex_unlock(&cuda_event_mutex);
}

void *recv_thread_fun(void *)
{
    char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

    // Variables for timeout detection
    time_t last_recv_time = time(0);
    bool timeout_occurred = false;

    // Variables for intermediate trace timeout
    time_t dump_start_time = time(0);
    bool dump_timeout_reached = false;

    while (recv_thread_done == RecvThreadState::WORKING)
    {
        // Check for timeout
        time_t current_time = time(0);
        if (difftime(current_time, last_recv_time) > deadlock_timeout)
        {
            printf("\n!!! POTENTIAL DEADLOCK DETECTED: No data received for %d seconds !!!\n",
                   deadlock_timeout);
            timeout_occurred = true;
            break; // Exit the loop to proceed with logging
        }

        // Use the original recv function without timeout parameter
        uint32_t num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE);

        if (num_recv_bytes > 0)
        {
            // Reset the timeout timer when data is received
            last_recv_time = time(0);

            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes)
            {
                reg_info_t *ri =
                    (reg_info_t *)&recv_buffer[num_processed_bytes];

                /* when we get this cta_id_x it means the kernel has completed
                 */
                if (ri->cta_id_x == -1)
                {
                    // Kernel completed, dump the currently collected traces
                    dump_trace_logs(warp_traces, false, current_kernel_name);
                    // Clear traces to prepare for the next kernel
                    warp_traces.clear();
                    // Reset the dump timeout when a new kernel starts
                    dump_start_time = time(0);
                    dump_timeout_reached = false;
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
                for (int reg_idx = 0; reg_idx < ri->num_regs; reg_idx++)
                {
                    trace.reg_values[reg_idx].resize(32);
                    for (int i = 0; i < 32; i++)
                    {
                        trace.reg_values[reg_idx][i] = ri->reg_vals[i][reg_idx];
                    }
                }

                // Add trace to the warp's trace vector
                if (store_last_traces_only)
                {
                    // If we're only keeping the last trace in memory, clear any existing traces
                    // and just add this one
                    if (warp_traces.find(key) != warp_traces.end())
                    {
                        warp_traces[key].clear(); // Clear existing traces
                    }
                    warp_traces[key].push_back(trace); // Add the new trace
                }
                else
                {
                    // Normal mode: keep all traces in memory
                    warp_traces[key].push_back(trace);
                }

                // Check if we should dump intermediate trace
                if (dump_intermedia_trace && !dump_timeout_reached)
                {
                    // Check if dump timeout has been reached
                    if (dump_intermedia_trace_timeout > 0)
                    {
                        current_time = time(0);
                        if (difftime(current_time, dump_start_time) > dump_intermedia_trace_timeout)
                        {
                            if (!dump_timeout_reached)
                            {
                                printf("\n!!! INTERMEDIATE TRACE DUMPING STOPPED: Timeout of %d seconds reached !!!\n\n",
                                       dump_intermedia_trace_timeout);
                                dump_timeout_reached = true;
                            }
                        }
                    }

                    // Only dump if timeout not reached
                    if (!dump_timeout_reached)
                    {
                        printf("INTERMEDIATE TRACE - CTA %d,%d,%d - warp %d:\n",
                               key.cta_id_x, key.cta_id_y, key.cta_id_z, key.warp_id);
                        // To match with the PC offset in ncu reports
                        printf("  %s - PC Offset %ld (0x%lx)\n",
                               id_to_sass_map[trace.opcode_id].c_str(), (trace.pc / 16) + 1, trace.pc);

                        for (size_t reg_idx = 0; reg_idx < trace.reg_values.size(); reg_idx++)
                        {
                            printf("  * ");
                            for (int i = 0; i < 32; i++)
                            {
                                printf("Reg%zu_T%d: 0x%08x ",
                                       reg_idx, i, trace.reg_values[reg_idx][i]);
                            }
                            printf("\n");
                        }
                        printf("\n");
                    }
                }

                num_processed_bytes += sizeof(reg_info_t);
            }
        }
        else
        {
            // If no data received, sleep for a short time to avoid busy waiting
            usleep(100000); // Sleep for 100ms
        }
    }

    // If timeout occurred, dump the currently collected traces
    if (timeout_occurred)
    {
        dump_trace_logs(warp_traces, true, current_kernel_name);
    }

    // Clear the map after printing
    warp_traces.clear();

    free(recv_buffer);
    recv_thread_done = RecvThreadState::FINISHED;
    return NULL;
}

void nvbit_tool_init(CUcontext ctx)
{
    /* set mutex as recursive */
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&cuda_event_mutex, &attr);

    recv_thread_done = RecvThreadState::WORKING;
    channel_host.init(0, CHANNEL_SIZE, &channel_dev, recv_thread_fun, NULL);
    nvbit_set_tool_pthread(channel_host.get_thread());
}

void nvbit_at_ctx_term(CUcontext ctx)
{
    skip_callback_flag = true;
    /* Notify receiver thread and wait for receiver thread to
     * notify back */
    recv_thread_done = RecvThreadState::STOP;
    while (recv_thread_done != RecvThreadState::FINISHED)
        ;
    channel_host.destroy(false);
    skip_callback_flag = false;
}

// Add a function to handle logging
void dump_trace_logs(const std::map<WarpKey, std::vector<TraceRecord>> &traces, bool timeout_occurred, const char *kernel_name)
{
    // Only create log files if logging is enabled
    if (!enable_logging)
    {
        // If logging is disabled but a timeout occurred, still print a warning to stdout
        if (timeout_occurred)
        {
            printf("\n!!! POTENTIAL DEADLOCK DETECTED: No data received for %d seconds !!!\n",
                   deadlock_timeout);
            printf("Logging is disabled. Set ENABLE_LOGGING=1 to generate detailed logs.\n\n");
        }
        return;
    }

    // Generate filename with current date and time
    time_t now = time(0);
    struct tm *timeinfo = localtime(&now);
    char timestamp[40]; // Reduced size to ensure it fits
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", timeinfo);

    // If kernel name is provided, add it to the filename
    char kernel_suffix[256] = "";
    if (kernel_name)
    {
        snprintf(kernel_suffix, sizeof(kernel_suffix), "_%s", kernel_name);
    }

    FILE *logfile = stdout;          // Default to stdout
    FILE *last_traces_file = stdout; // Default to stdout

    if (!log_to_stdout)
    {
        // Use snprintf instead of sprintf to avoid buffer overflow
        char filename[256]; // Increased buffer size
        snprintf(filename, sizeof(filename), "deadlock_detection%s_%s.log", kernel_suffix, timestamp);

        // Open file for writing
        logfile = fopen(filename, "w");
        if (!logfile)
        {
            printf("Error: Could not open log file %s for writing\n", filename);
            logfile = stdout; // Fallback to stdout if file can't be opened
        }
        else
        {
            printf("Writing trace information to %s\n", filename);
        }

        // Always create a file for the last traces, regardless of timeout
        char last_traces_filename[256]; // Increased buffer size
        snprintf(last_traces_filename, sizeof(last_traces_filename), "last_traces%s_%s.log", kernel_suffix, timestamp);

        last_traces_file = fopen(last_traces_filename, "w");
        if (!last_traces_file)
        {
            printf("Error: Could not open last traces file %s for writing\n", last_traces_filename);
            last_traces_file = stdout; // Fallback to stdout
        }
        else
        {
            printf("Writing last traces to %s\n", last_traces_filename);
        }
    }

    // Write last traces information
    fprintf(last_traces_file, "\n===== LAST TRACES FOR EACH WARP =====\n\n");

    // If kernel name is provided, add it to the log
    if (kernel_name)
    {
        fprintf(last_traces_file, "Kernel: %s\n\n", kernel_name);
    }

    // Only print deadlock message if timeout occurred
    if (timeout_occurred)
    {
        fprintf(last_traces_file, "!!! POTENTIAL DEADLOCK DETECTED: No data received for %d seconds !!!\n\n",
                deadlock_timeout);
    }

    // Print the last trace for each warp
    for (const auto &entry : traces)
    {
        const WarpKey &warp = entry.first;
        const std::vector<TraceRecord> &trace_records = entry.second;

        if (!trace_records.empty())
        {
            const TraceRecord &last_trace = trace_records.back();

            fprintf(last_traces_file, "CTA %d,%d,%d - warp %d - Last trace:\n",
                    warp.cta_id_x, warp.cta_id_y, warp.cta_id_z, warp.warp_id);

            fprintf(last_traces_file, "  %s - PC 0x%lx\n",
                    id_to_sass_map[last_trace.opcode_id].c_str(), last_trace.pc);

            for (size_t reg_idx = 0; reg_idx < last_trace.reg_values.size(); reg_idx++)
            {
                fprintf(last_traces_file, "  * ");
                for (int i = 0; i < 32; i++)
                {
                    fprintf(last_traces_file, "Reg%zu_T%d: 0x%08x ",
                            reg_idx, i, last_trace.reg_values[reg_idx][i]);
                }
                fprintf(last_traces_file, "\n");
            }
            fprintf(last_traces_file, "\n");
        }
    }

    // Close the last traces file if it's not stdout
    if (last_traces_file != stdout)
    {
        fclose(last_traces_file);
    }

    // Only print full trace information if not in log_last_traces_only mode
    if (!log_last_traces_only)
    {
        // Print all collected information to the main log file
        fprintf(logfile, "\n===== WARP TRACE INFORMATION =====\n\n");

        // If kernel name is provided, add it to the log
        if (kernel_name)
        {
            fprintf(logfile, "Kernel: %s\n\n", kernel_name);
        }

        if (timeout_occurred)
        {
            fprintf(logfile, "!!! POTENTIAL DEADLOCK DETECTED: No data received for %d seconds !!!\n\n",
                    deadlock_timeout);
        }

        for (const auto &entry : traces)
        {
            const WarpKey &warp = entry.first;
            const std::vector<TraceRecord> &trace_records = entry.second;

            fprintf(logfile, "CTA %d,%d,%d - warp %d - Total traces: %zu\n",
                    warp.cta_id_x, warp.cta_id_y, warp.cta_id_z,
                    warp.warp_id, trace_records.size());

            for (size_t trace_idx = 0; trace_idx < trace_records.size(); trace_idx++)
            {
                const TraceRecord &trace = trace_records[trace_idx];

                fprintf(logfile, "  Trace %zu: %s - PC 0x%lx\n",
                        trace_idx, id_to_sass_map[trace.opcode_id].c_str(), trace.pc);

                for (size_t reg_idx = 0; reg_idx < trace.reg_values.size(); reg_idx++)
                {
                    fprintf(logfile, "  * ");
                    for (int i = 0; i < 32; i++)
                    {
                        fprintf(logfile, "Reg%zu_T%d: 0x%08x ", reg_idx, i, trace.reg_values[reg_idx][i]);
                    }
                    fprintf(logfile, "\n");
                }
                fprintf(logfile, "\n");
            }
            fprintf(logfile, "\n");
        }
    }

    // Close the file if it's not stdout
    if (logfile != stdout)
    {
        fclose(logfile);
    }
}
