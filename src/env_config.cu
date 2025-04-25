#include "env_config.h"
#include "logger.h" // Include logger for print_config

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h> // For UINT32_MAX
#include <string>
#include <vector>
#include <iostream> // For error messages potentially

// We don't need to include nvbit_tool.h in every source file, just in main.cu
// We only use variables and functions defined in header files

// Configuration variable definitions
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
std::vector<std::pair<uint32_t, uint32_t>> instr_ranges;
bool use_instr_ranges = false;
std::string instr_ranges_str;
int deadlock_timeout = 60; // Default timeout 60 seconds
int enable_logging = 1; // Default enable logging
int log_last_traces_only = 0; // Default log all traces
int log_to_stdout = 0; // Default log to file
int store_last_traces_only = 0; // Default store all traces in memory
int dump_intermedia_trace = 0; // Default disable intermediate trace dump
int dump_intermedia_trace_timeout = 0; // Default no timeout for intermediate trace dump
int allow_reinstrument = 1; // Default allow reinstrumentation
uint32_t kernel_iter_begin = 0;
int single_kernel_trace = 0; // Default trace all kernels
uint64_t sampling_rate_warp = 1; // Default sampling rate per warp (1 = all)
uint64_t sampling_rate = 1; // Default sampling rate globally (1 = all)
int loop_win_size = 32; // Default window size
uint32_t loop_repeat_thresh = 16; // Default repeat threshold
int loop_hang_timeout = 3; // Default hang timeout in seconds
int loop_detection_enabled = 1; // Default enabled
int instrument_all_kernels = 0; // Default: require function filter match

// Function name patterns to filter
std::vector<std::string> function_patterns;
bool any_function_matched = false;

// Helper to get integer environment variable
static int get_env_int(const char *env_var, int default_val) {
    const char *val_str = getenv(env_var);
    if (val_str) {
        return atoi(val_str);
    }
    return default_val;
}

// Helper to get uint32 environment variable
static uint32_t get_env_uint32(const char *env_var, uint32_t default_val) {
    const char *val_str = getenv(env_var);
    if (val_str) {
        // Use strtoul for better error handling potential, though not fully utilized here
        char *endptr;
        unsigned long val = strtoul(val_str, &endptr, 10);
        if (*endptr == '\0') { // Ensure the whole string was parsed
            return (uint32_t)val;
        }
    }
    return default_val;
}

// Helper to get uint64 environment variable
static uint64_t get_env_uint64(const char *env_var, uint64_t default_val) {
    const char *val_str = getenv(env_var);
    if (val_str) {
        char *endptr;
        unsigned long long val = strtoull(val_str, &endptr, 10);
        if (*endptr == '\0') { // Ensure the whole string was parsed
            return (uint64_t)val;
        }
    }
    return default_val;
}

// Function to load configuration from environment variables
void load_config() {
    verbose = get_env_int(ENV_VERBOSE, 0);
    deadlock_timeout = get_env_int(ENV_DEADLOCK_TIMEOUT, 60);
    enable_logging = get_env_int(ENV_LOG_ENABLE, 1);
    log_last_traces_only = get_env_int(ENV_LOG_LAST_TRACES, 0);
    log_to_stdout = get_env_int(ENV_LOG_STDOUT, 0);
    store_last_traces_only = get_env_int(ENV_STORE_LAST_TRACES, 0);
    dump_intermedia_trace = get_env_int(ENV_DUMP_INTERMEDIA, 0);
    dump_intermedia_trace_timeout = get_env_int(ENV_DUMP_INTERMEDIA_TIMEOUT, 0);
    allow_reinstrument = get_env_int(ENV_ALLOW_REINSTRUMENT, 1);
    kernel_iter_begin = get_env_uint32(ENV_KERNEL_ITER_BEGIN, 0);
    single_kernel_trace = get_env_int(ENV_SINGLE_KERNEL_TRACE, 0);
    sampling_rate_warp = get_env_uint64(ENV_SAMPLING_RATE_WARP, 1);
    sampling_rate = get_env_uint64(ENV_SAMPLING_RATE, 1);
    loop_win_size = get_env_int(ENV_LOOP_WIN_SIZE, 32);
    loop_repeat_thresh = get_env_uint32(ENV_LOOP_REPEAT_THRESH, 16);
    loop_hang_timeout = get_env_int(ENV_LOOP_HANG_TIMEOUT, 3);
    loop_detection_enabled = get_env_int(ENV_LOOP_DETECTION_ENABLED, 1);
    instrument_all_kernels = get_env_int(ENV_INSTRUMENT_ALL_KERNELS, 0);

    // Instruction range parsing
    instr_begin_interval = get_env_uint32(ENV_INSTR_BEGIN, 0);
    instr_end_interval = get_env_uint32(ENV_INSTR_END, UINT32_MAX);

    // Parse function name patterns
    const char *patterns_env = getenv(ENV_TARGET_FUNCTIONS);
    if (patterns_env) {
        function_patterns.clear(); // Clear previous patterns if any
        std::string patterns_str(patterns_env);
        std::string delimiter = ",";
        size_t pos = 0;
        std::string token;
        while ((pos = patterns_str.find(delimiter)) != std::string::npos) {
            token = patterns_str.substr(0, pos);
            if (!token.empty()) {
                function_patterns.push_back(token);
            }
            patterns_str.erase(0, pos + delimiter.length());
        }
        if (!patterns_str.empty()) { // Add the last token
            function_patterns.push_back(patterns_str);
        }
    }
}

// Function to print the current configuration
void print_config() {
    // Use loprintf from logger.h (already included)
    loprintf("--- Configuration ---\n");
    loprintf("  Log Path: %s\n", getenv(ENV_LOG_PATH) ? getenv(ENV_LOG_PATH) : "(stdout)");
    loprintf("  Trace Path: %s\n", getenv(ENV_TRACE_PATH) ? getenv(ENV_TRACE_PATH) : "(stdout)");
    loprintf("  Function Filter: %s\n", getenv(ENV_TARGET_FUNCTIONS) ? getenv(ENV_TARGET_FUNCTIONS) : "(none)");
    loprintf("  Instrument All Kernels: %s\n", instrument_all_kernels ? "Yes" : "No");
    loprintf("  Instruction Range: [%u, %u]\n", instr_begin_interval, instr_end_interval);
    loprintf("  Verbose Level: %d\n", verbose);
    loprintf("  Deadlock Timeout: %d s\n", deadlock_timeout);
    loprintf("  Logging Enabled: %s\n", enable_logging ? "Yes" : "No");
    loprintf("  Log Last Traces Only: %s\n", log_last_traces_only ? "Yes" : "No");
    loprintf("  Log To Stdout: %s\n", log_to_stdout ? "Yes" : "No");
    loprintf("  Store Last Traces Only (Memory): %s\n", store_last_traces_only ? "Yes" : "No");
    loprintf("  Dump Intermediate Traces: %s\n", dump_intermedia_trace ? "Yes" : "No");
    if (dump_intermedia_trace) {
        loprintf("    Intermediate Dump Timeout: %d s\n", dump_intermedia_trace_timeout);
    }
    loprintf("  Allow Reinstrument: %s\n", allow_reinstrument ? "Yes" : "No");
    loprintf("  Kernel Iteration Begin: %u\n", kernel_iter_begin);
    loprintf("  Single Kernel Trace File: %s\n", single_kernel_trace ? "Yes" : "No");
    loprintf("  Warp Sampling Rate: %llu\n", sampling_rate_warp);
    loprintf("  Global Sampling Rate: %llu\n", sampling_rate);
    loprintf("  Loop Detection Enabled: %s\n", loop_detection_enabled ? "Yes" : "No");
    if (loop_detection_enabled) {
        loprintf("    Loop Window Size: %d\n", loop_win_size);
        loprintf("    Loop Repeat Threshold: %u\n", loop_repeat_thresh);
        loprintf("    Loop Hang Timeout: %d s\n", loop_hang_timeout);
    }
    loprintf("---------------------\n");
}

// Check if instrument_all_kernels is enabled
bool is_instrument_all_kernels_enabled() {
    return instrument_all_kernels;
}

// Check if an instruction count is within the specified range
bool is_instruction_in_ranges(uint32_t instr_cnt) {
    // Basic range check for now
    return instr_cnt >= instr_begin_interval && instr_cnt <= instr_end_interval;
    // TODO: Implement complex range check using instr_ranges vector if needed
}

// Initialize configuration (called from nvbit_at_init)
void init_config_from_env() {
    load_config();
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
