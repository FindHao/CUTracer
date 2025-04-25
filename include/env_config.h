#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <utility> // For std::pair

// --- Environment Variable Names ---
#define ENV_LOG_PATH "TOOL_LOG_PATH"
#define ENV_TRACE_PATH "TOOL_TRACE_PATH"
#define ENV_TARGET_FUNCTIONS "FUNC_NAME_FILTER"
#define ENV_INSTR_RANGES "INSTR_RANGES"
#define ENV_INSTR_BEGIN "INSTR_BEGIN"
#define ENV_INSTR_END "INSTR_END"
#define ENV_VERBOSE "VERBOSE"
#define ENV_DEADLOCK_TIMEOUT "DEADLOCK_TIMEOUT"
#define ENV_LOG_ENABLE "ENABLE_LOGGING"
#define ENV_LOG_LAST_TRACES "LOG_LAST_TRACES_ONLY"
#define ENV_LOG_STDOUT "LOG_TO_STDOUT"
#define ENV_STORE_LAST_TRACES "STORE_LAST_TRACES_ONLY"
#define ENV_DUMP_INTERMEDIA "DUMP_INTERMEDIA_TRACE"
#define ENV_DUMP_INTERMEDIA_TIMEOUT "DUMP_INTERMEDIA_TRACE_TIMEOUT"
#define ENV_ALLOW_REINSTRUMENT "ALLOW_REINSTRUMENT"
#define ENV_KERNEL_ITER_BEGIN "KERNEL_ITER_BEGIN"
#define ENV_SINGLE_KERNEL_TRACE "SINGLE_KERNEL_TRACE"
#define ENV_SAMPLING_RATE_WARP "SAMPLING_RATE_WARP"
#define ENV_SAMPLING_RATE "SAMPLING_RATE"
#define ENV_LOOP_WIN_SIZE "LOOP_WIN_SIZE"
#define ENV_LOOP_REPEAT_THRESH "LOOP_REPEAT_THRESH"
#define ENV_LOOP_HANG_TIMEOUT "LOOP_HANG_TIMEOUT"
#define ENV_LOOP_DETECTION_ENABLED "LOOP_DETECTION_ENABLED"
#define ENV_INSTRUMENT_ALL_KERNELS "INSTRUMENT_ALL_KERNELS"

// Configuration variables
extern uint32_t instr_begin_interval;
extern uint32_t instr_end_interval;
extern int verbose;
extern std::vector<std::pair<uint32_t, uint32_t>> instr_ranges;
extern bool use_instr_ranges;
extern std::string instr_ranges_str;
extern int deadlock_timeout;
extern int enable_logging;
extern int log_last_traces_only;
extern int log_to_stdout;
extern int store_last_traces_only;
extern int dump_intermedia_trace;
extern int dump_intermedia_trace_timeout;
extern int allow_reinstrument;
extern uint32_t kernel_iter_begin;
extern int single_kernel_trace;
extern int instrument_all_kernels;
extern uint64_t sampling_rate_warp;   // Sampling rate for trace dump based on warp (1=every instruction, N=every Nth instruction per warp)
extern uint64_t sampling_rate;        // Sampling rate for trace dump based on received data (1=every instruction, N=every Nth instruction)

// Loop detection configuration
extern int loop_win_size;             // Size of the PC window for loop detection (default: 32)
extern uint32_t loop_repeat_thresh;        // Threshold for repeat count to detect a loop (default: 16)
extern int loop_hang_timeout;         // Timeout in seconds for hang detection (default: 3)
extern int loop_detection_enabled;    // Enable/disable loop detection (default: 1)

// Function name patterns to filter
extern std::vector<std::string> function_patterns;
extern bool any_function_matched;

// Configuration functions
void load_config();
void print_config();
void init_config_from_env();
bool is_instrument_all_kernels_enabled();
bool is_instruction_in_ranges(uint32_t instr_cnt);
void truncate_mangled_name(const char *mangled_name, char *truncated_buffer, size_t buffer_size);
