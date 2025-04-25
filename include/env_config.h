#pragma once

#include <stdint.h>

#include <string>
#include <vector>

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

// Initialize configuration from environment variables
void init_config_from_env();

// Check if an instruction is within the specified ranges
bool is_instruction_in_ranges(uint32_t instr_cnt);
