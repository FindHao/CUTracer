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

// Function name patterns to filter
extern std::vector<std::string> function_patterns;
extern bool any_function_matched;

// Initialize configuration from environment variables
void init_config_from_env();

// Check if an instruction is within the specified ranges
bool is_instruction_in_ranges(uint32_t instr_cnt); 