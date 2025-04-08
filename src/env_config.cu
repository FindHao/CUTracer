#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>

// We don't need to include nvbit_tool.h in every source file, just in main.cu
// We only use variables and functions defined in header files
#include "env_config.h"

// Define configuration variables
// EVERY VARIABLE MUST BE INITIALIZED IN init_config_from_env()
uint32_t instr_begin_interval;
uint32_t instr_end_interval;
int verbose;
std::vector<std::pair<uint32_t, uint32_t>> instr_ranges;
bool use_instr_ranges = false;
std::string instr_ranges_str = "";
int deadlock_timeout;
int enable_logging;
int log_last_traces_only;
int log_to_stdout;
int store_last_traces_only;
int dump_intermedia_trace;
int dump_intermedia_trace_timeout;
int allow_reinstrument;      // if true, allow instrumenting the same kernel multiple times
uint32_t kernel_iter_begin;  // start instrumenting from this kernel iteration (0=first iteration)
int single_kernel_trace;
uint64_t sampling_rate;      // Sampling rate for trace dump (1=every instruction)

// Function name filters
std::vector<std::string> function_patterns;
bool any_function_matched = false;

// Check if instruction is within specified ranges
bool is_instruction_in_ranges(uint32_t instr_cnt) {
  if (!use_instr_ranges) {
    // Use the old interval-based logic
    return (instr_cnt >= instr_begin_interval && instr_cnt < instr_end_interval);
  }

  // Check if instruction count is in any specified range
  for (const auto &range : instr_ranges) {
    if (instr_cnt >= range.first && instr_cnt <= range.second) {
      return true;
    }
  }

  return false;
}

// Parse instruction ranges from environment variable
static void parse_instruction_ranges(const char *instr_filter) {
  if (!instr_filter) return;

  instr_ranges_str = std::string(instr_filter);
  use_instr_ranges = true;

  // Parse instruction ranges
  std::string ranges_str = instr_ranges_str;
  size_t pos = 0;
  std::string range;

  // Split by commas
  while ((pos = ranges_str.find(',')) != std::string::npos) {
    range = ranges_str.substr(0, pos);
    ranges_str.erase(0, pos + 1);

    // Parse individual range (could be a single number or a range with dash)
    size_t dash_pos = range.find('-');
    if (dash_pos != std::string::npos) {
      // This is a range (e.g., "1-10")
      try {
        uint32_t start = std::stoi(range.substr(0, dash_pos));
        uint32_t end = std::stoi(range.substr(dash_pos + 1));
        if (start > end) {
          printf("WARNING: Invalid range %s in INSTRS (start > end). Skipping this range.\n", range.c_str());
          continue;
        }
        instr_ranges.push_back(std::make_pair(start, end));
      } catch (const std::exception &e) {
        printf("ERROR: Failed to parse range %s in INSTRS: %s\n", range.c_str(), e.what());
        continue;
      }
    } else {
      // This is a single number (e.g., "13")
      try {
        uint32_t num = std::stoi(range);
        instr_ranges.push_back(std::make_pair(num, num));
      } catch (const std::exception &e) {
        printf("ERROR: Failed to parse instruction number %s in INSTRS: %s\n", range.c_str(), e.what());
        continue;
      }
    }
  }

  // Process the last range (after the last comma, or the entire string if no commas)
  if (!ranges_str.empty()) {
    size_t dash_pos = ranges_str.find('-');
    if (dash_pos != std::string::npos) {
      // This is a range (e.g., "1-10")
      try {
        uint32_t start = std::stoi(ranges_str.substr(0, dash_pos));
        uint32_t end = std::stoi(ranges_str.substr(dash_pos + 1));
        if (start > end) {
          printf("WARNING: Invalid range %s in INSTRS (start > end). Skipping this range.\n", ranges_str.c_str());
        } else {
          instr_ranges.push_back(std::make_pair(start, end));
        }
      } catch (const std::exception &e) {
        printf("ERROR: Failed to parse range %s in INSTRS: %s\n", ranges_str.c_str(), e.what());
      }
    } else {
      // This is a single number (e.g., "13")
      try {
        uint32_t num = std::stoi(ranges_str);
        instr_ranges.push_back(std::make_pair(num, num));
      } catch (const std::exception &e) {
        printf("ERROR: Failed to parse instruction number %s in INSTRS: %s\n", ranges_str.c_str(), e.what());
      }
    }
  }

  // Check if we have valid ranges
  if (instr_ranges.empty()) {
    printf("WARNING: No valid instruction ranges found in INSTRS=\"%s\". No instructions will be instrumented.\n",
           instr_filter);
  }

  // Backward compatibility: if there's only one number (no dash), set it as instr_begin_interval
  if (instr_ranges.size() == 1 && instr_ranges[0].first == instr_ranges[0].second) {
    instr_begin_interval = instr_ranges[0].first;
    if (verbose) {
      printf("Setting instruction begin interval to %u (backward compatibility mode)\n", instr_begin_interval);
    }
  }

  if (verbose) {
    printf("Instruction ranges: ");
    for (size_t i = 0; i < instr_ranges.size(); i++) {
      if (instr_ranges[i].first == instr_ranges[i].second) {
        printf("%u", instr_ranges[i].first);
      } else {
        printf("%u-%u", instr_ranges[i].first, instr_ranges[i].second);
      }
      if (i < instr_ranges.size() - 1) {
        printf(", ");
      }
    }
    printf("\n");
  }

  printf("Using instruction ranges filter: %s\n", instr_ranges_str.c_str());
}

// Parse function patterns from environment variable
static void parse_function_patterns(const char *patterns_env) {
  if (!patterns_env) return;

  std::string patterns_str(patterns_env);
  size_t pos = 0;
  std::string token;

  // Split by commas
  while ((pos = patterns_str.find(',')) != std::string::npos) {
    token = patterns_str.substr(0, pos);
    if (!token.empty()) {
      function_patterns.push_back(token);
    }
    patterns_str.erase(0, pos + 1);
  }

  // Add the last token (if it exists)
  if (!patterns_str.empty()) {
    function_patterns.push_back(patterns_str);
  }

  if (verbose) {
    printf("Function name filters to instrument:\n");
    for (const auto &pattern : function_patterns) {
      printf("  - %s\n", pattern.c_str());
    }
  }
}

// Helper function for reading environment variables
static void get_var_int(int &var, const char *env_name, int default_val, const char *description) {
  const char *env_val = getenv(env_name);
  if (env_val) {
    var = atoi(env_val);
  } else {
    var = default_val;
  }
  printf("%s = %d (%s)\n", env_name, var, description);
}

static void get_var_uint32(uint32_t &var, const char *env_name, uint32_t default_val, const char *description) {
  const char *env_val = getenv(env_name);
  if (env_val) {
    var = (uint32_t)atoll(env_val);
  } else {
    var = default_val;
  }
  printf("%s = %u (%s)\n", env_name, var, description);
}

static void get_var_uint64(uint64_t &var, const char *env_name, uint64_t default_val, const char *description) {
  const char *env_val = getenv(env_name);
  if (env_val) {
    var = (uint64_t)atoll(env_val);
  } else {
    var = default_val;
  }
  printf("%s = %lu (%s)\n", env_name, var, description);
}

// Initialize all configuration variables
void init_config_from_env() {
  // Enable device memory allocation
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

  // Get instruction range filter
  const char *instr_filter = getenv("INSTRS");
  if (instr_filter) {
    parse_instruction_ranges(instr_filter);
  } else {
    // If INSTRS is not set, fall back to the old INSTR_BEGIN/INSTR_END behavior
    get_var_uint32(instr_begin_interval, "INSTR_BEGIN", 0,
                   "Beginning of the instruction interval where to apply instrumentation");
    get_var_uint32(instr_end_interval, "INSTR_END", UINT32_MAX,
                   "End of the instruction interval where to apply instrumentation");
  }

  // Get other configuration variables
  get_var_int(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  get_var_int(deadlock_timeout, "DEADLOCK_TIMEOUT", 10, "Timeout in seconds to detect potential deadlocks");
  get_var_int(enable_logging, "ENABLE_LOGGING", 1, "Enable/disable logging (1=enabled, 0=disabled)");
  get_var_int(log_last_traces_only, "LOG_LAST_TRACES_ONLY", 0,
              "Only log the last trace for each warp (1=enabled, 0=disabled)");
  get_var_int(log_to_stdout, "LOG_TO_STDOUT", 1, "Log to stdout instead of files (1=enabled, 0=disabled)");
  get_var_int(store_last_traces_only, "STORE_LAST_TRACES_ONLY", 0,
              "Only store the last trace for each warp in memory (1=enabled, 0=disabled)");
  get_var_int(dump_intermedia_trace, "DUMP_INTERMEDIA_TRACE", 0,
              "Dump intermediate trace data to stdout (1=enabled, 0=disabled)");
  get_var_int(dump_intermedia_trace_timeout, "DUMP_INTERMEDIA_TRACE_TIMEOUT", 0,
              "Timeout in seconds for intermediate trace dumping (0=unlimited)");
  get_var_int(allow_reinstrument, "ALLOW_REINSTRUMENT", 0,
              "Allow instrumenting the same kernel multiple times (1=enabled, 0=disabled)");
  get_var_uint32(kernel_iter_begin, "KERNEL_ITER_BEGIN", 0,
                 "Start instrumenting from this kernel iteration (0=first iteration)");
  get_var_int(single_kernel_trace, "SINGLE_KERNEL_TRACE", 0, "Enable single kernel trace (1=enabled, 0=disabled)");
  get_var_uint64(sampling_rate, "SAMPLING_RATE", 1, 
                "Sampling rate for trace dump (1=every instruction, N=every Nth instruction)");

  // Get function name filter
  const char *patterns_env = getenv("FUNC_NAME_FILTER");
  if (patterns_env) {
    parse_function_patterns(patterns_env);
  } else if (verbose) {
    printf("WARNING: No function name filters specified. Instrumenting all functions.\n");
  }

  std::string pad(100, '-');
  printf("%s\n", pad.c_str());
}