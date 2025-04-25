#include "logger.h"

#include <stdio.h>
#include <stdarg.h>
#include <string> // For std::string potentially used later

// Static file handle for the log file
static FILE *log_handle = NULL;

// Base function for logging (internal implementation detail)
template <typename... Args>
static void base_fprintf(bool file_output, bool stdout_output, const char *format, Args... args) {
    // If no output destination is specified, return immediately.
    if (!file_output && !stdout_output) return;

    // Use a reasonably sized buffer for the formatted message.
    // Consider dynamic allocation or other strategies if messages can exceed this size.
    char output_buffer[2048];
    int written = snprintf(output_buffer, sizeof(output_buffer), format, args...);

    // Check for snprintf errors (optional but good practice)
    if (written < 0 || written >= sizeof(output_buffer)) {
        // Handle error: message truncated or encoding error
        fprintf(stderr, "Logger Error: Failed to format message or buffer too small.\\n");
        // Optionally log the truncated message or take other actions
        // For now, we'll just proceed with potentially truncated output
    }


    // Output to stdout if requested
    if (stdout_output) {
        fprintf(stdout, "%s", output_buffer);
        fflush(stdout); // Ensure immediate output
    }

    // Output to log file if requested and the handle is valid
    if (file_output && log_handle != NULL) {
        fprintf(log_handle, "%s", output_buffer);
        fflush(log_handle); // Ensure immediate write to file
    }
}

// --- Public API Implementation ---

void init_logger(const char *log_filepath) {
    if (log_handle != NULL && log_handle != stdout) {
        fclose(log_handle); // Close existing file if any
    }

    if (log_filepath != NULL && log_filepath[0] != '\\0') {
        log_handle = fopen(log_filepath, "w");
        if (log_handle == NULL) {
            // Failed to open file, fall back to stdout and print an error
            fprintf(stderr, "Error: Could not open log file '%s'. Logging to stdout instead.\\n", log_filepath);
            log_handle = stdout;
        }
    } else {
        // No path provided, default to stdout
        log_handle = stdout;
    }
}

void close_logger() {
    if (log_handle != NULL && log_handle != stdout) {
        fclose(log_handle);
        log_handle = NULL;
    }
}

// Explicit template instantiation is needed for template functions defined in a .cpp file
// We define them here so the linker can find them.

template <typename... Args>
void lprintf(const char *format, Args... args) {
    base_fprintf(true, false, format, args...);
}

template <typename... Args>
void oprintf(const char *format, Args... args) {
    base_fprintf(false, true, format, args...);
}

template <typename... Args>
void loprintf(const char *format, Args... args) {
    // If logging to stdout, don't duplicate output if file handle IS stdout
    bool file_is_stdout = (log_handle == stdout);
    base_fprintf(!file_is_stdout, true, format, args...);
}


// --- Explicit Instantiations ---
// Need to instantiate the templates for common types used in main.cu
// Add more instantiations if other types are used with logging functions

// Example instantiations (adjust based on actual usage in main.cu)
// For loprintf/lprintf/oprintf(const char*)
template void lprintf<>(const char* format);
template void oprintf<>(const char* format);
template void loprintf<>(const char* format);

// For loprintf/lprintf/oprintf(const char*, int)
template void lprintf<int>(const char* format, int);
template void oprintf<int>(const char* format, int);
template void loprintf<int>(const char* format, int);

// For loprintf/lprintf/oprintf(const char*, unsigned int)
template void lprintf<unsigned int>(const char* format, unsigned int);
template void oprintf<unsigned int>(const char* format, unsigned int);
template void loprintf<unsigned int>(const char* format, unsigned int);

// For loprintf/lprintf/oprintf(const char*, long)
template void lprintf<long>(const char* format, long);
template void oprintf<long>(const char* format, long);
template void loprintf<long>(const char* format, long);

// For loprintf/lprintf/oprintf(const char*, unsigned long)
template void lprintf<unsigned long>(const char* format, unsigned long);
template void oprintf<unsigned long>(const char* format, unsigned long);
template void loprintf<unsigned long>(const char* format, unsigned long);

// For loprintf/lprintf/oprintf(const char*, long long)
template void lprintf<long long>(const char* format, long long);
template void oprintf<long long>(const char* format, long long);
template void loprintf<long long>(const char* format, long long);

// For loprintf/lprintf/oprintf(const char*, unsigned long long)
template void lprintf<unsigned long long>(const char* format, unsigned long long);
template void oprintf<unsigned long long>(const char* format, unsigned long long);
template void loprintf<unsigned long long>(const char* format, unsigned long long);

// For loprintf/lprintf/oprintf(const char*, double)
template void lprintf<double>(const char* format, double);
template void oprintf<double>(const char* format, double);
template void loprintf<double>(const char* format, double);

// For loprintf/lprintf/oprintf(const char*, const char*)
template void lprintf<const char*>(const char* format, const char*);
template void oprintf<const char*>(const char* format, const char*);
template void loprintf<const char*>(const char* format, const char*);

// For loprintf/lprintf/oprintf(const char*, void*)
template void lprintf<void*>(const char* format, void*);
template void oprintf<void*>(const char* format, void*);
template void loprintf<void*>(const char* format, void*);

// Add combinations if necessary, e.g., (const char*, int, const char*)
template void loprintf<int, const char*>(const char* format, int, const char*);
template void loprintf<long long, long long>(const char* format, long long, long long);
template void loprintf<unsigned long long, unsigned long long>(const char* format, unsigned long long, unsigned long long);
template void loprintf<const char*, unsigned long long>(const char* format, const char*, unsigned long long);
template void loprintf<const char*, int, int, int, int>(const char* format, const char*, int, int, int, int); // For loop detected message
template void loprintf<int, int, int, int, unsigned long long, const char*, uint64_t>(const char* format, int, int, int, int, unsigned long long, const char*, uint64_t); // For recv_thread loop processing
template void loprintf<int, int, int, int, unsigned long long, const char*, uint64_t, int>(const char* format, int, int, int, int, unsigned long long, const char*, uint64_t, int); // For recv_thread reg info
template void loprintf<int, int, int, int, unsigned long long, const char*, uint64_t, int, int>(const char* format, int, int, int, int, unsigned long long, const char*, uint64_t, int, int); // For recv_thread reg info more detail
template void loprintf<int, int, int, int, unsigned long long, const char*, uint64_t, int, int, uint32_t>(const char* format, int, int, int, int, unsigned long long, const char*, uint64_t, int, int, uint32_t); // For recv_thread reg values
template void loprintf<int, int, int, int, unsigned long long, const char*, uint64_t, int, uint32_t>(const char* format, int, int, int, int, unsigned long long, const char*, uint64_t, int, uint32_t); // For recv_thread ureg info
template void loprintf<int, int, int, int, unsigned long long, const char*, uint64_t, int, uint64_t>(const char* format, int, int, int, int, unsigned long long, const char*, uint64_t, int, uint64_t); // For recv_thread mem access
template void loprintf<const char*, int, int, int, int, double>(const char* format, const char*, int, int, int, int, double); // For thread summary
template void loprintf<const char*, const char*, int, int, int, int>(const char* format, const char*, const char*, int, int, int, int); // For onKernelLaunchStart
template void lprintf<const char*, int, int, int, int>(const char* format, const char*, int, int, int, int); // For shouldDumpTrace

// Need instantiations for other types if used e.g. std::string, etc.
// If you encounter linker errors about undefined references to lprintf, oprintf, or loprintf,
// you need to add explicit instantiations here for the exact argument types used in the failing call site. 