#ifndef LOGGER_H
#define LOGGER_H

#include <stdarg.h> // Required for va_list in template declarations if used directly, though better encapsulated in .cpp

// Initializes the logger. If log_filepath is NULL or empty, logs to stdout.
// Otherwise, attempts to open log_filepath for writing.
// If opening the file fails, logs an error to stderr and falls back to stdout.
void init_logger(const char *log_filepath);

// Closes the log file if it was opened.
void close_logger();

/**
 * lprintf - print to log file only (log print)
 */
template <typename... Args>
void lprintf(const char *format, Args... args);

/**
 * oprintf - print to stdout only (output print)
 */
template <typename... Args>
void oprintf(const char *format, Args... args);

/**
 * loprintf - print to log file and stdout (log and output print)
 */
template <typename... Args>
void loprintf(const char *format, Args... args);

#endif // LOGGER_H 