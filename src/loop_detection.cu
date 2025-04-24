/*
 * SPDX-FileCopyrightText: Copyright (c) 2023
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <map>
#include <set>
#include <time.h>
#include <vector>

#include "common.h"
#include "env_config.h"
#include "loop_detection.h"

// Forward declarations
extern void loprintf(const char *format, ...);
extern std::map<int, std::string> id_to_sass_map;

// Maps for tracking warp loop states
std::map<WarpKey, WarpLoopState> loop_states;
std::set<WarpKey> active_warps;

// Last time we checked for kernel hangs
static time_t last_hang_check_time = 0;

/**
 * Updates the loop state for a warp based on its current PC
 * 
 * @param key The warp key
 * @param pc The program counter
 */
void update_loop_state(const WarpKey& key, uint64_t pc) {
    if (!loop_detection_enabled) {
        return;
    }

    // If key doesn't exist, add it with the right window size
    if (loop_states.find(key) == loop_states.end()) {
        loop_states.emplace(key, WarpLoopState(loop_win_size));
    }

    auto &state = loop_states[key];
    state.pcs[state.head] = pc;
    state.head = (state.head + 1) % loop_win_size;
    active_warps.insert(key);

    if (state.head != 0) {
        return;  // Buffer not full yet
    }

    // FNV-1a hash
    uint64_t sig = 14695981039346656037ULL;
    for (int i = 0; i < loop_win_size; ++i) {
        sig = (sig ^ state.pcs[i]) * 1099511628211ULL;
    }

    if (sig == state.last_sig) {
        if (++state.repeat_cnt >= (uint32_t)loop_repeat_thresh && !state.loop_flag) {
            state.loop_flag = true;
            state.first_loop_time = time(nullptr);
            
            if (verbose >= 2) {
                loprintf("Warp loop detected: CTA %d,%d,%d warp %d\n", 
                        key.cta_id_x, key.cta_id_y, key.cta_id_z, key.warp_id);
            }
        }
    } else {
        state.repeat_cnt = 0;
        state.last_sig = sig;
        state.loop_flag = false;
    }
}

/**
 * Clears loop detection state when a kernel completes
 */
void clear_loop_state() {
    if (!loop_detection_enabled) {
        return;
    }
    
    loop_states.clear();
    active_warps.clear();
    last_hang_check_time = time(nullptr);
}

/**
 * Checks if all active warps are stuck in loops
 * 
 * @param warp_traces Trace records for each warp
 * @param force_check Force check even if the periodic timer hasn't elapsed
 * @return true if a kernel hang is detected
 */
bool check_kernel_hang(const std::map<WarpKey, std::vector<TraceRecord>>& warp_traces, bool force_check) {
    if (!loop_detection_enabled || active_warps.empty()) {
        return false;
    }
    
    time_t now = time(nullptr);
    
    // Only check once per second unless forced
    if (!force_check && (now - last_hang_check_time < 1)) {
        return false;
    }
    
    last_hang_check_time = now;
    
    // Check if all active warps are in loops
    bool all_loops = true;
    time_t oldest_loop_time = now;
    
    for (const auto& key : active_warps) {
        auto it = loop_states.find(key);
        if (it == loop_states.end() || !it->second.loop_flag) {
            all_loops = false;
            break;
        }
        
        oldest_loop_time = std::min(oldest_loop_time, it->second.first_loop_time);
    }
    
    // If all warps are looping and the oldest loop has been going for at least HANG_TIMEOUT seconds
    if (all_loops && (now - oldest_loop_time >= loop_hang_timeout)) {
        loprintf("\n!!! KERNEL HANG DETECTED after %d seconds !!!\n\n", loop_hang_timeout);
        
        // Print detailed information about each looping warp
        for (const auto& key : active_warps) {
            const auto& warp_state = loop_states[key];
            
            // Make sure we have trace data for this warp
            auto trace_it = warp_traces.find(key);
            if (trace_it == warp_traces.end() || trace_it->second.empty()) {
                continue;
            }
            
            // Get the last trace record for this warp
            const auto& trace = trace_it->second.back();
            
            // Print warp information, SASS instruction, and PC offset
            loprintf("  CTA %d,%d,%d warp %d | %s | PCoff %ld (0x%lx)\n",
                     key.cta_id_x, key.cta_id_y, key.cta_id_z, key.warp_id,
                     id_to_sass_map[trace.opcode_id].c_str(), trace.pc/16, trace.pc);
                     
            // If we have register values, print them
            if (!trace.reg_values.empty()) {
                loprintf("    Registers (thread 0): ");
                for (size_t i = 0; i < std::min(size_t(4), trace.reg_values.size()); i++) {
                    if (!trace.reg_values[i].empty()) {
                        loprintf("R%zu:0x%08x ", i, trace.reg_values[i][0]);
                    }
                }
                loprintf("\n");
            }
            
            // If we have unified register values, print them
            if (!trace.ureg_values.empty()) {
                loprintf("    Unified Registers: ");
                
                // Check for barrier sync operations (look for SYNC or BAR in the instruction)
                std::string sass_str = id_to_sass_map[trace.opcode_id];
                bool is_sync = sass_str.find("SYNC") != std::string::npos || 
                               sass_str.find("BAR") != std::string::npos;
                
                // Special handling for synchronization instructions
                if (is_sync) {
                    loprintf("*** SYNC OPERATION DETECTED ***\n");
                    
                    for (size_t i = 0; i < std::min(size_t(8), trace.ureg_values.size()); i++) {
                        loprintf("    UR%zu:0x%08x ", i, trace.ureg_values[i]);
                        
                        // Add special annotations for barrier registers
                        if (i == 7 && is_sync) {
                            loprintf("(possible barrier ticket address: 0x%08x+offset)\n", 
                                    trace.ureg_values[i]);
                        }
                    }
                } else {
                    // Standard output for non-sync operations
                    for (size_t i = 0; i < std::min(size_t(8), trace.ureg_values.size()); i++) {
                        if (i == 7 || i == 6) {  // Special attention to UR7 and UR6 which often contain barrier info
                            loprintf("UR%zu:0x%08x ", i, trace.ureg_values[i]);
                        }
                    }
                    loprintf("\n");
                }
            }
        }
        
        return true;
    }
    
    return false;
} 