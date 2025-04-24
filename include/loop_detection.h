/*
 * SPDX-FileCopyrightText: Copyright (c) 2023
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <map>
#include <set>
#include <vector>
#include <time.h>

#include "common.h"

/**
 * Structure to track the loop state of a warp
 */
struct WarpLoopState {
    std::vector<uint64_t> pcs;  // Circular buffer of PCs
    uint8_t head;               // Current position in circular buffer
    uint64_t last_sig;          // Last computed signature
    uint32_t repeat_cnt;        // Number of consecutive pattern repetitions
    bool loop_flag;             // Flag indicating loop detection
    time_t first_loop_time;     // Time when loop was first detected

    // Default constructor (using a default window size of 32)
    WarpLoopState() 
        : pcs(32, 0), head(0), last_sig(0), repeat_cnt(0), 
          loop_flag(false), first_loop_time(0) {}
          
    WarpLoopState(int window_size) 
        : pcs(window_size, 0), head(0), last_sig(0), repeat_cnt(0), 
          loop_flag(false), first_loop_time(0) {}
};

// Maps for tracking warp loop states
extern std::map<WarpKey, WarpLoopState> loop_states;
extern std::set<WarpKey> active_warps;

/**
 * Updates the loop state for a warp based on its current PC
 * 
 * @param key The warp key
 * @param pc The program counter
 */
void update_loop_state(const WarpKey& key, uint64_t pc);

/**
 * Checks if all active warps are stuck in loops
 * 
 * @param warp_traces Trace records for each warp
 * @param force_check Force check even if the periodic timer hasn't elapsed
 * @return true if a kernel hang is detected
 */
bool check_kernel_hang(const std::map<WarpKey, std::vector<TraceRecord>>& warp_traces, bool force_check = false);

/**
 * Clears loop detection state when a kernel completes
 */
void clear_loop_state(); 