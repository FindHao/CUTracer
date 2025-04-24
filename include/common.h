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

#ifndef CUTRACER_COMMON_H
#define CUTRACER_COMMON_H

#include <stdint.h>
#include <vector>

/* Message type enum to identify different message types */
typedef enum { MSG_TYPE_REG_INFO = 0, MSG_TYPE_MEM_ACCESS = 1 } message_type_t;

/* Common header for all message types */
typedef struct {
  message_type_t type;  // Type of the message
} message_header_t;

/* Structure to uniquely identify a warp */
struct WarpKey {
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  int warp_id;

  // Operator for map comparison
  bool operator<(const WarpKey &other) const {
    if (cta_id_x != other.cta_id_x) return cta_id_x < other.cta_id_x;
    if (cta_id_y != other.cta_id_y) return cta_id_y < other.cta_id_y;
    if (cta_id_z != other.cta_id_z) return cta_id_z < other.cta_id_z;
    return warp_id < other.warp_id;
  }
  
  // Hash function for unordered_map
  struct Hash {
    size_t operator()(const WarpKey& k) const {
      return (size_t)k.cta_id_x ^ 
             ((size_t)k.cta_id_y << 10) ^ 
             ((size_t)k.cta_id_z << 20) ^ 
             ((size_t)k.warp_id << 30);
    }
  };
  
  // Equality operator for unordered_map
  bool operator==(const WarpKey& other) const {
    return cta_id_x == other.cta_id_x && 
           cta_id_y == other.cta_id_y && 
           cta_id_z == other.cta_id_z && 
           warp_id == other.warp_id;
  }
};

/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
typedef struct {
  message_header_t header;  // Common header with type=MSG_TYPE_REG_INFO
  int32_t cta_id_x;
  int32_t cta_id_y;
  int32_t cta_id_z;
  int32_t warp_id;
  int32_t opcode_id;
  uint64_t pc;  // Program counter for the instruction
  int32_t num_regs;
  /* 32 lanes, each thread can store up to 8 register values */
  uint32_t reg_vals[32][8];
  int32_t num_uregs;
  // unified registers shared by all threads in the same warp
  uint32_t ureg_vals[8];
} reg_info_t;

/* memory access information collected and passed
 * on the channel from the GPU to the CPU */
typedef struct {
  message_header_t header;  // Common header with type=MSG_TYPE_MEM_ACCESS
  uint64_t grid_launch_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  uint64_t pc;
  int warp_id;
  int opcode_id;
  uint64_t addrs[32];
} mem_access_t;

/* Structure to represent a single trace record */
struct TraceRecord {
  int opcode_id;
  uint64_t pc;
  std::vector<std::vector<uint32_t>> reg_values;  // [reg_idx][thread_idx]
  std::vector<uint32_t> ureg_values;              // [ureg_idx]
  std::vector<std::vector<uint64_t>> addrs;       // [thread_idx][addr_idx]
};

#endif // CUTRACER_COMMON_H
