//RSQF.cuh
/*
 * Copyright 2021 Regents of the University of California
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef RSQF_GPU_H
#define RSQF_GPU_H

#define RBITS 5
#define SLOTS_PER_BLOCK 64

struct __attribute__ ((packed)) cqf_gpu_block
{
    unsigned char offset;
    long unsigned int occupieds;
    long unsigned int runEnds;
    long unsigned int remainders[RBITS];
};

struct countingQuotientFilterGPU
{
    unsigned int qbits;
    unsigned int numBlocks;
    cqf_gpu_block* blocks; 
};

__host__ __device__ size_t calcNumBlocksGPU(unsigned int q);

__host__ void initCQFGPU(struct countingQuotientFilterGPU *cqf, unsigned int q);
    /*  Allocates memory for the counting quotient filter on the GPU
     *  based on number of quotient bits.
     *  Filter Capacity = 2 ^ q */

__host__ void printGPUFilter(struct countingQuotientFilterGPU *cqf);

__device__ __host__ unsigned int Normal_APHashGPU(unsigned int value, unsigned int maxHashValue);

__host__ float launchLookups(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_lookupValues, int* slotValuesArray);
    /* Looks up value in RSQF.
    *  Hashes all inputs and sorts before performing lookups, then sorts again to match with inputs.
    *  Returns the location of the remainder in slotValuesArray if it is found.
    *  Returns -1 if the remainder is not found. 
    *  Return value is time for CUDA code.  */

__host__ float launchUnsortedLookups(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_lookupValues, int* d_slotValuesArray);
    /* Looks up value in RSQF.
    *  This operation generally has lower throughput than the sorted version.
    *  Returns the location of the remainder in slotValuesArray if it is found.
    *  Returns -1 if the remainder is not found. 
    *  Return value is time for CUDA code.  */

__host__ float insertGPU(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_insertValues, int* d_returnValues);
    /* Inserts values into RSQF.
    *  Returns the final location of the remainders in d_returnValues.
    *  Return value is time for CUDA code.  */

#endif
