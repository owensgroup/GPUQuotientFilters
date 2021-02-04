//RSQF.cu
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

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include "RSQF.cuh"

#define LOW_BIT_MASK(n) ((1U << n) - 1U)
#define LOW_BIT_MASKLL(n) (n >= 64 ? 0xFFFFFFFFFFFFFFFF : (1ULL << n) - 1ULL)

__host__ __device__ size_t calcNumBlocksGPU(unsigned int q)
{
        return (((1 << q) + (SLOTS_PER_BLOCK - 1)) / SLOTS_PER_BLOCK) + 1;
}

__host__ void initCQFGPU(struct countingQuotientFilterGPU* cqf, unsigned int q)
{
    cqf->qbits = q;
    cqf->numBlocks = calcNumBlocksGPU(q);
    cqf_gpu_block* d_filterBlocks;
    cudaMalloc((void**) &d_filterBlocks, cqf->numBlocks * sizeof(cqf_gpu_block));
    cudaMemset(d_filterBlocks, 0, cqf->numBlocks * sizeof(cqf_gpu_block));
    cqf->blocks = d_filterBlocks;
}

__host__ __device__ bool isOccupiedGPU(long long unsigned int occupieds, unsigned int slotNumber)
{
    return (1ULL << slotNumber) & occupieds;
}

__device__ long long unsigned int setOccupiedGPU(long long unsigned int occupieds, unsigned int slotNumber)
{
    return (1ULL << slotNumber) | occupieds;
}

__host__ __device__ bool isRunEndGPU(long long unsigned int runends, unsigned int slotNumber)
{
    return (1ULL << slotNumber) & runends;
}

__device__ long long unsigned int setRunEndGPU(long long unsigned int runends, unsigned int slotNumber)
{
    return (1ULL << slotNumber) | runends;
}

__device__ long long unsigned int clearRunEndGPU(long long unsigned int runends, unsigned int slotNumber)
{
    return ~(1ULL << slotNumber) & runends;
}

__device__ unsigned int findBlockNumberGPU(unsigned int globalSlotNumber)
{
    return globalSlotNumber / SLOTS_PER_BLOCK;
}

__device__ unsigned int findPositionInBlockGPU(unsigned int globalSlotNumber)
{
    return globalSlotNumber % SLOTS_PER_BLOCK;
}

__host__ __device__ unsigned int findRemainderIntSlotGPU(unsigned int blockPosition)
{
    return blockPosition * RBITS / 64;
}

__host__ __device__ unsigned int findRemainderStartBitGPU(unsigned int blockPosition)
{
    return (blockPosition * RBITS) % 64;
}

__device__ unsigned int globalSlotIndexGPU(unsigned int blockNum, unsigned int slotNumInBlock){
    return (slotNumInBlock + (blockNum * SLOTS_PER_BLOCK));
}

__host__ __device__ unsigned int getRemainderGPU(struct countingQuotientFilterGPU* cqf, unsigned int blockNum, unsigned int slotNum)
{
    unsigned int integerSlot = findRemainderIntSlotGPU(slotNum);
    unsigned int startBit = findRemainderStartBitGPU(slotNum);
    int spillover = startBit + RBITS - 64;
    unsigned int remainder;
    if(spillover <= 0){
        remainder = (cqf->blocks[blockNum].remainders[integerSlot] >> startBit) & LOW_BIT_MASKLL(RBITS);
    }
    else{
        unsigned int mainBlockBits = RBITS - spillover;
        remainder = (cqf->blocks[blockNum].remainders[integerSlot] >> startBit) & LOW_BIT_MASKLL(mainBlockBits);
        unsigned int spilledOverBits = (cqf->blocks[blockNum].remainders[integerSlot + 1] & LOW_BIT_MASKLL(spillover)) << mainBlockBits;
        remainder = remainder | spilledOverBits;
    }
    return remainder;
}

__device__ void setRemainderGPU(struct countingQuotientFilterGPU* cqf, unsigned int blockNum, unsigned int slotNum, unsigned int value)
{
    unsigned int integerSlot = findRemainderIntSlotGPU(slotNum);
    unsigned int startBit = findRemainderStartBitGPU(slotNum);
    int spillover = startBit + RBITS - 64;
    if(spillover <= 0){
        cqf->blocks[blockNum].remainders[integerSlot] &= ~(LOW_BIT_MASKLL(RBITS) << startBit);
        cqf->blocks[blockNum].remainders[integerSlot] |= ((long long unsigned int)value << startBit);
    }
    else{
        unsigned int mainBlockBits = RBITS - spillover;
        cqf->blocks[blockNum].remainders[integerSlot] &= ~(LOW_BIT_MASKLL(mainBlockBits) << startBit);
        cqf->blocks[blockNum].remainders[integerSlot] |= (LOW_BIT_MASKLL(mainBlockBits) & (long long unsigned int)value) << startBit;
        cqf->blocks[blockNum].remainders[integerSlot + 1] &= ~(LOW_BIT_MASKLL(spillover));
        cqf->blocks[blockNum].remainders[integerSlot + 1] |= (long long unsigned int)value >> mainBlockBits;
    }
}

__host__ void printGPUFilter(struct countingQuotientFilterGPU* cqf)
{
    cqf_gpu_block* h_filterBlocks = new cqf_gpu_block[cqf->numBlocks];
    cudaMemcpy(h_filterBlocks, cqf->blocks, cqf->numBlocks * sizeof(cqf_gpu_block), cudaMemcpyDeviceToHost);
    cqf_gpu_block* d_filterBlocks = cqf->blocks;
    cqf->blocks = h_filterBlocks;
    printf("Filter contents:\n");
    for(int i = 0; i < cqf->numBlocks; i++){
        printf("block: %i\t", i);
        printf("offset: %u\n", cqf->blocks[i].offset);
        int rowsPerBlock = SLOTS_PER_BLOCK / 10;
        for(int j = 0; j < rowsPerBlock; j++){
            printf("\noccupieds:\t");
            for(int k = 0; k < 10; k++){
                printf("%u\t", isOccupiedGPU(cqf->blocks[i].occupieds, 10 * j + k));
            }
            printf("\nrunEnds:\t");
            for(int k = 0; k < 10; k++){
                printf("%u\t", isRunEndGPU(cqf->blocks[i].runEnds, 10 * j + k));
            }
            printf("\nremainders:\t");
            for(int k = 0; k < 10; k++){
                printf("%u\t", getRemainderGPU(cqf, i, 10 * j + k));
            }
            printf("\n");
        }
        if(SLOTS_PER_BLOCK % 10 != 0){
            int numLeft = SLOTS_PER_BLOCK % 10;
            printf("\noccupieds:\t");
            for(int k = 0; k < numLeft; k++){
                printf("%u\t", isOccupiedGPU(cqf->blocks[i].occupieds, rowsPerBlock * 10 + k));
            }
            printf("\nrunEnds:\t");
            for(int k = 0; k < numLeft; k++){
                printf("%u\t", isRunEndGPU(cqf->blocks[i].runEnds, rowsPerBlock * 10 + k));
            }
            printf("\nremainders:\t");
            for(int k = 0; k < numLeft; k++){
                printf("%u\t", getRemainderGPU(cqf, i, rowsPerBlock * 10 + k));
            }
            printf("\n");
        }
        printf("\n --------------------------------------------------------------------- \n");
    }
	cqf->blocks = d_filterBlocks;
}

__device__ __host__ unsigned int Normal_APHashGPU(unsigned int value, unsigned int maxHashValue)
{
    unsigned char p[4];
    p[0] = (value >> 24) & 0xFF;
    p[1] = (value >> 16) & 0xFF;
    p[2] = (value >> 8) & 0xFF;
    p[3] = value & 0xFF; 

    unsigned int hash = 0xAAAAAAAA;

    for (int i = 0; i < 4; i++){
        hash ^= ((i & 1) == 0) ? ((hash << 7) ^ p[i] ^ (hash >> 3)) : (~((hash << 11) ^ p[i] ^ (hash >> 5)));
    }     

    return hash % maxHashValue;
}

__device__ unsigned int rankBitGPU(long long unsigned int bitArray, unsigned int index)
{
    unsigned int rank = __popcll(bitArray & LOW_BIT_MASKLL(index + 1));
    return rank;
}

__device__ unsigned int selectBitGPU_old(long long unsigned int bitArray, unsigned int rank)
{
    //using iterative method for first basic implementation
    if(rank == 0){
        return 0;
    }

    unsigned int nextOne = 0;
    for(int i = 1; i <= rank; i++){
        if(bitArray == 0) return UINT_MAX;  //if runEnd is in next block
        nextOne = __ffsll(bitArray);
        bitArray &= ~LOW_BIT_MASKLL(nextOne);
    }

    return nextOne - 1;
}

__device__ const unsigned char kSelectInByte[2048] = {
        8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0,
        1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0,
        2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
        1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0,
        3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0,
        1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
        2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0,
        1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
        4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0,
        1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 8, 8, 8, 1,
        8, 2, 2, 1, 8, 3, 3, 1, 3, 2, 2, 1, 8, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2,
        2, 1, 8, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1,
        4, 3, 3, 1, 3, 2, 2, 1, 8, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4,
        4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1,
        3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 8, 7, 7, 1, 7, 2,
        2, 1, 7, 3, 3, 1, 3, 2, 2, 1, 7, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
        7, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3,
        3, 1, 3, 2, 2, 1, 7, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4, 4, 1,
        4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2,
        2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 8, 8, 8, 8, 8, 8, 8, 2,
        8, 8, 8, 3, 8, 3, 3, 2, 8, 8, 8, 4, 8, 4, 4, 2, 8, 4, 4, 3, 4, 3, 3, 2, 8, 8,
        8, 5, 8, 5, 5, 2, 8, 5, 5, 3, 5, 3, 3, 2, 8, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3,
        4, 3, 3, 2, 8, 8, 8, 6, 8, 6, 6, 2, 8, 6, 6, 3, 6, 3, 3, 2, 8, 6, 6, 4, 6, 4,
        4, 2, 6, 4, 4, 3, 4, 3, 3, 2, 8, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2,
        6, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2, 8, 8, 8, 7, 8, 7, 7, 2, 8, 7,
        7, 3, 7, 3, 3, 2, 8, 7, 7, 4, 7, 4, 4, 2, 7, 4, 4, 3, 4, 3, 3, 2, 8, 7, 7, 5,
        7, 5, 5, 2, 7, 5, 5, 3, 5, 3, 3, 2, 7, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3,
        3, 2, 8, 7, 7, 6, 7, 6, 6, 2, 7, 6, 6, 3, 6, 3, 3, 2, 7, 6, 6, 4, 6, 4, 4, 2,
        6, 4, 4, 3, 4, 3, 3, 2, 7, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2, 6, 5,
        5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 4, 8, 4, 4, 3, 8, 8, 8, 8, 8, 8,
        8, 5, 8, 8, 8, 5, 8, 5, 5, 3, 8, 8, 8, 5, 8, 5, 5, 4, 8, 5, 5, 4, 5, 4, 4, 3,
        8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 3, 8, 8, 8, 6, 8, 6, 6, 4, 8, 6,
        6, 4, 6, 4, 4, 3, 8, 8, 8, 6, 8, 6, 6, 5, 8, 6, 6, 5, 6, 5, 5, 3, 8, 6, 6, 5,
        6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7,
        7, 3, 8, 8, 8, 7, 8, 7, 7, 4, 8, 7, 7, 4, 7, 4, 4, 3, 8, 8, 8, 7, 8, 7, 7, 5,
        8, 7, 7, 5, 7, 5, 5, 3, 8, 7, 7, 5, 7, 5, 5, 4, 7, 5, 5, 4, 5, 4, 4, 3, 8, 8,
        8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 3, 8, 7, 7, 6, 7, 6, 6, 4, 7, 6, 6, 4,
        6, 4, 4, 3, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 3, 7, 6, 6, 5, 6, 5,
        5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 5, 8, 5, 5, 4, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6,
        6, 4, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 5, 8, 8, 8, 6, 8, 6, 6, 5,
        8, 6, 6, 5, 6, 5, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8,
        8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 4, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7,
        8, 7, 7, 5, 8, 8, 8, 7, 8, 7, 7, 5, 8, 7, 7, 5, 7, 5, 5, 4, 8, 8, 8, 8, 8, 8,
        8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 4,
        8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 5, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6,
        6, 5, 6, 5, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6,
        8, 6, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7,
        8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 8,
        8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6,
        6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7
};

__device__ unsigned int selectBitGPU(long long unsigned int x, unsigned int k)
{
    if(k == 0)return 0;

    k--;
    if (k >= __popcll(x)) { return UINT_MAX; }

    const long long unsigned int kOnesStep4  = 0x1111111111111111ULL;
    const long long unsigned int kOnesStep8  = 0x0101010101010101ULL;
    const long long unsigned int kMSBsStep8  = 0x80ULL * kOnesStep8;

    long long unsigned int s = x;
    s = s - ((s & 0xA * kOnesStep4) >> 1);
    s = (s & 0x3 * kOnesStep4) + ((s >> 2) & 0x3 * kOnesStep4);
    s = (s + (s >> 4)) & 0xF * kOnesStep8;
    long long unsigned int byteSums = s * kOnesStep8;

    long long unsigned int kStep8 = k * kOnesStep8;
    long long unsigned int geqKStep8 = (((kStep8 | kMSBsStep8) - byteSums) & kMSBsStep8);
    long long unsigned int place = __popcll(geqKStep8) * 8;
    long long unsigned int byteRank = k - (((byteSums << 8) >> place) & (long long unsigned int)(0xFF));
    return place + kSelectInByte[((x >> place) & 0xFF) | (byteRank << 8)];
}

__global__ void lookupGPU(int numItems, struct countingQuotientFilterGPU cqf, unsigned int* hashValues, int* slotValues)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x +blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    //compute hash value
    unsigned int q = cqf.qbits;
    unsigned int hashValue = hashValues[idx];

    //separate into quotient and remainder
    unsigned int fq = (hashValue >> RBITS) & LOW_BIT_MASK(q);
    unsigned int fr = hashValue & LOW_BIT_MASK(RBITS);

    unsigned int blockNum = findBlockNumberGPU(fq);
    unsigned int slotNum = findPositionInBlockGPU(fq);
    //check occupied bit
    if(!isOccupiedGPU(cqf.blocks[blockNum].occupieds, slotNum)){
        slotValues[idx] = -1;
        return; 
    }   

    //find rank of quotient slot
    unsigned char blockOffset = cqf.blocks[blockNum].offset;
    unsigned int rank = rankBitGPU(cqf.blocks[blockNum].occupieds, slotNum);

    //select slot with runEnd rank = quotient rank
    //mask off the runEnds for any runs in blocks i-1 or earlier
    unsigned int endOfRun = selectBitGPU((cqf.blocks[blockNum].runEnds & ~LOW_BIT_MASKLL(blockOffset)), rank);

    //if end of run is in next block
    while(endOfRun == UINT_MAX){
        rank -= __popcll(cqf.blocks[blockNum].runEnds & ~LOW_BIT_MASKLL(blockOffset));
        if(blockOffset - SLOTS_PER_BLOCK > 0){
            blockOffset = blockOffset - SLOTS_PER_BLOCK;
        }
        else{
            blockOffset = 0;
        }
        blockNum++;
        if(blockNum > cqf.numBlocks){
            slotValues[idx] = -1;
            return;
	}
        //select on remaining rank
        endOfRun = selectBitGPU((cqf.blocks[blockNum].runEnds & ~LOW_BIT_MASKLL(blockOffset)), rank);
    }

    //endOfRun now points to runEnd for correct quotient
    //search backwards through run
    //end search if: we reach another set runEnd bit; we find the remainder; we reach canonical slot
    unsigned int currentRemainder = getRemainderGPU(&cqf, blockNum, endOfRun);
//    printf("remainder in last run slot: %u\n", currentRemainder);
    unsigned int currentSlot = endOfRun;
    do{
        if(currentRemainder == fr){
            //return index of slot where remainder is stored
	    slotValues[idx] = globalSlotIndexGPU(blockNum, currentSlot);
	    return;
        }
        if(currentRemainder < fr){
	    slotValues[idx] = -1;
            return;
        }
        if(currentSlot > 0){
            currentSlot--;
        }
        else{
            currentSlot = SLOTS_PER_BLOCK - 1;
            if(blockNum == 0){
	        slotValues[idx] = -1;
		return;
	    }
            blockNum--;
        }
        currentRemainder = getRemainderGPU(&cqf, blockNum, currentSlot);
    }while(!isRunEndGPU(cqf.blocks[blockNum].runEnds, currentSlot) && (globalSlotIndexGPU(blockNum, currentSlot) >= fq));

    slotValues[idx] = -1;
    return;
}

__global__ void hashInputs(int numItems, struct countingQuotientFilterGPU cqf, unsigned int* insertValues, unsigned int* fingerprints)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    //hash values to get fingerprints
    unsigned int hashValue = Normal_APHashGPU(insertValues[idx], (1 << (cqf.qbits + RBITS)));
    fingerprints[idx] = hashValue;
}

__host__ float launchLookups(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_lookupValues, int* d_slotValuesArray)
{
    thrust::device_vector<unsigned int> d_hashValues(numValues);
    thrust::fill(d_hashValues.begin(), d_hashValues.end(), 0); 
    unsigned int* d_hashValuesArray = thrust::raw_pointer_cast(&d_hashValues[0]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //Hash items
    int numBlocks = (numValues + 1023) / 1024; 
    dim3 hashBlockDims((numBlocks + 31) / 32, 32); 
    hashInputs<<<hashBlockDims, 1024>>>(numValues, cqf, d_lookupValues, d_hashValuesArray);//was 128

    //Create index array to track inputs -> outputs
    thrust::device_vector<unsigned int> d_indices(numValues);
    thrust::fill(d_indices.begin(), d_indices.end(), 1);
    thrust::exclusive_scan(d_indices.begin(), d_indices.end(), d_indices.begin(), 0);

    //Sort by fingerprint
    thrust::sort_by_key(d_hashValues.begin(), d_hashValues.end(), d_indices.begin());

    //Launch lookup kernel
    numBlocks = (numValues + 511) / 512;
    dim3 blockDims((numBlocks + 31) / 32, 32);
    lookupGPU<<<blockDims, 512>>>(numValues, cqf, d_hashValuesArray, d_slotValuesArray); //was 1024

    //Sort outputs
    thrust::device_ptr<int> d_slotValues(d_slotValuesArray);
    thrust::sort_by_key(d_indices.begin(), d_indices.end(), d_slotValues);

    cudaEventRecord(stop);
    //Calculate timing results
    cudaEventSynchronize(stop);
    float lookupTime = 0;
    cudaEventElapsedTime(&lookupTime, start, stop);

    //Free Memory
    d_hashValues.~device_vector<unsigned int>();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return lookupTime;
}


__global__ void hashAndLookupGPU(int numItems, struct countingQuotientFilterGPU cqf, unsigned int* lookupValues, int* slotValues)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x +blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    //compute hash value
    unsigned int q = cqf.qbits;
    unsigned int hashValue = Normal_APHashGPU(lookupValues[idx], (1 << (q + RBITS)));

    //separate into quotient and remainder
    unsigned int fq = (hashValue >> RBITS) & LOW_BIT_MASK(q);
    unsigned int fr = hashValue & LOW_BIT_MASK(RBITS);

    unsigned int blockNum = findBlockNumberGPU(fq);
    unsigned int slotNum = findPositionInBlockGPU(fq);
    //check occupied bit
    if(!isOccupiedGPU(cqf.blocks[blockNum].occupieds, slotNum)){
        slotValues[idx] = -1;
        return; 
    }   

    //find rank of quotient slot
    unsigned char blockOffset = cqf.blocks[blockNum].offset;
    unsigned int rank = rankBitGPU(cqf.blocks[blockNum].occupieds, slotNum);

    //select slot with runEnd rank = quotient rank
    //mask off the runEnds for any runs in blocks i-1 or earlier
    unsigned int endOfRun = selectBitGPU((cqf.blocks[blockNum].runEnds & ~LOW_BIT_MASKLL(blockOffset)), rank);

    //if end of run is in next block
    while(endOfRun == UINT_MAX){
        rank -= __popcll(cqf.blocks[blockNum].runEnds & ~LOW_BIT_MASKLL(blockOffset));
        if(blockOffset - SLOTS_PER_BLOCK > 0){
            blockOffset = blockOffset - SLOTS_PER_BLOCK;
        }
        else{
            blockOffset = 0;
        }
        blockNum++;
        if(blockNum > cqf.numBlocks){
            slotValues[idx] = -1;
            return;
	}
        //select on remaining rank
        endOfRun = selectBitGPU((cqf.blocks[blockNum].runEnds & ~LOW_BIT_MASKLL(blockOffset)), rank);
    }

    //endOfRun now points to runEnd for correct quotient
    //search backwards through run
    //end search if: we reach another set runEnd bit; we find the remainder; we reach canonical slot
    unsigned int currentRemainder = getRemainderGPU(&cqf, blockNum, endOfRun);
//    printf("remainder in last run slot: %u\n", currentRemainder);
    unsigned int currentSlot = endOfRun;
    do{
        if(currentRemainder == fr){
            //return index of slot where remainder is stored
	    slotValues[idx] = globalSlotIndexGPU(blockNum, currentSlot);
	    return;
        }
        if(currentRemainder < fr){
	    slotValues[idx] = -1;
            return;
        }
        if(currentSlot > 0){
            currentSlot--;
        }
        else{
            currentSlot = SLOTS_PER_BLOCK - 1;
            if(blockNum == 0){
	        slotValues[idx] = -1;
		return;
	    }
            blockNum--;
        }
        currentRemainder = getRemainderGPU(&cqf, blockNum, currentSlot);
    }while(!isRunEndGPU(cqf.blocks[blockNum].runEnds, currentSlot) && (globalSlotIndexGPU(blockNum, currentSlot) >= fq));

    slotValues[idx] = -1;
    return;
}

__host__ float launchUnsortedLookups(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_lookupValues, int* d_slotValuesArray)
{

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //Launch lookup kernel
    int numBlocks = (numValues + 511) / 512;
    dim3 blockDims((numBlocks + 31) / 32, 32);
    hashAndLookupGPU<<<blockDims, 512>>>(numValues, cqf, d_lookupValues, d_slotValuesArray); //was 1024

    cudaEventRecord(stop);
    //Calculate timing results
    cudaEventSynchronize(stop);
    float lookupTime = 0;
    cudaEventElapsedTime(&lookupTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return lookupTime;
}

__device__ unsigned int findFirstUnusedSlotGPU(struct countingQuotientFilterGPU* cqf, int* blockNum, unsigned int currentSlot)
{
    long long unsigned int occupieds = cqf->blocks[*blockNum].occupieds;
    long long unsigned int runEnds = cqf->blocks[*blockNum].runEnds;
    unsigned char offset = cqf->blocks[*blockNum].offset;
    unsigned int rank = rankBitGPU(occupieds, currentSlot);
    unsigned int select = selectBitGPU((runEnds & ~LOW_BIT_MASKLL(offset)), rank);
    if(rank == 0){ 
        select = offset;
    }   
    while(currentSlot <= select){
        if(select == UINT_MAX || select == SLOTS_PER_BLOCK - 1){ 
            (*blockNum)++;
            if(*blockNum > cqf->numBlocks) return UINT_MAX;
            occupieds = cqf->blocks[*blockNum].occupieds;
            runEnds = cqf->blocks[*blockNum].runEnds;
            offset = cqf->blocks[*blockNum].offset;
            select = offset - 1;    //want currentSlot to be first slot after offset values
        }
        currentSlot = select + 1;
        rank = rankBitGPU(occupieds, currentSlot);
        select = selectBitGPU((runEnds & ~LOW_BIT_MASKLL(offset)), rank);
    }

    return currentSlot;
}

__global__ void quotienting(int numItems, unsigned int qbits, unsigned int* quotients, unsigned int* remainders)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    //return quotients and remainders
    unsigned int hashValue = quotients[idx]; //quotients array initially stores the fingerprint values
    quotients[idx] = (hashValue >> RBITS) & LOW_BIT_MASK(qbits);
    remainders[idx] = hashValue & LOW_BIT_MASK(RBITS);
}

__global__ void findBlockStartIndices(int numItems, unsigned int* quotients, unsigned int* blockStarts)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    unsigned int blockNumber = findBlockNumberGPU(quotients[idx]);

    if(idx == 0){
        blockStarts[blockNumber] = 0;
        return;
    }

    unsigned int previousItemBlock = findBlockNumberGPU(quotients[idx - 1]);
    if(blockNumber != previousItemBlock){
        blockStarts[blockNumber] = idx;
    }
}

__device__ void incrementQueuePointer(unsigned int nextValue, unsigned int* blockInsertQueues, int blockNum, unsigned int lastRegionBlock, unsigned int* blockStarts, int numBlocks, bool* itemsLeft)
{
    int nextBlockNum = blockNum + 1;
    unsigned int nextBlockStart = blockStarts[nextBlockNum];
    while(nextBlockStart == UINT_MAX && nextBlockNum < (numBlocks - 1)){
        nextBlockNum++;
        nextBlockStart = blockStarts[nextBlockNum];
    }
    if(nextValue + 1 < nextBlockStart){
        blockInsertQueues[blockNum]++;
        itemsLeft[0] = true;
    }
    else{
        blockInsertQueues[blockNum] = UINT_MAX;
        while(blockNum < lastRegionBlock){
            blockNum++;
            if(blockInsertQueues[blockNum] != UINT_MAX){
                itemsLeft[0] = true;
                return;
            }
        }
    }
}

__global__ void insertIntoRegions(int numRegions, int numBlocksPerRegion, int numItems, struct countingQuotientFilterGPU cqf, unsigned int* blockStarts, unsigned int* nextItems, unsigned int* quotients, unsigned int* remainders, int* finalSlotValues, bool* itemsLeft)
{
//TODO: reduce resources used

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x +blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numRegions) return;

    //find next item to insert for region
    //find start block for region
    unsigned int firstRegionBlock = idx * numBlocksPerRegion;
    unsigned int lastRegionBlock = (idx * numBlocksPerRegion) + numBlocksPerRegion - 1;
    if(lastRegionBlock >= cqf.numBlocks) lastRegionBlock = cqf.numBlocks - 1;
    unsigned int nextValue = nextItems[firstRegionBlock];
    int blockNum = firstRegionBlock;
    while(nextValue == UINT_MAX && blockNum < lastRegionBlock){
        blockNum++;    
        nextValue = nextItems[blockNum];
    }
    if(nextValue >= numItems) return;

//    printf("index of item to be inserted=%u\n", nextValue);
    unsigned int fq = quotients[nextValue];
    unsigned int fr = remainders[nextValue];

    int homeBlockNum = blockNum;
    unsigned int homeSlotNum = findPositionInBlockGPU(fq);
//    printf("quotient: %u\tslot:%u\tremainder: %u\n", fq, homeSlotNum, fr);
    long long unsigned int occupieds = cqf.blocks[blockNum].occupieds;
//    printf("blockNum = %u\t lastRegionBlock=%u\n", blockNum, lastRegionBlock);
//    printf("homeSlotNum = %u\n", homeSlotNum);
    //check occupied bit
    bool occupiedBit = isOccupiedGPU(occupieds, homeSlotNum);
//    printf("occupied? %u\n", (unsigned int)occupiedBit);

    //find rank of quotient slot
    unsigned char blockOffset = cqf.blocks[blockNum].offset;
//    printf("offset = %u\n", blockOffset);
    unsigned int rank = rankBitGPU(occupieds, homeSlotNum);
//    printf("rank = %u\n", rank);

    //select slot with runEnd rank = quotient rank
    //mask off the runEnds for any runs in blocks i-1 or earlier
    long long unsigned int runEnds = cqf.blocks[blockNum].runEnds;
    //printf("runEnds = %llu\n", runEnds);
    unsigned int endOfRun = selectBitGPU((runEnds & ~LOW_BIT_MASKLL(blockOffset)), rank);
    if(rank == 0){
        if(blockOffset == 0){
            endOfRun = 0;
        }
        else{
            endOfRun = blockOffset - 1;
        }
    }
//    printf("select(rank) = %u\n", endOfRun);

    //if end of run is in next block
    while(endOfRun == UINT_MAX){
        rank -= __popcll(cqf.blocks[blockNum].runEnds & ~LOW_BIT_MASKLL(blockOffset));
        if(blockOffset - SLOTS_PER_BLOCK > 0){
            blockOffset = blockOffset - SLOTS_PER_BLOCK;
        }
        else{
            blockOffset = 0;
        }
        blockNum++;
        //select on remaining rank
        endOfRun = selectBitGPU((cqf.blocks[blockNum].runEnds & ~LOW_BIT_MASKLL(blockOffset)), rank);
    }
    //TODO: block num check during or after loop?
    if(blockNum > lastRegionBlock){
        //the insert will affect the next region
        itemsLeft[0] = true;
        return;
    }

   //endOfRun now points to runEnd for correct quotient
    //if select returns location earlier than fq, slot is empty and we can insert the item there
    //(also if there are no occupied slots at all in the start block)
    if(globalSlotIndexGPU(blockNum, endOfRun) < globalSlotIndexGPU(homeBlockNum, homeSlotNum) | blockOffset + rank == 0){
        cqf.blocks[homeBlockNum].runEnds = setRunEndGPU(runEnds, homeSlotNum);
        setRemainderGPU(&cqf, homeBlockNum, homeSlotNum, fr);
        cqf.blocks[homeBlockNum].occupieds = setOccupiedGPU(occupieds, homeSlotNum);
        finalSlotValues[nextValue] = globalSlotIndexGPU(homeBlockNum, homeSlotNum);
        //move pointer to next item in queue
        incrementQueuePointer(nextValue, nextItems, homeBlockNum, lastRegionBlock,  blockStarts, cqf.numBlocks, itemsLeft);
        return;
    }

    //if slot is not empty, search through the filter for the first empty slot
    else{
        endOfRun++;
        if(endOfRun == SLOTS_PER_BLOCK){
            endOfRun = 0;
            blockNum++;
            if(blockNum > lastRegionBlock){
                //the insert will affect the next region
                itemsLeft[0] = true;
                return;
            }
            if(blockNum > cqf.numBlocks){     //insert fails
		finalSlotValues[nextValue] = -1;
                incrementQueuePointer(nextValue, nextItems, homeBlockNum, lastRegionBlock,  blockStarts, cqf.numBlocks, itemsLeft);
                return;
            }
        }
        unsigned int runEndBlock = blockNum;
        unsigned int unusedSlot = findFirstUnusedSlotGPU(&cqf, &blockNum, endOfRun);
        if(unusedSlot == UINT_MAX){     //insert fails
	    finalSlotValues[nextValue] = -1;
            incrementQueuePointer(nextValue, nextItems, homeBlockNum, lastRegionBlock,  blockStarts, cqf.numBlocks, itemsLeft);
	    return;
	}
        if(blockNum > lastRegionBlock){
            //the insert will affect the next region
            itemsLeft[0] = true;
            return;
        }
        if(blockNum > homeBlockNum){
            for(int i = 0; i < blockNum - homeBlockNum; i++){
                cqf.blocks[blockNum - i].offset++;
            }
        }
//        printf("unused slot idx = %u\n", unusedSlot);
//        printf("usused slot block = %u\n", blockNum);
//        printf("canonical end of run (block, slot) = (%u, %u)\n", runEndBlock, endOfRun);
        //move items over until we get back to the run the item belongs in
        while(globalSlotIndexGPU(blockNum, unusedSlot) > globalSlotIndexGPU(runEndBlock, endOfRun)){
//            printf("next slot: %u\n", unusedSlot);
            if(unusedSlot == 0){
                int nextBlock = blockNum - 1;
                unsigned int nextSlot = SLOTS_PER_BLOCK - 1;
                setRemainderGPU(&cqf, blockNum, unusedSlot, getRemainderGPU(&cqf, nextBlock, nextSlot));
                if(isRunEndGPU(cqf.blocks[nextBlock].runEnds, nextSlot)){
                    cqf.blocks[blockNum].runEnds = setRunEndGPU(cqf.blocks[blockNum].runEnds, unusedSlot);
                }
                else{
                    cqf.blocks[blockNum].runEnds = clearRunEndGPU(cqf.blocks[blockNum].runEnds, unusedSlot);
                }
                unusedSlot = SLOTS_PER_BLOCK - 1;
                blockNum--;
            }
            else{
                setRemainderGPU(&cqf, blockNum, unusedSlot, getRemainderGPU(&cqf, blockNum, (unusedSlot - 1)));
                if(isRunEndGPU(cqf.blocks[blockNum].runEnds, (unusedSlot - 1))){
                    cqf.blocks[blockNum].runEnds = setRunEndGPU(cqf.blocks[blockNum].runEnds, unusedSlot);
                }
                else{
                    cqf.blocks[blockNum].runEnds = clearRunEndGPU(cqf.blocks[blockNum].runEnds, unusedSlot);
                }
                unusedSlot--;
            }
        }

        //if the home slot was not previously occupied, then new item is its run
        if(!isOccupiedGPU(cqf.blocks[homeBlockNum].occupieds, homeSlotNum)){
            setRemainderGPU(&cqf, blockNum, unusedSlot, fr);
            cqf.blocks[blockNum].runEnds = setRunEndGPU(cqf.blocks[blockNum].runEnds, unusedSlot);
            cqf.blocks[homeBlockNum].occupieds = setOccupiedGPU(cqf.blocks[homeBlockNum].occupieds, homeSlotNum);
	    finalSlotValues[nextValue] = globalSlotIndexGPU(blockNum, unusedSlot);
            incrementQueuePointer(nextValue, nextItems, homeBlockNum, lastRegionBlock,  blockStarts, cqf.numBlocks, itemsLeft);
	    return;
        }
        //if home slot already has a run, put new item in correct sequential location
        else{
            //move run end over by one slot
            unsigned int nextSlot = unusedSlot - 1;
            int nextBlock = blockNum;
            if(unusedSlot == 0){
                nextSlot = SLOTS_PER_BLOCK - 1;
                nextBlock = blockNum - 1;
                if(nextBlock < 0){      //insert fails
	            finalSlotValues[nextValue] = -1;
                    incrementQueuePointer(nextValue, nextItems, homeBlockNum, lastRegionBlock,  blockStarts, cqf.numBlocks, itemsLeft);
		    return;
		}
            }
//            printf("nextSlot: %u\tnextBlock:%u\n", nextSlot, nextBlock);
//            printf("unusedSlot: %u\tblockNum: %u\n", unusedSlot, blockNum);
            cqf.blocks[blockNum].runEnds = setRunEndGPU(cqf.blocks[blockNum].runEnds, unusedSlot);
            cqf.blocks[nextBlock].runEnds = clearRunEndGPU(cqf.blocks[nextBlock].runEnds, nextSlot);
            //search backwards through run
            //end search if: we reach another set runEnd bit; we find remainder <= new remainder; we reach canonical slot
            unsigned int nextRemainder = getRemainderGPU(&cqf, nextBlock, nextSlot);
//            printf("remainder in last run slot: %u\n", nextRemainder);
            do{
                if(nextRemainder <= fr){
//                    printf("setting remainder in block %u, slot %u.\n", blockNum, unusedSlot); 
                    setRemainderGPU(&cqf, blockNum, unusedSlot, fr);
                    //this stores duplicates
                    //return index of slot where remainder is stored
                    finalSlotValues[nextValue] = globalSlotIndexGPU(blockNum, unusedSlot);
                    incrementQueuePointer(nextValue, nextItems, homeBlockNum, lastRegionBlock,  blockStarts, cqf.numBlocks, itemsLeft);
		    return;
                }
                setRemainderGPU(&cqf, blockNum, unusedSlot, nextRemainder);
                if(unusedSlot > 0){
                    unusedSlot--;
                    if(unusedSlot == 0){
                        if(blockNum == 0){
                            setRemainderGPU(&cqf, blockNum, unusedSlot, fr);
                            finalSlotValues[nextValue] = globalSlotIndexGPU(blockNum, unusedSlot);
                            incrementQueuePointer(nextValue, nextItems, homeBlockNum, lastRegionBlock,  blockStarts, cqf.numBlocks, itemsLeft);
                            return;
                        }
                        nextSlot = SLOTS_PER_BLOCK - 1;
                        nextBlock--;
                    }
                    else{
                        nextSlot = unusedSlot - 1;
                    }
                }
                else{
                    unusedSlot = SLOTS_PER_BLOCK - 1;
                    blockNum--;
                    if(blockNum < 0){   //insert fails
		        finalSlotValues[nextValue] = -1;
                        incrementQueuePointer(nextValue, nextItems, homeBlockNum, lastRegionBlock,  blockStarts, cqf.numBlocks, itemsLeft);
			return;
		    }
                    nextSlot = unusedSlot - 1;
                }
                nextRemainder = getRemainderGPU(&cqf, nextBlock, nextSlot);
            }while(!isRunEndGPU(cqf.blocks[nextBlock].runEnds, nextSlot) && (globalSlotIndexGPU(nextBlock, nextSlot) >= fq));
            //unusedSlot is now head of run. Insert the remainder there.
            setRemainderGPU(&cqf, blockNum, unusedSlot, fr);
	    finalSlotValues[nextValue] = globalSlotIndexGPU(blockNum, unusedSlot);
            incrementQueuePointer(nextValue, nextItems, homeBlockNum, lastRegionBlock,  blockStarts, cqf.numBlocks, itemsLeft);
	    return;
        }
    }
}

//Some possible versions:
// 1. Sort items and divide them up to be inserted into groups of blocks. Groups grow with more iterations.
// 2. Items bid to be inserted into groups of blocks. Flag items that succeed insert and compact them out.
//TODO: Groupings of blocks could also have the start of the group change indices, rather than just changing group sizes. This would keep the number of threads high, while still avoiding too many reptitions on insert failures for less full filters.

__host__ float insertGPU(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_insertValues, int* d_returnValues)
{
    //Allocate memory
    thrust::device_vector<unsigned int> d_quotients(numValues);
    thrust::fill(d_quotients.begin(), d_quotients.end(), 0);
    unsigned int* d_quotientsArray = thrust::raw_pointer_cast(&d_quotients[0]);
    unsigned int* d_remaindersArray;
    cudaMalloc((void**) &d_remaindersArray, numValues * sizeof(unsigned int));
    cudaMemset(d_remaindersArray, 0, numValues * sizeof(unsigned int));
    unsigned int* d_blockStarts;
    cudaMalloc((void**) &d_blockStarts, cqf.numBlocks * sizeof(unsigned int));
    cudaMemset(d_blockStarts, 0xFF, cqf.numBlocks * sizeof(unsigned int));
    unsigned int* d_nextItems;
    cudaMalloc((void**) &d_nextItems, cqf.numBlocks * sizeof(unsigned int));
    cudaMemset(d_nextItems, 0xFF, cqf.numBlocks * sizeof(unsigned int));

    bool* h_itemsLeft = new bool[1];
    h_itemsLeft[0] = 1;
    bool* d_itemsLeft;
    cudaMalloc((void**) &d_itemsLeft, sizeof(bool));
    cudaMemset(d_itemsLeft, 0, sizeof(bool));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //Hash items
    hashInputs<<<(numValues + 1023)/1024, 1024>>>(numValues, cqf, d_insertValues, d_quotientsArray);

    //Sort by fingerprint
    thrust::sort(d_quotients.begin(), d_quotients.end());
    
    //Split fingerprints into quotients and remainders
    quotienting<<<(numValues + 1023)/1024, 1024>>>(numValues, cqf.qbits, d_quotientsArray, d_remaindersArray);

    //Compute block ID & write to region start array if first item in region
    findBlockStartIndices<<<(numValues + 1023)/1024, 1024>>>(numValues, d_quotientsArray, d_blockStarts);
    cudaMemcpy(d_nextItems, d_blockStarts, cqf.numBlocks * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
   
/*    unsigned int* h_printBlockHolder = new unsigned int[cqf.numBlocks];
    cudaMemcpy(h_printBlockHolder, d_blockStarts, cqf.numBlocks * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("blockStarts after kernel:\n");
    for(int i = 0; i < cqf.numBlocks; i++){
        printf("%i\t", h_printBlockHolder[i]);
    }
    printf("\n");

    unsigned int* h_quotientsArray = new unsigned int[numValues];
    cudaMemcpy(h_quotientsArray, d_quotientsArray, numValues * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* h_remaindersArray = new unsigned int[numValues];
    cudaMemcpy(h_remaindersArray, d_remaindersArray, numValues * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    int numBlockItems = 0;
    printf("numBlocks=%u\n", cqf.numBlocks);
    for(int i = 0; i < cqf.numBlocks - 1; i++){
        printf("\n***Block %i***:\n", i);
        if(i == cqf.numBlocks - 2) numBlockItems = numValues - h_printBlockHolder[i];
        else{
            numBlockItems = h_printBlockHolder[i+1] - h_printBlockHolder[i];
        }
        if(h_printBlockHolder[i] == UINT_MAX) numBlockItems = 0;
        for(int j = 0; j < numBlockItems; j++){
            printf("quotient: %u\t remainder: %u\n", h_quotientsArray[h_printBlockHolder[i] + j], h_remaindersArray[h_printBlockHolder[i] + j]);
        }
    }
*/
    //Loop over insert kernel
    //If insert overflows, then next iteration has same region size, thread should return (all later items in the region will also overflow, since they are sorted)
    int numIterations = 0;
    int blocksPerRegion = 1;
    int numRegions = cqf.numBlocks;
    while(h_itemsLeft[0] == 1){
//        printf("--------------------\niteration #: %i\n", numIterations);

        numRegions = (cqf.numBlocks + blocksPerRegion - 1) / blocksPerRegion;

        //Launch insert kernel with one thread per insert region
        insertIntoRegions<<<(numRegions + 127)/128, 128>>>(numRegions, blocksPerRegion, numValues, cqf, d_blockStarts, d_nextItems, d_quotientsArray, d_remaindersArray, d_returnValues, d_itemsLeft);

        cudaMemcpy(h_itemsLeft, d_itemsLeft, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemset(d_itemsLeft, 0, sizeof(bool));

        numIterations++;
        blocksPerRegion = numIterations / 16 + 1;

//        printGPUFilter(&cqf);
    }

    cudaEventRecord(stop);

    //Calculate and print timing results
    cudaEventSynchronize(stop);
    float insertTime = 0;
    cudaEventElapsedTime(&insertTime, start, stop);

//    printf("total iterations: %i\n", numIterations);

    //Free memory
    d_quotients.~device_vector<unsigned int>();
    cudaFree(d_remaindersArray);
    cudaFree(d_blockStarts);
    cudaFree(d_nextItems);
    delete[] h_itemsLeft;
    cudaFree(d_itemsLeft);

    return insertTime;
}
