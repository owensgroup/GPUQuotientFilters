//RSQF.cpp
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
#include <string.h>
#include <limits.h>

#include "RSQF.h"

#define LOW_BIT_MASK(n) ((1U << n) - 1U)
#define LOW_BIT_MASKLL(n) (n >= 64 ? 0xFFFFFFFFFFFFFFFF : (1ULL << n) - 1ULL)

void initCQF(struct countingQuotientFilter* cqf, unsigned int q)
{
    cqf->qbits = q;
    size_t numBlocks = calcNumBlocks(q);
    cqf->blocks = (cqf_block*)calloc(numBlocks, sizeof(cqf_block));
    memset(cqf->blocks, 0, numBlocks * sizeof(cqf_block));
}

size_t calcNumBlocks(unsigned int q)
{
    return (((1 << q) + (SLOTS_PER_BLOCK - 1)) / SLOTS_PER_BLOCK) + 1;
}

bool isOccupied(long long unsigned int occupieds, unsigned int slotNumber)
{
   return (1ULL << slotNumber) & occupieds;
}

long long unsigned int setOccupied(long long unsigned int occupieds, unsigned int slotNumber)
{
    return (1ULL << slotNumber) | occupieds;
}

bool isRunEnd(long long unsigned int runends, unsigned int slotNumber)
{
   return (1ULL << slotNumber) & runends;
}

long long unsigned int setRunEnd(long long unsigned int runends, unsigned int slotNumber)
{
    return (1ULL << slotNumber) | runends;
}

long long unsigned int clearRunEnd(long long unsigned int runends, unsigned int slotNumber)
{
    return ~(1ULL << slotNumber) & runends;
}

unsigned int findBlockNumber(unsigned int globalSlotNumber)
{
    return globalSlotNumber / SLOTS_PER_BLOCK;
}

unsigned int findPositionInBlock(unsigned int globalSlotNumber)
{
    return globalSlotNumber % SLOTS_PER_BLOCK;
}

unsigned int findRemainderIntSlot(unsigned int blockPosition)
{
    return blockPosition * RBITS / 64;
}

unsigned int findRemainderStartBit(unsigned int blockPosition)
{
    return (blockPosition * RBITS) % 64;
}

unsigned int globalSlotIndex(unsigned int blockNum, unsigned int slotNumInBlock){
    return (slotNumInBlock + (blockNum * SLOTS_PER_BLOCK));
}

unsigned int getRemainder(struct countingQuotientFilter* cqf, unsigned int blockNum, unsigned int slotNum)
{
    unsigned int integerSlot = findRemainderIntSlot(slotNum);
    unsigned int startBit = findRemainderStartBit(slotNum);
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

void setRemainder(struct countingQuotientFilter* cqf, unsigned int blockNum, unsigned int slotNum, unsigned int value)
{
    unsigned int integerSlot = findRemainderIntSlot(slotNum);
    unsigned int startBit = findRemainderStartBit(slotNum);
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

void printFilter(struct countingQuotientFilter* cqf)
{
    printf("Filter contents:\n");
    int numBlocks = calcNumBlocks(cqf->qbits);
    for(int i = 0; i < numBlocks; i++){
        printf("block: %i\t", i);
        printf("offset: %u\n", cqf->blocks[i].offset);
        int rowsPerBlock = SLOTS_PER_BLOCK / 10;
        for(int j = 0; j < rowsPerBlock; j++){
            printf("\noccupieds:\t");
            for(int k = 0; k < 10; k++){
                printf("%u\t", isOccupied(cqf->blocks[i].occupieds, 10 * j + k));
            }
            printf("\nrunEnds:\t");
            for(int k = 0; k < 10; k++){
                printf("%u\t", isRunEnd(cqf->blocks[i].runEnds, 10 * j + k));
            }
            printf("\nremainders:\t");
            for(int k = 0; k < 10; k++){
                printf("%u\t", getRemainder(cqf, i, 10 * j + k));
            }
            printf("\n");
        }
        if(SLOTS_PER_BLOCK % 10 != 0){
            int numLeft = SLOTS_PER_BLOCK % 10;
            printf("\noccupieds:\t");
            for(int k = 0; k < numLeft; k++){
                printf("%u\t", isOccupied(cqf->blocks[i].occupieds, rowsPerBlock * 10 + k));
            }
            printf("\nrunEnds:\t");
            for(int k = 0; k < numLeft; k++){
                printf("%u\t", isRunEnd(cqf->blocks[i].runEnds, rowsPerBlock * 10 + k));
            }
            printf("\nremainders:\t");
            for(int k = 0; k < numLeft; k++){
                printf("%u\t", getRemainder(cqf, i, rowsPerBlock * 10 + k));
            }
            printf("\n");
        }
        printf("\n --------------------------------------------------------------------- \n");
    }
}

unsigned int Normal_APHash(unsigned int value, unsigned int maxHashValue)
{
    unsigned char p[4];
    p[0] = (value >> 24) & 0xFF;
    p[1] = (value >> 16) & 0xFF;
    p[2] = (value >> 8) & 0xFF;
    p[3] = value & 0xFF;

    unsigned int hash = 0xAAAAAAAA;

    for(int i = 0; i < 4; i++){
        hash ^= ((i & 1) == 0) ? ((hash << 7) ^ p[i] ^ (hash >> 3)) : (~((hash << 11) ^ p[i] ^ (hash >> 5)));
    }    

    return hash % maxHashValue;
}

unsigned int rankBit(long long unsigned int bitArray, unsigned int index)
{
    unsigned int rank = __builtin_popcountll(bitArray & LOW_BIT_MASKLL(index + 1));
    return rank;
}

unsigned int selectBit_old(long long unsigned int bitArray, unsigned int rank)
{
    //using iterative method for first try/basic implementation
    if(rank == 0){
        return 0;
    }
   
    unsigned int nextOne = 0;
    for(int i = 1; i <= rank; i++){
        if(bitArray == 0) return UINT_MAX;  //if runEnd is in next block
        nextOne = __builtin_ffsll(bitArray);
        bitArray &= ~LOW_BIT_MASKLL(nextOne);
    }

    return nextOne - 1;
}

const unsigned char kSelectInByte[2048] = {
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

unsigned int selectBit(long long unsigned int x, unsigned int k)
{
    if(k == 0)return 0;

    k--;
    if (k >= __builtin_popcountll(x)) { return UINT_MAX; }
        
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
    long long unsigned int place = __builtin_popcountll(geqKStep8) * 8;  
    long long unsigned int byteRank = k - (((byteSums << 8) >> place) & (long long unsigned int)(0xFF));
    return place + kSelectInByte[((x >> place) & 0xFF) | (byteRank << 8)]; 
}

int lookup(struct countingQuotientFilter* cqf, unsigned int value)
{
    //compute hash value
    unsigned int q = cqf->qbits;
    unsigned int hashValue = Normal_APHash(value, (1 << (q + RBITS)));

    //separate into quotient and remainder
    unsigned int fq = (hashValue >> RBITS) & LOW_BIT_MASK(q);
    unsigned int fr = hashValue & LOW_BIT_MASK(RBITS);

    unsigned int blockNum = findBlockNumber(fq);
    unsigned int slotNum = findPositionInBlock(fq);
    long long unsigned int occupieds = cqf->blocks[blockNum].occupieds;
    //check occupied bit
    if(!isOccupied(occupieds, slotNum)){
        return -1;
    }

    //find rank of quotient slot
    unsigned char blockOffset = cqf->blocks[blockNum].offset;
    unsigned int rank = rankBit(occupieds, slotNum);

    //select slot with runEnd rank = quotient rank
    //mask off the runEnds for any runs in blocks i-1 or earlier
    long long unsigned int runEnds = cqf->blocks[blockNum].runEnds;
    unsigned int endOfRun = selectBit((runEnds & ~LOW_BIT_MASKLL(blockOffset)), rank);

    //if end of run is in next block
    while(endOfRun == UINT_MAX){
        rank -= __builtin_popcountll(cqf->blocks[blockNum].runEnds & ~LOW_BIT_MASKLL(blockOffset));
        if(blockOffset - SLOTS_PER_BLOCK > 0){
            blockOffset = blockOffset - SLOTS_PER_BLOCK;
        }
        else{
            blockOffset = 0;
        }
        blockNum++;
        if(blockNum > calcNumBlocks(q)) return -1;
        //select on remaining rank
        endOfRun = selectBit((cqf->blocks[blockNum].runEnds & ~LOW_BIT_MASKLL(blockOffset)), rank);
    }

    //endOfRun now points to runEnd for correct quotient
    //search backwards through run
    //end search if: we reach another set runEnd bit; we find the remainder; we reach canonical slot
    unsigned int currentRemainder = getRemainder(cqf, blockNum, endOfRun);
    unsigned int currentSlot = endOfRun;
    do{
        if(currentRemainder == fr){
            //return index of slot where remainder is stored
            return globalSlotIndex(blockNum, currentSlot);
        }
        if(currentRemainder < fr){
            return -1;
        }
        if(currentSlot > 0){
            currentSlot--;
        }
        else{
            currentSlot = SLOTS_PER_BLOCK - 1;
            if(blockNum == 0) return -1;
            blockNum--;
        }
        currentRemainder = getRemainder(cqf, blockNum, currentSlot);
    }while(!isRunEnd(cqf->blocks[blockNum].runEnds, currentSlot) && (globalSlotIndex(blockNum, currentSlot) >= fq));

    return -1;
}

unsigned int findFirstUnusedSlot(struct countingQuotientFilter* cqf, int* blockNum, unsigned int startSlot)
{
    unsigned int currentSlot = startSlot;
    long long unsigned int occupieds = cqf->blocks[*blockNum].occupieds;
    long long unsigned int runEnds = cqf->blocks[*blockNum].runEnds;
    unsigned char offset = cqf->blocks[*blockNum].offset;
    unsigned int rank = rankBit(occupieds, startSlot);
    unsigned int select = selectBit((runEnds & ~LOW_BIT_MASKLL(offset)), rank);
    if(rank == 0){
        select = offset;
    }
    while(currentSlot <= select){
        if(select == UINT_MAX || select == SLOTS_PER_BLOCK - 1){
            (*blockNum)++;
            if(*blockNum > calcNumBlocks(cqf->qbits)) return UINT_MAX;
            occupieds = cqf->blocks[*blockNum].occupieds;
            runEnds = cqf->blocks[*blockNum].runEnds;
            offset = cqf->blocks[*blockNum].offset;
            select = offset - 1;    //want currentSlot to be first slot after offset values
        }
        currentSlot = select + 1;
        rank = rankBit(occupieds, currentSlot);
        select = selectBit((runEnds & ~LOW_BIT_MASKLL(offset)), rank);
    }

    return currentSlot;
}

int insert(struct countingQuotientFilter* cqf, unsigned int value)
{
    //compute hash value
    unsigned int q = cqf->qbits;
    unsigned int hashValue = Normal_APHash(value, (1 << (q + RBITS)));

    //separate into quotient and remainder
    unsigned int fq = (hashValue >> RBITS) & LOW_BIT_MASK(q);
    unsigned int fr = hashValue & LOW_BIT_MASK(RBITS);

    int homeBlockNum = findBlockNumber(fq);
    int blockNum = homeBlockNum;
    unsigned int homeSlotNum = findPositionInBlock(fq);
    long long unsigned int occupieds = cqf->blocks[blockNum].occupieds;

    //find rank of quotient slot
    unsigned char blockOffset = cqf->blocks[blockNum].offset;
    unsigned int rank = rankBit(occupieds, homeSlotNum);

    //select slot with runEnd rank = quotient rank
    //mask off the runEnds for any runs in blocks i-1 or earlier
    long long unsigned int runEnds = cqf->blocks[blockNum].runEnds;
    unsigned int endOfRun = selectBit((runEnds & ~LOW_BIT_MASKLL(blockOffset)), rank);
    if(rank == 0){
        if(blockOffset == 0){
            endOfRun = 0;
        }
        else{
            endOfRun = blockOffset - 1;
        }
    }

    //if end of run is in next block
    while(endOfRun == UINT_MAX){
        rank -= __builtin_popcountll(cqf->blocks[blockNum].runEnds & ~LOW_BIT_MASKLL(blockOffset));
        if(blockOffset - SLOTS_PER_BLOCK > 0){
            blockOffset = blockOffset - SLOTS_PER_BLOCK;
        }
        else{
            blockOffset = 0;
        }
        blockNum++;
        //select on remaining rank
        endOfRun = selectBit((cqf->blocks[blockNum].runEnds & ~LOW_BIT_MASKLL(blockOffset)), rank);
    }


    //endOfRun now points to runEnd for correct quotient
    //if select returns location earlier than fq, slot is empty and we can insert the item there
    //(also if there are no occupied slots at all in the start block)
    if(globalSlotIndex(blockNum, endOfRun) < globalSlotIndex(homeBlockNum, homeSlotNum) | blockOffset + rank == 0){
        cqf->blocks[homeBlockNum].runEnds = setRunEnd(runEnds, homeSlotNum);
        setRemainder(cqf, homeBlockNum, homeSlotNum, fr); 
        cqf->blocks[homeBlockNum].occupieds = setOccupied(occupieds, homeSlotNum);
        return globalSlotIndex(homeBlockNum, homeSlotNum);
    }

    //if slot is not empty, search through the filter for the first empty slot
    else{
        endOfRun++;
        if(endOfRun == SLOTS_PER_BLOCK){
            endOfRun = 0;
            blockNum++;
            if(blockNum > calcNumBlocks(cqf->qbits)) return -1;
        }
        unsigned int runEndBlock = blockNum;
        unsigned int unusedSlot = findFirstUnusedSlot(cqf, &blockNum, endOfRun);
        if(unusedSlot == UINT_MAX) return -1;
        if(blockNum > homeBlockNum){
            for(int i = 0; i < blockNum - homeBlockNum; i++){
                cqf->blocks[blockNum - i].offset++;
            }
        }
        //move items over until we get back to the run the item belongs in
        while(globalSlotIndex(blockNum, unusedSlot) > globalSlotIndex(runEndBlock, endOfRun)){
            if(unusedSlot == 0){
                int nextBlock = blockNum - 1;
                unsigned int nextSlot = SLOTS_PER_BLOCK - 1;
                setRemainder(cqf, blockNum, unusedSlot, getRemainder(cqf, nextBlock, nextSlot)); 
                if(isRunEnd(cqf->blocks[nextBlock].runEnds, nextSlot)){
                    cqf->blocks[blockNum].runEnds = setRunEnd(cqf->blocks[blockNum].runEnds, unusedSlot);
                }
                else{
                    cqf->blocks[blockNum].runEnds = clearRunEnd(cqf->blocks[blockNum].runEnds, unusedSlot);
                }
                unusedSlot = SLOTS_PER_BLOCK - 1;
                blockNum--;
            }
            else{
                setRemainder(cqf, blockNum, unusedSlot, getRemainder(cqf, blockNum, (unusedSlot - 1))); 
                if(isRunEnd(cqf->blocks[blockNum].runEnds, (unusedSlot - 1))){
                    cqf->blocks[blockNum].runEnds = setRunEnd(cqf->blocks[blockNum].runEnds, unusedSlot);
                }
                else{
                    cqf->blocks[blockNum].runEnds = clearRunEnd(cqf->blocks[blockNum].runEnds, unusedSlot);
                }
                unusedSlot--;
            }
        }
        
        //if the home slot was not previously occupied, then new item is its run
        if(!isOccupied(cqf->blocks[homeBlockNum].occupieds, homeSlotNum)){
            setRemainder(cqf, blockNum, unusedSlot, fr);
            cqf->blocks[blockNum].runEnds = setRunEnd(cqf->blocks[blockNum].runEnds, unusedSlot);
            cqf->blocks[homeBlockNum].occupieds = setOccupied(cqf->blocks[homeBlockNum].occupieds, homeSlotNum);
            return globalSlotIndex(blockNum, unusedSlot);
        }
        //if home slot already has a run, put new item in correct sequential location
        else{
            //move run end over by one slot
            unsigned int nextSlot = unusedSlot - 1;
            int nextBlock = blockNum;
            if(unusedSlot == 0){
                nextSlot = SLOTS_PER_BLOCK - 1;
                nextBlock = blockNum - 1;
                if(nextBlock < 0) return -1;
            }
            cqf->blocks[blockNum].runEnds = setRunEnd(cqf->blocks[blockNum].runEnds, unusedSlot);
            cqf->blocks[nextBlock].runEnds = clearRunEnd(cqf->blocks[nextBlock].runEnds, nextSlot);
            //search backwards through run
            //end search if: we reach another set runEnd bit; we find remainder <= new remainder; we reach canonical slot
            unsigned int nextRemainder = getRemainder(cqf, nextBlock, nextSlot);
//            printf("remainder in last run slot: %u\n", nextRemainder);
            do{
                if(nextRemainder <= fr){
                    setRemainder(cqf, blockNum, unusedSlot, fr);
                    //this stores duplicates
                    //return index of slot where remainder is stored
                    return globalSlotIndex(blockNum, unusedSlot);
                }
                setRemainder(cqf, blockNum, unusedSlot, nextRemainder);
                if(unusedSlot > 0){
                    unusedSlot--;
                    if(unusedSlot == 0){
                        if(blockNum == 0){
                            setRemainder(cqf, blockNum, unusedSlot, fr);
                            return globalSlotIndex(blockNum, unusedSlot);
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
                    if(blockNum < 0) return -1;
                    nextSlot = unusedSlot - 1;
                }
                nextRemainder = getRemainder(cqf, nextBlock, nextSlot);
            }while(!isRunEnd(cqf->blocks[nextBlock].runEnds, nextSlot) && (globalSlotIndex(nextBlock, nextSlot) >= fq));
            //unusedSlot is now head of run. Insert the remainder there.
            setRemainder(cqf, blockNum, unusedSlot, fr);
            return globalSlotIndex(blockNum, unusedSlot);
        }
    }
}
