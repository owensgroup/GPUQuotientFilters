//RSQF.h
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

#include <stdlib.h>

#ifndef RSQF_CPU_H
#define RSQF_CPU_H

#define RBITS 5
#define SLOTS_PER_BLOCK 64

struct __attribute__ ((packed)) cqf_block
{
    unsigned char offset;
    long long unsigned int occupieds;
    long long unsigned int runEnds;
    long long unsigned int remainders[RBITS];
};

struct countingQuotientFilter
{
    unsigned int qbits;
    cqf_block* blocks; 
};

size_t calcNumBlocks(unsigned int q);

void initCQF(struct countingQuotientFilter *cqf, unsigned int q);
     /*  Allocates memory for the counting quotient filter based on inputs q and r.
     *   Filter Capacity = 2 ^ q */

unsigned int Normal_APHash(unsigned int value, unsigned int maxHashValue);
 
void printFilter(struct countingQuotientFilter *cqf);

int lookup(struct countingQuotientFilter *cqf, unsigned int value);
    /*  Looks up value in counting QF.
     *  Returns the location of the remainder if it is found.
     *  Returns -1 if the remainder is not found. */

int insert(struct countingQuotientFilter *cqf, unsigned int value);
    /*  Inserts value into counting QF.
     *  Returns the final location of the remainder in the filter.  */

#endif
