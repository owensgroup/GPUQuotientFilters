//quotientFilter.h
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

#ifndef QUOTIENT_FILTER_H
#define QUOTIENT_FILTER_H

struct quotient_filter
{
	unsigned int qbits;
	unsigned int rbits;
	unsigned int* table;
};

size_t calcNumSlots(unsigned int q, unsigned int r);
        /*  Calculates the size of the array of unsigned integers needed
        *   to store the filter */

void initFilter(struct quotient_filter *qf, unsigned int q, unsigned int r);
        /*  Allocates memory for the quotient filter based on inputs q and r.
        *   Filter Capacity = 2 ^ q
        *   Slot size = r + 3   */

void printElement(struct quotient_filter *qf, unsigned int index);
	/*  Prints element to the screen.
        *   Shows remainder + metadata stored in filter at the given index. */

void printQuotientFilter(struct quotient_filter *qf);
        /*  Prints the metadata and the remainder for every slot in the filter. */

int mayContain(struct quotient_filter *qf, unsigned int hashValue);
        /*  Returns location of the given value in the quotient filter.
        *   Returns -1 if value is not in filter.
        *   Probability of false positives. */

unsigned int insertItem(struct quotient_filter *qf, unsigned int hashValue);
        /*  Inserts item into filter and returns the index. */

unsigned int fnv_hash(unsigned int value, unsigned int tableSize);
        /*  Returns the Fowler-Noll-Vo hash for a given value. */

#endif
