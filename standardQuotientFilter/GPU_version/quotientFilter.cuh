//quotientFilter.cuh
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

#ifndef QUOTIENT_FILTER_CUH
#define QUOTIENT_FILTER_CUH

#ifndef NOT_FOUND
#define NOT_FOUND UINT_MAX
#endif

struct quotient_filter
{
    unsigned int qbits;
    unsigned int rbits;
    unsigned int bytesPerElement;
    unsigned char* table;
};

__host__ void initFilterGPU(struct quotient_filter *qf, unsigned int q, unsigned int r);
    /*  Allocates memory on GPU for the quotient filter based on inputs q and r.
    *   Filter Capacity = 2 ^ q
    *   Slot size = r + 3   */

__device__ __host__ size_t calcNumSlotsGPU(unsigned int q, unsigned int r);
    /*  Calculates the size of the array of chars needed
    *   to store the filter */

//__device__ __host__ unsigned int FNVhashGPU(unsigned int value, unsigned int tableSize);

__device__ __host__ unsigned int Normal_APHash(unsigned int value, unsigned int tableSize);

__host__ float launchSortedLookups(quotient_filter qfilter, int numValues, unsigned int* d_lookupValues, unsigned int* d_returnValues);

__host__ float launchUnsortedLookups(quotient_filter qfilter, int numValues, unsigned int* d_lookupValues, unsigned int* d_returnValues);

__host__ void printQuotientFilterGPU(struct quotient_filter *qf);

__host__ float insert(quotient_filter qfilter, int numValues, unsigned int* d_insertValues);

__host__ float bulkBuildParallelMerging(quotient_filter qfilter, int numValues, unsigned int* d_insertValues, bool NoDuplicates);

__host__ float bulkBuildSequentialShifts(quotient_filter qfilter, int numValues, unsigned int* d_insertValues, bool NoDuplicates);

__host__ float bulkBuildSegmentedLayouts(quotient_filter qfilter, int numValues, unsigned int* d_insertValues, bool NoDuplicates);

__host__ float superclusterDeletes(quotient_filter qfilter, int numValues, unsigned int* d_deleteValues);

__host__ float insertViaMerge(quotient_filter qfilter, unsigned int* d_insertedValues, int numOldValues, unsigned int* d_newValues, int numNewValues, bool NoDuplicates);

__host__ float mergeTwoFilters(quotient_filter qfilter1, quotient_filter qfilter2, bool NoDuplicates);

#endif
