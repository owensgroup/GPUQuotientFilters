//quotientFilter.cu
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
#include <limits.h>
#include <assert.h>
#include <cuda_profiler_api.h>
#include "quotientFilter.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include "../../cub-1.7.4/cub/cub.cuh"
#include "../../moderngpu/src/moderngpu/kernel_merge.hxx"

#ifndef LOW_BIT_MASK
#define LOW_BIT_MASK(n) ((1U << n) - 1U)
#endif
#ifndef NOT_FOUND
#define NOT_FOUND UINT_MAX
#endif

__device__ __host__ size_t calcNumSlotsGPU(unsigned int q, unsigned int r)
{
    size_t tableBits = (1 << q) * (r + 3);
    size_t tableSlots = tableBits / 8;
    return tableSlots * 1.1;    //allow an extra 10% for overflow
}

__host__ void initFilterGPU(struct quotient_filter* qf, unsigned int q, unsigned int r)
{
    assert((q + r) <= 32);  //need to be able to store fingerprints in unsigned int
    assert(((r + 3) % 8) == 0);  //slot size is one or more full bytes
    qf->qbits = q;
    qf->rbits = r;
    qf->bytesPerElement = (r + 3) / 8;
    size_t slots = calcNumSlotsGPU(q, r);
    unsigned char* d_filterTable;
    cudaMalloc((void**) &d_filterTable, slots * sizeof(unsigned char));
    qf->table = d_filterTable;
}

__device__ bool isOccupiedGPU(unsigned int element)
{
    return element & 4;
}

__device__ bool isContinuationGPU(unsigned int element)
{
    return element & 2;
}

__device__ bool isShiftedGPU(unsigned int element)
{
    return element & 1;
}

__device__ bool isEmptyGPU(unsigned int element)
{
    return ((element & 7) == 0);
}

__device__ unsigned int setOccupiedGPU(unsigned int element)
{
    return element | 4;
}

__device__ unsigned int clearOccupiedGPU(unsigned int element)
{
    return element & ~4;
}

__device__ unsigned int setContinuationGPU(unsigned int element)
{
    return element | 2;
}

__device__ unsigned int clearContinuationGPU(unsigned int element)
{
    return element & ~2;
}

__device__ unsigned int setShiftedGPU(unsigned int element)
{
    return element | 1;
}

__device__ unsigned int clearShiftedGPU(unsigned int element)
{
    return element & ~1;
}

__device__ __host__ unsigned int getRemainderGPU(unsigned int element)
{
    return element >> 3;
}

__device__ unsigned int isolateOccupiedBit(unsigned int element)
{
    return element & 4;
}

__device__ __host__ unsigned int FNVhashGPU(unsigned int value, unsigned int tableSize)
{
    unsigned char p[4];
    p[0] = (value >> 24) & 0xFF;
    p[1] = (value >> 16) & 0xFF;
    p[2] = (value >> 8) & 0xFF;
    p[3] = value & 0xFF;

    unsigned int h = 2166136261;

    for (int i = 0; i < 4; i++){
        h = (h * 16777619) ^ p[i];
    }

    return h % tableSize;
}

__device__ __host__ unsigned int Normal_APHash(unsigned int value, unsigned int tableSize)
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

    return hash % tableSize;
}

__device__ __host__ unsigned int getElementGPU(struct quotient_filter* qf, unsigned int index)
{
    unsigned int startSlot = index * qf->bytesPerElement;
    unsigned int element = qf->table[startSlot];
    for (int i = 1; i < qf->bytesPerElement; i++){
        element = (element << 8) | qf->table[startSlot + i];
    }

    return element;
}

__device__ void setElementGPU(struct quotient_filter* qf, unsigned int index, unsigned int value)
{
    unsigned int startSlot = index * qf->bytesPerElement;
    for (int i = 0; i < qf->bytesPerElement; i++){
        unsigned int shift = qf->bytesPerElement - 1 - i;
        qf->table[startSlot + i] = (value >> (8 * shift)) & LOW_BIT_MASK(8);
    }

}

__device__ unsigned int findRunStartGPU(struct quotient_filter* qf, unsigned int fq)
{
    unsigned int numElements = (1 << qf->qbits) * 1.1;
    //start bucket is fq
    unsigned int b = fq;
    //find beginning of cluster:
    while(isShiftedGPU(getElementGPU(qf, b))){
        b--;
    }

    //find start of run we're interested in:
    //slot counter starts at beginning of cluster
    unsigned int s = b;
    while(b != fq){
        do{
            s++;
        }while((isContinuationGPU(getElementGPU(qf, s))) && (s < numElements));   //find end of current run
        do{
            b++;
        }while((!isOccupiedGPU(getElementGPU(qf, b))) && (b < numElements));  //count number of runs passed
    }

    //now s is first value in correct run
    return s;
}

__device__ void insertItemHereGPU(struct quotient_filter* qf, unsigned int index, unsigned int value)
{
    unsigned int previousElement;
    unsigned int newElement = value;
    bool empty = false;

    while(!empty){
        previousElement = getElementGPU(qf, index);
        empty = isEmptyGPU(previousElement);

        previousElement = setShiftedGPU(previousElement);

        if(isOccupiedGPU(previousElement)){
            //Need to preserve correct is_occupied bits
            previousElement = clearOccupiedGPU(previousElement);
            newElement = setOccupiedGPU(newElement);
        }

        setElementGPU(qf, index, newElement);
        newElement = previousElement;
        index++;
    }
}

__global__ void lookUp(int numItems, struct quotient_filter qfilter, unsigned int* hashValues, unsigned int* slotValues)
{
    //returns NOT_FOUND (UINT_MAX) in slotValues[idx] if value is not in the filter, and returns the location of the remainder if it is in the filter

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    unsigned int hashValue = hashValues[idx];

    //separate into quotient and remainder
    unsigned int fq = (hashValue >> qfilter.rbits) & LOW_BIT_MASK(qfilter.qbits);
    unsigned int fr = hashValue & LOW_BIT_MASK(qfilter.rbits);

    unsigned int element = getElementGPU(&qfilter, fq);

    if(!isOccupiedGPU(element)){
        slotValues[idx] = NOT_FOUND;
        return;
    }

    unsigned int s = findRunStartGPU(&qfilter, fq);

    //search through elements in run
    do{
        unsigned int remainder = getRemainderGPU(getElementGPU(&qfilter, s));
        if(remainder == fr){
            slotValues[idx] = s;
            return;
        }
        else if(remainder > fr){
            slotValues[idx] = NOT_FOUND;
            return;
        }
        s++;
    }while(isContinuationGPU(getElementGPU(&qfilter, s)));

    slotValues[idx] = NOT_FOUND;
}

__global__ void hashInputs(int numItems, quotient_filter qfilter, unsigned int* insertValues, unsigned int* fingerprints)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    //hash values to get fingerprints
//    unsigned int hashValue = FNVhashGPU(insertValues[idx], (1 << (qfilter.qbits + qfilter.rbits)));
    unsigned int hashValue = Normal_APHash(insertValues[idx], (1 << (qfilter.qbits + qfilter.rbits)));
    fingerprints[idx] = hashValue;
}

__host__ float launchSortedLookups(quotient_filter qfilter, int numValues, unsigned int* d_lookupValues, unsigned int* d_returnValuesArray)
{
    //Allocate array for hash values
    thrust::device_vector<unsigned int> d_hashValues(numValues);
    thrust::fill(d_hashValues.begin(), d_hashValues.end(), 0); 
    unsigned int* d_hashValuesArray = thrust::raw_pointer_cast(&d_hashValues[0]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //Hash input values
    int numBlocks = (numValues + 127) / 128; 
    dim3 hashBlockDims((numBlocks + 31) / 32, 32); 
    hashInputs<<<hashBlockDims, 128>>>(numValues, qfilter, d_lookupValues, d_hashValuesArray);

    //Create index array to track inputs -> outputs
    thrust::device_vector<unsigned int> d_indices(numValues);
    thrust::fill(d_indices.begin(), d_indices.end(), 1); 
    thrust::exclusive_scan(d_indices.begin(), d_indices.end(), d_indices.begin(), 0);

    //Sort by fingerprint
    thrust::sort_by_key(d_hashValues.begin(), d_hashValues.end(), d_indices.begin());

    //Launch lookup kernel
    numBlocks = (numValues + 1023) / 1024;
    dim3 blockDims((numBlocks + 31) / 32, 32);
    lookUp<<<blockDims, 1024>>>(numValues, qfilter, d_hashValuesArray, d_returnValuesArray);

    //Sort outputs
    thrust::device_ptr<unsigned int> d_returnValues(d_returnValuesArray);
    thrust::sort_by_key(d_indices.begin(), d_indices.end(), d_returnValues);

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

__global__ void hashAndLookUp(int numItems, struct quotient_filter qfilter, unsigned int* lookupValues, unsigned int* slotValues)
{
    //returns NOT_FOUND (UINT_MAX) in slotValues[idx] if value is not in the filter, and returns the location of the remainder if it is in the filter

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    unsigned int hashValue = Normal_APHash(lookupValues[idx], (1 << (qfilter.qbits + qfilter.rbits)));

    //separate into quotient and remainder
    unsigned int fq = (hashValue >> qfilter.rbits) & LOW_BIT_MASK(qfilter.qbits);
    unsigned int fr = hashValue & LOW_BIT_MASK(qfilter.rbits);

    unsigned int element = getElementGPU(&qfilter, fq);

    if(!isOccupiedGPU(element)){
        slotValues[idx] = NOT_FOUND;
        return;
    }

    unsigned int s = findRunStartGPU(&qfilter, fq);

    //search through elements in run
    do{
        unsigned int remainder = getRemainderGPU(getElementGPU(&qfilter, s));
        if(remainder == fr){
            slotValues[idx] = s;
            return;
        }
        else if(remainder > fr){
            slotValues[idx] = NOT_FOUND;
            return;
        }
        s++;
    }while(isContinuationGPU(getElementGPU(&qfilter, s)));

    slotValues[idx] = NOT_FOUND;
}

__host__ float launchUnsortedLookups(quotient_filter qfilter, int numValues, unsigned int* d_lookupValues, unsigned int* d_returnValuesArray)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //Launch lookup kernel
    int numBlocks = (numValues + 1023) / 1024;
    dim3 blockDims((numBlocks + 31) / 32, 32);
    hashAndLookUp<<<blockDims, 1024>>>(numValues, qfilter, d_lookupValues, d_returnValuesArray);

    cudaEventRecord(stop);
    //Calculate timing results
    cudaEventSynchronize(stop);
    float lookupTime = 0;
    cudaEventElapsedTime(&lookupTime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return lookupTime;
}

__host__ void printQuotientFilterGPU(struct quotient_filter* qf)
{
    unsigned char* h_filterTable = new unsigned char[calcNumSlotsGPU(qf->qbits, qf->rbits)];
    cudaMemcpy(h_filterTable, qf->table, calcNumSlotsGPU(qf->qbits, qf->rbits) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    unsigned char* d_filterTable = qf->table;
    qf->table = h_filterTable;
    int filterSize = (1 << qf->qbits) * 1.1;
    printf("Printing metadata and remainders:\n");
    for(int i = 0; i < filterSize/10; i++){
        for(int j = 0; j < 10; j++){
            int element = getElementGPU(qf, 10*i + j);
            printf("%u \t", element & 7);
        }
        printf("\n");
        for(int j = 0; j < 10; j++){
            int element = getElementGPU(qf, 10*i + j);
            printf("%u \t", getRemainderGPU(element));
        }
        printf("\n --------------------------------------------------------------------- \n");
    }
    printf("\n");
    qf->table = d_filterTable;
}

__global__ void locateInsertSuperclusters(int numItems, quotient_filter qfilter, unsigned int* superclusterStarts)
{
    //marks the beginning of each supercluster by looking for empty slots
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    superclusterStarts[idx] = 0;

    if(idx == 0) return;

    if(isEmptyGPU(getElementGPU(&qfilter, idx - 1))){
        superclusterStarts[idx] = 1;
    }
}

__global__ void superclusterBidding(int numItems, quotient_filter qfilter, unsigned int* insertValues, unsigned int* superclusterIDs, bool* insertFlags, unsigned int* slotWinners) 
{
    //Outputs an array with one value per supercluster. These values can be inserted in parallel without collisions.
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    //initialize insert flags
    insertFlags[idx] = 1;

    if(insertValues[idx] == NOT_FOUND){
        return;
    }

    //calculate fingerprint
//    unsigned int hashValue = FNVhashGPU(insertValues[idx], (1 << (qfilter.qbits + qfilter.rbits)));
    unsigned int hashValue = Normal_APHash(insertValues[idx], (1 << (qfilter.qbits + qfilter.rbits)));

    //separate out the quotient/canonical slot bits
    unsigned int fq = (hashValue >> qfilter.rbits) & LOW_BIT_MASK(qfilter.qbits);

    //determine which supercluster the item belongs in
    unsigned int superclusterNumber = superclusterIDs[fq];

    //write the item's index to the supercluster slot to bid for insert
    slotWinners[superclusterNumber] = idx;
}

__global__ void insertItemGPU(int numItems, quotient_filter qfilter, unsigned int* insertValues, unsigned int* winnerIndices, unsigned int* finalLocationValues, bool* insertFlags)
{
    //inserts items into the filter, returning their slot locations in slotValues[idx]
    //if the item is already in the filter, it still returns the item location, although no changes are made

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    //check that there is an item to insert for this supercluster
    if(winnerIndices[idx] == NOT_FOUND){
        finalLocationValues[idx] = NOT_FOUND;
        return;
    }

    //determine which value is being added to the QF
    unsigned int originalIndex = winnerIndices[idx];
    //reset winnerIndices for next bidding round
    winnerIndices[idx] = NOT_FOUND;
    insertFlags[originalIndex] = 0;     //want to remove this item from insert queue
    unsigned int value = insertValues[originalIndex];

    //calculate fingerprint
//    unsigned int hashValue = FNVhashGPU(value, (1 << (qfilter.qbits + qfilter.rbits)));
    unsigned int hashValue = Normal_APHash(value, (1 << (qfilter.qbits + qfilter.rbits)));

    //separate into quotient and remainder
    unsigned int fq = (hashValue >> qfilter.rbits) & LOW_BIT_MASK(qfilter.qbits);
    unsigned int fr = hashValue & LOW_BIT_MASK(qfilter.rbits);

    unsigned int canonElement = getElementGPU(&qfilter, fq);
    unsigned int newElement = fr << 3;

    if(isEmptyGPU(canonElement)){
        setElementGPU(&qfilter, fq, setOccupiedGPU(newElement));
        finalLocationValues[idx] = fq;
        return;
    }
    
    if(!isOccupiedGPU(canonElement)){
        //set is_occupied to show that there is now a run for this slot
        setElementGPU(&qfilter, fq, setOccupiedGPU(canonElement));
    }

    //Find beginning of item's run
    unsigned int runStart = findRunStartGPU(&qfilter, fq);
    unsigned int s = runStart;

    if(isOccupiedGPU(canonElement)){
        //If slot already has a run, search through its elements.
        do{
            unsigned int remainder = getRemainderGPU(getElementGPU(&qfilter, s));
            if(remainder == fr){
                //the item is already in the filter
                finalLocationValues[idx] = s;
                return;
            }
            else if(remainder > fr){
                //s now points to where item goes
                break;
            }
            s++;
        }while(isContinuationGPU(getElementGPU(&qfilter, s)));

        if(s == runStart){
            //The new element is now the start of the run, but we must move old start over, so it will be continuation
            unsigned int oldStartElement = getElementGPU(&qfilter, runStart);
            setElementGPU(&qfilter, runStart, setContinuationGPU(oldStartElement));
        }
        else{
            //New element is not the start, so set its continuation bit
            newElement = setContinuationGPU(newElement);
        }
    }

    if(s != fq){
        //If it's not being inserted into the canonical slot, the element is shifted.
        newElement = setShiftedGPU(newElement);
    }

    insertItemHereGPU(&qfilter, s, newElement);
    finalLocationValues[idx] = s;
    return;
}

__host__ float insert(quotient_filter qfilter, int numValues, unsigned int* d_insertValues)
{
    int filterSize = (1 << qfilter.qbits) * 1.1; //number of (r + 3)-bit slots in the filter

    //Allocate all necessary memory for inserts
    int* h_numItemsLeft = new int[1];   //counts number of elements in insert queue
    h_numItemsLeft[0] = numValues;
    int* d_numItemsLeft;
    cudaMalloc((void**) &d_numItemsLeft, sizeof(int));
    unsigned int* d_superclusterIndicators; //stores bits marking beginning of superclusters
    cudaMalloc((void**) &d_superclusterIndicators, filterSize * sizeof(unsigned int));
    //Variables for CUB function temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    unsigned int* d_superclusterLabels = NULL;  //labels each slot with its supercluster number
    cudaMalloc((void**) &d_superclusterLabels, filterSize * sizeof(unsigned int));
    int* h_lastSuperclusterLabel = new int[1];
    int maxNumSuperclusters = calcNumSlotsGPU(qfilter.qbits, qfilter.rbits) + 1;
    unsigned int* d_slotWinners;
    cudaMalloc((void**) &d_slotWinners, maxNumSuperclusters * sizeof(unsigned int));
    unsigned int* h_slotWinners = new unsigned int[maxNumSuperclusters];
    for(int i = 0; i < maxNumSuperclusters; i++){
        h_slotWinners[i] = NOT_FOUND;
    }
    cudaMemcpy(d_slotWinners, h_slotWinners, maxNumSuperclusters * sizeof(unsigned int), cudaMemcpyHostToDevice);
    unsigned int* d_insertLocations;    //Output for actual locations where items are inserted
    cudaMalloc((void**) &d_insertLocations, maxNumSuperclusters * sizeof(unsigned int));
    bool* d_insertFlags;    //Flags for removing items from insert queue
    cudaMalloc((void**) &d_insertFlags, numValues * sizeof(bool));
    unsigned int* d_insertItemsQueue;
    cudaMalloc((void**) &d_insertItemsQueue, numValues * sizeof(unsigned int));
    cudaMemcpy(d_insertItemsQueue, d_insertValues, numValues * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //cudaProfilerStart();
    cudaEventRecord(start);

    do{
        //TODO: could consider marking superclusters from previous rounds with no items to insert so that we don't continue to launch threads for these superclusters to do no work

        //Find supercluster array:
        int numBlocks = (filterSize + 1023) / 1024;
        dim3 SCBlockDims((numBlocks + 31) / 32, 32);
        locateInsertSuperclusters<<<SCBlockDims, 1024>>>(filterSize, qfilter, d_superclusterIndicators);

        //CUB Inclusive Prefix Sum
        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_superclusterIndicators, d_superclusterLabels, filterSize));
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_superclusterIndicators, d_superclusterLabels, filterSize));

        //Determine how many superclusters there are
        cudaMemcpy(h_lastSuperclusterLabel, d_superclusterLabels + (filterSize - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost);
        int numSuperclusters = h_lastSuperclusterLabel[0] + 1;

        //Pick one element per supercluster to insert
        numBlocks = (h_numItemsLeft[0] + 127) / 128;
        dim3 biddingBlockDims((numBlocks + 31) / 32, 32);
        superclusterBidding<<<biddingBlockDims, 128>>>(h_numItemsLeft[0], qfilter, d_insertItemsQueue, d_superclusterLabels, d_insertFlags, d_slotWinners);

        //Insert items into QF
        numBlocks = (numSuperclusters + 255) / 256;
        dim3 insertBlockDims((numBlocks + 31) / 32, 32);
        insertItemGPU<<<insertBlockDims, 256>>>(numSuperclusters, qfilter, d_insertItemsQueue, d_slotWinners, d_insertLocations, d_insertFlags);

        //Remove successfully inserted items from the queue
        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_insertItemsQueue, d_insertFlags, d_insertItemsQueue, d_numItemsLeft, h_numItemsLeft[0]));
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_insertItemsQueue, d_insertFlags, d_insertItemsQueue, d_numItemsLeft, h_numItemsLeft[0]));
        cudaMemcpy(h_numItemsLeft, d_numItemsLeft, sizeof(int), cudaMemcpyDeviceToHost);
    }while(h_numItemsLeft[0] > 0);

    cudaEventRecord(stop);
    //cudaProfilerStop();
    //Calculate timing results
    cudaEventSynchronize(stop);
    float insertTime = 0;
    cudaEventElapsedTime(&insertTime, start, stop);

    //Free memory
    delete[] h_numItemsLeft;
    cudaFree(d_numItemsLeft);
    cudaFree(d_superclusterIndicators);
    cudaFree(d_temp_storage);
    cudaFree(d_superclusterLabels);
    cudaFree(d_slotWinners);
    delete[] h_slotWinners;
    cudaFree(d_insertLocations);
    cudaFree(d_insertFlags);
    cudaFree(d_insertItemsQueue);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return insertTime;
}

__global__ void quotienting(int numItems, unsigned int qbits, unsigned int rbits, unsigned int* quotients, unsigned int* remainders)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    //return quotients and remainders
    unsigned int hashValue = quotients[idx]; //quotients array initially stores the fingerprint values
    unsigned int canonicalSlot = (hashValue >> rbits) & LOW_BIT_MASK(qbits);
    quotients[idx] = canonicalSlot;
    unsigned int remainderBits = hashValue & LOW_BIT_MASK(rbits);
    remainders[idx] = remainderBits;
}

__global__ void findSegmentHeads(int numItems, unsigned int* quotients, unsigned int* segStarts)
{
    //locate the beginnings of segments

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    if(idx != 0){
        if(quotients[idx] != quotients[idx - 1]){
            segStarts[idx] = 1;
        }
    }
}

__global__ void calcOffsets(int numItems, unsigned int* locations, unsigned int* segLabels, int* offsets, int* credits, int* creditCarryover)
{
    //compute the shift/credits for a group of elements when merging their segments
   
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems || idx ==0) return;

    unsigned int segmentIdx = segLabels[idx];
    if((segmentIdx != segLabels[idx - 1]) && (segmentIdx % 2 == 1)){
        offsets[segmentIdx] = locations[idx - 1] - locations[idx] + 1;
        creditCarryover[segmentIdx] = credits[idx - 1];
    }
}

__global__ void shiftElements(int numItems, int* offsets, int* credits, unsigned int* locations, int* creditCarryover, unsigned int* segLabels)
{
    //calculate the shifts for merging segments

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems || idx == 0) return;

    unsigned int segmentIdx = segLabels[idx];
    int netShift = offsets[segmentIdx] - credits[idx];
    int newCredits = 0;
    if(netShift > 0){       //merging the segments causes the items to shift.
        locations[idx] += netShift;
        newCredits = 0;
    }
    else{   //there are extra slots between segments. Track these with credits.
        newCredits = -netShift;
    }

    credits[idx] = newCredits + creditCarryover[segmentIdx];

    segLabels[idx] /= 2;
}

__global__ void setMetadata(int numItems, unsigned int* remainders, unsigned int* quotients, unsigned int* locations)
{
    //set is_continuation and is_shifted bits for each item in the filter
    
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    unsigned int element = remainders[idx];

    element = element << 3; //shift remainder left to make room for metadata bits
   
    //is_continuation: check if quotient[i-1] = quotient[i] (this is already stored in segStarts)
    if(idx != 0 && quotients[idx] == quotients[idx - 1]) element = setContinuationGPU(element);

    //is_shifted: if location > quotient
    if(locations[idx] != quotients[idx]) element = setShiftedGPU(element);

    remainders[idx] = element;
}

__global__ void writeRemainders(int numItems, quotient_filter qfilter, unsigned int* remainders, unsigned int* locations)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;
    
    setElementGPU(&qfilter, locations[idx], remainders[idx]);
}

__global__ void setOccupiedBits(int numItems, quotient_filter qfilter, unsigned int* quotients)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    unsigned int element = getElementGPU(&qfilter, quotients[idx]);
    setElementGPU(&qfilter, quotients[idx], setOccupiedGPU(element));
}

__host__ float bulkBuildParallelMerging(quotient_filter qfilter, int numValues, unsigned int* d_insertValues, bool NoDuplicates)
{
    //build a quotient filter by inserting all (or a large batch of) items all at once
    /*  1. Compute all fingerprints
        2. Sort list of fingerprints
        3. Quotienting
        4. Segmented scan of array of 1's
        5. Add to quotients
        6. Create array of credits[number items], initialized to 0
        7. "Associative scan" with saturating operator to find end positions
        8. Write values to filter
    */

    //Memory Allocation
    thrust::device_vector<unsigned int> d_quotients(numValues);
    thrust::fill(d_quotients.begin(), d_quotients.end(), 0);
    unsigned int* d_quotientsArray = thrust::raw_pointer_cast(&d_quotients[0]);
    thrust::device_vector<unsigned int> d_locations(numValues);
    thrust::fill_n(d_locations.begin(), numValues, 1);
    unsigned int* d_locationsArray = thrust::raw_pointer_cast(&d_locations[0]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //cudaProfilerStart();
    cudaEventRecord(start);

    //Hash input values
    int numBlocks = (numValues + 127) / 128;
    dim3 hashBlockDims((numBlocks + 31) / 32, 32);
    hashInputs<<<hashBlockDims, 128>>>(numValues, qfilter, d_insertValues, d_quotientsArray);  //store fingerprints in quotients array

    //Sort by fingerprint
    thrust::sort(d_quotients.begin(), d_quotients.end());

    //Remove duplicates, if desired
    if(NoDuplicates == true){
        thrust::detail::normal_iterator< thrust::device_ptr<unsigned int> > fingerprintEnd = thrust::unique(d_quotients.begin(), d_quotients.end());
        d_quotients.erase(fingerprintEnd, d_quotients.end());
        numValues = d_quotients.end() - d_quotients.begin();
    }

    //Divide fingerprints into quotients and remainders
    unsigned int* d_remaindersArray;
    cudaMalloc((void**) &d_remaindersArray, numValues * sizeof(unsigned int));
    cudaMemset(d_remaindersArray, 0, numValues * sizeof(unsigned int));
    numBlocks = (numValues + 767) / 768;
    dim3 quotientingBlockDims((numBlocks + 31) / 32, 32);
    quotienting<<<quotientingBlockDims, 768>>>(numValues, qfilter.qbits, qfilter.rbits, d_quotientsArray, d_remaindersArray);

    //Segmented scan of array of 1's
    thrust::exclusive_scan_by_key(d_quotients.begin(), d_quotients.end(), d_locations.begin(), d_locations.begin());

    //Add scanned values to quotients to find initial locations before shifts
    thrust::transform(d_quotients.begin(), d_quotients.end(), d_locations.begin(), d_locations.begin(), thrust::plus<unsigned int>());

    //Associative scans:
        //1. Each quotient is a segment
        //2. Pair up segments
        //3. offset = L_tail - R_head + 1
        //4. net shift = offset - credit
        //5. for each element:
            //if (net shift > 0): shift = net shift; credit = 0
            //if (net shift < 0): shift = 0; credit = -net shift
            //segmentLabel = segmentLabel/2

    thrust::device_vector<unsigned int> d_segStarts(numValues);
    thrust::fill(d_segStarts.begin(), d_segStarts.end(), 0);
    unsigned int* d_segStartsArray = thrust::raw_pointer_cast(&d_segStarts[0]);
    thrust::device_vector<unsigned int> d_segLabels(numValues);
    unsigned int* d_segLabelsArray = thrust::raw_pointer_cast(&d_segLabels[0]);

    //Label segments for grouping items
    numBlocks = (numValues + 767) / 768;
    dim3 findSegHeadsBlockDims((numBlocks + 31) / 32, 32);
    findSegmentHeads<<<findSegHeadsBlockDims, 768>>>(numValues, d_quotientsArray, d_segStartsArray);
    thrust::inclusive_scan(d_segStarts.begin(), d_segStarts.end(), d_segLabels.begin());
    d_segStarts.~device_vector<unsigned int>();

    //Join segments, calculating shifts along the way
    int numSegments = d_segLabels[numValues - 1] + 1;
    int numLoops = (int) ceil(log2((float)numSegments));
    int* d_offsets;
    cudaMalloc((void**) &d_offsets, numSegments * sizeof(int));
    int* d_creditCarryover;
    cudaMalloc((void**) &d_creditCarryover, numSegments * sizeof(int));
    int* d_credits;
    cudaMalloc((void**) &d_credits, numValues * sizeof(int));
    cudaMemset(d_credits, 0, numValues * sizeof(int));

    for(int i = 0; i < numLoops; i++){
        cudaMemset(d_offsets, 0, numSegments * sizeof(int));
        cudaMemset(d_creditCarryover, 0, numSegments * sizeof(int));
        
        //Calculate offsets between segments
        numBlocks = (numValues + 255) / 256;
        dim3 findSegHeadsBlockDims((numBlocks + 31) / 32, 32);
        calcOffsets<<<findSegHeadsBlockDims, 256>>>(numValues, d_locationsArray, d_segLabelsArray, d_offsets, d_credits, d_creditCarryover);

        //Calculate the shifts/credits for each item in this round of merging
        //Relabel segments so that pairs have now merged
        numBlocks = (numValues + 767) / 768;
        dim3 shiftElementsBlockDims((numBlocks + 31) / 32, 32);
        shiftElements<<<shiftElementsBlockDims, 768>>>(numValues, d_offsets, d_credits, d_locationsArray, d_creditCarryover, d_segLabelsArray);
    }

    //Shift the remainder values to left to make room for metadata
    //Then determine metadata bits and set them
    numBlocks = (numValues + 1023) / 1024;
    dim3 setMetadataBlockDims((numBlocks + 31) / 32, 32);
    setMetadata<<<setMetadataBlockDims, 1024>>>(numValues, d_remaindersArray, d_quotientsArray, d_locationsArray);

    //Scatter remainder values to the filter
    numBlocks = (numValues + 1023) / 1024;
    dim3 writeRemaindersBlockDims((numBlocks + 31) / 32, 32);
    writeRemainders<<<writeRemaindersBlockDims, 1024>>>(numValues, qfilter, d_remaindersArray, d_locationsArray);

    //Set the is_occupied bits
    numBlocks = (numValues + 511) / 512;
    dim3 setOccupiedBitsBlockDims((numBlocks + 31) / 32, 32);
    setOccupiedBits<<<setOccupiedBitsBlockDims, 512>>>(numValues, qfilter, d_quotientsArray);

    //Calculate and print timing results
    cudaEventRecord(stop);
    //cudaProfilerStop();
    cudaEventSynchronize(stop);
    float filterBuildTime = 0;
    cudaEventElapsedTime(&filterBuildTime, start, stop);

    //Free memory
    d_quotients.~device_vector<unsigned int>();
    cudaFree(d_remaindersArray);
    d_locations.~device_vector<unsigned int>();
    d_segLabels.~device_vector<unsigned int>();
    cudaFree(d_credits);
    cudaFree(d_offsets);
    cudaFree(d_creditCarryover);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return filterBuildTime;
}

__global__ void segmentStartLocations(int numItems, unsigned int* segLabels, unsigned int* segStartLocations)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;
    if(idx == 0){
        segStartLocations[0] = 0;
        return;
    }

    if(segLabels[idx] == segLabels[idx-1]) return;

    segStartLocations[segLabels[idx]] = idx;
}

__global__ void shiftSegments(int numItems, unsigned int* segStartLocations, unsigned int* locations, int numValues, bool* changesMade)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems || idx ==0) return;

    int arrayIndex = segStartLocations[idx];
    int shift = locations[arrayIndex-1] - locations[arrayIndex] + 1;
    if(shift > 0){
        int segmentLength;
        if(idx == (numItems - 1)){
            segmentLength = numValues - segStartLocations[idx];
        } 
        else{
            segmentLength = segStartLocations[idx+1] - segStartLocations[idx];
        }
        for(int i = 0; i < segmentLength; i++){
            locations[arrayIndex + i] += shift;
        }
        changesMade[0] = 1;
    }
}

void printArray(int numValues, int* array)
{   
    for(int i = 0; i < numValues/10; i++){
        for(int j = 0; j < 10; j++){
            printf("%i\t", array[i*10 + j]);
        }
        printf("\n");
    }
    for(int i = 0; i < numValues % 10; i++){
        printf("%i\t", array[((numValues/10)*10) + i]);
    }
    printf("\n");
}

__host__ float bulkBuildSequentialShifts(quotient_filter qfilter, int numValues, unsigned int* d_insertValues, bool NoDuplicates)
{
    //build a quotient filter by inserting all (or a large batch of) items all at once
    //compute locations by shifting one run at a time
    //exit when shifting stops
    /*  1. Compute all quotients & remainders
           List of pairs of (fq, fr)
        2. Sort list by fq, then by fr within groups of same fq (or maybe in reverse order?)
        3. Segmented scan of array of 1's
        4. Add to quotients
        5. Iterate:
            a. Compute shift at every boundary between runs and shift all items in run if needed
            b. Write to Bool to indicate shift happened
            b. Check if a shift happened; if not, done!
        6. Write values to filter
    */

    //Memory Allocation
    thrust::device_vector<unsigned int> d_quotients(numValues);
    thrust::fill(d_quotients.begin(), d_quotients.end(), 0);
    unsigned int* d_quotientsArray = thrust::raw_pointer_cast(&d_quotients[0]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //Hash input values
    int numBlocks = (numValues + 127) / 128;
    dim3 hashBlockDims((numBlocks + 31) / 32, 32);
    hashInputs<<<hashBlockDims, 128>>>(numValues, qfilter, d_insertValues, d_quotientsArray);  //store fingerprints in quotients array

    //Sort by fingerprint
    thrust::sort(d_quotients.begin(), d_quotients.end());

    //Remove duplicates, if desired
    if(NoDuplicates == true){
        thrust::detail::normal_iterator< thrust::device_ptr<unsigned int> > fingerprintEnd = thrust::unique(d_quotients.begin(), d_quotients.end());
        d_quotients.erase(fingerprintEnd, d_quotients.end());
        numValues = d_quotients.end() - d_quotients.begin();
    }

    //Divide fingerprints into quotients and remainders
    unsigned int* d_remaindersArray;
    cudaMalloc((void**) &d_remaindersArray, numValues * sizeof(unsigned int));
    cudaMemset(d_remaindersArray, 0, numValues * sizeof(unsigned int));
    thrust::device_vector<unsigned int> d_locations(numValues);
    thrust::fill_n(d_locations.begin(), numValues, 1);
    unsigned int* d_locationsArray = thrust::raw_pointer_cast(&d_locations[0]);

    numBlocks = (numValues + 767) / 768;
    dim3 quotientingBlockDims((numBlocks + 31) / 32, 32);
    quotienting<<<quotientingBlockDims, 768>>>(numValues, qfilter.qbits, qfilter.rbits, d_quotientsArray, d_remaindersArray);

    //Segmented scan of array of 1's
    thrust::exclusive_scan_by_key(d_quotients.begin(), d_quotients.end(), d_locations.begin(), d_locations.begin());

    //Add scanned values to quotients to find initial locations before shifts
    thrust::transform(d_quotients.begin(), d_quotients.end(), d_locations.begin(), d_locations.begin(), thrust::plus<unsigned int>());

    //Label segments for grouping items
    thrust::device_vector<unsigned int> d_segStarts(numValues);
    thrust::fill(d_segStarts.begin(), d_segStarts.end(), 0);
    unsigned int* d_segStartsArray = thrust::raw_pointer_cast(&d_segStarts[0]);
    thrust::device_vector<unsigned int> d_segLabels(numValues);
    unsigned int* d_segLabelsArray = thrust::raw_pointer_cast(&d_segLabels[0]);
    numBlocks = (numValues + 767) / 768;
    dim3 findSegHeadsBlockDims((numBlocks + 31) / 32, 32);
    findSegmentHeads<<<findSegHeadsBlockDims, 768>>>(numValues, d_quotientsArray, d_segStartsArray);
    thrust::inclusive_scan(d_segStarts.begin(), d_segStarts.end(), d_segLabels.begin());
    d_segStarts.~device_vector<unsigned int>();

    int numSegments = d_segLabels[numValues - 1] + 1;
    unsigned int* d_segStartLocations;
    cudaMalloc((void**) &d_segStartLocations, numSegments * sizeof(unsigned int));
    //Create array with the location of first item in each run
    numBlocks = (numValues + 1023) / 1024;
    dim3 segStartLocationsBlockDims((numBlocks + 31) / 32, 32);
    segmentStartLocations<<<findSegHeadsBlockDims, 1024>>>(numValues, d_segLabelsArray, d_segStartLocations);

    bool* h_changesMade = new bool[1];
    h_changesMade[0] = 1;
    bool* d_changesMade;
    cudaMalloc((void**) &d_changesMade, sizeof(bool));
    cudaMemset(d_changesMade, 0, sizeof(bool));
    while(h_changesMade[0] == 1){
        h_changesMade[0] = 0;
        numBlocks = (numSegments + 191) / 192;
        dim3 shiftSegsBlockDims((numBlocks + 31) / 32, 32);
        shiftSegments<<<shiftSegsBlockDims, 192>>>(numSegments, d_segStartLocations, d_locationsArray, numValues, d_changesMade);
        cudaMemcpy(h_changesMade, d_changesMade, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemset(d_changesMade, 0, sizeof(bool));
    }

    //Shift the remainder values to left to make room for metadata
    //Then determine metadata bits and set them
    numBlocks = (numValues + 1023) / 1024;
    dim3 setMetadataBlockDims((numBlocks + 31) / 32, 32);
    setMetadata<<<setMetadataBlockDims, 1024>>>(numValues, d_remaindersArray, d_quotientsArray, d_locationsArray);

    //Scatter remainder values to the filter
    numBlocks = (numValues + 1023) / 1024;
    dim3 writeRemaindersBlockDims((numBlocks + 31) / 32, 32);
    writeRemainders<<<writeRemaindersBlockDims, 1024>>>(numValues, qfilter, d_remaindersArray, d_locationsArray);

    //Set the is_occupied bits
    numBlocks = (numValues + 511) / 512;
    dim3 setOccupiedBitsBlockDims((numBlocks + 31) / 32, 32);
    setOccupiedBits<<<setOccupiedBitsBlockDims, 512>>>(numValues, qfilter, d_quotientsArray);

    //Calculate and print timing results
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float filterBuildTime = 0;
    cudaEventElapsedTime(&filterBuildTime, start, stop);

    //Free memory
    d_quotients.~device_vector<unsigned int>();
    cudaFree(d_remaindersArray);
    d_locations.~device_vector<unsigned int>();
    d_segLabels.~device_vector<unsigned int>();
    cudaFree(d_segStartLocations);
    delete[] h_changesMade;
    cudaFree(d_changesMade);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return filterBuildTime;
}

__global__ void locateDeleteSuperclusters(int numItems, quotient_filter qfilter, unsigned int* superclusterStarts)
{
    //marks the beginning of each "supercluster" -> really, for deletes this is same as a cluster
    
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    superclusterStarts[idx] = 0;

    if(idx == 0) return;

    if((!isEmptyGPU(getElementGPU(&qfilter, idx))) && (!isShiftedGPU(getElementGPU(&qfilter, idx)))){
        superclusterStarts[idx] = 1;
    }
}

__global__ void deleteItemGPU(int numItems, quotient_filter qfilter, unsigned int* deleteValues, unsigned int* winnerIndices, bool* deleteFlags)
{
    //deletes items from the quotient filter, shifting other items left if required
    
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    //check that there is an item assigned to be deleted for supercluster idx
    if(winnerIndices[idx] == NOT_FOUND){
        return;
    }

   //determine which value is being added to the QF
    unsigned int originalIndex = winnerIndices[idx];
    //reset winnerIndices for next bidding round
    winnerIndices[idx] = NOT_FOUND;
    deleteFlags[originalIndex] = 0;     //want to remove this item from delete queue after this finishes
    unsigned int value = deleteValues[originalIndex];

    //calculate fingerprint
//    unsigned int hashValue = FNVhashGPU(value, (1 << (qfilter.qbits + qfilter.rbits)));
    unsigned int hashValue = Normal_APHash(value, (1 << (qfilter.qbits + qfilter.rbits)));

    //separate into quotient and remainder
    unsigned int fq = (hashValue >> qfilter.rbits) & LOW_BIT_MASK(qfilter.qbits);
    unsigned int fr = hashValue & LOW_BIT_MASK(qfilter.rbits);

    unsigned int canonElement = getElementGPU(&qfilter, fq);
    if(!isOccupiedGPU(canonElement)){
        //if canonical slot is not occupied, the item isn't in the filter; we're done.
        return;
    }

    //start bucket is fq
    unsigned int b = fq;
    //find beginning of cluster:
    while(isShiftedGPU(getElementGPU(&qfilter, b))){
        b--;
    }   

    //find start of run we're interested in:
    //slot counter starts at beginning of cluster
    unsigned int s = b;
    while(b != fq){
        do{
            s++;
        }while((isContinuationGPU(getElementGPU(&qfilter, s))));   //find end of current run
        do{
            b++;
        }while((!isOccupiedGPU(getElementGPU(&qfilter, b))));  //count number of runs passed
    }

    //now s is first value in run of item to be deleted 
    unsigned int runStart = s;
    //Search through the run's elements to find item needing to be deleted
    unsigned int remainder;
    do{
        remainder = getRemainderGPU(getElementGPU(&qfilter, s));
        if(remainder == fr){
            //found it!
            break;
        }
        else if(remainder > fr){
            //the item is not in the filter
            //nothing to delete here
            return;
        }
        s++;
    }while(isContinuationGPU(getElementGPU(&qfilter, s)));

    //If we searched entire run without finding it:
    if(remainder != fr){
        return; //the item is not in the filter
    }

    if(!isContinuationGPU(getElementGPU(&qfilter, (s + 1)))){
        do{
            //if next item is a new run, add to run count
            b++;
        }while(!isOccupiedGPU(getElementGPU(&qfilter, b)));
    }

    //We have now located the item that needs to be deleted, stored in slot s.
    //Special conditions for deleted run starts
    if(s == runStart){
        if(!isContinuationGPU(getElementGPU(&qfilter, (s + 1)))){
            //the run is empty; clear the occupied bit
            setElementGPU(&qfilter, fq, clearOccupiedGPU(getElementGPU(&qfilter, fq)));
        }
        else{
            //next item is now the first in the run
            setElementGPU(&qfilter, (s + 1), clearContinuationGPU(getElementGPU(&qfilter, (s + 1))));
        }
    }

    //now check the item to the right to see whether it will need to be moved
    //if it was shifted, it is part of the same cluster and can be shifted left
    while(isShiftedGPU(getElementGPU(&qfilter, (s + 1)))){
        //want to check if s = b for clearing shifted bit
        if(b == s){  //in this case, run about to be shifted into its correct slot -> unshifted
           setElementGPU(&qfilter, (s + 1), clearShiftedGPU(getElementGPU(&qfilter, (s + 1))));
        }

        do{
            unsigned int nextElement = getElementGPU(&qfilter, (s + 1));
            if(isOccupiedGPU(getElementGPU(&qfilter, s))){
                setElementGPU(&qfilter, s, setOccupiedGPU(nextElement));
            }
            else{
                setElementGPU(&qfilter, s, clearOccupiedGPU(nextElement));
            }
            s++;
        }while((isContinuationGPU(getElementGPU(&qfilter, (s + 1))))); //shift the items in current run

        do{
            b++;
        }while(!isOccupiedGPU(getElementGPU(&qfilter, b)));  //keep track of current run

    }

    //Last item is always a new empty slot
    setElementGPU(&qfilter, s, 0);

    return;
}

__host__ float superclusterDeletes(quotient_filter qfilter, int numValues, unsigned int* d_deleteValues)
{
    int filterSize = (1 << qfilter.qbits) * 1.1; //number of (r + 3)-bit slots in the filter

    //Allocate all necessary memory for deletes
    int* h_numItemsLeft = new int[1];   //counts number of elements in delete queue
    h_numItemsLeft[0] = numValues;
    int* d_numItemsLeft;
    cudaMalloc((void**) &d_numItemsLeft, sizeof(int));
    unsigned int* d_superclusterIndicators; //stores bits marking beginning of superclusters
    cudaMalloc((void**) &d_superclusterIndicators, filterSize * sizeof(unsigned int));
    //Variables for CUB function temporary storage
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    unsigned int* d_superclusterLabels = NULL;  //labels each slot with its supercluster number
    cudaMalloc((void**) &d_superclusterLabels, filterSize * sizeof(unsigned int));
    int* h_lastSuperclusterLabel = new int[1];
    int maxNumSuperclusters = calcNumSlotsGPU(qfilter.qbits, qfilter.rbits) + 1;
    unsigned int* d_slotWinners;
    cudaMalloc((void**) &d_slotWinners, maxNumSuperclusters * sizeof(unsigned int));
    unsigned int* h_slotWinners = new unsigned int[maxNumSuperclusters];
    for(int i = 0; i < maxNumSuperclusters; i++){
        h_slotWinners[i] = NOT_FOUND;
    }
    cudaMemcpy(d_slotWinners, h_slotWinners, maxNumSuperclusters * sizeof(unsigned int), cudaMemcpyHostToDevice);
    bool* d_deleteFlags;    //Flags for removing items from delete queue
    cudaMalloc((void**) &d_deleteFlags, numValues * sizeof(bool));
    unsigned int* d_deleteItemsQueue;
    cudaMalloc((void**) &d_deleteItemsQueue, numValues * sizeof(unsigned int));
    cudaMemcpy(d_deleteItemsQueue, d_deleteValues, numValues * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    do{
        //Find supercluster array:
        int numBlocks = (filterSize + 1023) / 1024;
        dim3 deleteSCBlockDims((numBlocks + 31) / 32, 32);
        locateDeleteSuperclusters<<<deleteSCBlockDims, 1024>>>(filterSize, qfilter, d_superclusterIndicators);

        //CUB Inclusive Prefix Sum
        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_superclusterIndicators, d_superclusterLabels, filterSize));
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_superclusterIndicators, d_superclusterLabels, filterSize));

        //Determine how many superclusters there are
        cudaMemcpy(h_lastSuperclusterLabel, d_superclusterLabels + (filterSize - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost);
        int numSuperclusters = h_lastSuperclusterLabel[0] + 1;

        //Pick one element per supercluster to delete
        numBlocks = (h_numItemsLeft[0] + 127) / 128;
        dim3 biddingBlockDims((numBlocks + 31) / 32, 32);
        superclusterBidding<<<biddingBlockDims, 128>>>(h_numItemsLeft[0], qfilter, d_deleteItemsQueue, d_superclusterLabels, d_deleteFlags, d_slotWinners);

        //Insert items into QF
        numBlocks = (numSuperclusters + 1023) / 1024;
        dim3 deleteBlockDims((numBlocks + 31) / 32, 32);
        deleteItemGPU<<<deleteBlockDims, 1024>>>(numSuperclusters, qfilter, d_deleteItemsQueue, d_slotWinners, d_deleteFlags);

        //Remove successfully deleted items from the queue
        d_temp_storage = NULL;
        temp_storage_bytes = 0;

        CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_deleteItemsQueue, d_deleteFlags, d_deleteItemsQueue, d_numItemsLeft, h_numItemsLeft[0]));
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_deleteItemsQueue, d_deleteFlags, d_deleteItemsQueue, d_numItemsLeft, h_numItemsLeft[0]));
        cudaMemcpy(h_numItemsLeft, d_numItemsLeft, sizeof(int), cudaMemcpyDeviceToHost);

    }while(h_numItemsLeft[0] > 0);

    cudaEventRecord(stop);
    //Calculate timing results
    cudaEventSynchronize(stop);
    float deleteTime = 0;
    cudaEventElapsedTime(&deleteTime, start, stop);

    //Free memory
    delete[] h_numItemsLeft;
    cudaFree(d_numItemsLeft);
    cudaFree(d_superclusterIndicators);
    cudaFree(d_temp_storage);
    cudaFree(d_superclusterLabels);
    cudaFree(d_slotWinners);
    delete[] h_slotWinners;
    cudaFree(d_deleteFlags);
    cudaFree(d_deleteItemsQueue);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return deleteTime;
}

__global__ void findSegmentStarts(int numItems, unsigned int q, unsigned int* quotients, unsigned int* segmentStarts)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems || idx ==0) return;

    unsigned int currentSegment = quotients[idx] / q;
    unsigned int previousSegment = quotients[idx - 1] / q;
    if(currentSegment != previousSegment){
        segmentStarts[currentSegment] = idx;
    }
}

__global__ void layout(int numItems, unsigned int qbits, unsigned int* quotients, unsigned int* segmentAssignments, int* shift, int* overflow, bool* changesMade, int numInsertValues)
{
    //computes layout for the idx-th segment of the filter

    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    int firstNonemptySegment = quotients[0] / qbits;
    if(idx < firstNonemptySegment){
        overflow[idx] = 0;
        return;
    }
    int segStart;
    int firstItemIdx;
    if(idx == 0){
        segStart = 0;
        if(quotients[0] < qbits){
            firstItemIdx = 0;
        }
        else{   //no items in segment 0
            overflow[0] = 0;
            return;
        }
    }
    else{
        segStart = idx * qbits + shift[idx-1];  //start the layout to right of shifted values from previous segment
        firstItemIdx = segmentAssignments[idx];
        if(firstItemIdx == 0 && (quotients[0] < segStart)){  //the segment has no items
            overflow[idx] = 0;
            return;
        }
    }
    int lastItemIdx;
    if(idx == (numItems - 1)){  //last segment
        lastItemIdx = numInsertValues - 1;
    }
    else{
        lastItemIdx = segmentAssignments[idx + 1] - 1;
        int j = idx + 1;
        while(lastItemIdx == -1 && j < numItems){    //in case of empty segments to the right
            if(j == numItems - 1){
                lastItemIdx = numInsertValues - 1;
            }
            else{
                lastItemIdx = segmentAssignments[j] - 1;
            }
            j++;
        }
    }
    int numSegItems = lastItemIdx - firstItemIdx + 1;
    if(numSegItems <= 0){
        overflow[idx] = 0;
        return;
    }

    int maxSlot = segStart; //maxSlot = next open slot
    for(int i = firstItemIdx; i <= lastItemIdx; i++){
        if(quotients[i] > maxSlot) maxSlot = quotients[i];
        maxSlot++;
    }

    int segEnd = ((idx + 1) * qbits) - 1;
    int segmentOverflow = (maxSlot - 1) - segEnd;
    if(segmentOverflow > 0){
        overflow[idx] = segmentOverflow;
        if(segmentOverflow > shift[idx]){   //check if there has been change from last iteration
            changesMade[0] = 1;
        }
    }
    else{
        overflow[idx] = 0;
    }
}

__global__ void segmentedQFWrite(int numItems, quotient_filter qfilter, unsigned int* quotients, unsigned int* remainders, unsigned int* segmentAssignments, int* shift, int numInsertValues)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    int segStart;
    int firstItemIdx;
    if(idx == 0){
        segStart = 0;
        if(quotients[0] < qfilter.qbits){
            firstItemIdx = 0;
        }
        else{   //no items in segment 0
            return;
        }
    }
    else{
        segStart = idx * qfilter.qbits + shift[idx-1];  //start the layout to right of shifted values from previous segment
        firstItemIdx = segmentAssignments[idx];
        if(firstItemIdx == 0 && (quotients[0] < segStart)){  //the segment has no items
            return;
        }
    }
    int lastItemIdx;
    if(idx == (numItems - 1)){
        lastItemIdx = numInsertValues - 1;
    }
    else{
        lastItemIdx = segmentAssignments[idx + 1] - 1;
        int j = idx + 1;
        while(lastItemIdx == -1 && j < numItems){    //in case of empty segments to the right
            if(j == numItems - 1){
                lastItemIdx = numInsertValues - 1;
            }
            else{
                lastItemIdx = segmentAssignments[j] - 1;
            }
            j++;
        }
    }
    int numSegItems = lastItemIdx - firstItemIdx + 1;
    if(numSegItems <= 0){
        return;
    }

    int maxSlot = segStart;    //maxSlot = location of last/currently inserted item
    for(int i = firstItemIdx; i <= lastItemIdx; i++){
        unsigned int currentRemainder = remainders[i] << 3;
        if(quotients[i] >= maxSlot){    //item is not shifted
            maxSlot = quotients[i];
            setElementGPU(&qfilter, maxSlot, currentRemainder);
        }
        else{
            currentRemainder = setShiftedGPU(currentRemainder);
            if(quotients[i] == quotients[i - 1]) currentRemainder = setContinuationGPU(currentRemainder);
            setElementGPU(&qfilter, maxSlot, currentRemainder);
        }
        maxSlot++;
    }
}

__host__ float bulkBuildSegmentedLayouts(quotient_filter qfilter, int numValues, unsigned int* d_insertValues, bool NoDuplicates)
{
    //build a quotient filter by partitioning into segments, inserting items into segments, then computing overflow
    /*  1. Compute all fingerprints
        2. Sort list of fingerprints
        3. Quotienting
        4. Assign items to a segment
        5. Compute layouts and overflow
        6. Repeat until convergence
        7. Write final values to filter
    */

    //Memory Allocation
    thrust::device_vector<unsigned int> d_quotients(numValues);
    thrust::fill(d_quotients.begin(), d_quotients.end(), 0);
    unsigned int* d_quotientsArray = thrust::raw_pointer_cast(&d_quotients[0]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //cudaProfilerStart();
    cudaEventRecord(start);

    //Hash input values
    int numBlocks = (numValues + 127) / 128;
    dim3 hashBlockDims((numBlocks + 31) / 32, 32);
    hashInputs<<<hashBlockDims, 128>>>(numValues, qfilter, d_insertValues, d_quotientsArray);  //store fingerprints in quotients array

    //Sort by fingerprint
    thrust::sort(d_quotients.begin(), d_quotients.end());

    //Remove duplicates, if desired
    if(NoDuplicates == true){
        thrust::detail::normal_iterator< thrust::device_ptr<unsigned int> > fingerprintEnd = thrust::unique(d_quotients.begin(), d_quotients.end());
        d_quotients.erase(fingerprintEnd, d_quotients.end());
        numValues = d_quotients.end() - d_quotients.begin();
    }

    //Divide fingerprints into quotients and remainders
    unsigned int* d_remaindersArray;
    cudaMalloc((void**) &d_remaindersArray, numValues * sizeof(unsigned int));
    cudaMemset(d_remaindersArray, 0, numValues * sizeof(unsigned int));
    numBlocks = (numValues + 767) / 768;
    dim3 quotientingBlockDims((numBlocks + 31) / 32, 32);
    quotienting<<<quotientingBlockDims, 768>>>(numValues, qfilter.qbits, qfilter.rbits, d_quotientsArray, d_remaindersArray);

    unsigned int q = qfilter.qbits;
    int numSegments = ((1 << q) / q) + 1;

    //Determine which items belong in each segment
    unsigned int* d_segmentStarts;
    cudaMalloc((void**) &d_segmentStarts, numSegments * sizeof(unsigned int));
    cudaMemset(d_segmentStarts, 0, numSegments * sizeof(unsigned int));
    numBlocks = (numValues + 255) / 256;
    dim3 findSegStartsBlockDims((numBlocks + 31) / 32, 32);
    findSegmentStarts<<<findSegStartsBlockDims, 256>>>(numValues, q, d_quotientsArray, d_segmentStarts);

    //Each segment has an input shift value and outputs overflow value
    int* d_shifts;
    cudaMalloc((void**) &d_shifts, numSegments * sizeof(int));
    cudaMemset(d_shifts, 0, numSegments * sizeof(int));
    int* d_overflows;
    cudaMalloc((void**) &d_overflows, numSegments * sizeof(int));
    cudaMemset(d_overflows, 0, numSegments * sizeof(int));

    bool* h_changesMade = new bool[1];
    h_changesMade[0] = 1;
    bool* d_changesMade;
    cudaMalloc((void**) &d_changesMade, sizeof(bool));
    cudaMemset(d_changesMade, 0, sizeof(bool));
    while(h_changesMade[0] == 1){
        h_changesMade[0] = 0;   //since I set d_changesMade to 0 already might not need this
        //copy overflows into shifts
        //shifts[idx] represents the shift caused by segment idx, to be carried over into segment idx+1
        cudaMemcpy(d_shifts, d_overflows, sizeof(int) * numSegments, cudaMemcpyDeviceToDevice);

        //Launch one thread per segment
        //Layout
        numBlocks = (numSegments + 255) / 256;
        dim3 layoutBlockDims((numBlocks + 31) / 32, 32);
        layout<<<layoutBlockDims, 256>>>(numSegments, qfilter.qbits, d_quotientsArray, d_segmentStarts, d_shifts, d_overflows, d_changesMade, numValues);

        cudaMemcpy(h_changesMade, d_changesMade, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemset(d_changesMade, 0, sizeof(bool));
    }

    //Write final values to filter
    numBlocks = (numSegments + 127) / 128;
    dim3 segmentedQFWriteBlockDims((numBlocks + 31) / 32, 32);
    segmentedQFWrite<<<segmentedQFWriteBlockDims, 128>>>(numSegments, qfilter, d_quotientsArray, d_remaindersArray, d_segmentStarts, d_shifts, numValues);

    numBlocks = (numValues + 511) / 512;
    dim3 setOccupiedBitsBlockDims((numBlocks + 31) / 32, 32);
    setOccupiedBits<<<setOccupiedBitsBlockDims, 512>>>(numValues, qfilter, d_quotientsArray);

    cudaEventRecord(stop);
    //Calculate timing results
    cudaEventSynchronize(stop);
    float buildTime = 0;
    cudaEventElapsedTime(&buildTime, start, stop);

    //Free memory
    d_quotients.~device_vector<unsigned int>();
    cudaFree(d_remaindersArray);
    cudaFree(d_segmentStarts);
    cudaFree(d_shifts);
    cudaFree(d_overflows);
    delete[] h_changesMade;
    cudaFree(d_changesMade);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return buildTime;
}

__global__ void extractQuotients(int numItems, quotient_filter qfilter, unsigned int* fingerprints, bool* emptySlotFlags)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * gridDim.x * blockDim.x;
    if(idx >= numItems) return;

    unsigned int element = getElementGPU(&qfilter, idx);
   
    //Empty slots:
    if(isEmptyGPU(element)){
        emptySlotFlags[idx] = true;
        return;
    }

    //Unshifted elements(beginning of cluster): 
    if(!isShiftedGPU(element)){
        fingerprints[idx] = (idx << qfilter.rbits) | getRemainderGPU(element);
        return;
    }    

    //Shifted elements:
    //Find beginning of cluster:
    unsigned int b = idx;
    do{
        b--;
    }while(isShiftedGPU(getElementGPU(&qfilter, b)));

    //Step through cluster, counting the runs:
    unsigned int s = b;
    while(s <= idx){
        do{
            s++;
        }while((isContinuationGPU(getElementGPU(&qfilter, s))));    //find end of each run
        if(s > idx) break; 
        do{
            b++;
        }while((!isOccupiedGPU(getElementGPU(&qfilter, b))));    //keeping track of canonical slot
    }
    
    fingerprints[idx] = (b << qfilter.rbits) | getRemainderGPU(element);
}

__host__ float insertViaMerge(quotient_filter qfilter, unsigned int* d_insertedValues, int numOldValues, unsigned int* d_newValues, int numNewValues, bool NoDuplicates)
{
    //d_insertedValues and numOldValues are just for checking results of fingerprint extraction. They are not needed for the merge operation.

//    printQuotientFilterGPU(&qfilter);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

//Extract fingerprints from quotient filter
    int numSlots = calcNumSlotsGPU(qfilter.qbits, qfilter.rbits);
    thrust::device_vector<bool> d_emptySlotFlags(numSlots);
    thrust::fill(d_emptySlotFlags.begin(), d_emptySlotFlags.end(), 0);
    bool* d_emptySlotFlagsArray = thrust::raw_pointer_cast(&d_emptySlotFlags[0]);

    thrust::device_vector<unsigned int> d_fingerprintsBySlot(numSlots);
    thrust::fill(d_fingerprintsBySlot.begin(), d_fingerprintsBySlot.end(), 0);
    unsigned int* d_fingerprintsBySlotArray = thrust::raw_pointer_cast(&d_fingerprintsBySlot[0]);

    int numBlocks = (numSlots + 191) / 192;
    dim3 extractQuotientsBlockDims((numBlocks + 31) / 32, 32);
    extractQuotients<<<extractQuotientsBlockDims, 192>>>(numSlots, qfilter, d_fingerprintsBySlotArray, d_emptySlotFlagsArray);

    thrust::detail::normal_iterator< thrust::device_ptr<unsigned int> > fingerprintsEnd = thrust::remove_if(d_fingerprintsBySlot.begin(), d_fingerprintsBySlot.end(), d_emptySlotFlags.begin(), thrust::identity<bool>());
    d_fingerprintsBySlot.erase(fingerprintsEnd, d_fingerprintsBySlot.end());
    int numExtractedValues = d_fingerprintsBySlot.end() - d_fingerprintsBySlot.begin();

//Merge with new array 
    //Hash and quotientize new values to insert
    thrust::device_vector<unsigned int> d_newFingerprints(numNewValues);
    thrust::fill(d_newFingerprints.begin(), d_newFingerprints.end(), 0);
    unsigned int* d_newFingerprintsArray = thrust::raw_pointer_cast(&d_newFingerprints[0]);
    numBlocks = (numNewValues + 127) / 128;
    dim3 hashBlockDims((numBlocks + 31) / 32, 32);
    hashInputs<<<hashBlockDims, 128>>>(numNewValues, qfilter, d_newValues, d_newFingerprintsArray);

    //Sort by fingerprint
    thrust::sort(d_newFingerprints.begin(), d_newFingerprints.end());

    //Merge d_newValues with extracted quotients and remainders
    mgpu::standard_context_t context(false);

    int outputSize = numExtractedValues + numNewValues;
    mgpu::mem_t<unsigned int> d_fingerprintsOutput(outputSize, context);
    mgpu::mem_t<unsigned int> d_newFingerprintsMem = copy_to_mem(d_newFingerprintsArray, numNewValues, context);
    mgpu::mem_t<unsigned int> d_extractedFingerprintsMem = copy_to_mem(d_fingerprintsBySlotArray, numExtractedValues, context);

    mgpu::merge(d_extractedFingerprintsMem.data(), numExtractedValues, d_newFingerprintsMem.data(), numNewValues, d_fingerprintsOutput.data(), mgpu::less_t<unsigned int>(), context);

    unsigned int* d_combinedQuotients = d_fingerprintsOutput.data();

//Rebuild filter using segmented layouts method
    //Clear old filter
    cudaMemset(qfilter.table, 0, numSlots * sizeof(unsigned char));

    //Remove duplicates, if desired
    thrust::device_vector<unsigned int> d_thrustQuotients(d_combinedQuotients, d_combinedQuotients + outputSize);   //must copy values to get them into thrust device_vector
    if(NoDuplicates == true){
        thrust::detail::normal_iterator< thrust::device_ptr<unsigned int> > fingerprintEnd = thrust::unique(d_thrustQuotients.begin(), d_thrustQuotients.end());
        d_thrustQuotients.erase(fingerprintEnd, d_thrustQuotients.end());
        outputSize = d_thrustQuotients.end() - d_thrustQuotients.begin();
    }
    d_combinedQuotients = thrust::raw_pointer_cast(&d_thrustQuotients[0]);

    //Divide fingerprints into quotients and remainders
    unsigned int* d_combinedRemainders;
    cudaMalloc((void**) &d_combinedRemainders, outputSize * sizeof(unsigned int));
    cudaMemset(d_combinedRemainders, 0, outputSize * sizeof(unsigned int));
    numBlocks = (outputSize + 767) / 768;
    dim3 quotientingBlockDims((numBlocks + 31) / 32, 32);
    quotienting<<<quotientingBlockDims, 768>>>(outputSize, qfilter.qbits, qfilter.rbits, d_combinedQuotients, d_combinedRemainders);

    unsigned int q = qfilter.qbits;
    int numSegments = ((1 << q) / q) + 1;

    //Determine which items belong in each segment
    unsigned int* d_segmentStarts;
    cudaMalloc((void**) &d_segmentStarts, numSegments * sizeof(unsigned int));
    cudaMemset(d_segmentStarts, 0, numSegments * sizeof(unsigned int));
    numBlocks = (outputSize + 255) / 256;
    dim3 findSegStartsBlockDims((numBlocks + 31) / 32, 32);
    findSegmentStarts<<<findSegStartsBlockDims, 256>>>(outputSize, q, d_combinedQuotients, d_segmentStarts);

    //Each segment has an input shift value and outputs overflow value
    int* d_shifts;
    cudaMalloc((void**) &d_shifts, numSegments * sizeof(int));
    cudaMemset(d_shifts, 0, numSegments * sizeof(int));
    int* d_overflows;
    cudaMalloc((void**) &d_overflows, numSegments * sizeof(int));
    cudaMemset(d_overflows, 0, numSegments * sizeof(int));

    bool* h_changesMade = new bool[1];
    h_changesMade[0] = 1;
    bool* d_changesMade;
    cudaMalloc((void**) &d_changesMade, sizeof(bool));
    cudaMemset(d_changesMade, 0, sizeof(bool));
    while(h_changesMade[0] == 1){
        //copy overflows into shifts
        cudaMemcpy(d_shifts, d_overflows, sizeof(int) * numSegments, cudaMemcpyDeviceToDevice);

        //Layout
        numBlocks = (numSegments + 255) / 256;
        dim3 layoutBlockDims((numBlocks + 31) / 32, 32);
        layout<<<layoutBlockDims, 256>>>(numSegments, qfilter.qbits, d_combinedQuotients, d_segmentStarts, d_shifts, d_overflows, d_changesMade, outputSize);

        cudaMemcpy(h_changesMade, d_changesMade, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemset(d_changesMade, 0, sizeof(bool));
    }
    //Write final values to filter
    numBlocks = (numSegments + 127) / 128;
    dim3 segmentedQFWriteBlockDims((numBlocks + 31) / 32, 32);
    segmentedQFWrite<<<segmentedQFWriteBlockDims, 128>>>(numSegments, qfilter, d_combinedQuotients, d_combinedRemainders, d_segmentStarts, d_shifts, outputSize);

    numBlocks = (outputSize + 511) / 512;
    dim3 setOccupiedBitsBlockDims((numBlocks + 31) / 32, 32);
    setOccupiedBits<<<setOccupiedBitsBlockDims, 512>>>(outputSize, qfilter, d_combinedQuotients);

    //Timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float rebuildTime = 0;
    cudaEventElapsedTime(&rebuildTime, start, stop);

    //Free Memory
    d_emptySlotFlags.~device_vector<bool>();
    d_fingerprintsBySlot.~device_vector<unsigned int>();
    d_newFingerprints.~device_vector<unsigned int>();
    d_fingerprintsOutput.~mem_t<unsigned int>();
    d_newFingerprintsMem.~mem_t<unsigned int>();
    d_extractedFingerprintsMem.~mem_t<unsigned int>();
    cudaFree(d_combinedRemainders);
    cudaFree(d_segmentStarts);
    cudaFree(d_shifts);
    cudaFree(d_overflows);
    delete[] h_changesMade;
    cudaFree(d_changesMade);

    return rebuildTime;
}

__host__ float mergeTwoFilters(quotient_filter qfilter1, quotient_filter qfilter2, bool NoDuplicates)
{
    //merges filters qfilter1 and qfilter2 and outputs the result to qfilter1

    //Check that filters are the same size
    if(qfilter1.qbits != qfilter2.qbits || qfilter1.rbits != qfilter2.rbits){
        printf("Error: two filters to be merged must have same number of quotient and remainder bits\n");
        return 0.0f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

//Extract fingerprints from first quotient filter
    int numSlots = calcNumSlotsGPU(qfilter1.qbits, qfilter1.rbits);
    thrust::device_vector<bool> d_emptySlotFlags(numSlots);
    thrust::fill(d_emptySlotFlags.begin(), d_emptySlotFlags.end(), 0);
    bool* d_emptySlotFlagsArray = thrust::raw_pointer_cast(&d_emptySlotFlags[0]);

    thrust::device_vector<unsigned int> d_fingerprintsBySlot1(numSlots);
    thrust::fill(d_fingerprintsBySlot1.begin(), d_fingerprintsBySlot1.end(), 0);
    unsigned int* d_fingerprintsBySlotArray1 = thrust::raw_pointer_cast(&d_fingerprintsBySlot1[0]);

    int numBlocks = (numSlots + 191) / 192;
    dim3 extractQuotientsBlockDims1((numBlocks + 31) / 32, 32);
    extractQuotients<<<extractQuotientsBlockDims1, 192>>>(numSlots, qfilter1, d_fingerprintsBySlotArray1, d_emptySlotFlagsArray);

    thrust::detail::normal_iterator< thrust::device_ptr<unsigned int> > fingerprintsEnd = thrust::remove_if(d_fingerprintsBySlot1.begin(), d_fingerprintsBySlot1.end(), d_emptySlotFlags.begin(), thrust::identity<bool>());
    d_fingerprintsBySlot1.erase(fingerprintsEnd, d_fingerprintsBySlot1.end());
    int numExtractedValues1 = d_fingerprintsBySlot1.end() - d_fingerprintsBySlot1.begin();

//Extract fingerprints from second quotient filter
    thrust::fill(d_emptySlotFlags.begin(), d_emptySlotFlags.end(), 0);

    thrust::device_vector<unsigned int> d_fingerprintsBySlot2(numSlots);
    thrust::fill(d_fingerprintsBySlot2.begin(), d_fingerprintsBySlot2.end(), 0);
    unsigned int* d_fingerprintsBySlotArray2 = thrust::raw_pointer_cast(&d_fingerprintsBySlot2[0]);

    numBlocks = (numSlots + 191) / 192;
    dim3 extractQuotientsBlockDims2((numBlocks + 31) / 32, 32);
    extractQuotients<<<extractQuotientsBlockDims2, 192>>>(numSlots, qfilter2, d_fingerprintsBySlotArray2, d_emptySlotFlagsArray);

    fingerprintsEnd = thrust::remove_if(d_fingerprintsBySlot2.begin(), d_fingerprintsBySlot2.end(), d_emptySlotFlags.begin(), thrust::identity<bool>());
    d_fingerprintsBySlot2.erase(fingerprintsEnd, d_fingerprintsBySlot2.end());
    int numExtractedValues2 = d_fingerprintsBySlot2.end() - d_fingerprintsBySlot2.begin();

//Merge arrays of extracted values 
    mgpu::standard_context_t context(false);

    int outputSize = numExtractedValues1 + numExtractedValues2;
    mgpu::mem_t<unsigned int> d_fingerprintsOutput(outputSize, context);
    mgpu::mem_t<unsigned int> d_extractedFingerprintsMem1 = copy_to_mem(d_fingerprintsBySlotArray1, numExtractedValues1, context);
    mgpu::mem_t<unsigned int> d_extractedFingerprintsMem2 = copy_to_mem(d_fingerprintsBySlotArray2, numExtractedValues2, context);

    mgpu::merge(d_extractedFingerprintsMem1.data(), numExtractedValues1, d_extractedFingerprintsMem2.data(), numExtractedValues2, d_fingerprintsOutput.data(), mgpu::less_t<unsigned int>(), context);

    unsigned int* d_combinedQuotients = d_fingerprintsOutput.data();

//Rebuild filter using segmented layouts method
    //Clear old filter
    cudaMemset(qfilter1.table, 0, numSlots * sizeof(unsigned char));

    //Remove duplicates, if desired
    thrust::device_vector<unsigned int> d_thrustQuotients(d_combinedQuotients, d_combinedQuotients + outputSize);   //must copy values to get them into thrust device_vector
    if(NoDuplicates == true){
        thrust::detail::normal_iterator< thrust::device_ptr<unsigned int> > fingerprintEnd = thrust::unique(d_thrustQuotients.begin(), d_thrustQuotients.end());
        d_thrustQuotients.erase(fingerprintEnd, d_thrustQuotients.end());
        outputSize = d_thrustQuotients.end() - d_thrustQuotients.begin();
    }
    d_combinedQuotients = thrust::raw_pointer_cast(&d_thrustQuotients[0]);

    //Divide fingerprints into quotients and remainders
    unsigned int* d_combinedRemainders;
    cudaMalloc((void**) &d_combinedRemainders, outputSize * sizeof(unsigned int));
    cudaMemset(d_combinedRemainders, 0, outputSize * sizeof(unsigned int));
    numBlocks = (outputSize + 767) / 768;
    dim3 quotientingBlockDims((numBlocks + 31) / 32, 32);
    quotienting<<<quotientingBlockDims, 768>>>(outputSize, qfilter1.qbits, qfilter1.rbits, d_combinedQuotients, d_combinedRemainders);

    unsigned int q = qfilter1.qbits;
    int numSegments = ((1 << q) / q) + 1;

    //Determine which items belong in each segment
    unsigned int* d_segmentStarts;
    cudaMalloc((void**) &d_segmentStarts, numSegments * sizeof(unsigned int));
    cudaMemset(d_segmentStarts, 0, numSegments * sizeof(unsigned int));
    numBlocks = (outputSize + 255) / 256;
    dim3 findSegStartsBlockDims((numBlocks + 31) / 32, 32);
    findSegmentStarts<<<findSegStartsBlockDims, 256>>>(outputSize, q, d_combinedQuotients, d_segmentStarts);

    //Each segment has an input shift value and outputs overflow value
    int* d_shifts;
    cudaMalloc((void**) &d_shifts, numSegments * sizeof(int));
    cudaMemset(d_shifts, 0, numSegments * sizeof(int));
    int* d_overflows;
    cudaMalloc((void**) &d_overflows, numSegments * sizeof(int));
    cudaMemset(d_overflows, 0, numSegments * sizeof(int));

    bool* h_changesMade = new bool[1];
    h_changesMade[0] = 1;
    bool* d_changesMade;
    cudaMalloc((void**) &d_changesMade, sizeof(bool));
    cudaMemset(d_changesMade, 0, sizeof(bool));
    while(h_changesMade[0] == 1){
        //copy overflows into shifts
        cudaMemcpy(d_shifts, d_overflows, sizeof(int) * numSegments, cudaMemcpyDeviceToDevice);

        //Layout
        numBlocks = (numSegments + 255) / 256;
        dim3 layoutBlockDims((numBlocks + 31) / 32, 32);
        layout<<<layoutBlockDims, 256>>>(numSegments, qfilter1.qbits, d_combinedQuotients, d_segmentStarts, d_shifts, d_overflows, d_changesMade, outputSize);

        cudaMemcpy(h_changesMade, d_changesMade, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemset(d_changesMade, 0, sizeof(bool));
    }
    //Write final values to filter
    numBlocks = (numSegments + 127) / 128;
    dim3 segmentedQFWriteBlockDims((numBlocks + 31) / 32, 32);
    segmentedQFWrite<<<segmentedQFWriteBlockDims, 128>>>(numSegments, qfilter1, d_combinedQuotients, d_combinedRemainders, d_segmentStarts, d_shifts, outputSize);

    numBlocks = (outputSize + 511) / 512;
    dim3 setOccupiedBitsBlockDims((numBlocks + 31) / 32, 32);
    setOccupiedBits<<<setOccupiedBitsBlockDims, 512>>>(outputSize, qfilter1, d_combinedQuotients);

    //Timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float rebuildTime = 0;
    cudaEventElapsedTime(&rebuildTime, start, stop);

    //Free Memory
    d_emptySlotFlags.~device_vector<bool>();
    d_fingerprintsBySlot1.~device_vector<unsigned int>();
    d_fingerprintsBySlot2.~device_vector<unsigned int>();
    d_fingerprintsOutput.~mem_t<unsigned int>();
    d_extractedFingerprintsMem1.~mem_t<unsigned int>();
    d_extractedFingerprintsMem2.~mem_t<unsigned int>();
    cudaFree(d_combinedRemainders);
    cudaFree(d_segmentStarts);
    cudaFree(d_shifts);
    cudaFree(d_overflows);
    delete[] h_changesMade;
    cudaFree(d_changesMade);

    return rebuildTime;
}
