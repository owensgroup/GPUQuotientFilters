//deletesTest.cu
//Performs batch deletes on quotient filter based on superclusters.

#include <stdio.h>
#include <assert.h>
#include <cuda_profiler_api.h>

#include "../../mt19937ar.h"
#include "quotientFilter.cuh"

#ifndef LOW_BIT_MASK
#define LOW_BIT_MASK(n) ((1U << n) - 1U)
#endif

void generateRandomNumbers(unsigned int *numberArray, int n)
{
    for (int i = 0; i < n; i++){
        numberArray[i] = genrand_int32();
    }
}

void CUDAErrorCheck()
{
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    errSync = cudaGetLastError();
    errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess)
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    if (errAsync != cudaSuccess)
        printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
}

int main(int argc, char* argv[])
{
    assert(argc == 5);
    int q = atoi(argv[1]);
    int r = atoi(argv[2]);
    float alpha = atof(argv[3]);    //initial fill %
    int batchSize = atoi(argv[4]);  //size of batch to delete after build

    //Initialize filter
    struct quotient_filter d_qfilter;
    initFilterGPU(&d_qfilter, q, r);
    cudaMemset(d_qfilter.table, 0, calcNumSlotsGPU(q, r) * sizeof(unsigned char));

    //Generate set of random numbers
    int numValues = alpha * (1 << q);
    printf("Generating filter with %i slots and inserting %i values.\n", (1<<q), numValues);
    unsigned int* h_randomValues = new unsigned int[numValues];
    init_genrand(time(NULL));   //initialize random number generator
    generateRandomNumbers(h_randomValues, numValues);
    unsigned int* d_randomValues;
    cudaMalloc((void**) &d_randomValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_randomValues, h_randomValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

//Random Inserts
    CUDAErrorCheck();
    float filterBuildTime = bulkBuildSegmentedLayouts(d_qfilter, numValues, d_randomValues, true);
    CUDAErrorCheck();
    printf("Insert rate = %f million ops/sec\n", numValues / filterBuildTime / 1000);

//Test that lookups on all inserted items succeed
    //Array of lookup values
    unsigned int* d_successfulLookupValues;
    cudaMalloc((void**) &d_successfulLookupValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_successfulLookupValues, h_randomValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Allocate output array
    unsigned int* d_returnValues;
    cudaMalloc((void**) &d_returnValues, numValues * sizeof(unsigned int));
    cudaMemset(d_returnValues, 0, numValues * sizeof(unsigned int));

    //Launch lookup kernel
    CUDAErrorCheck();
    float lookupTime = launchUnsortedLookups(d_qfilter, numValues, d_successfulLookupValues, d_returnValues);
    CUDAErrorCheck();

    //Transfer back results of lookups and check that they succeeded
    unsigned int* h_returnValues = new unsigned int[numValues];
    for(int i = 0; i < numValues; i++){
        h_returnValues[i] = 0;
    }
    cudaMemcpy(h_returnValues, d_returnValues, numValues * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Checking that all items are there before deletes:\n");
    int insertFailures = 0;
    for(int i = 0; i < numValues; i++){
        if (h_returnValues[i] == NOT_FOUND){
            printf("ERROR: %dth value not found.\n", i);
            printf("Value: %u \n", h_randomValues[i]);
            insertFailures++;
        }
    }
    printf("%d inserted values were not found.\n", insertFailures);

//Delete Some of the Items
    unsigned int* d_deleteValues;
    cudaMalloc((void**) &d_deleteValues, batchSize * sizeof(unsigned int));
    cudaMemcpy(d_deleteValues, h_randomValues, batchSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    //Delete kernel
    CUDAErrorCheck();
    float deleteTime = superclusterDeletes(d_qfilter, batchSize, d_deleteValues);
    CUDAErrorCheck();
    printf("Delete rate = %f million ops/sec\n", batchSize / deleteTime / 1000);

    //Lookup kernel on deleted items to check that they are not there.
    unsigned int* d_deleteLookupReturns;
    cudaMalloc((void**) &d_deleteLookupReturns, numValues * sizeof(unsigned int));
    cudaMemset(d_deleteLookupReturns, 0, numValues * sizeof(unsigned int));

    //Launch lookup kernel
    CUDAErrorCheck();
    lookupTime = launchUnsortedLookups(d_qfilter, numValues, d_randomValues, d_deleteLookupReturns);
    CUDAErrorCheck();

    //Transfer back results of lookups and check that they failed
    unsigned int* h_deleteLookupReturns = new unsigned int[numValues];
    for(int i = 0; i < numValues; i++){
        h_deleteLookupReturns[i] = 0;
    }
    cudaMemcpy(h_deleteLookupReturns, d_deleteLookupReturns, numValues * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("Checking that all items have been deleted:\n");
    int deleteFailures = 0;
    for(int i = 0; i < batchSize; i++){
        if (h_deleteLookupReturns[i] != NOT_FOUND){
            printf("ERROR: %dth deleted value found.\n", i);
            printf("return value: %u \n", h_deleteLookupReturns[i]);
            printf("Value: %u \n", h_randomValues[i]);
            unsigned int hashValue = Normal_APHash(h_randomValues[i], (1 << (q + r)));
            //unsigned int hashValue = FNVhashGPU(h_randomValues[i], (1 << (q + r)));
            unsigned int fq = (hashValue >> r) & LOW_BIT_MASK(q);
            unsigned int fr = hashValue & LOW_BIT_MASK(r);
            printf("Quotient: %u \t Remainder: %u \n", fq, fr);
            deleteFailures++;
        }
    }
    printf("%d deleted values still found.\n", deleteFailures);

//Check the remaining (numValues - batchSize) items to check that they are still there. 
    //If two items withs same hash value were inserted, a lookup will fail for either.
    printf("Checking that items that were not deleted may still be found:\n");
    int lookupFailures = 0;
    for(int i = batchSize; i < numValues; i++){
        if (h_deleteLookupReturns[i] == NOT_FOUND){
            printf("return value[%d]: %u \n", i, h_deleteLookupReturns[i]);
            //unsigned int hashValue = FNVhashGPU(h_randomValues[i], (1 << (q + r)));
            unsigned int hashValue = Normal_APHash(h_randomValues[i], (1 << (q + r)));
            bool falseNegative = false;
            for(int j = 0; j < batchSize; j++){
                if(h_randomValues[j] = h_randomValues[i]) falseNegative = true;
            }
            if(falseNegative == false){
                printf("ERROR: %dth value not found.\n", i);
                printf("Value: %u \n", h_randomValues[i]);
                unsigned int fq = (hashValue >> r) & LOW_BIT_MASK(q);
                unsigned int fr = hashValue & LOW_BIT_MASK(r);
                printf("Quotient: %u \t Remainder: %u \n", fq, fr);
                lookupFailures++;
            }
        }
    }
    printf("%d values not found.\n", lookupFailures);

    //Free Memory
    cudaFree(d_qfilter.table);
    cudaFree(d_successfulLookupValues);
    cudaFree(d_returnValues);
    delete[] h_returnValues;
    cudaFree(d_randomValues);
    delete[] h_randomValues;
    cudaFree(d_deleteValues);
    cudaFree(d_deleteLookupReturns);
    cudaDeviceReset();

    return 0;
}

