//insertViaMergeTest.cu

#include <stdio.h>
#include <assert.h>
#include <cuda_profiler_api.h>

#include "../../mt19937ar.h"
#include "quotientFilter.cuh"

#ifndef LOW_BIT_MASK
#define LOW_BIT_MASK(n) ((1U << n) -1U)
#endif

void generateRandomNumbers(unsigned int *numberArray, unsigned int n)
{
    for (int i = 0; i < n; i++){
        numberArray[i] = genrand_int32();
    }
}

void generateNewRandomNumbers(unsigned int *newNumberArray, unsigned int *comparisonNumberArray, int n, int n_comp)
{
    generateRandomNumbers(newNumberArray, n);
    unsigned int numNew = 0;
    while (numNew < n){
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n_comp; j++){
                if (newNumberArray[i] == comparisonNumberArray[j]){
                    newNumberArray[i] = genrand_int32();
                    j = 0;
                }
            }
            numNew++;
        }
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

void arrayPrint(int numValues, int* array)
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

int main(int argc, char* argv[])
{
    assert(argc == 6);
    int q = atoi(argv[1]);
    int r = atoi(argv[2]);
    float alpha = atof(argv[3]);    //initial fill %
    int batchSize = atoi(argv[4]);  //size of batch to insert after build
    bool NoDuplicates;
    std::string dup(argv[5]);
    if(dup == "Dup") NoDuplicates = false;
    else if(dup == "NoDup") NoDuplicates = true;
    else{
        printf("ERROR: Last argument should be Dup or NoDup. \nPlease indicate whether you want to leave duplicate items or remove them.");
        return 0;
    }

    //Initialize filter
    printf("Building filter with 2^%u slots and %u-bit remainders....\n", q, r);
    struct quotient_filter d_qfilter;
    initFilterGPU(&d_qfilter, q, r);
    cudaMemset(d_qfilter.table, 0, calcNumSlotsGPU(q, r) * sizeof(unsigned char));
    
    //Generate set of random numbers
    printf("Generating random numbers to fill the filter to %3.1f percent full...\n", (alpha * 100));
    int numValues = alpha * (1 << q);
    unsigned int* h_randomValues = new unsigned int[numValues];
    init_genrand(time(NULL));   //initialize random number generator
    generateRandomNumbers(h_randomValues, numValues);
    printf("%d random numbers generated.\n", numValues);
    unsigned int* d_randomValues;
    cudaMalloc((void**) &d_randomValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_randomValues, h_randomValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Build filter
    CUDAErrorCheck();
    float filterBuildTime = bulkBuildSegmentedLayouts(d_qfilter, numValues, d_randomValues, true);
    CUDAErrorCheck();

    //Generate new set of random numbers to insert
    unsigned int* h_newValues = new unsigned int[batchSize];
    generateNewRandomNumbers(h_newValues, h_randomValues, batchSize, numValues);
    printf("%d random new numbers generated.\n", batchSize);
    unsigned int* d_newValues;
    cudaMalloc((void**) &d_newValues, batchSize * sizeof(unsigned int));
    cudaMemcpy(d_newValues, h_newValues, batchSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

//Read out old values and rebuild filter with new values added
    CUDAErrorCheck();
    float filterMergeTime = insertViaMerge(d_qfilter, d_randomValues, numValues, d_newValues, batchSize, NoDuplicates);
    CUDAErrorCheck();
    printf("Insert rate = %f million ops/sec\n", batchSize / filterMergeTime / 1000);

//Lookup original values
    unsigned int* d_lookupReturnValues;
    cudaMalloc((void**) &d_lookupReturnValues, numValues * sizeof(unsigned int));
    cudaMemset(d_lookupReturnValues, 0, numValues * sizeof(unsigned int));

    CUDAErrorCheck();
    float lookupTime = launchUnsortedLookups(d_qfilter, numValues, d_randomValues, d_lookupReturnValues);
    CUDAErrorCheck();

    //Transfer back results of lookups and check that they succeeded
    unsigned int* h_lookupReturnValues = new unsigned int[numValues];
    for(int i = 0; i < numValues; i++){
        h_lookupReturnValues[i] = 0;
    }
    cudaMemcpy(h_lookupReturnValues, d_lookupReturnValues, numValues * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    int insertFailures = 0;
    for(int i = 0; i < numValues; i++){
        if (h_lookupReturnValues[i] == NOT_FOUND){
            printf("ERROR: %dth value not found.\n", i);
            printf("Value: %u \n", h_randomValues[i]);
            //unsigned int hashValue = FNVhashGPU(h_randomValues[i], (1 << (q + r)));
            unsigned int hashValue = Normal_APHash(h_newValues[i], (1 << (q + r)));
            unsigned int fq = (hashValue >> r) & LOW_BIT_MASK(q);
            unsigned int fr = hashValue & LOW_BIT_MASK(r);
            unsigned int fp = hashValue & LOW_BIT_MASK(r + q);
            printf("Quotient = %u \t Remainder = %u\t, Fingerprint = %u\n", fq, fr, fp);
            insertFailures++;
        }
    }
    printf("%d inserted values (old filter) were not found.\n", insertFailures);

//Lookup added batch values
    unsigned int* d_batchLookupReturnValues;
    cudaMalloc((void**) &d_batchLookupReturnValues, batchSize * sizeof(unsigned int));
    cudaMemset(d_batchLookupReturnValues, 0, batchSize * sizeof(unsigned int));

    CUDAErrorCheck();
    lookupTime = launchUnsortedLookups(d_qfilter, batchSize, d_newValues, d_batchLookupReturnValues);
    CUDAErrorCheck();

    //Transfer back results of lookups and check that they succeeded
    unsigned int* h_batchLookupReturnValues = new unsigned int[batchSize];
    for(int i = 0; i < batchSize; i++){
        h_batchLookupReturnValues[i] = 0;
    }
    cudaMemcpy(h_batchLookupReturnValues, d_batchLookupReturnValues, batchSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    insertFailures = 0;
    for(int i = 0; i < batchSize; i++){
        if (h_batchLookupReturnValues[i] == NOT_FOUND){
            printf("ERROR: %dth value not found.\n", i);
            printf("Value: %u \n", h_newValues[i]);
            //unsigned int hashValue = FNVhashGPU(h_newValues[i], (1 << (q + r)));
            unsigned int hashValue = Normal_APHash(h_newValues[i], (1 << (q + r)));
            unsigned int fq = (hashValue >> r) & LOW_BIT_MASK(q);
            unsigned int fr = hashValue & LOW_BIT_MASK(r);
            unsigned int fp = hashValue & LOW_BIT_MASK(r + q);
            printf("Quotient = %u \t Remainder = %u\t, Fingerprint = %u\n", fq, fr, fp);
            insertFailures++;
        }
    }
    printf("%d inserted values (new batch) were not found.\n", insertFailures);

    //Free Memory
    cudaFree(d_qfilter.table);
    delete[] h_randomValues;
    cudaFree(d_randomValues);
    delete[] h_newValues;
    cudaFree(d_newValues);
    cudaFree(d_lookupReturnValues);
    delete[] h_lookupReturnValues;
    cudaFree(d_batchLookupReturnValues);
    delete[] h_batchLookupReturnValues;
    cudaDeviceReset();

    return 0;
}
