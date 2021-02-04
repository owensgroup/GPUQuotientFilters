//superclusterTest.cu

#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <cuda_profiler_api.h>

#include "../../mt19937ar.h"
#include "quotientFilter.cuh"

#ifndef NOT_FOUND
#define NOT_FOUND UINT_MAX
#endif

void generateRandomNumbers(unsigned int *numberArray, int n)
{
    for (int i = 0; i < n; i++){
        numberArray[i] = genrand_int32();
    }
}

void generateNewRandomNumbers(unsigned int *newNumberArray, unsigned int *comparisonNumberArray, int n)
{
    generateRandomNumbers(newNumberArray, n);
    unsigned int numNew = 0;
    while (numNew < n){
        for (int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
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

int main(int argc, char* argv[])
{
    assert(argc == 4);
    unsigned int q = atoi(argv[1]);
    unsigned int r = atoi(argv[2]);
    float alpha = atof(argv[3]);

    //Initialize filter
    printf("Building filter with 2^%u slots and %u-bit remainders....\n", q, r);
    struct quotient_filter d_qfilter;
    initFilterGPU(&d_qfilter, q, r);
    cudaMemset(d_qfilter.table, 0, calcNumSlotsGPU(q, r) * sizeof(unsigned char));

    //Generate random numbers to build filter
    printf("Generating random numbers to fill the filter to %3.1f percent full...\n", (alpha * 100));
    int numValues = alpha * (1 << q);
    unsigned int* h_randomValues = new unsigned int[numValues];
    init_genrand(time(NULL));   //initialize random number generator
    generateRandomNumbers(h_randomValues, numValues);
    printf("%d random numbers generated.\n", numValues);
    unsigned int* d_randomValues;
    cudaMalloc((void**) &d_randomValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_randomValues, h_randomValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

//Insert items
    //Build filter on GPU
    CUDAErrorCheck();
    float filterBuildTime = insert(d_qfilter, numValues, d_randomValues);
    CUDAErrorCheck();

//Test that lookups on all inserted items succeed
    //Array of lookup values
    unsigned int* d_successfulLookupValues;
    cudaMalloc((void**) &d_successfulLookupValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_successfulLookupValues, h_randomValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Allocate output array
    unsigned int* d_returnValues;
    cudaMalloc((void**) &d_returnValues, numValues * sizeof(unsigned int));
    cudaMemset(d_returnValues, 0, numValues * sizeof(unsigned int));

    CUDAErrorCheck();
    float lookupTime = launchUnsortedLookups(d_qfilter, numValues, d_successfulLookupValues, d_returnValues);
    CUDAErrorCheck();

    //Transfer back results of lookups and check that they succeeded
    unsigned int* h_returnValues = new unsigned int[numValues];
    for(int i = 0; i < numValues; i++){
        h_returnValues[i] = 0;
    }
    cudaMemcpy(h_returnValues, d_returnValues, numValues * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    int insertFailures = 0;
    for(int i = 0; i < numValues; i++){
        if (h_returnValues[i] == NOT_FOUND){
            insertFailures++;
        }
    }
    printf("%d inserted values were not found.\n", insertFailures);

//Calculate false positive rate
    //Generate array of values not in filter
    unsigned int* h_failedLookupValues = new unsigned int[numValues];
    generateNewRandomNumbers(h_failedLookupValues, h_randomValues, numValues);
    unsigned int* d_failedLookupValues;
    cudaMalloc((void**) &d_failedLookupValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_failedLookupValues, h_failedLookupValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Allocate output array
    unsigned int* d_failureReturnValues;
    cudaMalloc((void**) &d_failureReturnValues, numValues * sizeof(unsigned int));
    cudaMemset(d_failureReturnValues, 0, numValues * sizeof(unsigned int));

    CUDAErrorCheck();
    lookupTime = launchUnsortedLookups(d_qfilter, numValues, d_failedLookupValues, d_failureReturnValues);
    CUDAErrorCheck();

    //Transfer data back and find false positive rate
    unsigned int* h_failureReturnValues = new unsigned int[numValues];
    for(int i = 0; i < numValues; i++){
        h_failureReturnValues[i] = 0;
    }
    cudaMemcpy(h_failureReturnValues, d_failureReturnValues, numValues * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    int falsePositives = 0;
    for(int i = 0; i < numValues; i++){
        if(h_failureReturnValues[i] != NOT_FOUND){
            falsePositives++;
        }
    }
    printf("False positive rate: %f \n", ((float)falsePositives)/numValues);
//    printf("%f\n", ((float)falsePositives)/numValues);

    //Free Memory
    cudaFree(d_qfilter.table);
    cudaFree(d_successfulLookupValues);
    cudaFree(d_returnValues);
    delete[] h_returnValues;
    cudaFree(d_randomValues);
    delete[] h_randomValues;
    cudaFree(d_failedLookupValues);
    cudaFree(d_failureReturnValues);
    delete[] h_failedLookupValues;
    delete[] h_failureReturnValues;
    cudaDeviceReset();

    return 0;
}
