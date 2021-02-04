//bulkBuildViaMergingTest.cu
//Builds a quotient filter all at once, removing duplicates.

#include <stdio.h>
#include <assert.h>
#include <string>

#include "../../mt19937ar.h"
#include "quotientFilter.cuh"

#ifndef LOW_BIT_MASK
#define LOW_BIT_MASK(n) ((1U << n) -1U)
#endif

#ifndef NOT_FOUND
#define NOT_FOUND UINT_MAX
#endif

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

int main(int argc, char* argv[])
{
    assert(argc == 5);
    unsigned int q = atoi(argv[1]);
    unsigned int r = atoi(argv[2]);
    float alpha = atof(argv[3]);
    bool NoDuplicates;
    std::string dup(argv[4]);
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
    float filterBuildTime = bulkBuildParallelMerging(d_qfilter, numValues, d_randomValues, NoDuplicates);
    CUDAErrorCheck();

    //Check filter accuracy
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
            unsigned int hashValue = Normal_APHash(h_randomValues[i], (1 << (q + r)));
            //unsigned int hashValue = FNVhashGPU(h_randomValues[i], (1 << (d_qfilter.qbits + d_qfilter.rbits)));
            unsigned int fq = (hashValue >> d_qfilter.rbits) & LOW_BIT_MASK(d_qfilter.qbits);
            unsigned int fr = hashValue & LOW_BIT_MASK(d_qfilter.rbits);
            printf("quotient: %u\t remainder:%u\n", fq, fr);
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

    //Perform lookups to find false positive rate
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

    //Free memory
    delete[] h_randomValues;
    cudaFree(d_randomValues);
    cudaFree(d_qfilter.table);
    delete[] h_lookupReturnValues;
    cudaFree(d_lookupReturnValues);
    delete[] h_failureReturnValues;
    cudaFree(d_failureReturnValues);
    cudaDeviceReset();
    return 0;
}
