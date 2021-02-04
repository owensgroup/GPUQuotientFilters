//merge2FiltersTest.cu

#include <stdio.h>
#include <assert.h>
#include <cuda_profiler_api.h>

#include "../../mt19937ar.h"
#include "quotientFilter.cuh"

#ifndef LOW_BIT_MASK
#define LOW_BIT_MASK(n) ((1U << n) -1U)
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

void generateRandomNumbers(unsigned int *numberArray, unsigned int n)
{
    for (int i = 0; i < n; i++){
        numberArray[i] = genrand_int32();
    }
}

int main(int argc, char* argv[])
{
    assert(argc == 6);
    int q = atoi(argv[1]);
    int r = atoi(argv[2]);
    float alpha1 = atof(argv[3]);   //initial fill % for filter 1
    float alpha2 = atof(argv[4]);   //initial fill % for filter 2
    bool NoDuplicates;
    std::string dup(argv[5]);
    if(dup == "Dup") NoDuplicates = false;
    else if(dup == "NoDup") NoDuplicates = true;
    else{
        printf("ERROR: Last argument should be Dup or NoDup. \nPlease indicate whether you want to leave duplicate items or remove them.");
        return 0;
    }

//Build the Two Filters
    //Initialize filter 1
    printf("Building filter with 2^%u slots and %u-bit remainders....\n", q, r);
    struct quotient_filter d_qfilter1;
    initFilterGPU(&d_qfilter1, q, r);
    cudaMemset(d_qfilter1.table, 0, calcNumSlotsGPU(q, r) * sizeof(unsigned char));

    //Generate first set of random numbers
    printf("Generating random numbers to fill the filter to %3.1f percent full...\n", (alpha1 * 100));
    int numValues1 = alpha1 * (1 << q);
    unsigned int* h_filter1Values = new unsigned int[numValues1];
    init_genrand(time(NULL));   //initialize random number generator
    generateRandomNumbers(h_filter1Values, numValues1);
    printf("%d random numbers generated.\n", numValues1);
    unsigned int* d_filter1Values;
    cudaMalloc((void**) &d_filter1Values, numValues1 * sizeof(unsigned int));
    cudaMemcpy(d_filter1Values, h_filter1Values, numValues1 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Build filter 1
    CUDAErrorCheck();
    float filter1BuildTime = bulkBuildSegmentedLayouts(d_qfilter1, numValues1, d_filter1Values, true);
    CUDAErrorCheck();

    //Initialize filter 2
    printf("Building filter with 2^%u slots and %u-bit remainders....\n", q, r);
    struct quotient_filter d_qfilter2;
    initFilterGPU(&d_qfilter2, q, r);
    cudaMemset(d_qfilter2.table, 0, calcNumSlotsGPU(q, r) * sizeof(unsigned char));

    //Generate second set of random numbers
    printf("Generating random numbers to fill the filter to %3.1f percent full...\n", (alpha2 * 100));
    int numValues2 = alpha2 * (1 << q);
    unsigned int* h_filter2Values = new unsigned int[numValues2];
    generateRandomNumbers(h_filter2Values, numValues2);
    printf("%d random numbers generated.\n", numValues2);
    unsigned int* d_filter2Values;
    cudaMalloc((void**) &d_filter2Values, numValues2 * sizeof(unsigned int));
    cudaMemcpy(d_filter2Values, h_filter2Values, numValues2 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Build filter 2
    CUDAErrorCheck();
    float filter2BuildTime = bulkBuildSegmentedLayouts(d_qfilter2, numValues2, d_filter2Values, true);
    CUDAErrorCheck();

//Merge Filters
    CUDAErrorCheck();
    float filterMergeTime = mergeTwoFilters(d_qfilter1, d_qfilter2, NoDuplicates);
    CUDAErrorCheck();
    //d_qfilter1 now holds the result of merging the two filters
    printf("Merge througput = %f million items/sec\n", (numValues1 + numValues2) / filterMergeTime / 1000);

//Lookups to Verify Correctness
    //Filter 1 Values:
    unsigned int* d_lookupOutput1;
    cudaMalloc((void**) &d_lookupOutput1, numValues1 * sizeof(unsigned int));
    cudaMemset(d_lookupOutput1, 0, numValues1 * sizeof(unsigned int));
    CUDAErrorCheck();
    float lookupTime = launchUnsortedLookups(d_qfilter1, numValues1, d_filter1Values, d_lookupOutput1);
    CUDAErrorCheck();

   //Transfer back results of lookups and check that they succeeded
    unsigned int* h_lookupOutput1 = new unsigned int[numValues1];
    for(int i = 0; i < numValues1; i++){
        h_lookupOutput1[i] = 0;
    }
    cudaMemcpy(h_lookupOutput1, d_lookupOutput1, numValues1 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    int insertFailures = 0;
    for(int i = 0; i < numValues1; i++){
        if (h_lookupOutput1[i] == NOT_FOUND){
            printf("ERROR: %dth value not found.\n", i);
            printf("Value: %u \n", h_filter1Values[i]);
            //unsigned int hashValue = FNVhashGPU(h_filter1Values[i], (1 << (q + r)));
            unsigned int hashValue = Normal_APHash(h_filter1Values[i], (1 << (q + r)));
            unsigned int fq = (hashValue >> r) & LOW_BIT_MASK(q);
            unsigned int fr = hashValue & LOW_BIT_MASK(r);
            unsigned int fp = hashValue & LOW_BIT_MASK(r + q);
            printf("Quotient = %u \t Remainder = %u\t, Fingerprint = %u\n", fq, fr, fp);
            insertFailures++;
        }
    }
    printf("%d inserted values from filter 1 were not found.\n", insertFailures);

    //Filter 2 Values:
    unsigned int* d_lookupOutput2;
    cudaMalloc((void**) &d_lookupOutput2, numValues2 * sizeof(unsigned int));
    cudaMemset(d_lookupOutput2, 0, numValues2 * sizeof(unsigned int));
    CUDAErrorCheck();
    lookupTime = launchUnsortedLookups(d_qfilter2, numValues2, d_filter2Values, d_lookupOutput2);
    CUDAErrorCheck();

   //Transfer back results of lookups and check that they succeeded
    unsigned int* h_lookupOutput2 = new unsigned int[numValues2];
    for(int i = 0; i < numValues2; i++){
        h_lookupOutput2[i] = 0;
    }
    cudaMemcpy(h_lookupOutput2, d_lookupOutput2, numValues2 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    insertFailures = 0;
    for(int i = 0; i < numValues2; i++){
        if (h_lookupOutput2[i] == NOT_FOUND){
            printf("ERROR: %dth value not found.\n", i);
            printf("Value: %u \n", h_filter2Values[i]);
            //unsigned int hashValue = FNVhashGPU(h_filter2Values[i], (1 << (q + r)));
            unsigned int hashValue = Normal_APHash(h_filter2Values[i], (1 << (q + r)));
            unsigned int fq = (hashValue >> r) & LOW_BIT_MASK(q);
            unsigned int fr = hashValue & LOW_BIT_MASK(r);
            unsigned int fp = hashValue & LOW_BIT_MASK(r + q);
            printf("Quotient = %u \t Remainder = %u\t, Fingerprint = %u\n", fq, fr, fp);
            insertFailures++;
        }
    }
    printf("%d inserted values from filter 2 were not found.\n", insertFailures);

    //Free Memory
    cudaFree(d_qfilter1.table);
    cudaFree(d_qfilter2.table);
    delete[] h_filter1Values;
    cudaFree(d_filter1Values);
    delete[] h_filter2Values;
    cudaFree(d_filter2Values);
    delete[] h_lookupOutput1;
    cudaFree(d_lookupOutput1);
    delete[] h_lookupOutput2;
    cudaFree(d_lookupOutput2);
    cudaDeviceReset();

    return 0;
}
