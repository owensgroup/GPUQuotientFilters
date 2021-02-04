//bulkBuildSequentialShiftsPerf.cu
//Builds a quotient filter all at once, only looping as long as it takes to shift all elements to their final locations. This version also removes duplicates from the dataset.

#include <stdio.h>
#include <assert.h>
#include <string>
#include <cuda_profiler_api.h>

#include "../../mt19937ar.h"
#include "quotientFilter.cuh"

#ifndef LOW_BIT_MASK
#define LOW_BIT_MASK(n) ((1U << n) -1U)
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
    struct quotient_filter d_qfilter;
    initFilterGPU(&d_qfilter, q, r);
    cudaMemset(d_qfilter.table, 0, calcNumSlotsGPU(q, r) * sizeof(unsigned char));

    //Generate set of random numbers
    int numValues = alpha * (1 << q);
    unsigned int* h_randomValues = new unsigned int[numValues];
    init_genrand(time(NULL));   //initialize random number generator
    generateRandomNumbers(h_randomValues, numValues);
    unsigned int* d_randomValues;
    cudaMalloc((void**) &d_randomValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_randomValues, h_randomValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Build filter
    CUDAErrorCheck();
    float filterBuildTime = bulkBuildSequentialShifts(d_qfilter, numValues, d_randomValues, NoDuplicates);
    CUDAErrorCheck();
    printf("Insert rate = %f million ops/sec\n", numValues / filterBuildTime / 1000);
    //printf("%f\n", numValues / filterBuildTime / 1000);

    //Free memory
    delete[] h_randomValues;
    cudaFree(d_randomValues);
    cudaFree(d_qfilter.table);
    cudaDeviceReset();

    return 0;
}
