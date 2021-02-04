//bulkBuildSegmentedLayoutPerf.cu

#include <stdio.h>
#include <assert.h>
#include <string>
#include <cuda_profiler_api.h>

#include "../../mt19937ar.h"
#include "quotientFilter.cuh"

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
    float filterBuildTime = bulkBuildSegmentedLayouts(d_qfilter, numValues, d_randomValues, NoDuplicates);
    CUDAErrorCheck();
    printf("Insert rate = %f million ops/sec\n", numValues / filterBuildTime / 1000);
    //printf("%f\n", numValues / filterBuildTime / 1000);

//Successful lookups
    unsigned int* d_lookupReturnValues;
    cudaMalloc((void**) &d_lookupReturnValues, numValues * sizeof(unsigned int));
    cudaMemset(d_lookupReturnValues, 0, numValues * sizeof(unsigned int));

    //Launch lookup kernel
    CUDAErrorCheck();
    float successfulLookupTime = launchUnsortedLookups(d_qfilter, numValues, d_randomValues, d_lookupReturnValues);
    CUDAErrorCheck();

    //Print timing results
    printf("Successful lookup rate = %f million ops/sec\n", numValues / successfulLookupTime / 1000);

//Random lookups
    //Generate values for random lookups
    unsigned int* h_randomLookupValues = new unsigned int[numValues];
    generateRandomNumbers(h_randomLookupValues, numValues);

    //Array of lookup values
    unsigned int* d_randomLookupValues;
    cudaMalloc((void**) &d_randomLookupValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_randomLookupValues, h_randomLookupValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Output array
    unsigned int* d_randomReturnValues;
    cudaMalloc((void**) &d_randomReturnValues, numValues * sizeof(unsigned int));
    cudaMemset(&d_randomReturnValues, 0, numValues * sizeof(unsigned int));

    //Launch lookup kernel
    CUDAErrorCheck();
    float randomLookupTime = launchUnsortedLookups(d_qfilter, numValues, d_randomLookupValues, d_randomReturnValues);
    CUDAErrorCheck();

    //Print timing results
    printf("Random lookup rate = %f million ops/sec\n", numValues / randomLookupTime / 1000);

    //Free memory
    delete[] h_randomValues;
    cudaFree(d_randomValues);
    cudaFree(d_qfilter.table);
    cudaFree(d_lookupReturnValues);
    delete[] h_randomLookupValues;
    cudaFree(d_randomLookupValues);
    cudaFree(d_randomReturnValues);
    cudaDeviceReset();

    return 0;
}
