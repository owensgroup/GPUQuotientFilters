//merge2FiltersPerf.cu

#include <stdio.h>
#include <assert.h>
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
    struct quotient_filter d_qfilter1;
    initFilterGPU(&d_qfilter1, q, r);
    cudaMemset(d_qfilter1.table, 0, calcNumSlotsGPU(q, r) * sizeof(unsigned char));

    //Generate first set of random numbers
    int numValues1 = alpha1 * (1 << q);
    unsigned int* h_filter1Values = new unsigned int[numValues1];
    init_genrand(time(NULL));   //initialize random number generator
    generateRandomNumbers(h_filter1Values, numValues1);
    unsigned int* d_filter1Values;
    cudaMalloc((void**) &d_filter1Values, numValues1 * sizeof(unsigned int));
    cudaMemcpy(d_filter1Values, h_filter1Values, numValues1 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Build filter 1
    CUDAErrorCheck();
    float filter1BuildTime = bulkBuildSegmentedLayouts(d_qfilter1, numValues1, d_filter1Values, true);
    CUDAErrorCheck();

    //Initialize filter 2
    struct quotient_filter d_qfilter2;
    initFilterGPU(&d_qfilter2, q, r);
    cudaMemset(d_qfilter2.table, 0, calcNumSlotsGPU(q, r) * sizeof(unsigned char));

    //Generate second set of random numbers
    int numValues2 = alpha2 * (1 << q);
    unsigned int* h_filter2Values = new unsigned int[numValues2];
    generateRandomNumbers(h_filter2Values, numValues2);
    unsigned int* d_filter2Values;
    cudaMalloc((void**) &d_filter2Values, numValues2 * sizeof(unsigned int));
    cudaMemcpy(d_filter2Values, h_filter2Values, numValues2 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Build filter 2
    CUDAErrorCheck();
    float filter2BuildTime = bulkBuildSegmentedLayouts(d_qfilter2, numValues2, d_filter2Values, true);
    CUDAErrorCheck();

//Merge Filters
    CUDAErrorCheck();
    float filterMergeTime = mergeTwoFilters(d_qfilter1, d_qfilter2, NoDuplicates);  //d_qfilter1 now holds the result of merging the two filters
    CUDAErrorCheck();

    //printf("%f\n", (numValues1 + numValues2) / filterMergeTime / 1000);
    printf("Merge througput = %f million items/sec\n", (numValues1 + numValues2) / filterMergeTime / 1000);

    //Free Memory
    cudaFree(d_qfilter1.table);
    cudaFree(d_qfilter2.table);
    delete[] h_filter1Values;
    cudaFree(d_filter1Values);
    delete[] h_filter2Values;
    cudaFree(d_filter2Values);
    cudaDeviceReset();

    return 0;
}
