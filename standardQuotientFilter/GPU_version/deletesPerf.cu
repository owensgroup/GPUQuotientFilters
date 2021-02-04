//deletesPerf.cu

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
    unsigned int* h_randomValues = new unsigned int[numValues];
    init_genrand(time(NULL));   //initialize random number generator
    generateRandomNumbers(h_randomValues, numValues);
    unsigned int* d_randomValues;
    cudaMalloc((void**) &d_randomValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_randomValues, h_randomValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

//Random Inserts
    float filterBuildTime = bulkBuildSegmentedLayouts(d_qfilter, numValues, d_randomValues, true);

//Delete Some of the Items
    unsigned int* d_deleteValues;
    cudaMalloc((void**) &d_deleteValues, batchSize * sizeof(unsigned int));
    cudaMemcpy(d_deleteValues, h_randomValues, batchSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Delete kernel
    float deleteTime = superclusterDeletes(d_qfilter, batchSize, d_deleteValues);
    printf("Delete rate = %f million ops/sec\n", batchSize / deleteTime / 1000);
    //printf("%f\n", batchSize / deleteTime / 1000);

    //Free Memory
    cudaFree(d_qfilter.table);
    cudaFree(d_randomValues);
    delete[] h_randomValues;
    cudaFree(d_deleteValues);
    cudaDeviceReset();

    return 0;
}
