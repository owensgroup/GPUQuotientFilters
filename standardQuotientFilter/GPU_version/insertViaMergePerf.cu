//insertViaMergePerf.cu

#include <stdio.h>
#include <assert.h>
#include <cuda_profiler_api.h>

#include "../../mt19937ar.h"
#include "quotientFilter.cuh"

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
    float filterBuildTime = bulkBuildSegmentedLayouts(d_qfilter, numValues, d_randomValues, true);
    CUDAErrorCheck();

    //Generate new set of random numbers to insert
    unsigned int* h_newValues = new unsigned int[batchSize];
    generateRandomNumbers(h_newValues, batchSize);
    unsigned int* d_newValues;
    cudaMalloc((void**) &d_newValues, batchSize * sizeof(unsigned int));
    cudaMemcpy(d_newValues, h_newValues, batchSize * sizeof(unsigned int), cudaMemcpyHostToDevice);

//Read out old values and rebuild filter with new values added
    CUDAErrorCheck();
    float filterMergeTime = insertViaMerge(d_qfilter, d_randomValues, numValues, d_newValues, batchSize, NoDuplicates);
    CUDAErrorCheck();

    //printf("%f\n", batchSize / filterMergeTime / 1000);
    printf("Insert rate = %f million ops/sec\n", batchSize / filterMergeTime / 1000);

    //Free Memory
    cudaFree(d_qfilter.table);
    delete[] h_randomValues;
    cudaFree(d_randomValues);
    delete[] h_newValues;
    cudaFree(d_newValues);
    cudaDeviceReset();

    return 0;
}
