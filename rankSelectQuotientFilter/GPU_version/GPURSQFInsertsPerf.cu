//GPURSQFInsertsPerf.cu

#include <stdio.h>
#include <assert.h>

#include "../../mt19937ar.h"
#include "RSQF.cuh"

void generateRandomNumbers(unsigned int *numberArray, unsigned int n)
{
    for (int i = 0; i < n; i++){
        numberArray[i] = genrand_int32();
    }   
}

int main(int argc, char* argv[])
{
    assert(argc == 3); 
    int q = atoi(argv[1]);
    float alpha = atof(argv[2]);

//Initialize the filter:
    struct countingQuotientFilterGPU test_cqf_gpu;
    initCQFGPU(&test_cqf_gpu, q); 

    //Generate random numbers:
    unsigned int numValues = alpha * (1 << q); 
    unsigned int* h_randomValues = new unsigned int[numValues];
    init_genrand(time(NULL));       //initialize random number generator
    generateRandomNumbers(h_randomValues, numValues);
    unsigned int* d_randomValues;
    cudaMalloc((void**) &d_randomValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_randomValues, h_randomValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

//Inserts
    //Allocate output array
    int* d_insertReturnValues;
    cudaMalloc((void**) &d_insertReturnValues, numValues * sizeof(int));

    //Insert kernel
    float insertTime = insertGPU(test_cqf_gpu, numValues, d_randomValues, d_insertReturnValues);
    printf("%f\n", numValues / insertTime / 1000);

    //Free Memory
    cudaFree(test_cqf_gpu.blocks);
    delete[] h_randomValues;
    cudaFree(d_randomValues);
    cudaFree(d_insertReturnValues);
    cudaDeviceReset();

    return 0;
}
