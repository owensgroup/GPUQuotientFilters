//testGPURSQF.cu

#include "RSQF.cuh"
#include "../../mt19937ar.h"

#include <algorithm>
#include <stdio.h>
#include <assert.h>

#define LOW_BIT_MASK(n) ((1U << n) - 1U)

void generateRandomNumbers(unsigned int *numberArray, unsigned int n)
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
    assert(argc == 3);
    int q = atoi(argv[1]);
    float alpha = atof(argv[2]);

//Initialize the filter:
    struct countingQuotientFilterGPU test_cqf_gpu;
    printf("Building filter with 2^%u slots and %u-bit remainders....\n", q, RBITS); 
    initCQFGPU(&test_cqf_gpu, q);

    //Generate random numbers:
    printf("Generating random numbers to fill the filter to %3.1f percent full...\n", (alpha * 100));
    unsigned int numValues = alpha * (1 << q);
    unsigned int* h_randomValues = new unsigned int[numValues];
    init_genrand(time(NULL));       //initialize random number generator
    generateRandomNumbers(h_randomValues, numValues);
    printf("%d random numbers generated.\n", numValues);
    unsigned int* d_randomValues;
    cudaMalloc((void**) &d_randomValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_randomValues, h_randomValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Allocate output array
    int* d_insertReturnValues;
    cudaMalloc((void**) &d_insertReturnValues, numValues * sizeof(int));

//Inserts
    float insertTime = insertGPU(test_cqf_gpu, numValues, d_randomValues, d_insertReturnValues);

    int* h_insertReturnValues = new int[numValues];
    cudaMemcpy(h_insertReturnValues, d_insertReturnValues, numValues * sizeof(int), cudaMemcpyDeviceToHost);

    //Check for overflow failures
    int overflow = 0;
    for(int i = 0; i < numValues; i++){
        if(h_insertReturnValues[i] < 0){
            printf("ERROR: insert for item %i, value=%u failed.\n", i, h_randomValues[i]);
            overflow++;
        }
    }
    printf("%i items overflowed.\n", overflow);
    if(overflow > 0){
        printf("Need to upsize filter.\n");
        return 0;
    }
    
    printf("Inserts complete.\n");

//Lookups
    unsigned int* d_successfulLookupValues;
    cudaMalloc((void**) &d_successfulLookupValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_successfulLookupValues, h_randomValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

   //Allocate output array
    int* d_successfulLookupReturnValues;
    cudaMalloc((void**) &d_successfulLookupReturnValues, numValues * sizeof(int));
    cudaMemset(d_successfulLookupReturnValues, 0, numValues * sizeof(int));

    float successfulLookupTime = launchLookups(test_cqf_gpu, numValues, d_successfulLookupValues, d_successfulLookupReturnValues);
    //float successfulLookupTime = launchUnsortedLookups(test_cqf_gpu, numValues, d_successfulLookupValues, d_successfulLookupReturnValues);

    //Transfer back results of lookups and check that they succeeded
    int* h_successfulLookupReturnValues = new int[numValues];
    for(int i = 0; i < numValues; i++){
        h_successfulLookupReturnValues[i] = 0;
    }
    cudaMemcpy(h_successfulLookupReturnValues, d_successfulLookupReturnValues, numValues * sizeof(int), cudaMemcpyDeviceToHost);
    int insertFailures = 0;
    unsigned int* failedHashes = new unsigned int[numValues];
    for(int i = 0; i < numValues; i++){
        if (h_successfulLookupReturnValues[i] == -1){
            failedHashes[insertFailures] = Normal_APHashGPU(h_randomValues[i], (1 << (q + RBITS)));
            insertFailures++;
        }
    }
    printf("%d inserted values were not found.\n", insertFailures);

    if(insertFailures != 0){ 
        //print all inputs to reuse in debugging
        printf("input values:\n");
        for(int i = 0; i < numValues; i++){
            printf("%u, ", h_randomValues[i]);
        }
        printf("\n");
        std::sort(failedHashes, failedHashes + insertFailures);
        printf("Failed values:\n");
        for(int i = 0; i < insertFailures; i++){
            unsigned int quotient = (failedHashes[i] >> RBITS) & LOW_BIT_MASK(q);
            printf("quotient:%u\tblock:%u\tslot:%u\tremainder:%u\n", quotient, quotient / SLOTS_PER_BLOCK, quotient % SLOTS_PER_BLOCK, failedHashes[i] & LOW_BIT_MASK(RBITS));
        }
        printGPUFilter(&test_cqf_gpu);
    }

    //Calculate and print timing results
    printf("Successful lookup rate = %f million ops/sec\n", numValues / successfulLookupTime / 1000);
    printf("Insert throughput = %f million ops/sec\n", numValues / insertTime / 1000);

//Calculate false positive rate
    //Generate array of values not in filter
    unsigned int* h_failedLookupValues = new unsigned int[numValues];
    generateNewRandomNumbers(h_failedLookupValues, h_randomValues, numValues);
    unsigned int* d_failedLookupValues;
    cudaMalloc((void**) &d_failedLookupValues, numValues * sizeof(unsigned int));
    cudaMemcpy(d_failedLookupValues, h_failedLookupValues, numValues * sizeof(unsigned int), cudaMemcpyHostToDevice);

    //Allocate output array
    int* d_failureReturnValues;
    cudaMalloc((void**) &d_failureReturnValues, numValues * sizeof(int));
    cudaMemset(d_failureReturnValues, 0, numValues * sizeof(int));

    float failedLookupTime = launchLookups(test_cqf_gpu, numValues, d_failedLookupValues, d_failureReturnValues);
    //float failedLookupTime = launchUnsortedLookups(test_cqf_gpu, numValues, d_failedLookupValues, d_failureReturnValues);

    //Transfer data back and find false positive rate
    int* h_failureReturnValues = new int[numValues];
    for(int i = 0; i < numValues; i++){
        h_failureReturnValues[i] = 0;
    }   
    cudaMemcpy(h_failureReturnValues, d_failureReturnValues, numValues * sizeof(int), cudaMemcpyDeviceToHost);
    int falsePositives = 0;
    for(int i = 0; i < numValues; i++){
        if(h_failureReturnValues[i] != -1){
            falsePositives++;
        }
    }   
    printf("False positive rate: %f \n", ((float)falsePositives)/numValues);

    //Calculate and print timing results
    printf("Failed lookup rate = %f million ops/sec\n", numValues / failedLookupTime / 1000);

    //Free memory
    cudaFree(test_cqf_gpu.blocks);
    delete[] h_randomValues;
    cudaFree(d_randomValues);
    cudaFree(d_insertReturnValues);
    delete[] h_insertReturnValues;
    cudaFree(d_successfulLookupValues);
    cudaFree(d_successfulLookupReturnValues);
    delete[] h_successfulLookupReturnValues;
    delete[] failedHashes;
    cudaFree(d_failedLookupValues);
    cudaFree(d_failureReturnValues);
    delete[] h_failedLookupValues;
    delete[] h_failureReturnValues;
    cudaDeviceReset();
	
    return 0;
}
