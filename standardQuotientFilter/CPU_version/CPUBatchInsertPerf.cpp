//CPUBatchInsertPerf.cpp

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "quotientFilter.h"
#include "../../mt19937ar.h"

long long unsigned int timespec2usec(struct timespec ts) 
{
    return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

void generateRandomNumbers(unsigned int *numberArray, unsigned int n)
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
    float alpha = atof(argv[3]);
    int batchSize = atoi(argv[4]);

    //Initialize filter:
    struct quotient_filter qfilterPerfTest;
    initFilter(&qfilterPerfTest, q, r);

    //Generate set of random numbers:
    unsigned int numValues = alpha * (1 << q);
    unsigned int *randomValues = new unsigned int[numValues];
    init_genrand(time(NULL));   //initialize random number generator
    generateRandomNumbers(randomValues, numValues);

    //Random Insertions
    struct timespec startTime, endTime;
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    for(int i = 0; i < numValues; i++){
        int slotValue =  insertItem(&qfilterPerfTest, randomValues[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    double insertRate = (double) numValues / (timespec2usec(endTime) - timespec2usec(startTime));

    //Additional batch of inserts
    unsigned int *batchValues = new unsigned int[batchSize];
    generateRandomNumbers(batchValues, batchSize);
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    for(int i = 0; i < batchSize; i++){
        int slotValue = insertItem(&qfilterPerfTest, batchValues[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    double batchInsertRate = (double) batchSize / (timespec2usec(endTime) - timespec2usec(startTime));
    printf("%f\n", batchInsertRate);

    delete[] randomValues;
    delete[] batchValues;

    return 0;
}
