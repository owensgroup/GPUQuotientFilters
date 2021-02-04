//CPURSQFLookupPerf.cpp

#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "RSQF.h"
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
    assert(argc == 3); 
    int q = atoi(argv[1]);
    float alpha = atof(argv[2]);

    //Initialize the filter:
    struct countingQuotientFilter test_cqf;
    initCQF(&test_cqf, q); 

    //Generate random numbers:
    unsigned int numValues = alpha * (1 << q); 
    unsigned int *randomValues = new unsigned int[numValues];
    init_genrand(time(NULL));       //initialize random number generator
    generateRandomNumbers(randomValues, numValues);

    //Random Insertions
    for(int i = 0; i < numValues; i++){
        int slotValue =  insert(&test_cqf, randomValues[i]);
    }   

    //Successful lookups
    struct timespec startTime, endTime;
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    for(int i = 0; i < numValues; i++){
        int slotValue = lookup(&test_cqf, randomValues[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    double successfulLookupRate = (double) numValues / (timespec2usec(endTime) - timespec2usec(startTime));
    printf("%f\n", successfulLookupRate);

    delete[] randomValues;

    return 0;
}

