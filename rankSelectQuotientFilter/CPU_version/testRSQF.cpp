//testRSQF.cpp

#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "RSQF.h"
#include "../../mt19937ar.h"

#include <algorithm>

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
    struct countingQuotientFilter test_cqf;
    printf("Building filter with 2^%u slots and %u-bit remainders....\n", q, RBITS); 
    initCQF(&test_cqf, q); 

    //Generate random numbers:
    printf("Generating random numbers to fill the filter to %3.1f percent full...\n", (alpha * 100));
    unsigned int numValues = alpha * (1 << q);
    unsigned int *randomValues = new unsigned int[numValues];
    init_genrand(time(NULL));       //initialize random number generator
    generateRandomNumbers(randomValues, numValues);
    printf("%d random numbers generated.\n", numValues);

    //Insert values into quotient filter:
    int overflow = 0;
    for (int i = 0; i < numValues; i++){
        int insertSlot = insert(&test_cqf, randomValues[i]);
        if(insertSlot < 0){
            printf("ERROR: insert for item %i, value=%u failed.\n", i, randomValues[i]);
            overflow++;
        }
    }
    printf("%i items overflowed.\n", overflow);
    if(overflow > 0){
        printf("Need to upsize filter.\n");
        return 0;
    }

//	printFilter(&test_cqf);
    printf("Inserts complete.\n");

    int maxOffset = 0;
    for(int i = 0; i < calcNumBlocks(q); i++){
        if(test_cqf.blocks[i].offset > maxOffset){
            maxOffset = test_cqf.blocks[i].offset;
        }
    }
    printf("maximum offset value: %i\n", maxOffset);
    
    //Test that lookups say items are actually there:
    int slotValue;
    int insertFailures = 0;
    unsigned int *failedHashes = new unsigned int[numValues];
    for (int i = 0; i < numValues; i++){
        slotValue = lookup(&test_cqf, randomValues[i]);
        if (slotValue == -1){
            printf("ERROR: Value not found!\n");
            failedHashes[insertFailures] = Normal_APHash(randomValues[i], (1 << (q + RBITS)));
            unsigned int quotient = (failedHashes[insertFailures] >> RBITS) & LOW_BIT_MASK(q);
            printf("quotient:%u\tblock:%u\tslot:%u\tremainder:%u\n", quotient, quotient / SLOTS_PER_BLOCK, quotient % SLOTS_PER_BLOCK, failedHashes[insertFailures] & LOW_BIT_MASK(RBITS));
            insertFailures++;
        }
    }
    printf("%d inserted values were not found.\n", insertFailures);

    if(insertFailures != 0){
        for(int i = 0; i < numValues; i++){
            printf("%u, ", randomValues[i]);
        }
        printf("\n");
        std::sort(failedHashes, failedHashes + insertFailures);
        printf("Failed values:\n");
        for(int i = 0; i < insertFailures; i++){
            unsigned int quotient = (failedHashes[i] >> RBITS) & LOW_BIT_MASK(q);
            printf("quotient:%u\tblock:%u\tslot:%u\tremainder:%u\n", quotient, quotient / SLOTS_PER_BLOCK, quotient % SLOTS_PER_BLOCK, failedHashes[i] & LOW_BIT_MASK(RBITS));
	}
        printFilter(&test_cqf);
    }

    //Calculate false positive rate
    //Generate new items not in filter
    unsigned int *newNumbers = new unsigned int[numValues];
    generateNewRandomNumbers(newNumbers, randomValues, numValues);
    int falsePositives = 0;
    for(int i = 0; i < numValues; i++){
        int slotValue = lookup(&test_cqf, newNumbers[i]);
        if(slotValue != -1){
            falsePositives++; 
        }
    }
    printf("False positive rate: %f \n", ((float)falsePositives/numValues));

    delete[] randomValues;
    delete[] newNumbers;
    delete[] failedHashes;

    return 0;
}
