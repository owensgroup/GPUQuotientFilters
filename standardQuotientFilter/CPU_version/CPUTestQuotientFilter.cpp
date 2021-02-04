//CPUTestQuotientFilter.cpp

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "quotientFilter.h"
#include "../../mt19937ar.h"

void generateRandomNumbers(unsigned int *numberArray, unsigned int n)
{
    for (int i = 0; i < n; i++){
        numberArray[i] = genrand_int32();
    }
}

void generateNewRandomNumbers(unsigned int *newNumberArray, unsigned int *comparisonNumberArray, unsigned int n)
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

void printQF(struct quotient_filter* qf)
{
    for(int i = 0; i < ((1 << qf->qbits) * 1.1); i++){
        printElement(qf, i);
    }
}

int main(int argc, char* argv[])
{
    assert(argc == 4);
    int q = atoi(argv[1]);
    int r = atoi(argv[2]);
    float alpha = atof(argv[3]);

    //Initialize the filter:
    struct quotient_filter qfilterTest1;
    printf("Building filter with 2^%u slots and %u-bit remainders....\n", q, r);
    initFilter(&qfilterTest1, q, r);

    //Generate random numbers:
    printf("Generating random numbers to fill the filter to %3.1f percent full...\n", (alpha * 100));
    unsigned int numValues = alpha * (1 << q);
    unsigned int *randomValues = new unsigned int[numValues];
    init_genrand(time(NULL));	//initialize random number generator
    generateRandomNumbers(randomValues, numValues);
    printf("%d random numbers generated.\n", numValues);

    //Insert values into quotient filter:
    for (int i = 0; i < numValues; i++){
        int slotValue = insertItem(&qfilterTest1, randomValues[i]);
    }

    //Test that lookups say items are actually there:
    int slotValue;
    int insertFailures = 0;
    for (int i = 0; i < numValues; i++){
        slotValue = mayContain(&qfilterTest1, randomValues[i]);
        if (slotValue == -1){
            printf("ERROR: Value not found!\n");
            insertFailures++;
        }
    }
    printf("%d inserted values were not found.\n", insertFailures);

    //printQF(&qfilterTest1);

    //Check some not inserted items to see the false positive rate:
    unsigned int *newNumbers = new unsigned int[numValues]; 
    generateNewRandomNumbers(newNumbers, randomValues, numValues);
    int falsePositives = 0;
    for (int i = 0; i < numValues; i++){
        slotValue = mayContain(&qfilterTest1, newNumbers[i]);
        if (slotValue != -1){
            falsePositives++; 
        }
    }
    printf("False positive rate: %f \n", ((float)falsePositives/numValues));

    delete[] randomValues;
    delete[] newNumbers;

    return 0;
}
