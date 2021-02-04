//quotientFilter.cpp
/*
 * Copyright 2021 Regents of the University of California
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "quotientFilter.h"

#define LOW_BIT_MASK(n) ((1U << n) - 1U)

size_t calcNumSlots(unsigned int q, unsigned int r)
{
    size_t tableBits = (1 << q) * (r + 3);
    size_t tableSlots = tableBits/32;
    if ((tableSlots % 32) > 1) tableSlots += 1;
    return tableSlots * 1.1;	//allow an extra 10% for overflow
}

void initFilter(struct quotient_filter *qf, unsigned int q, unsigned int r)
{
    qf->qbits = q;
    qf->rbits = r;
    size_t slots = calcNumSlots(q, r);
    qf->table = (unsigned int*)calloc(slots, sizeof(int));
}

bool isOccupied(unsigned int element)
{
    return element & 4;
}

bool isContinuation(unsigned int element)
{
    return element & 2;
}

bool isShifted(unsigned int element)
{
    return element & 1;
}

bool isEmpty(unsigned int element)
{
    return ((element & 7) == 0);
}

unsigned int setOccupied(unsigned int element)
{
    return element | 4;
}

unsigned int clearOccupied(unsigned int element)
{
    return element & ~4;
}

unsigned int setContinuation(unsigned int element)
{
    return element | 2;
}

unsigned int clearContinuation(unsigned int element)
{
    return element & ~2;
}

unsigned int setShifted(unsigned int element)
{
    return element | 1;
}

unsigned int clearShifted(unsigned int element)
{
    return element & ~1;
}

unsigned int getRemainder(unsigned int element)
{
    return element >> 3;
}

unsigned int fnv_hash(unsigned int value, unsigned int tableSize)
{
    unsigned char p[4];
    p[0] = (value >> 24) & 0xFF;
    p[1] = (value >> 16) & 0xFF;
    p[2] = (value >> 8) & 0xFF;
    p[3] = value & 0xFF;

    unsigned int h = 2166136261;

    for (int i = 0; i < 4; i++){
        h = (h * 16777619) ^ p[i];
    }

    return h % tableSize;
}

unsigned int getElement(struct quotient_filter *qf, unsigned int index)
{
    unsigned int bitLocation = index * (qf->rbits + 3);
    unsigned int startSlot = bitLocation / 32;
    unsigned int slotPosition = bitLocation % 32;
    int spillover = slotPosition + (qf->rbits + 3) - 32;
    unsigned int element;
    if (spillover <= 0){
        element = (qf->table[startSlot] >> abs(spillover)) & LOW_BIT_MASK(qf->rbits + 3);
    }
    else{
        element = (qf->table[startSlot] << spillover) & LOW_BIT_MASK(qf->rbits + 3);
        unsigned int spilloverBits = qf->table[startSlot + 1] >> (32 - spillover);
        element = element | spilloverBits;
    }
    return element;
}

void printElement(struct quotient_filter *qf, unsigned int index)
{
    unsigned int element = getElement(qf, index);
    printf("%-10s%-15s%-20s%-20s%-20s\n", "Element", "Remainder", "is_occupied", "is_continuation", "is_shifted");
    printf("%-10u%-15u%-20u%-20u%-20u\n", index, getRemainder(element), isOccupied(element), isContinuation(element), isShifted(element));
}

void printQuotientFilter(struct quotient_filter *qf)
{
    int filterSize = (1 << qf->qbits) * 1.1;
    printf("Printing metadata and remainders:\n");
    for(int i = 0; i < filterSize/10; i++){
        for(int j = 0; j < 10; j++){
            int element = getElement(qf, 10*i + j);
            printf("%u \t", element & 7);
        }
        printf("\n");
        for(int j = 0; j < 10; j++){
            int element = getElement(qf, 10*i + j);
            printf("%u \t", getRemainder(element));
        }
        printf("\n --------------------------------------------------------------------- \n");
    }
    printf("\n");
}

void setElement(struct quotient_filter *qf, unsigned int index, unsigned int value)
{
    unsigned int bitLocation = index * (qf->rbits + 3);
    unsigned int startSlot = bitLocation / 32;
    unsigned int slotPosition = bitLocation % 32;
    int spillover = slotPosition + (qf->rbits + 3) - 32;
    if (spillover <= 0){
        qf->table[startSlot] &= ~(LOW_BIT_MASK(qf->rbits + 3) << -spillover);
        qf->table[startSlot] |= value << -spillover;
    }
    else{
        qf->table[startSlot] &= ~(LOW_BIT_MASK(qf->rbits + 3 - spillover));
        qf->table[startSlot] |= value >> spillover;
        qf->table[startSlot + 1] &= LOW_BIT_MASK(32 - spillover);
        qf->table[startSlot + 1] |= ((value & LOW_BIT_MASK(spillover)) << (32 - spillover));
    }
}

int findRunStart(struct quotient_filter *qf, unsigned int fq)
{
    //start bucket is fq
    int b = fq;
    //find beginning of cluster:
    while(isShifted(getElement(qf, b))){
        b--;
    }

    //find start of run we're interested in:
    //slot counter starts at beginning of cluster
    int s = b;
    while(b != fq){
        do{
            s++;
        }while(isContinuation(getElement(qf,s)));   //find end of current run
        do{
            b++;
        }while(!isOccupied(getElement(qf,b)));  //count number of runs passed
    }

    //now s is first value in correct run
    return s;
}


int mayContain(struct quotient_filter *qf, unsigned int value)
{
    //returns -1 if value is not in the filter, and returns the location of the remainder if it is probably in the filter

    //calculate fingerprint
    unsigned int hashValue = fnv_hash(value, (1 << (qf->qbits + qf->rbits)));

    //hash to get quotient and remainder
    unsigned int fq = (hashValue >> qf->rbits) & LOW_BIT_MASK(qf->qbits);
    unsigned int fr = hashValue & LOW_BIT_MASK(qf->rbits);

    unsigned int element = getElement(qf, fq);
	
    if(!isOccupied(element)){
        return -1;
    }

    int s = findRunStart(qf, fq);

    //search through elements in run
    do{
        unsigned int remainder = getRemainder(getElement(qf,s));
        if(remainder == fr){
            return s;
        }
        else if(remainder > fr){
            return -1;
        }
        s++;
    }while(isContinuation(getElement(qf, s)));

    return -1;
}

void insertItemHere(struct quotient_filter *qf, unsigned int index, unsigned int value)
{
    unsigned int previousElement;
    unsigned int newElement = value;
    bool empty = false;	

    while(!empty){
        previousElement = getElement(qf, index);
        empty = isEmpty(previousElement);

        previousElement = setShifted(previousElement);

        if(isOccupied(previousElement)){
            //Need to preserve correct is_occupied bits
            previousElement = clearOccupied(previousElement);
            newElement = setOccupied(newElement);
        }

        setElement(qf, index, newElement);
        newElement = previousElement;
        index++;
    }
}

unsigned int insertItem(struct quotient_filter *qf, unsigned int value)
{
    //returns the location of inserted element

    //calculate fingerprint
    unsigned int hashValue = fnv_hash(value, (1 << (qf->qbits + qf->rbits)));

    //hash to get quotient and remainder
    unsigned int fq = (hashValue >> qf->rbits) & LOW_BIT_MASK(qf->qbits);
    unsigned int fr = hashValue & LOW_BIT_MASK(qf->rbits);

    unsigned int canonElement = getElement(qf, fq);
    unsigned int newElement = fr << 3;

    //meta-data bits: is_occupied  is_continuation  is_shifted
    //bits in each element left to right: r bits of remainder, is_occupied, is_continuation, is_shifted

    //simplest case: if empty, insert it
    if(isEmpty(canonElement)){
        //set filter element value and is_occupied
        setElement(qf, fq, setOccupied(newElement));
        return fq;
    }

    if(!isOccupied(canonElement)){
        //set is_occupied to show that there is now a run for this slot
        setElement(qf, fq, setOccupied(canonElement));
    }

    //Find beginning of item's run
    int runStart = findRunStart(qf, fq);
    int s = runStart;

    if(isOccupied(canonElement)){
    //If slot already has a run, search through its elements.
        do{
            unsigned int remainder = getRemainder(getElement(qf, s));
            if(remainder == fr){
                return s;
            }
            else if(remainder > fr){
                //s now points to where item goes
                break;
            }
            s++;
        }while(isContinuation(getElement(qf, s)));

        if(s == runStart){
            //The new element is now the start of the run, but we must move old start over, so it will be continuation
            unsigned int oldStartElement = getElement(qf, runStart);
            setElement(qf, runStart, setContinuation(oldStartElement));
        }
        else{
            //New element is not the start, so set its continuation bit
            newElement = setContinuation(newElement);
        }
    }

    if(s != fq){
        //If it's not being inserted into the canonical slot, the element is shifted.
        newElement = setShifted(newElement);
    }

    insertItemHere(qf, s, newElement);
    return s;
}
