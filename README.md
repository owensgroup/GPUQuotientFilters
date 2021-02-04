# Overview
This repo includes GPU implementations of two different quotient filters: 

1. Standard Quotient Filter (SQF), originally described by [Bender, et al.](http://vldb.org/pvldb/vol5/p1627_michaelabender_vldb2012.pdf)
2. Rank-and-Select-Based Quotient Filter (RSQF), originally described by [Pandey, et al.](https://dl.acm.org/doi/10.1145/3035918.3035963)

## Paper
[Afton Geil, Martin Farach-Colton and John D. Owens, "Quotient Filters: Approximate Membership Queries on the GPU," 2018 IEEE International Parallel and Distributed Processing Symposium (IPDPS), Vancouver, BC, 2018, pp. 451-462, doi: 10.1109/IPDPS.2018.00055.](https://escholarship.org/uc/item/3v12f7dn)

## Building The Code
1. `mkdir build`
2. `cmake ..`
3. `make`
4. `make test` <- This only checks whether all of the verification tests run, not whether the outputs are correct. To check correctness, run the tests individually. (e.g.: `./bin/GPUSuperclusterVerificationTest 15 5 .6`) 

## Performance Testing
You can generate the throughput data from the paper using the `*Perf.cu` files, which will output the throughput for the operation of interest (e.g. delete throughput for `deletesPerf.cu`) in millions of ops per second. Command line inputs for these tests are `quotientBits remainderBits fillFraction`. For tests that use a batch size, there is a fourth `batchSize` input. For bulk build tests, specify whether or not to deduplicate inputs before build using `NoDup` (remove duplicate values) or `Dup` (allow duplicate values).

All tests generate random numbers (generated using the [Mersenne Twister RNG](http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/emt.html)) to use as inputs to the quotient filters.

\*\*\***Note:** *The implementation of lookups has changed since the paper was published.* At the time, I overlooked the necessity of tracking the items' orginal indices and sorting the outputs in order to match lookup results to the inputs, which I had sorted by hash value to achieve greater locality in the lookup kernel itself. After this correction, presorting still results in a higher throughput for the RSQF, due to the way the metadata values are shared across a block of slots; however, for the SQF, the cost of the additional bookkeeping outweighs the benefits of greater locality in accessing the data structure.

## GPU Quotient Filter Operations
### Standard Quotient Filter
Code for the GPU implementation of the standard quotient filter is `standardQuotientFilter/GPU_version/quotientFilter.cu`.

#### Initialize Filter
`__host__ void initFilterGPU(struct quotient_filter *qf, unsigned int q, unsigned int r);`

To initialize a `quotient_filter` struct, use this function, where `q` is the number of quotient bits (filter size is 2^q), and `r` is the number of remainder bits (bits stored in the filter per item).

Unfortunately, in order to increase parallelism in filter operations, ranges of the sizes and false positive rates for my implementation of the standard quotient filter are quite limited. The SQF code will only work for quotient and remainder sizes such that `(r + 3) % 8 = 0`, and `q + r < 32`. I would use r = 5, which is a false positive rate of ~3.1%. The code should also work for r = 13, or a false positive rate of ~0.012%, if q < 19.

#### Build Options
`__host__ float bulkBuildParallelMerging(quotient_filter qfilter, int numValues, unsigned int* d_insertValues, bool NoDuplicates);`

`__host__ float bulkBuildEarlyExit(quotient_filter qfilter, int numValues, unsigned int* d_insertValues, bool NoDuplicates);`

`__host__ float bulkBuildSegmentedLayouts(quotient_filter qfilter, int numValues, unsigned int* d_insertValues, bool NoDuplicates);`

I implemented 4 different algorithms for the initial construction of a quotient filter from a given input data set. One option is to simply use the `insert` operation to insert the items in the same way as you would for a filter that already contains some number of items; however, we devised alternative methods to construct the filter in parallel, in an attempt to leverage the more complete knowledge of the data that we have in the initial build. The relative performance of the build methods is dependent on the final filter fill fraction, but the "segmented layouts" method is generally the fastest, so if you only try one method, I would suggest that one. Each build function gives the option (the `NoDuplicates` parameter) to remove any items with the same hash value (with a relatively small performance cost), or to leave them in. The regular insert algorithm performs deduplication inherently. All of the functions return the time used for the operation.

#### Inserts
`__host__ float insert(quotient_filter qfilter, int numValues, unsigned int* d_insertValues);`

Inserts new items into the SQF, moving around the items already in the filter as necessary. This is the function that should be used for incremental updates to the filter.

`__host__ float insertViaMerge(quotient_filter qfilter, unsigned int* d_insertedValues, int numOldValues, unsigned int* d_newValues, int numNewValues, bool NoDuplicates);`

Inserts new items by extracting the hash values from the filter, merging these values with the new items to be inserted, then rebuilding the filter. This is usually significantly slower than the regular insert function.

#### Lookups (Queries)
`__host__ float launchSortedLookups(quotient_filter qfilter, int numValues, unsigned int* d_lookupValues, unsigned int* d_returnValuesArray);`

Queries for a batch of items by first hashing and sorting them before searching for them in the SQF data structure. This gives greater locality in the data structure operations, at the cost of additional sorting and bookkeeping before and after. Returns the locations of the items in `returnValuesArray` (missing items return `UINT_MAX`).

`__host__ float launchUnsortedLookups(quotient_filter qfilter, int numValues, unsigned int* d_lookupValues, unsigned int* d_returnValuesArray);`

Queries for a batch of items without doing any presorting. This will generally yield higher throughput than the sorted version for the SQF. Returns the locations of the items in `returnValuesArray` (missing items return `UINT_MAX`).

#### Deletes
`__host__ float superclusterDeletes(quotient_filter qfilter, int numValues, unsigned int* d_deleteValues);`

Removes items' fingerprints from the filter. (Note that because only fingerprints are stored in QF, if two different items, A and B, with the same fingerprint are inserted, then item A is deleted, item B will return a false negative in future queries.)

#### Merging Filters
`__host__ float mergeTwoFilters(quotient_filter qfilter1, quotient_filter qfilter2, bool NoDuplicates);`

Combines two quotient filters (of the same size, i.e., q1 == q2) into one filter.

### Rank-and-Select-Based Quotient Filter
Code for the GPU implementation of the rank-and-select-based quotient filter is `rankSelectQuotientFilter/GPU_version/RSQF.cu`.

I did not implement counters, so this is not a counting quotient filter, though it should be fairly straightforward to add them. Addtionally, I did not implement any bulk build operations to use with the RSQF, but it should be possible to use any of the ones I used for the standard quotient filter with a few modifications to the code. Similarly, I did not implement deletes or merging filters for the RSQF.

#### Initialize Filter
`__host__ void initCQFGPU(struct countingQuotientFilterGPU *cqf, unsigned int q);`

The number of remainder bits is hard-coded in `RSQF.cuh`, but, unlike the SQF, this filter implementation works with a variety of `RBITS` values.

#### Inserts
`__host__ float insertGPU(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_insertValues, int* d_returnValues);`

Inserts new items into the RSQF, moving around the items already in the filter as necessary.

#### Lookups (Queries)
`__host__ float launchLookups(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_lookupValues, int* slotValuesArray);`

Queries for a batch of items by first hashing and sorting them before searching for them in the RSQF data structure. This gives greater locality in the data structure operations, at the cost of additional sorting and bookkeeping before and after. For the RSQF, this version generally yields higher throughput than the unsorted version. Returns the locations of the items in `slotValuesArray` (missing items return `UINT_MAX`).

`__host__ float launchUnsortedLookups(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_lookupValues, int* slotValuesArray);`

Queries for a batch of items without performing any presorting. Returns the locations of the items in `slotValuesArray` (missing items return `UINT_MAX`).
