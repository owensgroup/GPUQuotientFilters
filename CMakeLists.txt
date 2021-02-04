cmake_minimum_required (VERSION 2.8)

project (QuotientFilter)

set (QuotientFilter_VERSION_MAJOR 0)
set (QuotientFilter_VERSION_MINOR 0)

find_package(CUDA REQUIRED)

set(CPU_SQF_SOURCE_DIR
    ${CMAKE_SOURCE_DIR}/standardQuotientFilter/CPU_version
    CACHE PATH
    "Directory for CPU version of quotient filter")

set(GPU_SQF_SOURCE_DIR
    ${CMAKE_SOURCE_DIR}/standardQuotientFilter/GPU_version
    CACHE PATH
    "Directory for GPU version of quotient filter")

set(MODERNGPU_SOURCE_DIR
    ${CMAKE_SOURCE_DIR}/moderngpu/src
    CACHE PATH
    "Directory for Modern GPU source")

set(CPU_RSQF_SOURCE_DIR
    ${CMAKE_SOURCE_DIR}/rankSelectQuotientFilter/CPU_version
    CACHE PATH
    "Directory for CPU version of RSQF")

set(GPU_RSQF_SOURCE_DIR
    ${CMAKE_SOURCE_DIR}/rankSelectQuotientFilter/GPU_version
    CACHE PATH
    "Directory for GPU version of RSQF")

## Set the directory where the binaries will be stored
set(EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executables will be stored")

## Set the directory where the libraries will be stored
set(LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH
  "Directory where all the libraries will be stored")

set(CUDA_NVCC_FLAGS -std=c++11 --expt-extended-lambda -use_fast_math -Wno-deprecated-gpu-targets -gencode arch=compute_61,code=compute_61 -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35)

##Executables for correctness tests
add_executable (CPUSQFverificationTest ${CPU_SQF_SOURCE_DIR}/quotientFilter.cpp mt19937ar.cpp ${CPU_SQF_SOURCE_DIR}/CPUTestQuotientFilter.cpp)
cuda_add_executable (GPUSuperclusterVerificationTest mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/superclusterTest.cu)
cuda_add_executable (GPUBulkBuildSequentialShiftsVerificationTest mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/bulkBuildSequentialShiftsTest.cu)
cuda_add_executable (GPUBulkBuildParallelMergingVerificationTest mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/bulkBuildViaMergingTest.cu)
cuda_add_executable (GPUBulkBuildSegmentedLayoutsVerificationTest mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/bulkBuildSegmentedLayoutTest.cu)
cuda_add_executable (GPUDeletesVerificationTest mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/deletesTest.cu)
cuda_add_executable (GPUMergeInsertVerificationTest mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/insertViaMergeTest.cu)
cuda_add_executable (GPUMergeFiltersVerificationTest mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/merge2FiltersTest.cu)
add_executable (CPURSQFverificationTest ${CPU_RSQF_SOURCE_DIR}/RSQF.cpp mt19937ar.cpp ${CPU_RSQF_SOURCE_DIR}/testRSQF.cpp)
cuda_add_executable (GPURSQFverificationTest ${GPU_RSQF_SOURCE_DIR}/RSQF.cu mt19937ar.cpp ${GPU_RSQF_SOURCE_DIR}/testGPURSQF.cu)

## Executables for perf graphs
add_executable (CPULookups.o ${CPU_SQF_SOURCE_DIR}/quotientFilter.cpp mt19937ar.cpp ${CPU_SQF_SOURCE_DIR}/CPULookupPerf.cpp)
add_executable (CPUInserts.o ${CPU_SQF_SOURCE_DIR}/quotientFilter.cpp mt19937ar.cpp ${CPU_SQF_SOURCE_DIR}/CPUInsertPerf.cpp)
add_executable (CPUBatchLookups.o ${CPU_SQF_SOURCE_DIR}/quotientFilter.cpp mt19937ar.cpp ${CPU_SQF_SOURCE_DIR}/CPUBatchLookupPerf.cpp)
add_executable (CPUBatchInserts.o ${CPU_SQF_SOURCE_DIR}/quotientFilter.cpp mt19937ar.cpp ${CPU_SQF_SOURCE_DIR}/CPUBatchInsertPerf.cpp)
cuda_add_executable (GPULookups.o mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/GPULookupsPerf.cu)
cuda_add_executable (GPUInserts.o mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/GPUInsertsPerf.cu)
cuda_add_executable (GPUBatchLookups.o mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/GPUBatchLookups.cu)
cuda_add_executable (GPUBatchInserts.o mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/GPUBatchInserts.cu)
cuda_add_executable (sequentialShiftsBuild.o mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/bulkBuildSequentialShiftsPerf.cu)
cuda_add_executable (parMergeBuild.o mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/bulkBuildViaMergingPerf.cu)
cuda_add_executable (segLayoutsBuild.o mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/bulkBuildSegmentedLayoutPerf.cu)
cuda_add_executable (GPUDeletes.o mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/deletesPerf.cu)
cuda_add_executable (GPUBatchDeletes.o mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/deletesBatchPerf.cu)
cuda_add_executable (GPUMergeInserts.o mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/insertViaMergePerf.cu)
cuda_add_executable (GPUMergeFilters.o mt19937ar.cpp ${GPU_SQF_SOURCE_DIR}/quotientFilter.cu ${GPU_SQF_SOURCE_DIR}/merge2FiltersPerf.cu)
add_executable (CPURSQFLookups.o mt19937ar.cpp ${CPU_RSQF_SOURCE_DIR}/RSQF.cpp ${CPU_RSQF_SOURCE_DIR}/CPURSQFLookupPerf.cpp)
add_executable (CPURSQFInserts.o mt19937ar.cpp ${CPU_RSQF_SOURCE_DIR}/RSQF.cpp ${CPU_RSQF_SOURCE_DIR}/CPURSQFInsertPerf.cpp)
add_executable (CPURSQFBatchLookups.o mt19937ar.cpp ${CPU_RSQF_SOURCE_DIR}/RSQF.cpp ${CPU_RSQF_SOURCE_DIR}/CPURSQFBatchLookupPerf.cpp)
add_executable (CPURSQFBatchInserts.o mt19937ar.cpp ${CPU_RSQF_SOURCE_DIR}/RSQF.cpp ${CPU_RSQF_SOURCE_DIR}/CPURSQFBatchInsertPerf.cpp)
cuda_add_executable (GPURSQFLookups.o mt19937ar.cpp ${GPU_RSQF_SOURCE_DIR}/RSQF.cu ${GPU_RSQF_SOURCE_DIR}/GPURSQFLookupsPerf.cu)
cuda_add_executable (GPURSQFInserts.o mt19937ar.cpp ${GPU_RSQF_SOURCE_DIR}/RSQF.cu ${GPU_RSQF_SOURCE_DIR}/GPURSQFInsertsPerf.cu)
cuda_add_executable (GPURSQFBatchLookups.o mt19937ar.cpp ${GPU_RSQF_SOURCE_DIR}/RSQF.cu ${GPU_RSQF_SOURCE_DIR}/GPURSQFBatchLookupsPerf.cu)
cuda_add_executable (GPURSQFBatchInserts.o mt19937ar.cpp ${GPU_RSQF_SOURCE_DIR}/RSQF.cu ${GPU_RSQF_SOURCE_DIR}/GPURSQFBatchInsertsPerf.cu)

enable_testing()

## Lists the commands used to run tests
add_test (NAME QuotientFilterCorrectnessCPU COMMAND CPUSQFverificationTest 15 5 .6)
add_test (NAME QuotientFilterSuperclusterCorrectnessGPU COMMAND GPUSuperclusterVerificationTest 15 5 .6)
add_test (NAME QuotientFilterBulkBuildSequentialShiftsNoDupCorrectnessGPU COMMAND GPUBulkBuildSequentialShiftsVerificationTest 15 5 .6 NoDup)
add_test (NAME QuotientFilterBulkBuildSequentialShiftsDupCorrectnessGPU COMMAND GPUBulkBuildSequentialShiftsVerificationTest 15 5 .6 Dup)
add_test (NAME QuotientFilterBulkBuildParallelMergingNoDupCorrectnessGPU COMMAND GPUBulkBuildParallelMergingVerificationTest 15 5 .6 NoDup)
add_test (NAME QuotientFilterBulkBuildParallelMergingDupCorrectnessGPU COMMAND GPUBulkBuildParallelMergingVerificationTest 15 5 .6 Dup)
add_test (NAME QuotientFilterBulkBuildSegmentedLayoutNoDupCorrectnessGPU COMMAND GPUBulkBuildSegmentedLayoutsVerificationTest 15 5 .6 NoDup)
add_test (NAME QuotientFilterBulkBuildSegmentedLayoutDupCorrectnessGPU COMMAND GPUBulkBuildSegmentedLayoutsVerificationTest 15 5 .6 Dup)
add_test (NAME QuotientFilterDeletesCorrectnessGPU COMMAND GPUDeletesVerificationTest 15 5 .6 5000)
add_test (NAME QuotientFilterMergeInsertCorrectnessGPU COMMAND GPUMergeInsertVerificationTest 15 5 .6 1000 NoDup)
add_test (NAME QuotientFilterMergeFiltersCorrectnessGPU COMMAND GPUMergeFiltersVerificationTest 15 5 .2 .3 NoDup)
add_test (NAME RSQFCorrectnessCPU COMMAND CPURSQFverificationTest 15 .6)
add_test (NAME RSQFCorrectnessGPU COMMAND GPURSQFverificationTest 15 .6)
