cmake_minimum_required (VERSION 3.5)
project(nccl)
find_package(CUDA 7.5 QUIET REQUIRED)

set(NCCL_MAJOR 1)
set(NCCL_MINOR 3)
set(NCCL_PATCH 4)

# Call add_subdirectory(nccl) after nvcc flags have been set in the parent project to propagate flags to nccl
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-DNCCL_MAJOR=${NCCL_MAJOR} -DNCCL_MINOR=${NCCL_MINOR} -DNCCL_PATCH=${NCCL_PATCH} -DCUDA_MAJOR=7 -DCUDA_MINOR=5")
file(GLOB SOURCES src/*.cu)

cuda_add_library(${PROJECT_NAME} STATIC ${SOURCES})
