cmake_minimum_required(VERSION 3.18)
project(cuda-cpds LANGUAGES C CXX CUDA)

# Makefile-style CUDA / NVCC variables (imported from your Makefile)
# These keep the same defaults you used in the Makefile but are editable via CMake cache.
set(CUDA_HOME "/usr/local/cuda" CACHE PATH "CUDA installation directory (from Makefile)")
# multiple gencode entries become a CMake list
set(CUDA_ARCH
  -gencode=arch=compute_86,code=sm_86
  -gencode=arch=compute_89,code=compute_89
)
set(NVCC "${CUDA_HOME}/bin/nvcc" CACHE PATH "nvcc compiler path (from Makefile)")
## Option to control whether -lineinfo is added to nvcc invocations. Default ON
option(ENABLE_NVCC_LINEINFO "Add -lineinfo to NVCC flags (useful for profiling/line mapping)" ON)

# NVCC flags as a list; we'll join into a single string for generator-expression use below
if(ENABLE_NVCC_LINEINFO)
  set(_NVCC_LINEINFO_FLAG -lineinfo)
else()
  set(_NVCC_LINEINFO_FLAG "")
endif()

set(NVCC_FLAGS_LIST
  ${_NVCC_LINEINFO_FLAG}
  -O3
  -Wno-deprecated-gpu-targets
  -I${CUDA_HOME}/include
  ${CUDA_ARCH}
  --ptxas-options=-v
  -I${CUDA_HOME}/sdk/CUDALibraries/common/inc
)
string(REPLACE ";" " " NVCC_FLAGS "${NVCC_FLAGS_LIST}")

# Program flags (preprocessor defines)
set(PROG_FLAGS "-DNumElem=1024")

# Prefer modern CUDAToolkit but fall back to legacy FindCUDA if needed
find_package(CUDAToolkit QUIET)

# Example executable that uses CUB (header-only) from the CUDA include path
add_executable(sample_cub sample_cub.cu)

# Link to CUDA runtime. Try imported targets first, then fall back to available variables
if (CUDAToolkit_FOUND)
  if (TARGET CUDAToolkit::cudart)
    target_link_libraries(sample_cub PRIVATE CUDAToolkit::cudart)
  elseif (DEFINED CUDAToolkit_LIBRARIES)
    target_link_libraries(sample_cub PRIVATE ${CUDAToolkit_LIBRARIES})
  else()
    message(STATUS "CUDAToolkit found but no imported target available; falling back to FindCUDA.")
    find_package(CUDA REQUIRED)
    target_include_directories(sample_cub PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(sample_cub PRIVATE ${CUDA_CUDART_LIBRARY})
  endif()
else()
  # Legacy fallback for older CMake/CUDAToolkit setups
  find_package(CUDA REQUIRED)
  target_include_directories(sample_cub PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(sample_cub PRIVATE ${CUDA_CUDART_LIBRARY})
endif()

# Use C++17 for host code
set_target_properties(sample_cub PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# Ensure the CUDA include path is available (CUDAToolkit provides include dirs)
if (DEFINED CUDAToolkit_INCLUDE_DIRS)
  target_include_directories(sample_cub PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# Helpful compile options (adjust as needed)
if (MSVC)
  target_compile_options(sample_cub PRIVATE /W4)
else()
  target_compile_options(sample_cub PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Apply NVCC / CUDA-specific flags (from NVCC_FLAGS) to CUDA compilation only
if (NVCC_FLAGS)
  # NVCC_FLAGS was created above by joining a list into a single space-separated string
  target_compile_options(sample_cub PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>
  )
endif()

# Add any include path from the Makefile SDK location (keeps parity with your Makefile)
target_include_directories(sample_cub PRIVATE ${CUDA_HOME}/sdk/CUDALibraries/common/inc)

# Add program define from PROG_FLAGS (NumElem)
target_compile_definitions(sample_cub PRIVATE NumElem=1024)

# Prefer imported CUDAToolkit target or FindCUDA variable for linking. If those
# are not available and CUDA_HOME provides a lib64, expose it as an rpath so
# the runtime loader can find libcuda/libcudart without embedding a broken
# combined flag string into the linker arguments.
if (NOT TARGET CUDAToolkit::cudart)
  if (EXISTS "${CUDA_HOME}/lib64")
    # add rpath so runtime can find CUDA libraries in the provided CUDA_HOME
    target_link_options(sample_cub PRIVATE "-Wl,-rpath,${CUDA_HOME}/lib64")
  endif()
endif()
