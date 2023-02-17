cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(closest_neighbour LANGUAGES C CXX CUDA)

set(HDR_DIR include)
set(SRC_DIR src)

# python lib
set(CLOSEST_NEIGHBOUR_PYTHON_LIB closest_neighbour)
add_subdirectory(pybind11)
pybind11_add_module(${CLOSEST_NEIGHBOUR_PYTHON_LIB} ${SRC_DIR}/closest_neighbour_cuda.cu)

# cpp test
if(BUILD_CPP_TEST)
  set(CLOSEST_NEIGHBOUR_LIB closest_neighbour_lib)
  add_library(${CLOSEST_NEIGHBOUR_LIB} OBJECT
    ${SRC_DIR}/closest_neighbour_cuda.cu
  )

  target_compile_options(${CLOSEST_NEIGHBOUR_LIB} PUBLIC "-Wno-deprecated-gpu-targets")
  target_include_directories(${CLOSEST_NEIGHBOUR_LIB} PUBLIC ${HDR_DIR})

  set(TARGET closest_neighbour_test)
  set(SRCS
    ${CMAKE_CURRENT_LIST_DIR}/test.cpp
  )

  add_executable(${TARGET} ${SRCS})
  target_link_libraries(${TARGET} PUBLIC ${CLOSEST_NEIGHBOUR_LIB_TEST})
endif()