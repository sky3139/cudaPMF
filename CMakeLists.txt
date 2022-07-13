cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(bare_earth )
SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_BUILD_TYPE "Debug")

find_package(PCL REQUIRED)
find_package(Boost REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)

if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}  -O3")
endif()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-prec-div=true;--expt-relaxed-constexpr;-w)#;-G;-g$

add_definitions(${PCL_DEFINITIONS})
include_directories(${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS}
./app
)
cuda_add_executable(main ground_detection.cu app/apmf.h app/tool.cu
app/InputReader.cpp
app/Timer.cc
)
target_link_libraries(main 
${PCL_LIBRARIES}
${OpenCV_LIBRARIES}
)

cuda_add_executable(sgd seteroGD.cu 
app/apmf.h 
app/tool.cu 
app/rw.cc
app/InputReader.cpp
app/Timer.cc
)
target_link_libraries(sgd 
${PCL_LIBRARIES} 
${OpenCV_LIBRARIES}
)
cuda_add_executable(opengf opengf.cu 
app/apmf.h 
app/tool.cu 
app/rw.cc
app/InputReader.cpp
app/Timer.cc
)
target_link_libraries(opengf 
${PCL_LIBRARIES} 
${OpenCV_LIBRARIES}
)