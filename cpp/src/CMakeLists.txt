cmake_minimum_required(VERSION 3.6)
project(openpifpaf LANGUAGES CXX)
find_package(CUDA REQUIRED)

set(CMAKE_BUILD_TYPE Debug)


set(CUDA_INSTALL_DIR /usr/local/cuda)

set(TARGET_NAME openpifpaf)

file(GLOB HEADER_LIST
        "${OpenPifPaf_SOURCE_DIR}/include//*.h"
        "${OpenPifPaf_SOURCE_DIR}/include/utils/*.h"
        "${OpenPifPaf_SOURCE_DIR}/include/utils/cuda/*.h")

file(GLOB SOURCE_LIST
        "${OpenPifPaf_SOURCE_DIR}/src/utils/*.cpp"
        "${OpenPifPaf_SOURCE_DIR}/src/utils/*.inl"
        "${OpenPifPaf_SOURCE_DIR}/src/utils/cuda/*.cu"
        "${OpenPifPaf_SOURCE_DIR}/src/calibration/*"
        "${OpenPifPaf_SOURCE_DIR}/src/*.cpp")



cuda_add_library(${TARGET_NAME} STATIC ${SOURCE_LIST} ${HEADER_LIST})
set_target_properties(${TARGET_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenCV)
find_library(CUDART_LIBRARY cudart HINTS /usr/local/cuda/lib64)

target_include_directories(${TARGET_NAME}
    PRIVATE stdc++fs
    PUBLIC ../include
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    )


set(SAMPLE_DEP_LIBS
    ${CUBLAS_LIB}
    ${CUDNN_LIB}
    ${OpenCV_LIBS}
    ${CUDART_LIBRARY}
    nvinfer
)

add_definitions(-O2 -pthread)

target_link_libraries(${TARGET_NAME} ${SAMPLE_DEP_LIBS})
target_compile_options(${TARGET_NAME} PUBLIC -Wreturn-local-addr -Wdeprecated-declarations -Wnarrowing)
