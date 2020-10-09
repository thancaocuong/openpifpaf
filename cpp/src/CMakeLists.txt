cmake_minimum_required(VERSION 3.6)
project(openpifpaf LANGUAGES CXX)
find_package(CUDA REQUIRED)
find_cuda_helper_libs(npp)
set(CMAKE_BUILD_TYPE Debug)


set(CUDA_INSTALL_DIR /usr/local/cuda)

set(TARGET_NAME openpifpaf)

file(GLOB HEADER_LIST
        "${OpenPifPaf_SOURCE_DIR}/include/*.h"
        "${OpenPifPaf_SOURCE_DIR}/include/utils/*.h"
        "${OpenPifPaf_SOURCE_DIR}/include/utils/camera/*.h"
        "${OpenPifPaf_SOURCE_DIR}/include/utils/threads/*"
        "${OpenPifPaf_SOURCE_DIR}/include/utils/codec/*"
        "${OpenPifPaf_SOURCE_DIR}/include/utils/display/*"
        "${OpenPifPaf_SOURCE_DIR}/include/utils/stb/*"
        "${OpenPifPaf_SOURCE_DIR}/include/utils/cuda/*.h")

file(GLOB SOURCE_LIST
        "${OpenPifPaf_SOURCE_DIR}/src/utils/*.cpp"
        "${OpenPifPaf_SOURCE_DIR}/src/utils/cuda/*"
        "${OpenPifPaf_SOURCE_DIR}/src/utils/display/*"
        "${OpenPifPaf_SOURCE_DIR}/src/utils/codec/*"
        "${OpenPifPaf_SOURCE_DIR}/src/utils/camera/*"
        "${OpenPifPaf_SOURCE_DIR}/src/utils/threads/*"
        "${OpenPifPaf_SOURCE_DIR}/src/*.cpp"
        "${OpenPifPaf_SOURCE_DIR}/src/*.cu")

cuda_add_library(${TARGET_NAME} STATIC ${SOURCE_LIST} ${HEADER_LIST})
set_target_properties(${TARGET_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenCV)
find_library(CUDART_LIBRARY cudart HINTS /usr/local/cuda/lib64)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED gstreamer-1.0>=1.8)

include_directories(/usr/include/gstreamer-1.0
                    /usr/include/glib-2.0
                    /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include
                    /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include/
                    /usr/lib/aarch64-linux-gnu/glib-2.0/include/
                    /usr/lib/x86_64-linux-gnu/glib-2.0/include/)

include_directories("${GST_INCLUDE_DIRS}")
link_libraries(${GST_LIBRARIES})
target_include_directories(${TARGET_NAME}
    PRIVATE stdc++fs
    PUBLIC ../include
    PUBLIC ${CUDA_INSTALL_DIR}/include
    PUBLIC ${OpenCV_INCLUDE_DIRS}
    PUBLIC /usr/lib/x86_64-linux-gnu/
    )


set(SAMPLE_DEP_LIBS
    ${CUBLAS_LIB}
    ${CUDNN_LIB}
    ${OpenCV_LIBS}
    ${CUDART_LIBRARY}
    nvinfer
    GL GLU
    GLEW
    X11
    gstapp-1.0
    gstpbutils-1.0
    ${GST_LIBRARIES}
    ${CUDA_npp_LIBRARY}
)

add_definitions(-O2 -pthread)

target_link_libraries(${TARGET_NAME} ${SAMPLE_DEP_LIBS})
target_compile_options(${TARGET_NAME} PUBLIC -Wreturn-local-addr -Wdeprecated-declarations -Wnarrowing)
