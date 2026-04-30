cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(orinflow_kernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
    set(TORCH_CUDA_ARCH_LIST "8.7;12.0;")
endif()

file(GLOB_RECURSE SRC_FILES
    "src/*.cc"
    "src/*.cu"
    "src/*.cpp"
)

add_library(${PROJECT_NAME} MODULE ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${TORCH_LIBRARIES}
    Python3::Module
)
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-D_GLIBCXX_USE_CXX11_ABI=1>
)

set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so")
endif()