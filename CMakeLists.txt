
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(orinflow_kernel LANGUAGES CXX CUDA)

# 强制使用 C++17 标准 (PyTorch 当前的主流要求)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 这里会依赖 setup.py 传过来的 CMAKE_PREFIX_PATH
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(NOT DEFINED TORCH_CUDA_ARCH_LIST)
    set(TORCH_CUDA_ARCH_LIST "8.7;12.0;")
endif()

# 自动递归搜索 src 目录下所有的 .cc 和 .cu 文件
file(GLOB_RECURSE SRC_FILES
    "src/*.cc"
    "src/*.cu"
    "src/*.cpp"
)

# 打印一下找到的文件，方便调试时看有没有漏掉
message(STATUS "找到的源码文件: ${SRC_FILES}")

# 5. 定义编译产物
# 作为一个给 Python 调用的动态库，标准做法是声明为 MODULE 而不是 SHARED
add_library(${PROJECT_NAME} MODULE ${SRC_FILES})

# 6. 设置包含目录和链接库
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# 链接 PyTorch 底层库 (libtorch, libtorch_python 等)
target_link_libraries(${PROJECT_NAME} PRIVATE 
    ${TORCH_LIBRARIES}
    Python3::Module
)

# 保证 C++ ABI 的兼容性（极度关键，防止和 PyTorch 预编译包冲突报错）
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-D_GLIBCXX_USE_CXX11_ABI=1>
)

# 在 Linux 下，CMake 默认会在库前面加 "lib" (生成 liborinflow_kernel.so)
# 但 Python 和 torch.ops.load_library 期望直接叫 orinflow_kernel.so，所以必须清空前缀
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")

# 根据操作系统自动推导后缀
if(WIN32)
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".pyd")
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".so")
endif()