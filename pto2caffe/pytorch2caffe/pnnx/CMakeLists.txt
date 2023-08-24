project(pnnx)
cmake_minimum_required(VERSION 3.10)

# c++14 is required for using torch headers
set(CMAKE_CXX_STANDARD 14)


option(PNNX_COVERAGE "build for coverage" OFF)

execute_process(COMMAND python3 -c "import torch;print((torch.__path__)[0])" OUTPUT_VARIABLE TORCH_INSTALL_DIR__ OUTPUT_STRIP_TRAILING_WHITESPACE)
set(Torch_INSTALL_DIR "${TORCH_INSTALL_DIR__}" CACHE STRING "")

set(Torch_DIR "${Torch_INSTALL_DIR}/share/cmake/Torch")
set(TorchVision_DIR "${TorchVision_INSTALL_DIR}/share/cmake/TorchVision")

find_package(Torch REQUIRED)

message(STATUS "Torch_VERSION = ${Torch_VERSION}")
message(STATUS "Torch_VERSION_MAJOR = ${Torch_VERSION_MAJOR}")
message(STATUS "Torch_VERSION_MINOR = ${Torch_VERSION_MINOR}")
message(STATUS "Torch_VERSION_PATCH = ${Torch_VERSION_PATCH}")

if(Torch_VERSION VERSION_LESS "1.8")
    message(FATAL_ERROR "pnnx only supports PyTorch >= 1.8")
endif()

if(TorchVision_FOUND)
    message(STATUS "Building with TorchVision")
else()
    message(WARNING "Building without TorchVision")
endif()

include_directories(${TORCH_INCLUDE_DIRS})

add_subdirectory(src)
