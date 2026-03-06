cmake_minimum_required(VERSION 3.20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
project(photon_backend LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 120)



find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# CPU backend
add_library(photon_core_cpu STATIC src/cpu/backend_float.cc)
target_include_directories(photon_core_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/cpu)
target_include_directories(photon_core_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu)

if(NOT MSVC)
    target_compile_options(photon_core_cpu PRIVATE -O3)
endif()

pybind11_add_module(backend_cpu MODULE src/cpu/bindings.cc)
target_link_libraries(backend_cpu PRIVATE photon_core_cpu)
install(TARGETS backend_cpu DESTINATION photon)


# GPU backend
add_library(photon_core_gpu STATIC src/gpu/backend_float.cu)
target_include_directories(photon_core_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu)
target_include_directories(photon_core_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu)


pybind11_add_module(backend_gpu MODULE src/gpu/bindings.cu)
target_link_libraries(backend_gpu PRIVATE photon_core_gpu)
install(TARGETS backend_gpu DESTINATION photon)



add_executable(cpp_test src/main.cc)
set_source_files_properties(src/main.cc PROPERTIES LANGUAGE CUDA)
target_include_directories(cpp_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/gpu)
target_link_libraries(cpp_test PRIVATE photon_core_cpu photon_core_gpu)
