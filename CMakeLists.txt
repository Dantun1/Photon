cmake_minimum_required(VERSION 3.20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(photon_backend LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)


add_library(photon_core STATIC src/cpu/backend_float.cc)
target_include_directories(photon_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(photon_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu)

if(NOT MSVC)
    target_compile_options(photon_core PRIVATE -O3)
endif()

pybind11_add_module(backend_cpu MODULE src/bindings.cc)
target_link_libraries(backend_cpu PRIVATE photon_core)
install(TARGETS backend_cpu DESTINATION photon)

add_executable(cpp_test src/main.cc)
target_link_libraries(cpp_test PRIVATE photon_core)
