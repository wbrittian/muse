cmake_minimum_required(VERSION 3.20)
project(model LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

find_package(Python3 3.12 COMPONENTS Interpreter Development EXACT REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

find_package(GTest REQUIRED)
enable_testing()

file(GLOB_RECURSE MUSEFORMER_SOURCES src/cppsrc/model/*.cpp)
file(GLOB_RECURSE MUSEFORMER_HEADERS src/cppsrc/model/*.hpp)
file(GLOB_RECURSE MUSEFORMER_TESTS src/cppsrc/test/*.cpp)

add_executable(museformer_tests ${MUSEFORMER_TESTS} ${MUSEFORMER_SOURCES} ${MUSEFORMER_HEADERS})
target_include_directories(museformer_tests PUBLIC src)

target_link_libraries(museformer_tests PRIVATE Eigen3::Eigen)
target_link_libraries(museformer_tests PRIVATE gtest::gtest)

pybind11_add_module(museformer src/cppsrc/main.cpp ${MUSEFORMER_SOURCES})
install(TARGETS museformer DESTINATION museformer)

target_include_directories(museformer
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cppsrc
)

target_link_libraries(museformer
  PRIVATE
    Eigen3::Eigen
)