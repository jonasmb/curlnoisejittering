cmake_minimum_required(VERSION 3.14)

project(dyadicnets)
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
add_executable(net-optimize-pointers net-optimize-pointers.cpp getopt.c)
target_compile_features(net-optimize-pointers PRIVATE cxx_std_17)
