cmake_minimum_required(VERSION 3.14)

project(dyadicnets)

add_executable(net-optimize-pointers net-optimize-pointers.cpp)
target_compile_features(net-optimize-pointers PRIVATE cxx_std_17)
