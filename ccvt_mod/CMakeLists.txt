cmake_minimum_required(VERSION 3.14)
project(ccvt)
set (CMAKE_CXX_STANDARD 11)
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIRECTORY}")
SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIRECTORY}")
add_executable (ccvt main.cpp)