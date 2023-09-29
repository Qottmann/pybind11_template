cmake_minimum_required(VERSION 3.20)
project(pybind_template)
# set(CMAKE_OSX_DEPLOYMENT_TARGET "11" CACHE STRING "Minimum OS X deployment version")
# set(CMAKE_CXX_STANDARD 20)
add_subdirectory(pybind11)
pybind11_add_module(my_cpp_module main.cpp)

# Add the -DACCELERATE_NEW_LAPACK compilation flag
# target_compile_definitions(my_cpp_module PRIVATE -DACCELERATE_NEW_LAPACK)