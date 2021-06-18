cmake_minimum_required(VERSION 3.14)

project(tsvd
    VERSION 1.0.0
    DESCRIPTION "A C++ implementation of truncated SVD algorithms"
    LANGUAGES CXX)

add_library(tsvd INTERFACE)

target_include_directories(tsvd INTERFACE include/)

#if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#    include(CTest)
#    if(BUILD_TESTING)
#        add_subdirectory(tests)
#    endif() 
#endif()
