cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

include(FetchContent)

FetchContent_Declare(
  pcg-cpp
  GIT_REPOSITORY https://github.com/LTLA/pcg-cpp
  GIT_TAG master
)

FetchContent_MakeAvailable(pcg-cpp)
