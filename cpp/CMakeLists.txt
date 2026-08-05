cmake_minimum_required(VERSION 3.20)
project(dedup_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/utf-8)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

add_executable(dedup_core dedup_core.cpp)
target_link_libraries(dedup_core PRIVATE ${OpenCV_LIBS})
target_include_directories(dedup_core PRIVATE ${OpenCV_INCLUDE_DIRS})
