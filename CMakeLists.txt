cmake_minimum_required(VERSION 3.26)
project(red_eye)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(red_eye main.cpp
        implementation/eye_detection.cpp
        implementation/eye_detection.h
        implementation/eye_detection.cpp)

target_link_libraries(red_eye ${OpenCV_LIBS})
