cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(bare_earth )
SET(CMAKE_BUILD_TYPE "Release")

add_executable(qtree ma1.cc)
add_executable(rmq rmq.cc)