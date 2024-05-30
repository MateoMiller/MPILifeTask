cmake_minimum_required(VERSION 3.26)
project(MPILifeTask)

set(CMAKE_CXX_STANDARD 14)

# Add MPI Package to Project
find_package(MPI REQUIRED)

add_executable(MPILifeTask main.cpp)

# Add libraries for code completion and compiling
target_link_libraries(MPILifeTask PUBLIC MPI::MPI_CXX)