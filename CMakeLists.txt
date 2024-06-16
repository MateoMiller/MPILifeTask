cmake_minimum_required(VERSION 3.26)
project(MPILifeTask)

set(CMAKE_CXX_STANDARD 14)

# Add MPI Package to Project
find_package(MPI REQUIRED)

add_executable(MPILifeTask main.cpp)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/input.txt
        $<TARGET_FILE_DIR:${PROJECT_NAME}>/input.txt
        COMMENT "Копирование input.txt в bin/debug"
)

target_link_libraries(MPILifeTask PUBLIC MPI::MPI_CXX)