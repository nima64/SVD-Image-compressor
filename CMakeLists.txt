cmake_minimum_required(VERSION 3.0)

project(main)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add your source files here
set(SOURCES
    main.cpp
)
include_directories(eigen lib)

# Generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# # Add debug flag
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Generate the executable
add_executable(${PROJECT_NAME} ${SOURCES})
