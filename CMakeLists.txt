cmake_minimum_required(VERSION 3.25)
project(PlanetCollision VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_CUDA "Build the CUDA gravity backend" OFF)
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
set(SANITIZER "" CACHE STRING "Sanitizer build: address (ASan+UBSan), thread (TSan), or empty")

# Sanitizers apply to everything (including GoogleTest) so instrumentation is consistent.
# Compile flags are limited to C++ so nvcc never sees gcc/clang-only options.
if(SANITIZER STREQUAL "address")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined;-fno-sanitize-recover=all;-fno-omit-frame-pointer>")
    add_link_options(-fsanitize=address,undefined)
elseif(SANITIZER STREQUAL "thread")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=thread;-fno-omit-frame-pointer>")
    add_link_options(-fsanitize=thread)
elseif(NOT SANITIZER STREQUAL "")
    message(FATAL_ERROR "Unknown SANITIZER '${SANITIZER}' (expected address, thread, or empty)")
endif()

# Apple clang ships without OpenMP; point CMake at Homebrew's libomp if it's installed.
if(APPLE AND NOT DEFINED OpenMP_ROOT AND NOT DEFINED ENV{OpenMP_ROOT})
    foreach(prefix /opt/homebrew/opt/libomp /usr/local/opt/libomp)
        if(EXISTS ${prefix})
            set(OpenMP_ROOT ${prefix})
            break()
        endif()
    endforeach()
endif()
find_package(OpenMP REQUIRED)

if(ENABLE_CUDA)
    enable_language(CUDA)
    message(STATUS "CUDA enabled (backend arrives in milestone 4)")
endif()

# Warnings for our own C++ targets only; third-party code (GoogleTest) and nvcc are left alone.
function(set_project_warnings target)
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wshadow;-Wconversion>")
    if(WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
    endif()
endfunction()

add_library(sim_core src/diagnostics.cpp)
target_include_directories(sim_core PUBLIC src)
target_link_libraries(sim_core PUBLIC OpenMP::OpenMP_CXX)
set_project_warnings(sim_core)

add_executable(planet_sim src/main.cpp)
target_link_libraries(planet_sim PRIVATE sim_core)
set_project_warnings(planet_sim)

enable_testing()
add_subdirectory(tests)
