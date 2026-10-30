cmake_minimum_required(VERSION 3.25)
project(PlanetCollision VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_HIP "Build the HIP (AMD GPU) gravity backend" OFF)
option(ENABLE_VIEWER "Build planet_view, the live 3D viewer (downloads raylib; needs X11/OpenGL)" OFF)
option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
set(SANITIZER "" CACHE STRING "Sanitizer build: address (ASan+UBSan), thread (TSan), or empty")

# Sanitizers apply to everything (including GoogleTest) so instrumentation is consistent.
# Compile flags are limited to C++ so the HIP compiler never sees host-only options.
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

if(ENABLE_HIP)
    # CMAKE_HIP_ARCHITECTURES (e.g. gfx1200 for the RX 9060 XT) comes from the hip preset.
    enable_language(HIP)
    set(CMAKE_HIP_STANDARD 17)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)
    message(STATUS "HIP enabled for ${CMAKE_HIP_ARCHITECTURES}")
endif()

# Warnings for our own targets (C++ and HIP; HIP is compiled by clang, so it takes the
# same flags). Third-party code (GoogleTest) is left alone.
function(set_project_warnings target)
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,HIP>:-Wall;-Wextra;-Wshadow;-Wconversion>")
    if(WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,HIP>:-Werror>")
    endif()
endfunction()

# Every src/*.cpp except main.cpp goes into the core library, so new modules need no
# CMake edits. CONFIGURE_DEPENDS re-checks the file list on each build.
file(GLOB SIM_CORE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM SIM_CORE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
add_library(sim_core ${SIM_CORE_SOURCES})
target_include_directories(sim_core PUBLIC src)
target_link_libraries(sim_core PUBLIC OpenMP::OpenMP_CXX)
set_project_warnings(sim_core)
if(ENABLE_HIP)
    target_sources(sim_core PRIVATE src/gravity_hip.hip)
    target_compile_definitions(sim_core PUBLIC SIM_HAVE_HIP)
    # The kernel's GPU list, so hip_init never picks a GPU without code for it (that
    # aborts inside the HIP runtime). Commas: a ';' would split the definition.
    string(REPLACE ";" "," sim_hip_archs "${CMAKE_HIP_ARCHITECTURES}")
    target_compile_definitions(sim_core PRIVATE "SIM_HIP_ARCHS=\"${sim_hip_archs}\"")
endif()

add_executable(planet_sim src/main.cpp)
target_link_libraries(planet_sim PRIVATE sim_core)
set_project_warnings(planet_sim)

# Micro-benchmark for the gravity backends (#19). Built everywhere so CI keeps it
# compiling; run it from the release preset for real numbers.
add_executable(bench_gravity bench/bench_gravity.cpp)
target_link_libraries(bench_gravity PRIVATE sim_core)
set_project_warnings(bench_gravity)

# Live 3D viewer (#45). raylib is pinned to a release tag and built from source, so
# every machine gets the same version; SYSTEM keeps our warnings off its headers.
if(ENABLE_VIEWER)
    include(FetchContent)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        raylib
        URL https://github.com/raysan5/raylib/archive/refs/tags/5.5.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SYSTEM
    )
    FetchContent_MakeAvailable(raylib)
    add_executable(planet_view viewer/planet_view.cpp)
    target_link_libraries(planet_view PRIVATE sim_core raylib)
    set_project_warnings(planet_view)
endif()

enable_testing()
add_subdirectory(tests)
