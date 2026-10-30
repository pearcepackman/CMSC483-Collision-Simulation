include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SYSTEM
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Every tests/*.cpp joins the one test binary, so new test files need no CMake edits.
file(GLOB SIM_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(sim_tests ${SIM_TEST_SOURCES})
target_link_libraries(sim_tests PRIVATE sim_core GTest::gtest_main)
set_project_warnings(sim_tests)
# PRE_TEST: list tests when ctest runs, not at build time, so a binary that can't
# run (e.g. a broken sanitizer runtime) fails the tests instead of the build.
gtest_discover_tests(sim_tests DISCOVERY_MODE PRE_TEST)

# GPU tests: tests/gpu/*.cpp in their own binary, only with ENABLE_HIP, every test
# labelled `gpu` so presets without a GPU can skip them (docs/notes.md).
if(ENABLE_HIP)
    file(GLOB SIM_GPU_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.cpp)
    add_executable(sim_tests_gpu ${SIM_GPU_TEST_SOURCES})
    target_link_libraries(sim_tests_gpu PRIVATE sim_core GTest::gtest_main)
    set_project_warnings(sim_tests_gpu)
    gtest_discover_tests(sim_tests_gpu DISCOVERY_MODE PRE_TEST PROPERTIES LABELS gpu)
endif()

# Tests that run the real planet_sim program (or PROGRAM, e.g. planet_view) and check
# its exit code and output (#14).
function(add_cli_test name expect_exit)
    cmake_parse_arguments(T "" "OUTPUT;MUST_NOT_EXIST;PROGRAM" "ARGS" ${ARGN})
    if(NOT T_PROGRAM)
        set(T_PROGRAM planet_sim)
    endif()
    list(JOIN T_ARGS "|" joined_args) # see check_run.cmake
    set(cmd -DCMD=$<TARGET_FILE:${T_PROGRAM}> "-DARGS=${joined_args}" -DEXPECT_EXIT=${expect_exit})
    if(T_OUTPUT)
        list(APPEND cmd "-DEXPECT_OUTPUT=${T_OUTPUT}")
    endif()
    if(T_MUST_NOT_EXIST)
        list(APPEND cmd "-DMUST_NOT_EXIST=${T_MUST_NOT_EXIST}")
    endif()
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} ${cmd} -P ${CMAKE_CURRENT_SOURCE_DIR}/check_run.cmake)
endfunction()

add_cli_test(Cli.OrbitBench 0
    ARGS --scenario orbit --steps 1000 --bench
    OUTPUT "bench backend=serial n=2 threads=1 steps=1000 seconds_per_step=")
add_cli_test(Cli.CloudBenchOmp 0
    ARGS --scenario cloud --n 200 --steps 20 --backend omp --bench
    OUTPUT "bench backend=omp n=200 ")
add_cli_test(Cli.BenchWritesNoFiles 0
    ARGS --scenario orbit --steps 10 --bench --out ${CMAKE_CURRENT_BINARY_DIR}/bench_out_must_not_exist
    MUST_NOT_EXIST ${CMAKE_CURRENT_BINARY_DIR}/bench_out_must_not_exist)
add_cli_test(Cli.RunsWithoutOutput 0
    ARGS --scenario figure8 --steps 500
    OUTPUT "ran 500 steps")
# Error cases check the specific message, so a different error can't pass them by luck.
add_cli_test(Cli.BadFlagIsUsageError 1
    ARGS --scenario orbit --bogus
    OUTPUT "unknown flag '--bogus'.*usage")
add_cli_test(Cli.UnknownScenarioIsUsageError 1
    ARGS --scenario nope
    OUTPUT "unknown scenario 'nope'")
add_cli_test(Cli.BlowUpExitsWithCode2 2
    ARGS --scenario orbit --dt 1e300 --steps 5
    OUTPUT "became NaN/Inf after step 1")
if(NOT ENABLE_HIP)
    add_cli_test(Cli.HipBackendNotInThisBuild 1
        ARGS --scenario orbit --backend hip
        OUTPUT "this build has no hip backend; build with: cmake --preset hip")
else()
    # The whole program on the GPU: device chosen, bench line, exit 0. Needs a GPU.
    add_cli_test(Cli.HipBackendRuns 0
        ARGS --scenario cloud --n 300 --steps 3 --backend hip --bench
        OUTPUT "bench backend=hip n=300 threads=0 .*hip device: ") # stdout, then stderr
    set_tests_properties(Cli.HipBackendRuns PROPERTIES LABELS gpu)
endif()
add_cli_test(Cli.InfiniteDtIsUsageError 1
    ARGS --scenario orbit --dt inf
    OUTPUT "--dt needs a finite number > 0")

# planet_view's argument handling. Parsing happens before any window opens, so these
# run without a display.
if(ENABLE_VIEWER)
    add_cli_test(Viewer.RejectsPlanetSimOnlyFlags 1 PROGRAM planet_view
        ARGS --scenario cloud --out somewhere
        OUTPUT "are planet_sim flags; the viewer runs until you close it")
    add_cli_test(Viewer.RejectsBadFrames 1 PROGRAM planet_view
        ARGS --scenario cloud --frames abc
        OUTPUT "--frames needs a whole number >= 1, got 'abc'")
    add_cli_test(Viewer.RejectsMissingFramesValue 1 PROGRAM planet_view
        ARGS --scenario cloud --frames
        OUTPUT "--frames needs a whole number >= 1, got ''")
    add_cli_test(Viewer.RejectsZeroFrames 1 PROGRAM planet_view
        ARGS --scenario cloud --frames 0
        OUTPUT "--frames needs a whole number >= 1, got '0'")
    add_cli_test(Viewer.RejectsNegativeFrames 1 PROGRAM planet_view
        ARGS --scenario cloud --frames -3
        OUTPUT "--frames needs a whole number >= 1, got '-3'")
    add_cli_test(Viewer.RejectsOverflowingFrames 1 PROGRAM planet_view
        ARGS --scenario cloud --frames 99999999999999999999999
        OUTPUT "--frames needs a whole number >= 1")
    add_cli_test(Viewer.RejectsNonPngScreenshot 1 PROGRAM planet_view
        ARGS --scenario cloud --screenshot shot.jpg
        OUTPUT "--screenshot needs a .png file name, got 'shot.jpg'")
    add_cli_test(Viewer.RejectsEmptyScreenshotName 1 PROGRAM planet_view
        ARGS --scenario cloud --screenshot ""
        OUTPUT "--screenshot needs a .png file name, got ''")
    add_cli_test(Viewer.UnknownScenarioIsSetupError 1 PROGRAM planet_view
        ARGS --scenario nope
        OUTPUT "unknown scenario 'nope'")
    if(NOT ENABLE_HIP)
        add_cli_test(Viewer.HipNotInThisBuild 1 PROGRAM planet_view
            ARGS --scenario cloud --backend hip
            OUTPUT "this build has no hip backend")
    endif()
    add_cli_test(Viewer.Help 0 PROGRAM planet_view
        ARGS --help
        OUTPUT "keys: 1/2/3 serial/OpenMP/HIP")
endif()
