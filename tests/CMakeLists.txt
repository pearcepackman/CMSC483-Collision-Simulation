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

add_executable(sim_tests test_diagnostics.cpp)
target_link_libraries(sim_tests PRIVATE sim_core GTest::gtest_main)
set_project_warnings(sim_tests)
# PRE_TEST: list tests when ctest runs, not at build time, so a binary that can't
# run (e.g. a broken sanitizer runtime) fails the tests instead of the build.
gtest_discover_tests(sim_tests DISCOVERY_MODE PRE_TEST)
