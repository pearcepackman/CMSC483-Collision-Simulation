# Runs a program and checks its exit code and output, for tests of the real planet_sim
# binary. Usage (from add_test):
#   cmake -DCMD=<exe> "-DARGS=a|b|c" -DEXPECT_EXIT=<code> [-DEXPECT_OUTPUT=<regex>]
#         [-DMUST_NOT_EXIST=<path>] [-DFRESH_DIR=<dir>] ["-DEXPECT_FILES=p|q"]
#         -P check_run.cmake
# ARGS and EXPECT_FILES use '|' as the separator: a ';' list would be split into
# separate arguments by add_test before reaching this script.
# FRESH_DIR is deleted before the run, so files from an earlier run can't pass the
# EXPECT_FILES check.
string(REPLACE "|" ";" ARGS "${ARGS}")
string(REPLACE "|" ";" EXPECT_FILES "${EXPECT_FILES}")
if(DEFINED MUST_NOT_EXIST)
    file(REMOVE_RECURSE "${MUST_NOT_EXIST}") # a leftover from an earlier failed run
endif()
if(DEFINED FRESH_DIR)
    file(REMOVE_RECURSE "${FRESH_DIR}")
endif()
execute_process(COMMAND ${CMD} ${ARGS} RESULT_VARIABLE code OUTPUT_VARIABLE out ERROR_VARIABLE err)
message("${out}${err}")
if(NOT code EQUAL EXPECT_EXIT)
    message(FATAL_ERROR "expected exit code ${EXPECT_EXIT}, got ${code}")
endif()
if(DEFINED EXPECT_OUTPUT AND NOT "${out}${err}" MATCHES "${EXPECT_OUTPUT}")
    message(FATAL_ERROR "output doesn't match: ${EXPECT_OUTPUT}")
endif()
if(DEFINED MUST_NOT_EXIST AND EXISTS "${MUST_NOT_EXIST}")
    message(FATAL_ERROR "${MUST_NOT_EXIST} should not have been created")
endif()
foreach(f IN LISTS EXPECT_FILES)
    if(NOT EXISTS "${f}")
        message(FATAL_ERROR "${f} should have been created")
    endif()
endforeach()
