# Runs a program and checks its exit code and output, for tests of the real planet_sim
# binary. Usage (from add_test):
#   cmake -DCMD=<exe> "-DARGS=a|b|c" -DEXPECT_EXIT=<code> [-DEXPECT_OUTPUT=<regex>]
#         [-DMUST_NOT_EXIST=<path>] -P check_run.cmake
# ARGS uses '|' as the separator: a ';' list would be split into separate arguments by
# add_test before reaching this script.
string(REPLACE "|" ";" ARGS "${ARGS}")
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
