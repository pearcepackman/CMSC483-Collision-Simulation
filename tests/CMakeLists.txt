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
# Lets tests load the real files in configs/ wherever ctest runs from (#13).
target_compile_definitions(sim_tests PRIVATE "SIM_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\"")
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
    cmake_parse_arguments(T "" "OUTPUT;MUST_NOT_EXIST;PROGRAM;FRESH_DIR" "ARGS;EXPECT_FILES" ${ARGN})
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
    if(T_FRESH_DIR)
        list(APPEND cmd "-DFRESH_DIR=${T_FRESH_DIR}")
    endif()
    if(T_EXPECT_FILES)
        list(JOIN T_EXPECT_FILES "|" joined_files)
        list(APPEND cmd "-DEXPECT_FILES=${joined_files}")
    endif()
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} ${cmd} -P ${CMAKE_CURRENT_SOURCE_DIR}/check_run.cmake)
endfunction()

add_cli_test(Cli.OrbitBench 0
    ARGS --scenario orbit --steps 1000 --bench
    OUTPUT "bench backend=serial n=2 threads=1 steps=1000 seconds_per_step=[^ ]+ precision=double\n")
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
        OUTPUT "bench backend=hip n=300 threads=0 .* precision=float\n.*hip device: ") # stdout, then stderr
    add_cli_test(Cli.HipBackendRunsInDouble 0
        ARGS --scenario cloud --n 300 --steps 3 --backend hip --gpu-precision double --bench
        OUTPUT " precision=double\n")
    set_tests_properties(Cli.HipBackendRunsInDouble PROPERTIES LABELS gpu)
    set_tests_properties(Cli.HipBackendRuns PROPERTIES LABELS gpu)
endif()
add_cli_test(Cli.InfiniteDtIsUsageError 1
    ARGS --scenario orbit --dt inf
    OUTPUT "--dt needs a finite number > 0")
# --out: a frame at steps 0, 10, 20 (numbered 0, 1, 2) plus energy.csv, and nothing more (#10).
set(out_dir ${CMAKE_CURRENT_BINARY_DIR}/cli_out_frames)
add_cli_test(Cli.OutWritesFramesAndEnergyLog 0
    ARGS --scenario orbit --steps 20 --output-every 10 --out ${out_dir}
    FRESH_DIR ${out_dir}
    EXPECT_FILES ${out_dir}/frame_00000.csv ${out_dir}/frame_00001.csv ${out_dir}/frame_00002.csv
                 ${out_dir}/energy.csv
    MUST_NOT_EXIST ${out_dir}/frame_00003.csv)
# The final step gets a frame even when it isn't a multiple of --output-every: 0, 10, 20, 25.
set(out_dir ${CMAKE_CURRENT_BINARY_DIR}/cli_out_final_frame)
add_cli_test(Cli.OutWritesFinalStepFrame 0
    ARGS --scenario orbit --steps 25 --output-every 10 --out ${out_dir}
    FRESH_DIR ${out_dir}
    EXPECT_FILES ${out_dir}/frame_00003.csv
    MUST_NOT_EXIST ${out_dir}/frame_00004.csv)

# --config: the CI smoke config loads, builds both planets and runs (#13). A bad config
# is a setup error (exit 1) that names the file, line and key.
# The particle count catches a planet silently dropped. smoke.cfg relaxes its planets
# (#15), so this also runs relax_planet end to end. head_on.cfg isn't run here: relaxing
# its 4,000 particles takes about a minute; Config.ShippedConfigsLoad checks its planets
# with planet_spec_ok instead.
add_cli_test(Cli.SmokeConfigRuns 0
    ARGS --config ${PROJECT_SOURCE_DIR}/configs/smoke.cfg --steps 2
    OUTPUT "ran 2 steps of 500 particles")
# Command-line overrides reach relaxation (#15): planets are built after --dt is applied.
# A dt of 1e300 makes relaxation itself blow up: it says so, and the starting state is
# already NaN/Inf ("after step 0"). If relaxation used the config's dt instead, the
# planets would be fine until the first real step ("after step 1").
# Sim.SetupRunAppliesOverridesBeforeRelaxing covers the same order for planet_view.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/relax_dt.cfg
    "output_every = 1\nsteps = 3\ndt = 0.002\nsoftening = 0.05\ncontact = true\n"
    "contact_k = 20\nrelax_drag = 5\nrelax_ke_threshold = 1e-6\nrelax_max_steps = 5\n"
    "planet0.n = 10\nplanet0.radius = 0.5\nplanet0.particle_radius = 0.06\nplanet0.mass = 0.01\n")
add_cli_test(Cli.DtOverrideReachesRelaxation 2
    ARGS --config ${CMAKE_CURRENT_BINARY_DIR}/relax_dt.cfg --dt 1e300
    OUTPUT "planet0 became NaN/Inf during relaxation, after 1 steps.*became NaN/Inf after step 0")
# A value make_planet can't build is a clean setup error naming the planet, not an abort.
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/bad_planet.cfg
    "output_every = 1\nsteps = 1\ndt = 0.01\nplanet0.n = 2000\nplanet0.radius = 1\n"
    "planet0.particle_radius = 0.07\nplanet0.mass = 2\n")
add_cli_test(Cli.BadPlanetConfigIsSetupError 1
    ARGS --config ${CMAKE_CURRENT_BINARY_DIR}/bad_planet.cfg
    OUTPUT "planet0: too many particles.*couldn't load config")
add_cli_test(Cli.MissingConfigIsSetupError 1
    ARGS --config ${CMAKE_CURRENT_BINARY_DIR}/no_such_config.cfg
    OUTPUT "no_such_config.cfg: can't open the file.*couldn't load config")

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
