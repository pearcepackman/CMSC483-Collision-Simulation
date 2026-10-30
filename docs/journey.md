# Project journey

How the simulator got here, one stage at a time: what was built, what it shows about parallel computing, and the evidence. Design details and full measurements live in [notes.md](notes.md); this file is the story (and the backbone of the presentation). **Add an entry with every PR that changes what the project can do or measure**, with numbers and a screenshot or figure where there's something to see. Screenshots go in [media/](media/), plots in [figures/](figures/).

All measurements: AMD Ryzen 5 7600X (6 cores / 12 threads), AMD Radeon RX 9060 XT (RDNA 4, `gfx1200`), Arch Linux, ROCm 7.2.4.

---

## Where it's going

Two particle planets collide, and the same O(N²) physics runs on three swappable backends: serial, OpenMP (CPU threads) and HIP (GPU). The final demo is the live viewer: switch backends mid-collision and watch the frame rate change. The presentation's speedup chart, experienced live.

The browser prototype that set the visual target (a JavaScript port of the same physics): https://claude.ai/artifact/HSeEFYtGkJCAVEtBv7Aj9f

**Still ahead:** planets, contact forces and relaxation (#9, #12, #15), the fast GPU kernels (float #32, tiled #26, data kept on the GPU #30), the glow look (#47), recorded runs to video (#49), full scaling plots (#34).

---

## Oct 22: Foundations (#1, #39, #40)

**Built:** CMake + Ninja build with presets, GoogleTest, CI on every PR (gcc and clang, AddressSanitizer, ThreadSanitizer), Struct-of-Arrays particle storage.

**Parallel-computing angle:** the testing strategy was set up before any parallel code existed. Parallel bugs (races, reordered floating-point sums) can look correct, so every backend is tested against one serial reference with relative tolerances, at 1 and 4 threads, and under ThreadSanitizer. Struct-of-Arrays was chosen so the force loop reads contiguous memory: SIMD-friendly on the CPU, coalesced loads on the GPU.

**Decision:** the GPU backend switched from CUDA to HIP, because the GPU we have is an AMD card. HIP is nearly line-for-line CUDA.

## Oct 22: Serial reference gravity (#41, #7)

**Built:** `gravity_serial.cpp`, the all-pairs reference every backend is tested against (never optimized).

**Evidence:** matches hand-computed 2- and 3-body forces to 1e-14; force equals minus the slope of the potential; 19 deliberately broken copies were all caught by the tests.

## Oct 22: Leapfrog instead of Euler (#8)

**Built:** kick-drift-kick leapfrog integrator.

**Evidence:** over 1,000 orbits leapfrog's energy error stays at 3.9e-10 while Euler's orbit spirals out to 6× its starting separation. Same cost (one force evaluation per step).

![Leapfrog vs Euler](figures/fig_leapfrog_vs_euler.png)

## Oct 22: OpenMP gravity, 6× on 6 cores (#42, #19)

**Built:** `#pragma omp parallel for schedule(static) default(none)` over particles. Each thread owns its particles' accelerations, so there are no races and no atomics.

**Evidence:** 5.9× at 6 threads (98% efficiency) at N = 10,000; 0 races under ThreadSanitizer + Archer, which does catch a deliberately racy canary.

**Parallel-computing lessons:**
- **8 threads is slower than 6:** static scheduling gives equal blocks, but two cores run two threads each, and everyone waits for them (load imbalance).
- **12 threads ≈ 6:** SMT (hyperthreads) share the same floating-point units, so a compute-bound loop gains nothing.
- **"123% efficiency" was measurement noise** (CPU boost/governor drift); the benchmark now interleaves configurations.

![OpenMP speedup](figures/fig_openmp_speedup.png)

## Oct 27: The simulation program (#43, #14)

**Built:** `planet_sim` with built-in scenarios (orbit, figure-8, random cloud), a benchmark mode, NaN/Inf detection with distinct exit codes, and an energy log.

**Evidence:** first end-to-end OpenMP number through the real program: 5,000 particles, 0.077 s/step serial vs 0.012 with 6 threads (6.4×).

## Oct 30: Project proposal (#44)

Proposal document and its figures. The collision figure is from an early prototype render, not our C++ code:

![Prototype collision render](figures/fig_collision.png)

## Oct 30: First GPU backend (#51, #24, #25)

**Built:** HIP kernel, one GPU thread per particle, in double precision; an optional `hip` build; a CI job that compiles it in an AMD ROCm container (CI has no GPU).

**Evidence:**
- Matches serial to 1e-12; 9 GPU tests; 8 deliberately broken kernels all caught.
- Whole-step speed vs OpenMP: 0.6× at 2k particles (slower), 1.3× at 10k, 2.1× at 30k.
- A scratch probe of the kernel alone: 25 ms in double vs **2.0 ms in float** at N = 10,000.

**Parallel-computing lessons:**
- **Precision matters enormously on consumer GPUs:** this card runs `double` about 13× slower than `float`, so in double it barely beats 6 CPU cores.
- **Small problems lose on the GPU:** launch and copy overheads dominate until N² work is large enough (a crossover point).
- **A real bug review found:** ROCm also lists the CPU's integrated GPU, and the runtime aborts a kernel launch if *any* visible GPU lacks code for it. Fixed by building for both chips and choosing the device by architecture.

## Oct 30: Live viewer with backend switching (#52, #45, #48)

**Built:** `planet_view`, a raylib 3D window running the same simulation. Keys 1/2/3 switch serial / OpenMP / GPU mid-run, with live ms/step and fps. CI runs it for 60 frames on a virtual screen.

**Evidence** (10,000-particle cloud, same moment, real display):

| Serial: 277 ms/step, 4 fps | OpenMP (12 threads): 47 ms/step, 21 fps | GPU (naive double): 28 ms/step, 36 fps |
|---|---|---|
| ![serial](media/viewer_10k_serial.png) | ![omp](media/viewer_10k_omp.png) | ![gpu](media/viewer_10k_hip.png) |

5,000 particles on the GPU at 60 fps, mid-collapse under their own gravity:

![collapse](media/viewer_5k_hip_collapse.png)

**Parallel-computing angle:** the viewer turns the speedup chart into something you can see: the same physics, the same frame, and the backend is the only thing that changes. The GPU numbers here are the slow first kernel; float and tiling come next.
