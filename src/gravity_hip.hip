#include "gravity.hpp"

#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

// The GPU architectures the kernel was compiled for (CMAKE_HIP_ARCHITECTURES, set by
// CMake as e.g. "gfx1200,gfx1036"). Launching on any other GPU aborts inside the runtime.
#ifndef SIM_HIP_ARCHS
#error "SIM_HIP_ARCHS must be defined by the build (see CMakeLists.txt)"
#endif

// Naive HIP gravity (#25): one GPU thread per particle, each looping over all others
// in global memory. The baseline every GPU optimization (#26 tiling, #30 keeping data
// on the GPU, #32 float) is measured against.

namespace {

const int BLOCK_SIZE = 256; // threads per block; a multiple of the 32-wide RDNA wavefront

// Every HIP call goes through this: an error stops the program with where it happened,
// instead of continuing with garbage accelerations.
void hip_check(hipError_t err, const char *call, const char *file, int line) {
    if (err != hipSuccess) {
        std::fprintf(stderr, "HIP error: %s\n  in %s\n  at %s:%d\n", hipGetErrorString(err), call,
                     file, line);
        std::abort();
    }
}
#define HIP_CHECK(call) hip_check((call), #call, __FILE__, __LINE__)

// Same formula and summation order as compute_gravity_serial: particle i adds the pull
// of j = 0, 1, ..., n-1 in turn, skipping itself.
__global__ void gravity_kernel(int n, double eps2, const double *__restrict__ x,
                               const double *__restrict__ y, const double *__restrict__ z,
                               const double *__restrict__ mass, double *__restrict__ ax,
                               double *__restrict__ ay, double *__restrict__ az) {
    int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) {
        return; // the last block is partly empty when n isn't a multiple of BLOCK_SIZE
    }
    double xi = x[i], yi = y[i], zi = z[i];
    double axi = 0.0, ayi = 0.0, azi = 0.0;
    for (int j = 0; j < n; ++j) {
        if (j == i) {
            continue;
        }
        double dx = x[j] - xi;
        double dy = y[j] - yi;
        double dz = z[j] - zi;
        double r2 = dx * dx + dy * dy + dz * dz + eps2;
        double inv_r3 = 1.0 / (r2 * sqrt(r2));
        axi += mass[j] * dx * inv_r3;
        ayi += mass[j] * dy * inv_r3;
        azi += mass[j] * dz * inv_r3;
    }
    ax[i] = axi;
    ay[i] = ayi;
    az[i] = azi;
}

// Chosen device and the GPU copies of the particle arrays (Struct-of-Arrays, like
// Particles), kept between calls and grown only when n exceeds the capacity.
int g_device = -1;
std::string g_device_info;
std::size_t g_capacity = 0;
double *d_x = nullptr, *d_y = nullptr, *d_z = nullptr, *d_mass = nullptr;
double *d_ax = nullptr, *d_ay = nullptr, *d_az = nullptr;

void ensure_capacity(std::size_t n) {
    if (n <= g_capacity) {
        return;
    }
    hip_release();
    const std::size_t bytes = n * sizeof(double);
    for (double **buf : {&d_x, &d_y, &d_z, &d_mass, &d_ax, &d_ay, &d_az}) {
        HIP_CHECK(hipMalloc(reinterpret_cast<void **>(buf), bytes));
    }
    g_capacity = n;
}

void to_device(double *dst, const std::vector<double> &src, std::size_t n) {
    HIP_CHECK(hipMemcpy(dst, src.data(), n * sizeof(double), hipMemcpyHostToDevice));
}

void to_host(std::vector<double> &dst, const double *src, std::size_t n) {
    HIP_CHECK(hipMemcpy(dst.data(), src, n * sizeof(double), hipMemcpyDeviceToHost));
}

} // namespace

bool hip_init(std::string *info) {
    if (g_device >= 0) {
        *info = g_device_info;
        return true;
    }
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count == 0) {
        *info = std::string("no usable GPU found (") +
                (err != hipSuccess ? hipGetErrorString(err) : "0 devices") +
                "); check HIP_VISIBLE_DEVICES and that ROCm sees the card (rocminfo)";
        return false;
    }
    // Most compute units among the GPUs the kernel was built for: the RX 9060 XT over the
    // Ryzen's integrated GPU, which ROCm also lists.
    std::vector<hipDeviceProp_t> props(static_cast<std::size_t>(count));
    std::vector<int> units(props.size(), -1);
    std::vector<const char *> archs(props.size(), "");
    std::string seen;
    for (std::size_t d = 0; d < props.size(); ++d) {
        if (hipGetDeviceProperties(&props[d], static_cast<int>(d)) == hipSuccess) {
            units[d] = props[d].multiProcessorCount;
            archs[d] = props[d].gcnArchName;
            seen += (seen.empty() ? "" : ", ") + std::string(archs[d]);
        }
    }
    const int best = pick_gpu(count, units.data(), archs.data(), SIM_HIP_ARCHS);
    if (best < 0) {
        *info = "no visible GPU matches the architectures this build supports (" +
                std::string(SIM_HIP_ARCHS) + "; found: " + seen +
                "); add yours to CMAKE_HIP_ARCHITECTURES in the hip preset";
        return false;
    }
    if (hipSetDevice(best) != hipSuccess) {
        *info = "no usable GPU found (couldn't select device " + std::to_string(best) + ")";
        return false;
    }
    const hipDeviceProp_t &chosen = props[static_cast<std::size_t>(best)];
    g_device = best;
    g_device_info = std::string(chosen.name) + " (" + chosen.gcnArchName + ", device " +
                    std::to_string(best) + " of " + std::to_string(count) + ")";
    *info = g_device_info;
    return true;
}

void compute_gravity_hip(Particles &p, const SimParams &params) {
    const std::size_t n = p.size();
    if (n == 0) {
        return; // a launch with zero blocks is an error, and there's nothing to compute
    }
    if (n > static_cast<std::size_t>(INT_MAX)) { // the kernel indexes particles with int
        std::fprintf(stderr,
                     "compute_gravity_hip: %zu particles is more than the kernel "
                     "can index\n",
                     n);
        std::abort();
    }
    if (g_device < 0) {
        std::string info;
        if (!hip_init(&info)) {
            std::fprintf(stderr, "compute_gravity_hip: %s\n", info.c_str());
            std::abort();
        }
    }
    // The current device belongs to each host thread; set it every call so a caller on
    // another thread (e.g. a viewer) can't silently launch on device 0.
    HIP_CHECK(hipSetDevice(g_device));
    ensure_capacity(n);
    to_device(d_x, p.x, n);
    to_device(d_y, p.y, n);
    to_device(d_z, p.z, n);
    to_device(d_mass, p.mass, n);

    const int count = static_cast<int>(n);
    const int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE; // round up: cover every particle
    const double eps2 = params.softening * params.softening;
    const dim3 grid(static_cast<unsigned>(blocks)), block(static_cast<unsigned>(BLOCK_SIZE));
    gravity_kernel<<<grid, block>>>(count, eps2, d_x, d_y, d_z, d_mass, d_ax, d_ay, d_az);
    HIP_CHECK(hipGetLastError()); // launch errors (e.g. no kernel for this GPU's architecture)

    // hipMemcpy waits for the kernel to finish, and reports errors from inside it.
    to_host(p.ax, d_ax, n);
    to_host(p.ay, d_ay, n);
    to_host(p.az, d_az, n);
}

void hip_release() {
    for (double **buf : {&d_x, &d_y, &d_z, &d_mass, &d_ax, &d_ay, &d_az}) {
        if (*buf != nullptr) {
            HIP_CHECK(hipFree(*buf));
            *buf = nullptr;
        }
    }
    g_capacity = 0;
}
