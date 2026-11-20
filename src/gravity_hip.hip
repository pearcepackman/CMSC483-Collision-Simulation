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

// GPU forces (#25, #26, #32, #66): one GPU thread per particle. Particles are read in
// tiles of BLOCK_SIZE through shared memory (fast on-chip memory each block shares), so a
// block reads each particle from GPU memory once per tile instead of once per thread.
// The same loop does gravity and, when contact is on, contact forces and heat, so the
// whole force step runs on the GPU. Float by default; double is the reference that
// matches the serial code to ~1e-12.

namespace {

const int BLOCK_SIZE = 256; // threads per block = particles per tile; a multiple of the
                            // 32-wide RDNA wavefront

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

// The kernel's inputs, one array each (Struct-of-Arrays, like Particles), in precision T.
// Positions come as a high and a low part, x = hi + lo: in float, the difference of two
// nearby positions far from the origin would otherwise lose the small overlaps contact
// depends on (at x = 2.5 float positions step by 2.4e-7, overlaps are ~1e-4). With
// (xj_hi - xi_hi) + (xj_lo - xi_lo) the separation keeps almost double accuracy. In
// double, lo is 0. Velocities stay plain float on purpose: the damper only needs their
// difference, and at our speeds (< ~1) its rounding is far below the measured error.
template <typename T> struct Inputs {
    const T *xh, *yh, *zh, *xl, *yl, *zl, *m, *r, *vx, *vy, *vz;
};

// Contact settings in precision T (contact = 0: gravity only, no heat).
template <typename T> struct ContactSettings {
    int contact;
    T k, c, dt;
};

// Gravity, the same formula as compute_gravity_serial, and contact with exactly
// compute_contact_bruteforce's rules (overlap test, spring-dashpot force, heat factor).
// The templates in this file only exist to build each piece in float and in double.
template <typename T>
__global__ void force_kernel(int n, T eps2, ContactSettings<T> cs, Inputs<T> in, T *__restrict__ ax,
                             T *__restrict__ ay, T *__restrict__ az, T *__restrict__ heat_rate) {
    __shared__ T s_xh[BLOCK_SIZE], s_yh[BLOCK_SIZE], s_zh[BLOCK_SIZE];
    __shared__ T s_xl[BLOCK_SIZE], s_yl[BLOCK_SIZE], s_zl[BLOCK_SIZE];
    __shared__ T s_m[BLOCK_SIZE], s_r[BLOCK_SIZE];
    __shared__ T s_vx[BLOCK_SIZE], s_vy[BLOCK_SIZE], s_vz[BLOCK_SIZE];

    const int t = static_cast<int>(threadIdx.x);
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    // Threads past n (the last block is partly empty) still load tiles and reach every
    // __syncthreads(); they just don't compute or write.
    const bool active = i < n;
    T xhi = 0, yhi = 0, zhi = 0, xli = 0, yli = 0, zli = 0, mi = 1, ri = 0, vxi = 0, vyi = 0,
      vzi = 0;
    if (active) {
        xhi = in.xh[i];
        yhi = in.yh[i];
        zhi = in.zh[i];
        xli = in.xl[i];
        yli = in.yl[i];
        zli = in.zl[i];
        mi = in.m[i];
        if (cs.contact) {
            ri = in.r[i];
            vxi = in.vx[i];
            vyi = in.vy[i];
            vzi = in.vz[i];
        }
    }
    T gx = 0, gy = 0, gz = 0; // gravity acceleration
    T fx = 0, fy = 0, fz = 0; // contact force (divided by mi at the end, like the CPU)
    T q = 0;                  // heating rate

    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + t;
        if (j < n) {
            s_xh[t] = in.xh[j];
            s_yh[t] = in.yh[j];
            s_zh[t] = in.zh[j];
            s_xl[t] = in.xl[j];
            s_yl[t] = in.yl[j];
            s_zl[t] = in.zl[j];
            s_m[t] = in.m[j];
            if (cs.contact) {
                s_r[t] = in.r[j];
                s_vx[t] = in.vx[j];
                s_vy[t] = in.vy[j];
                s_vz[t] = in.vz[j];
            }
        }
        __syncthreads();
        // Only the tile's real particles: padding past n is skipped by index, never read.
        const int count = n - base < BLOCK_SIZE ? n - base : BLOCK_SIZE;
        if (active) {
            for (int s = 0; s < count; ++s) {
                if (base + s == i) {
                    continue;
                }
                T dx = (s_xh[s] - xhi) + (s_xl[s] - xli); // r_j - r_i, as in the serial code
                T dy = (s_yh[s] - yhi) + (s_yl[s] - yli);
                T dz = (s_zh[s] - zhi) + (s_zl[s] - zli);
                T d2 = dx * dx + dy * dy + dz * dz;
                T r2 = d2 + eps2;
                T inv_r3 = T(1) / (r2 * sqrt(r2));
                gx += s_m[s] * dx * inv_r3;
                gy += s_m[s] * dy * inv_r3;
                gz += s_m[s] * dz * inv_r3;
                if (cs.contact) {
                    T reach = ri + s_r[s];
                    if (d2 < reach * reach && d2 != T(0)) {
                        T d = sqrt(d2);
                        T nx = -dx / d, ny = -dy / d, nz = -dz / d; // unit vector from j to i
                        T overlap = reach - d;
                        T vn = (vxi - s_vx[s]) * nx + (vyi - s_vy[s]) * ny + (vzi - s_vz[s]) * nz;
                        T f = cs.k * overlap - cs.c * vn;
                        fx += f * nx;
                        fy += f * ny;
                        fz += f * nz;
                        T m_eff = mi * s_m[s] / (mi + s_m[s]);
                        q += T(0.5) * cs.c * vn * vn * (T(1) - T(0.5) * cs.c * cs.dt / m_eff);
                    }
                }
            }
        }
        __syncthreads(); // the tile is read by everyone before it is overwritten
    }
    if (active) {
        ax[i] = gx + fx / mi;
        ay[i] = gy + fy / mi;
        az[i] = gz + fz / mi;
        heat_rate[i] = q;
    }
}

// Chosen device, precision, and the GPU buffers, kept between calls and grown only when
// n exceeds the capacity. Each buffer holds n doubles, enough for either precision.
int g_device = -1;
std::string g_device_info;
bool g_use_double = false; // float by default (hip_set_double)
std::size_t g_capacity = 0;
const int kBuffers = 15; // 11 inputs, 4 outputs
void *d_buf[kBuffers] = {};

void ensure_capacity(std::size_t n) {
    if (n <= g_capacity) {
        return;
    }
    hip_release();
    for (void *&buf : d_buf) {
        HIP_CHECK(hipMalloc(&buf, n * sizeof(double)));
    }
    g_capacity = n;
}

// Host copies in precision T, reused between calls.
template <typename T> std::vector<T> &staging(int which) {
    static std::vector<T> host[kBuffers];
    return host[which];
}

// Fills staging buffer `which` from a double array (hi part), and buffer `which_lo` with
// the remainder (lo part, 0 in double).
template <typename T>
void stage_split(const std::vector<double> &src, int which, int which_lo, std::size_t n) {
    std::vector<T> &hi = staging<T>(which);
    std::vector<T> &lo = staging<T>(which_lo);
    hi.resize(n);
    lo.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        hi[i] = static_cast<T>(src[i]);
        lo[i] = static_cast<T>(src[i] - static_cast<double>(hi[i]));
    }
}

template <typename T> void stage(const std::vector<double> &src, int which, std::size_t n) {
    std::vector<T> &dst = staging<T>(which);
    dst.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<T>(src[i]);
    }
}

template <typename T> void upload(int which, std::size_t n) {
    HIP_CHECK(
        hipMemcpy(d_buf[which], staging<T>(which).data(), n * sizeof(T), hipMemcpyHostToDevice));
}

template <typename T> void download(std::vector<double> &dst, int which, std::size_t n) {
    std::vector<T> &tmp = staging<T>(which);
    tmp.resize(n);
    HIP_CHECK(hipMemcpy(tmp.data(), d_buf[which], n * sizeof(T), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<double>(tmp[i]);
    }
}

enum { XH, YH, ZH, XL, YL, ZL, M, R, VX, VY, VZ, AX, AY, AZ, Q };

// One force evaluation in precision T. With contact, also sets heat_rate.
template <typename T> void run_forces(Particles &p, const SimParams &params, bool contact) {
    const std::size_t n = p.size();
    stage_split<T>(p.x, XH, XL, n);
    stage_split<T>(p.y, YH, YL, n);
    stage_split<T>(p.z, ZH, ZL, n);
    stage<T>(p.mass, M, n);
    for (int b : {XH, YH, ZH, XL, YL, ZL, M}) {
        upload<T>(b, n);
    }
    if (contact) {
        stage<T>(p.radius, R, n);
        stage<T>(p.vx, VX, n);
        stage<T>(p.vy, VY, n);
        stage<T>(p.vz, VZ, n);
        for (int b : {R, VX, VY, VZ}) {
            upload<T>(b, n);
        }
    }
    Inputs<T> in;
    const T **fields[] = {&in.xh, &in.yh, &in.zh, &in.xl, &in.yl, &in.zl,
                          &in.m,  &in.r,  &in.vx, &in.vy, &in.vz};
    for (int b = XH; b <= VZ; ++b) {
        *fields[b] = static_cast<const T *>(d_buf[b]);
    }
    ContactSettings<T> cs;
    cs.contact = contact ? 1 : 0;
    cs.k = static_cast<T>(params.contact_k);
    cs.c = static_cast<T>(params.contact_c);
    cs.dt = static_cast<T>(params.dt);

    const int count = static_cast<int>(n);
    const int blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE; // round up: cover every particle
    const T eps2 = static_cast<T>(params.softening * params.softening);
    const dim3 grid(static_cast<unsigned>(blocks)), block(static_cast<unsigned>(BLOCK_SIZE));
    force_kernel<T><<<grid, block>>>(count, eps2, cs, in, static_cast<T *>(d_buf[AX]),
                                     static_cast<T *>(d_buf[AY]), static_cast<T *>(d_buf[AZ]),
                                     static_cast<T *>(d_buf[Q]));
    HIP_CHECK(hipGetLastError()); // launch errors (e.g. no kernel for this GPU's architecture)

    // hipMemcpy waits for the kernel to finish, and reports errors from inside it.
    download<T>(p.ax, AX, n);
    download<T>(p.ay, AY, n);
    download<T>(p.az, AZ, n);
    if (contact) {
        download<T>(p.heat_rate, Q, n);
    }
}

// Checks n and the device before a launch; returns false if there is nothing to do.
bool prepare(std::size_t n, const char *who) {
    if (n == 0) {
        return false; // a launch with zero blocks is an error, and there's nothing to compute
    }
    if (n > static_cast<std::size_t>(INT_MAX)) { // the kernel indexes particles with int
        std::fprintf(stderr, "%s: %zu particles is more than the kernel can index\n", who, n);
        std::abort();
    }
    if (g_device < 0) {
        std::string info;
        if (!hip_init(&info)) {
            std::fprintf(stderr, "%s: %s\n", who, info.c_str());
            std::abort();
        }
    }
    // The current device belongs to each host thread; set it every call so a caller on
    // another thread (e.g. a viewer) can't silently launch on device 0.
    HIP_CHECK(hipSetDevice(g_device));
    ensure_capacity(n);
    return true;
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

void hip_set_double(bool use_double) {
    g_use_double = use_double;
}

bool hip_uses_double() {
    return g_use_double;
}

void compute_gravity_hip(Particles &p, const SimParams &params) {
    if (!prepare(p.size(), "compute_gravity_hip")) {
        return;
    }
    if (g_use_double) {
        run_forces<double>(p, params, false);
    } else {
        run_forces<float>(p, params, false);
    }
}

void compute_forces_hip(Particles &p, const SimParams &params) {
    if (!params.contact_enabled) {
        for (std::size_t i = 0; i < p.size(); ++i) {
            p.heat_rate[i] = 0.0; // only contact heats, so no stale rate from earlier
        }
    }
    if (!prepare(p.size(), "compute_forces_hip")) {
        return;
    }
    if (g_use_double) {
        run_forces<double>(p, params, params.contact_enabled);
    } else {
        run_forces<float>(p, params, params.contact_enabled);
    }
}

void hip_release() {
    for (void *&buf : d_buf) {
        if (buf != nullptr) {
            HIP_CHECK(hipFree(buf));
            buf = nullptr;
        }
    }
    g_capacity = 0;
}
