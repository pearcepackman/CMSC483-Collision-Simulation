#include "look.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

const char *const MODE_NAMES[COLOR_MODE_COUNT] = {"heat", "material", "planet", "speed", "plain"};

// Planet colors: orange for even bodies, blue for odd ones. Each has a rock (mantle) and
// an iron (core) shade; the iron ones are hotter-looking reds/magentas so cores stand out.
const Rgb ROCK[2] = {{0.80f, 0.66f, 0.48f}, {0.38f, 0.68f, 0.94f}};
const Rgb IRON[2] = {{1.00f, 0.36f, 0.12f}, {1.00f, 0.18f, 0.55f}};
const Rgb PLANET[2] = {{1.00f, 0.58f, 0.25f}, {0.27f, 0.74f, 1.00f}};
const Rgb PLAIN = {0.85f, 0.86f, 0.92f};
const float HEAT_BASE_DIM = 0.30f; // a cold particle in heat mode: its material, dimmed
const int MAX_BODY_ID = 1024;      // ids above this are ignored by look_scales

Rgb lerp(Rgb a, Rgb b, float t) {
    return Rgb{a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t};
}

// Piecewise-linear ramp through `n` evenly spaced stops; t is clamped to [0, 1].
Rgb ramp(const Rgb *stops, int n, double t) {
    t = std::min(1.0, std::max(0.0, t));
    double x = t * (n - 1);
    int k = std::min(n - 2, static_cast<int>(x));
    return lerp(stops[k], stops[k + 1], static_cast<float>(x - k));
}

// Glow of hot material: black, dark red, orange, yellow, white (roughly a blackbody).
const Rgb BLACKBODY[5] = {{0.00f, 0.00f, 0.00f},
                          {0.55f, 0.06f, 0.00f},
                          {1.00f, 0.45f, 0.05f},
                          {1.00f, 0.85f, 0.30f},
                          {1.00f, 1.00f, 1.00f}};

// Speed: dim violet (slow, still visible) to bright yellow-white (fast).
const Rgb SPEED[5] = {{0.18f, 0.06f, 0.36f},
                      {0.50f, 0.12f, 0.52f},
                      {0.86f, 0.28f, 0.30f},
                      {0.99f, 0.62f, 0.12f},
                      {1.00f, 0.96f, 0.70f}};

int parity(int body) {
    return body % 2 == 0 ? 0 : 1; // works for negative ids too
}

// Mass and extent of one body: center of mass, its velocity, and the farthest particle
// edge from the center.
struct Body {
    double m = 0.0, cx = 0.0, cy = 0.0, cz = 0.0, vx = 0.0, vy = 0.0, vz = 0.0, r = 0.0;
};

} // namespace

const char *color_mode_name(ColorMode m) {
    return m >= 0 && m < COLOR_MODE_COUNT ? MODE_NAMES[m] : "unknown";
}

bool color_mode_from_name(const char *name, ColorMode *out) {
    if (name == nullptr) {
        return false;
    }
    for (int m = 0; m < COLOR_MODE_COUNT; ++m) {
        if (std::strcmp(name, MODE_NAMES[m]) == 0) {
            *out = static_cast<ColorMode>(m);
            return true;
        }
    }
    return false;
}

ColorMode next_color_mode(ColorMode m) {
    return static_cast<ColorMode>((static_cast<int>(m) + 1) % COLOR_MODE_COUNT);
}

LookScales look_scales(const Particles &p) {
    // Bodies by id; ids are small (0, 1, ... in configs; 0 for the built-in scenarios).
    std::vector<Body> bodies;
    for (std::size_t i = 0; i < p.size(); ++i) {
        // Bodies are numbered from 0 per planet; an absurd id (from a hand-edited frame,
        // say) is skipped rather than making this vector huge.
        if (p.body[i] < 0 || p.body[i] > MAX_BODY_ID || !(p.mass[i] > 0.0)) {
            continue;
        }
        const std::size_t b = static_cast<std::size_t>(p.body[i]);
        if (b >= bodies.size()) {
            bodies.resize(b + 1);
        }
        Body &B = bodies[b];
        const double m = p.mass[i];
        B.m += m;
        B.cx += m * p.x[i];
        B.cy += m * p.y[i];
        B.cz += m * p.z[i];
        B.vx += m * p.vx[i];
        B.vy += m * p.vy[i];
        B.vz += m * p.vz[i];
    }
    for (Body &B : bodies) {
        if (B.m > 0.0) {
            B.cx /= B.m;
            B.cy /= B.m;
            B.cz /= B.m;
            B.vx /= B.m;
            B.vy /= B.m;
            B.vz /= B.m;
        }
    }
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (p.body[i] < 0 || p.body[i] > MAX_BODY_ID || !(p.mass[i] > 0.0)) {
            continue;
        }
        Body &B = bodies[static_cast<std::size_t>(p.body[i])];
        double dx = p.x[i] - B.cx, dy = p.y[i] - B.cy, dz = p.z[i] - B.cz;
        B.r = std::max(B.r, std::sqrt(dx * dx + dy * dy + dz * dz) + p.radius[i]);
    }
    // The two most massive bodies.
    std::sort(bodies.begin(), bodies.end(), [](const Body &a, const Body &b) { return a.m > b.m; });
    double speed = 0.0;
    if (bodies.size() >= 2 && bodies[1].m > 0.0) {
        const Body &a = bodies[0], &b = bodies[1];
        double dvx = a.vx - b.vx, dvy = a.vy - b.vy, dvz = a.vz - b.vz;
        double impact = std::sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
        double escape = a.r + b.r > 0.0 ? std::sqrt(2.0 * (a.m + b.m) / (a.r + b.r)) : 0.0;
        speed = std::max(impact, escape);
    } else if (!bodies.empty() && bodies[0].m > 0.0 && bodies[0].r > 0.0) {
        speed = std::sqrt(2.0 * bodies[0].m / bodies[0].r);
    }
    if (!(speed > 0.0) || !std::isfinite(speed)) {
        speed = 1.0;
    }
    return LookScales{speed, 0.5 * speed * speed};
}

double heat_glow(double mass, double heat, const LookScales &scales) {
    double per_mass = mass > 0.0 ? heat / mass : 0.0;
    double x = scales.heat_per_mass > 0.0 ? per_mass / scales.heat_per_mass : 0.0;
    if (!(x > 0.0) || !std::isfinite(x)) {
        return 0.0;
    }
    return std::min(1.0, std::max(0.0, (std::log10(x) + 2.5) / 2.5)); // 0.003 -> 0, 1 -> 1
}

Rgb particle_color(ColorMode mode, int body, int material, double mass, double speed, double heat,
                   const LookScales &scales) {
    const int k = parity(body);
    const Rgb base = material == MATERIAL_IRON ? IRON[k] : ROCK[k];
    switch (mode) {
    case COLOR_MATERIAL:
        return base;
    case COLOR_PLANET:
        return PLANET[k];
    case COLOR_SPEED:
        return ramp(SPEED, 5, speed / (scales.speed > 0.0 ? scales.speed : 1.0));
    case COLOR_PLAIN:
        return PLAIN;
    case COLOR_HEAT:
    default: {
        // Heat per unit mass against the scale, on a log ramp from 0.003x (barely warm) to
        // 1x (white-hot). The scale is about the impact's kinetic energy per unit mass, the
        // most a particle can be heated by, so white means "took the full impact". A
        // linear ramp would leave most of a collision dark; contact heat piles up on the
        // impact surfaces.
        Rgb c{base.r * HEAT_BASE_DIM, base.g * HEAT_BASE_DIM, base.b * HEAT_BASE_DIM};
        double t = heat_glow(mass, heat, scales);
        if (t > 0.0) {
            Rgb glow = ramp(BLACKBODY, 5, t);
            // Emission adds to the dim body color, so hotter is always brighter.
            c = Rgb{std::min(1.0f, c.r + glow.r), std::min(1.0f, c.g + glow.g),
                    std::min(1.0f, c.b + glow.b)};
        }
        return c;
    }
    }
}

float luma(Rgb c) {
    return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
}

float sprite_half_size(double radius, double glow_factor, double depth, double focal_px,
                       double min_pixels) {
    double h = glow_factor * radius;
    if (depth > 0.0 && focal_px > 0.0) {
        double min_world = min_pixels * depth / focal_px;
        h = std::max(h, min_world);
    }
    return static_cast<float>(h);
}
