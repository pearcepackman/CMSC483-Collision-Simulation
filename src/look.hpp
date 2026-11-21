#pragma once

#include <cstddef>

#include "particles.hpp"

// How the viewer colors and sizes particles (#47). Plain functions with no graphics
// library, so they are unit-tested in every build. planet_view draws with them.

// Color modes, in the order the C key cycles through them.
enum ColorMode {
    COLOR_HEAT,     // dim material color, plus a glow that rises with heat (the default):
                    // white at heat per mass = LookScales::heat_per_mass
    COLOR_MATERIAL, // iron core vs rock mantle, a pair of colors per planet
    COLOR_PLANET,   // which planet a particle started in
    COLOR_SPEED,    // |v| on a dark-to-bright ramp
    COLOR_PLAIN,    // one color, no glow: the "before" picture
    COLOR_MODE_COUNT
};

const char *color_mode_name(ColorMode m); // "heat", "material", "planet", "speed", "plain"

// Parses a name above (exact). Returns false and leaves *out alone if it's unknown.
bool color_mode_from_name(const char *name, ColorMode *out);

ColorMode next_color_mode(ColorMode m); // wraps around after the last one

struct Rgb {
    float r, g, b; // 0 to 1
};

// Scales that turn speed and heat into colors. Computed once per scenario (setup and
// restart), never per frame, so a color means the same thing for the whole run.
struct LookScales {
    double speed;         // a "fast" speed for this scenario
    double heat_per_mass; // white-hot heat per unit mass: 0.5 * speed^2, about the impact's
                          // kinetic energy per unit mass
};

// speed = the larger of the two biggest bodies' relative speed and their mutual surface
// escape speed sqrt(2 G (M1 + M2) / (R1 + R2)) (G = 1). With one body (e.g. the cloud
// scenario) it's that body's own surface escape speed. Falls back to 1 if there is no
// mass or no size.
LookScales look_scales(const Particles &p);

// One particle's color. heat is in energy units, as in Particles::heat; it is divided by
// mass, so a heavy iron particle and a light rock particle with the same heat per mass
// look the same. speed is |v| in the frame the caller chooses.
Rgb particle_color(ColorMode mode, int body, int material, double mass, double speed, double heat,
                   const LookScales &scales);

// How hot a particle looks, from 0 (no glow: below 0.003x the heat scale) to 1 (white-hot:
// at the scale or above), on the same log ramp heat mode colors with. The viewer makes
// hot particles emit more light, so a lone white-hot speck still reads as white.
double heat_glow(double mass, double heat, const LookScales &scales);

// Brightness used by the tests (Rec. 709 luma).
float luma(Rgb c);

// Half-size in world units of a particle's glow sprite: glow_factor * radius, but never
// smaller than min_pixels on screen. focal_px is pixels per world unit at depth 1, so a
// half-size h at depth d covers h * focal_px / d pixels. Non-positive depth: no minimum.
float sprite_half_size(double radius, double glow_factor, double depth, double focal_px,
                       double min_pixels);
