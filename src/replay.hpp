#pragma once

#include <string>
#include <vector>

#include "particles.hpp"

// Playing back a recorded run (#49): the parts of planet_view's replay and export that
// need no graphics, so they are unit-tested in every build.

struct ReplayRun {
    std::vector<std::string> paths; // <dir>/frame_*.csv, sorted by frame number
    std::vector<double> times;      // t of each frame, from energy.csv; empty if unusable
};

// Lists <dir>/frame_*.csv in frame-number order (frame_99999 before frame_100000) and
// reads energy.csv's t column when it has exactly one row per frame with step strictly
// increasing (planet_sim writes a frame and an energy row in the same call; the last gap
// can be shorter because the final step is always written). Otherwise times stays empty.
// Returns false with a message if dir has no frames, or if frame 0 was written before
// the radius column (#49), which replay needs for particle sizes.
bool replay_open(const char *dir, ReplayRun *run, std::string *error);

// Replay time of the first and last frame: times.front() and times.back(), or 0 and
// frames - 1 without times.
double replay_start(const ReplayRun &run);
double replay_end(const ReplayRun &run);

// Where replay time t falls: frame k and fraction s in [0, 1) toward frame k + 1. t is
// clamped to [start, end]; at the end (or with a single frame) k is the last frame and
// s = 0. Without times, frame k is at t = k.
void replay_locate(const ReplayRun &run, double t, int *k, double *s);

// The particles between frame a (at time ta) and frame b (at tb), at fraction s in
// [0, 1]: positions follow the cubic Hermite curve through both frames' positions and
// velocities (exact when the acceleration is constant between the frames), and
// velocities are that curve's derivative. With tb <= ta (no times) positions and
// velocities are linear in s. Heat is linear in s; mass, radius, body and material are
// a's. Returns false, leaving out alone, if a and b have different particle counts or
// body ids (not frames of the same run).
bool interpolate_frames(const Particles &a, const Particles &b, double ta, double tb, double s,
                        Particles *out);

// What happened when replay tried to show a moment of the run.
enum ReplayShow { REPLAY_OK, REPLAY_READ_FAILED, REPLAY_NOT_FINITE, REPLAY_MISMATCH };

// Reads one frame file into p (read_frame, or a stand-in in tests).
typedef bool (*FrameReader)(const char *path, Particles &p);

// The two saved frames around the moment shown, kept so replay reads a frame only when it
// moves past one. ka and kb are their frame indices; -1 means not loaded.
struct ReplayCache {
    Particles fa, fb;
    int ka = -1, kb = -1;
};

// Puts the run at replay time t into *out: the saved frame, or a point between two
// (interpolate_frames). Reads frames through `read` only when they aren't already in the
// cache: moving forward by one frame reuses the later frame as the earlier one. *k_shown
// is the earlier frame's index. On failure *error names the file(s), the failed frame is
// dropped from the cache (so the next call reads it again), and *out is left alone,
// except for frames that don't match each other, where *out becomes frame k alone.
ReplayShow replay_cache_show(const ReplayRun &run, double t, ReplayCache *cache, FrameReader read,
                             Particles *out, int *k_shown, std::string *error);

// Number of export images for replay times [t0, t1] at `speed` (replay time per second
// of video) and `fps`, both ends included: ceil((t1 - t0) / speed * fps - 1e-9) + 1. The
// last image's time is clamped to t1 (export_time), so a video always ends on the last
// frame, and the 1e-9 keeps floating-point error (0.3 / 0.1 = 2.9999999999999996, or
// 3.0000000000000004) from adding an image. 1 for a zero-length run. -1 if the inputs
// aren't finite and positive (t1 < t0 included) or the count would be over 1,000,000.
long export_frame_count(double t0, double t1, double speed, double fps);

// Replay time of export image i: t0 + i * speed / fps, never past t1.
double export_time(long i, double t0, double t1, double speed, double fps);

// True if every position, velocity and heat in p is finite.
bool frame_is_finite(const Particles &p);
