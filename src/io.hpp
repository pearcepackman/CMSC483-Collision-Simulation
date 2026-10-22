#pragma once

#include "particles.hpp"

// Writes <dir>/frame_<frame, 5 digits>.csv (e.g. frame_00042.csv) with columns
// id, x, y, z, body. Doubles use "%.17g" so they read back bit-for-bit. Does not
// create dir (the caller does). Returns false if the file can't be written.
bool write_frame(const Particles &p, const char *dir, int frame);

// Reads a file written by write_frame() into p: resizes p to the number of rows,
// fills positions and body ids, and leaves the other fields zero.
// Returns false if the file can't be read or is malformed.
bool read_frame(const char *path, Particles &p);
