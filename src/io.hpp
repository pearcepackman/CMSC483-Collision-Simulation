#pragma once

#include "particles.hpp"

// Writes <dir>/frame_<frame, 5 digits>.csv (e.g. frame_00042.csv): a header line
// "id,x,y,z,body", then one row per particle with those columns. Doubles use "%.17g"
// so they read back bit-for-bit; lines end in \n on every OS. Does not create dir
// (the caller does). Returns false if the file can't be written or frame < 0.
bool write_frame(const Particles &p, const char *dir, int frame);

// Reads a file written by write_frame() into p: resizes p to the number of rows,
// fills positions and body ids, and leaves the other fields zero.
// Returns false, leaving p unchanged, if the file can't be read or is malformed
// (wrong header, wrong number of columns, ids not 0, 1, 2, ... in order, a body that
// isn't an int). Accepts \r\n line endings and one blank line at the end.
bool read_frame(const char *path, Particles &p);
