#pragma once

#include "particles.hpp"

// Writes <dir>/frame_<frame, 5 digits>.csv (e.g. frame_00042.csv): a header line
// "id,x,y,z,vx,vy,vz,mass,body,material,heat", then one row per particle with those
// columns (about 170 bytes per particle, #50). Doubles use "%.17g"
// so they read back bit-for-bit; lines end in \n on every OS. Does not create dir
// (the caller does). Returns false if the file can't be written or frame < 0.
bool write_frame(const Particles &p, const char *dir, int frame);

// Reads a file written by write_frame() into p: resizes p to the number of rows, fills
// every field the frame holds, and leaves the others (accelerations, radius, heat_rate)
// zero.
// Returns false, leaving p unchanged, if the file can't be read or is malformed
// (wrong header, wrong number of columns, ids not 0, 1, 2, ... in order, a body or
// material that isn't an int, heat that isn't a number). Any int is accepted as a
// material: only MATERIAL_ROCK and MATERIAL_IRON exist today, and the reader doesn't
// police that. Accepts \r\n line endings and one blank line at the end.
bool read_frame(const char *path, Particles &p);
