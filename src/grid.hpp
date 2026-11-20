#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "particles.hpp"

// Hashed uniform grid (a "cell list") for finding nearby particles in O(N).
// Space is cut into cubes of side cell_size; a particle is in cell
// (floor(x / cell_size), floor(y / cell_size), floor(z / cell_size)). The three cell
// coordinates are hashed into a table of table_size slots, so memory stays O(N) however
// far apart the particles are (stragglers far from a collision don't stretch anything).
// Two different cells can land in the same slot; code that walks a slot must skip the
// particles whose cell coordinates (cx, cy, cz) aren't the cell it asked for.
//
// Layout (all Struct-of-Arrays, like Particles): the particles of slot s are
//   order[start[s]], ..., order[start[s + 1] - 1]
// in increasing particle index (a stable counting sort), so the result is the same on
// every run.
struct CellList {
    double cell_size = 0.0;
    std::size_t table_size = 0;           // a power of two
    std::vector<std::size_t> start;       // table_size + 1 entries
    std::vector<std::size_t> order;       // particle indices sorted by slot
    std::vector<std::int64_t> cx, cy, cz; // each particle's cell coordinates
    std::vector<std::size_t> slot;        // each particle's slot
    std::vector<std::size_t> cursor;      // scratch for the sort (table_size entries)
};

// Cell coordinate of x: floor(x / cell_size), clamped to +-2^40 so huge, infinite or NaN
// positions (NaN -> 0) can't overflow the int64 cast. Clamping only ever merges far-away
// cells, so particles that touch still end up in the same or neighboring cells.
std::int64_t cell_coord(double x, double cell_size);

// Slot of cell (ix, iy, iz) in a table of table_size slots (a power of two).
std::size_t cell_slot(std::int64_t ix, std::int64_t iy, std::int64_t iz, std::size_t table_size);

// The table size used for n particles: the smallest power of two >= 2n (at least 1), so
// most slots hold one cell.
std::size_t cell_table_size(std::size_t n);

// Fills grid for the particles in p. cell_size must be > 0 and table_size a power of two.
// Reuses grid's arrays (they only grow), so calling it every step allocates nothing
// after the first call.
void build_cell_list(const Particles &p, double cell_size, std::size_t table_size, CellList &grid);
