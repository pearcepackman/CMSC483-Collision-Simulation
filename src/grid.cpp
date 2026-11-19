#include "grid.hpp"

#include <cassert>
#include <cmath>

std::int64_t cell_coord(double x, double cell_size) {
    const double limit = 1099511627776.0; // 2^40: far from int64 overflow, even +-1
    double q = std::floor(x / cell_size);
    if (!(q > -limit)) { // also catches NaN (-> 0 below)
        return q != q ? 0 : -static_cast<std::int64_t>(limit);
    }
    if (q > limit) {
        return static_cast<std::int64_t>(limit);
    }
    return static_cast<std::int64_t>(q);
}

std::size_t cell_slot(std::int64_t ix, std::int64_t iy, std::int64_t iz, std::size_t table_size) {
    // Three large primes (Teschner et al. 2003), mixed with xor. Unsigned arithmetic, so
    // negative coordinates and overflow wrap around instead of being undefined.
    std::uint64_t h = static_cast<std::uint64_t>(ix) * 73856093u ^
                      static_cast<std::uint64_t>(iy) * 19349663u ^
                      static_cast<std::uint64_t>(iz) * 83492791u;
    return static_cast<std::size_t>(h & (table_size - 1));
}

std::size_t cell_table_size(std::size_t n) {
    std::size_t size = 1;
    while (size < 2 * n) {
        size *= 2;
    }
    return size;
}

void build_cell_list(const Particles &p, double cell_size, std::size_t table_size, CellList &grid) {
    // A size that isn't a power of two would still stay in bounds, but the mask would
    // leave some slots unused and pile particles into the rest.
    assert(table_size > 0 && (table_size & (table_size - 1)) == 0);
    const std::size_t n = p.size();
    grid.cell_size = cell_size;
    grid.table_size = table_size;
    grid.cx.resize(n);
    grid.cy.resize(n);
    grid.cz.resize(n);
    grid.slot.resize(n);
    grid.order.resize(n);
    grid.start.assign(table_size + 1, 0);

    // Counting sort by slot: count each slot, turn counts into start positions, place.
    for (std::size_t i = 0; i < n; ++i) {
        grid.cx[i] = cell_coord(p.x[i], cell_size);
        grid.cy[i] = cell_coord(p.y[i], cell_size);
        grid.cz[i] = cell_coord(p.z[i], cell_size);
        grid.slot[i] = cell_slot(grid.cx[i], grid.cy[i], grid.cz[i], table_size);
        ++grid.start[grid.slot[i] + 1];
    }
    for (std::size_t s = 0; s < table_size; ++s) {
        grid.start[s + 1] += grid.start[s];
    }
    // start[s] is now where slot s begins. Place each particle at its slot's cursor;
    // going through i in increasing order keeps each slot's particles in index order.
    grid.cursor.assign(grid.start.begin(), grid.start.end() - 1);
    for (std::size_t i = 0; i < n; ++i) {
        grid.order[grid.cursor[grid.slot[i]]++] = i;
    }
}
