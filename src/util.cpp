#include "util.hpp"

#include <cstdio>
#include <cstdlib>

void not_implemented(const char *what) {
    std::fprintf(stderr, "not implemented: %s\n", what);
    std::abort();
}
