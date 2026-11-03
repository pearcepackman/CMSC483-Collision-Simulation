#include "io.hpp"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

static const char *FRAME_HEADER = "id,x,y,z,body";

bool write_frame(const Particles &p, const char *dir, int frame) {
    if (dir == nullptr || frame < 0) {
        return false;
    }
    char name[32];
    std::snprintf(name, sizeof name, "/frame_%05d.csv", frame);
    const std::string path = std::string(dir) + name;
    // Binary mode, so lines end in \n on every OS (text mode on Windows writes \r\n).
    FILE *f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        return false;
    }
    std::fprintf(f, "%s\n", FRAME_HEADER);
    for (std::size_t i = 0; i < p.size(); ++i) {
        std::fprintf(f, "%zu,%.17g,%.17g,%.17g,%d\n", i, p.x[i], p.y[i], p.z[i], p.body[i]);
    }
    bool ok = std::ferror(f) == 0;
    if (std::fclose(f) != 0) { // fclose flushes, so a full disk often shows up here
        ok = false;
    }
    return ok;
}

enum LineResult { LINE_OK, LINE_END, LINE_BAD };

// Reads one line into buf and strips its line ending (\n or \r\n). LINE_BAD means
// the line didn't fit in buf; no real frame row comes close.
static LineResult read_line(FILE *f, char *buf, int size) {
    if (std::fgets(buf, size, f) == nullptr) {
        return LINE_END;
    }
    std::size_t len = std::strlen(buf);
    if (len == 0) { // a NUL byte at the start of the line: not a text file
        return LINE_BAD;
    }
    if (buf[len - 1] == '\n') {
        buf[--len] = '\0';
    } else if (!std::feof(f)) {
        return LINE_BAD;
    }
    if (len > 0 && buf[len - 1] == '\r') {
        buf[--len] = '\0';
    }
    return LINE_OK;
}

// Parses one row "id,x,y,z,body". Returns false unless the whole line is exactly that.
static bool parse_row(const char *line, unsigned long long *id, double *x, double *y, double *z,
                      int *body) {
    int used = 0; // characters parsed so far
    if (std::sscanf(line, "%llu,%lf,%lf,%lf,%n", id, x, y, z, &used) != 4 || used == 0) {
        return false;
    }
    // strtol, not %d: sscanf's behavior on a number too big for an int is undefined.
    const char *start = line + used;
    char *end = nullptr;
    errno = 0;
    long b = std::strtol(start, &end, 10);
    if (end == start || *end != '\0' || errno == ERANGE || b < INT_MIN || b > INT_MAX) {
        return false;
    }
    *body = static_cast<int>(b);
    return true;
}

bool read_frame(const char *path, Particles &p) {
    if (path == nullptr) {
        return false;
    }
    FILE *f = std::fopen(path, "rb");
    if (f == nullptr) {
        return false;
    }
    // Fill a copy, so p is untouched if the file turns out to be bad.
    Particles q;
    char line[256];
    bool ok = read_line(f, line, sizeof line) == LINE_OK && std::strcmp(line, FRAME_HEADER) == 0;
    while (ok) {
        LineResult r = read_line(f, line, sizeof line);
        if (r == LINE_END) {
            break;
        }
        if (r == LINE_OK && line[0] == '\0') {
            // Some editors add one blank line at the end; anywhere else it's an error.
            ok = read_line(f, line, sizeof line) == LINE_END;
            break;
        }
        unsigned long long id = 0;
        double x = 0.0, y = 0.0, z = 0.0;
        int body = 0;
        if (r == LINE_BAD || !parse_row(line, &id, &x, &y, &z, &body) || id != q.size()) {
            ok = false;
            break;
        }
        const std::size_t i = q.size();
        q.resize(i + 1);
        q.x[i] = x;
        q.y[i] = y;
        q.z[i] = z;
        q.body[i] = body;
    }
    if (std::ferror(f) != 0) {
        ok = false;
    }
    std::fclose(f);
    if (ok) {
        p = std::move(q);
    }
    return ok;
}
