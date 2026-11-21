#include "io.hpp"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

// Since #50 a frame holds everything the analysis (#67) and replay (#49) need, not just
// positions: one format change instead of one per feature.
static const char *FRAME_HEADER = "id,x,y,z,vx,vy,vz,mass,body,material,heat";

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
        std::fprintf(f, "%zu,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,%.17g\n", i, p.x[i],
                     p.y[i], p.z[i], p.vx[i], p.vy[i], p.vz[i], p.mass[i], p.body[i], p.material[i],
                     p.heat[i]);
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

// One parsed row of a frame.
struct FrameRow {
    unsigned long long id = 0;
    double x = 0.0, y = 0.0, z = 0.0, vx = 0.0, vy = 0.0, vz = 0.0, mass = 0.0, heat = 0.0;
    int body = 0, material = 0;
};

// An int at *s followed by `next` (',' or '\0'); moves *s past both. strtol, not %d:
// sscanf's behavior on a number too big for an int is undefined.
static bool parse_int_field(const char **s, char next, int *out) {
    char *end = nullptr;
    errno = 0;
    long v = std::strtol(*s, &end, 10);
    if (end == *s || *end != next || errno == ERANGE || v < INT_MIN || v > INT_MAX) {
        return false;
    }
    *out = static_cast<int>(v);
    *s = end + (next == '\0' ? 0 : 1);
    return true;
}

// Parses one row "id,x,y,z,vx,vy,vz,mass,body,material,heat". Returns false unless the
// whole line is exactly that.
static bool parse_row(const char *line, FrameRow *r) {
    int used = 0; // characters parsed so far
    if (std::sscanf(line, "%llu,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%n", &r->id, &r->x, &r->y, &r->z,
                    &r->vx, &r->vy, &r->vz, &r->mass, &used) != 8 ||
        used == 0) {
        return false;
    }
    const char *s = line + used;
    if (!parse_int_field(&s, ',', &r->body) || !parse_int_field(&s, ',', &r->material)) {
        return false;
    }
    char *end = nullptr;
    r->heat = std::strtod(s, &end);
    return end != s && *end == '\0';
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
    char line[512]; // a row is at most ~260 characters (8 doubles at %.17g, 2 ints, an id)
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
        FrameRow row;
        if (r == LINE_BAD || !parse_row(line, &row) || row.id != q.size()) {
            ok = false;
            break;
        }
        const std::size_t i = q.size();
        q.resize(i + 1);
        q.x[i] = row.x;
        q.y[i] = row.y;
        q.z[i] = row.z;
        q.vx[i] = row.vx;
        q.vy[i] = row.vy;
        q.vz[i] = row.vz;
        q.mass[i] = row.mass;
        q.body[i] = row.body;
        q.material[i] = row.material;
        q.heat[i] = row.heat;
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
