#include "config.hpp"

#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include "init.hpp"

namespace {

const int MAX_PLANETS = 1000; // planet0 .. planet999

// Bits for the planet keys a file has set, to check the required ones are there.
const unsigned HAS_N = 1u << 0;
const unsigned HAS_RADIUS = 1u << 1;
const unsigned HAS_PARTICLE_RADIUS = 1u << 2;
const unsigned HAS_MASS = 1u << 3;
const unsigned HAS_ANY = 1u << 4;
const unsigned REQUIRED = HAS_N | HAS_RADIUS | HAS_PARTICLE_RADIUS | HAS_MASS;

std::string trim(const std::string &s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

// Whole string must be a finite number. strtod accepts "nan" and "inf", so check.
bool parse_double(const std::string &s, double *out) {
    const char *start = s.c_str();
    char *end = nullptr;
    errno = 0;
    double v = std::strtod(start, &end);
    if (end == start || *end != '\0' || errno == ERANGE || !std::isfinite(v)) {
        return false;
    }
    *out = v;
    return true;
}

bool parse_long(const std::string &s, long long *out) {
    const char *start = s.c_str();
    char *end = nullptr;
    errno = 0;
    long long v = std::strtoll(start, &end, 10);
    if (end == start || *end != '\0' || errno == ERANGE) {
        return false;
    }
    *out = v;
    return true;
}

bool parse_int(const std::string &s, int *out) {
    long long v = 0;
    if (!parse_long(s, &v) || v < INT_MIN || v > INT_MAX) {
        return false;
    }
    *out = static_cast<int>(v);
    return true;
}

bool parse_unsigned(const std::string &s, unsigned *out) {
    long long v = 0;
    if (!parse_long(s, &v) || v < 0 || v > static_cast<long long>(UINT_MAX)) {
        return false;
    }
    *out = static_cast<unsigned>(v);
    return true;
}

bool parse_bool(const std::string &s, bool *out) {
    if (s == "true") {
        *out = true;
        return true;
    }
    if (s == "false") {
        *out = false;
        return true;
    }
    return false;
}

// Exactly three finite numbers separated by spaces, e.g. "-3 0 0.5".
bool parse_vec3(const std::string &s, Vec3 *out) {
    double v[3];
    const char *p = s.c_str();
    for (int i = 0; i < 3; ++i) {
        if (i > 0 && !std::isspace(static_cast<unsigned char>(*p))) {
            return false; // "1-2-3" would otherwise read as (1, -2, -3)
        }
        char *end = nullptr;
        errno = 0;
        v[i] = std::strtod(p, &end);
        if (end == p || errno == ERANGE || !std::isfinite(v[i])) {
            return false;
        }
        p = end;
    }
    while (std::isspace(static_cast<unsigned char>(*p))) {
        ++p;
    }
    if (*p != '\0') {
        return false; // a fourth number or trailing junk
    }
    *out = Vec3{v[0], v[1], v[2]};
    return true;
}

// What happened to one key = value pair.
enum KeyResult { KEY_OK, KEY_UNKNOWN, KEY_BAD_VALUE };

KeyResult set_run_key(const std::string &key, const std::string &val, Config *cfg,
                      bool *has_output_every) {
    SimParams &sp = cfg->params;
    bool ok;
    if (key == "dt") {
        ok = parse_double(val, &sp.dt);
    } else if (key == "softening") {
        ok = parse_double(val, &sp.softening);
    } else if (key == "contact") {
        ok = parse_bool(val, &sp.contact_enabled);
    } else if (key == "contact_k") {
        ok = parse_double(val, &sp.contact_k);
    } else if (key == "contact_c") {
        ok = parse_double(val, &sp.contact_c);
    } else if (key == "steps") {
        ok = parse_int(val, &cfg->steps);
    } else if (key == "output_every") {
        ok = parse_int(val, &cfg->output_every);
        *has_output_every = ok;
    } else if (key == "relax_ke_threshold") {
        ok = parse_double(val, &cfg->relax_ke_threshold);
    } else if (key == "relax_max_steps") {
        ok = parse_int(val, &cfg->relax_max_steps);
    } else {
        return KEY_UNKNOWN;
    }
    return ok ? KEY_OK : KEY_BAD_VALUE;
}

KeyResult set_planet_key(const std::string &field, const std::string &val, PlanetSpec *s,
                         unsigned *has) {
    bool ok;
    unsigned bit = 0;
    if (field == "n") {
        ok = parse_int(val, &s->n);
        bit = HAS_N;
    } else if (field == "radius") {
        ok = parse_double(val, &s->radius);
        bit = HAS_RADIUS;
    } else if (field == "particle_radius") {
        ok = parse_double(val, &s->particle_radius);
        bit = HAS_PARTICLE_RADIUS;
    } else if (field == "mass") {
        ok = parse_double(val, &s->mass);
        bit = HAS_MASS;
    } else if (field == "center") {
        ok = parse_vec3(val, &s->center);
    } else if (field == "velocity") {
        ok = parse_vec3(val, &s->velocity);
    } else if (field == "seed") {
        ok = parse_unsigned(val, &s->seed);
    } else {
        return KEY_UNKNOWN;
    }
    if (ok) {
        *has |= bit | HAS_ANY;
    }
    return ok ? KEY_OK : KEY_BAD_VALUE;
}

// "planet12.radius" -> index 12, field "radius". Returns false if the key doesn't start
// with "planet<digits>." at all (so it's a run key, or unknown). *bad_number is set when
// it does but the number is unusable (leading zero, too big).
bool split_planet_key(const std::string &key, int *index, std::string *field, bool *bad_number) {
    *bad_number = false;
    const std::size_t prefix = std::strlen("planet");
    if (key.compare(0, prefix, "planet") != 0) {
        return false;
    }
    std::size_t dot = key.find('.', prefix);
    if (dot == std::string::npos || dot == prefix) {
        return false;
    }
    std::string digits = key.substr(prefix, dot - prefix);
    for (char ch : digits) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            return false;
        }
    }
    // "planet01" would silently be a second name for planet1. Six digits at most, so
    // atoi can't overflow; MAX_PLANETS is the real limit.
    if ((digits.size() > 1 && digits[0] == '0') || digits.size() > 6) {
        *bad_number = true;
        return true;
    }
    *index = std::atoi(digits.c_str());
    if (*index >= MAX_PLANETS) {
        *bad_number = true;
        return true;
    }
    *field = key.substr(dot + 1);
    return true;
}

void line_error(const char *path, int line_no, const std::string &why, const std::string &line) {
    std::fprintf(stderr, "%s:%d: %s\n    %s\n", path, line_no, why.c_str(), line.c_str());
}

// Range checks the rest of the program doesn't already make. dt, steps and
// output_every are checked by apply_overrides, after command-line flags can replace
// them; planets by planet_spec_ok. Returns the problem, or nullptr.
const char *run_settings_error(const Config &c) {
    const SimParams &sp = c.params;
    if (sp.softening < 0.0) {
        return "softening must be >= 0";
    }
    if (sp.contact_enabled && sp.contact_k <= 0.0) {
        return "contact = true needs contact_k > 0 (a spring of 0 does nothing)";
    }
    if (sp.contact_c < 0.0) {
        return "contact_c must be >= 0 (a negative damper adds energy)";
    }
    if (c.relax_ke_threshold < 0.0 || c.relax_max_steps < 0) {
        return "relax_ke_threshold and relax_max_steps must be >= 0";
    }
    return nullptr;
}

} // namespace

bool load_config(const char *path, Config &cfg) {
    // A directory opens fine on Linux and reads as empty, which would report a
    // confusing "missing 'output_every'".
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec)) {
        std::fprintf(stderr, "%s: can't open the file (missing, or not a regular file)\n", path);
        return false;
    }
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "%s: can't open the file\n", path);
        return false;
    }

    // Fill a copy, so a file with an error leaves cfg as it was.
    Config out = cfg;
    out.planets.clear();
    std::vector<unsigned> planet_has; // HAS_* bits per planet index
    std::vector<std::string> seen;    // keys so far, to catch duplicates
    bool has_output_every = false;

    std::string raw;
    int line_no = 0;
    while (std::getline(in, raw)) {
        ++line_no;
        // Windows editors (and PowerShell 5.1's Out-File) may start the file with a UTF-8
        // byte order mark, which would otherwise become part of the first key.
        if (line_no == 1 && raw.compare(0, 3, "\xEF\xBB\xBF") == 0) {
            raw.erase(0, 3);
        }
        std::string line = raw;
        std::size_t hash = line.find('#');
        if (hash != std::string::npos) {
            line.erase(hash); // comment to the end of the line
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            line_error(path, line_no, "expected 'key = value'", trim(raw));
            return false;
        }
        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));
        if (key.empty()) {
            line_error(path, line_no, "missing key before '='", trim(raw));
            return false;
        }
        if (val.empty()) {
            line_error(path, line_no, "missing value for '" + key + "'", trim(raw));
            return false;
        }
        for (const std::string &k : seen) {
            if (k == key) {
                line_error(path, line_no, "'" + key + "' is set twice", trim(raw));
                return false;
            }
        }
        seen.push_back(key);

        KeyResult r;
        int index = 0;
        std::string field;
        bool bad_number = false;
        if (split_planet_key(key, &index, &field, &bad_number)) {
            if (bad_number) {
                line_error(path, line_no,
                           "planet numbers are 0 to " + std::to_string(MAX_PLANETS - 1) +
                               " with no leading zeros",
                           trim(raw));
                return false;
            }
            const std::size_t i = static_cast<std::size_t>(index);
            if (i >= out.planets.size()) {
                out.planets.resize(i + 1);
                planet_has.resize(i + 1, 0u);
            }
            r = set_planet_key(field, val, &out.planets[i], &planet_has[i]);
        } else {
            r = set_run_key(key, val, &out, &has_output_every);
        }
        if (r == KEY_UNKNOWN) {
            line_error(path, line_no, "unknown key '" + key + "'", trim(raw));
            return false;
        }
        if (r == KEY_BAD_VALUE) {
            line_error(path, line_no, "bad value for '" + key + "'", trim(raw));
            return false;
        }
    }

    if (in.bad()) {
        std::fprintf(stderr, "%s: read error after line %d\n", path, line_no);
        return false;
    }

    // Checks on the whole file. These have no single line to point at.
    if (!has_output_every) {
        std::fprintf(stderr,
                     "%s: missing 'output_every' (without it every step writes a frame and "
                     "computes the O(N^2) energy)\n",
                     path);
        return false;
    }
    if (out.planets.empty()) {
        std::fprintf(stderr, "%s: no planets (add planet0.n, planet0.radius, ...)\n", path);
        return false;
    }
    for (std::size_t i = 0; i < out.planets.size(); ++i) {
        if (planet_has[i] == 0u) {
            std::fprintf(stderr, "%s: planet%zu is missing (planets are numbered 0, 1, 2, ...)\n",
                         path, i);
            return false;
        }
        if ((planet_has[i] & REQUIRED) != REQUIRED) {
            std::fprintf(stderr, "%s: planet%zu needs n, radius, particle_radius and mass\n", path,
                         i);
            return false;
        }
        const char *why = nullptr;
        if (!planet_spec_ok(out.planets[i], &why)) {
            std::fprintf(stderr, "%s: planet%zu: %s\n", path, i, why);
            return false;
        }
        out.planets[i].body = static_cast<int>(i); // frames color particles by planet
    }
    const char *bad = run_settings_error(out);
    if (bad != nullptr) {
        std::fprintf(stderr, "%s: %s\n", path, bad);
        return false;
    }
    cfg = out;
    return true;
}
