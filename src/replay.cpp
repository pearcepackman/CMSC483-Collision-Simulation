#include "replay.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <utility>

namespace {

// The frame number in "frame_<digits>.csv", or -1 for any other name.
long long frame_number(const std::string &name) {
    const char *prefix = "frame_";
    const char *suffix = ".csv";
    const std::size_t lp = std::strlen(prefix), ls = std::strlen(suffix);
    if (name.size() <= lp + ls || name.compare(0, lp, prefix) != 0 ||
        name.compare(name.size() - ls, ls, suffix) != 0) {
        return -1;
    }
    const std::string digits = name.substr(lp, name.size() - lp - ls);
    if (digits.empty() || digits.size() > 15) {
        return -1;
    }
    for (char c : digits) {
        if (c < '0' || c > '9') {
            return -1;
        }
    }
    return std::atoll(digits.c_str());
}

// The first line of a file, without its line ending; empty if it can't be read.
std::string first_line(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return "";
    }
    char buf[256];
    std::string line;
    if (std::fgets(buf, sizeof buf, f) != nullptr) {
        line = buf;
    }
    std::fclose(f);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    return line;
}

// energy.csv's t column, if it has one row per frame and step strictly increases.
std::vector<double> read_times(const std::string &path, std::size_t frames) {
    std::vector<double> times;
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return times;
    }
    char line[1024];
    bool ok = std::fgets(line, sizeof line, f) != nullptr &&
              std::strncmp(line, "step,t,", std::strlen("step,t,")) == 0;
    double last_step = -1.0;
    while (ok && std::fgets(line, sizeof line, f) != nullptr) {
        if (line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        double step = 0.0, t = 0.0;
        if (std::sscanf(line, "%lf,%lf,", &step, &t) != 2 || !std::isfinite(t) ||
            !(step > last_step)) {
            ok = false;
            break;
        }
        last_step = step;
        times.push_back(t);
    }
    std::fclose(f);
    if (!ok || times.size() != frames) {
        times.clear();
    }
    return times;
}

} // namespace

bool replay_open(const char *dir, ReplayRun *run, std::string *error) {
    if (dir == nullptr) {
        *error = "no replay folder given";
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) {
        *error = std::string("'") + dir + "' is not a folder";
        return false;
    }
    std::vector<std::pair<long long, std::string>> found;
    for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
        const long long n = frame_number(entry.path().filename().string());
        if (n >= 0) {
            found.emplace_back(n, entry.path().string());
        }
    }
    if (ec) {
        *error = std::string("can't list '") + dir + "': " + ec.message();
        return false;
    }
    if (found.empty()) {
        *error =
            std::string("no frame_*.csv files in '") + dir + "' (write some with planet_sim --out)";
        return false;
    }
    std::sort(found.begin(), found.end());
    const std::string header = first_line(found.front().second);
    if (header == "id,x,y,z,vx,vy,vz,mass,body,material,heat" || header == "id,x,y,z,body") {
        *error = std::string("'") + found.front().second +
                 "' was written before frames had a radius column (#49); rerun planet_sim "
                 "to get frames the viewer can replay";
        return false;
    }
    ReplayRun r;
    for (auto &f : found) {
        r.paths.push_back(std::move(f.second));
    }
    r.times = read_times((std::filesystem::path(dir) / "energy.csv").string(), r.paths.size());
    *run = std::move(r);
    return true;
}

double replay_start(const ReplayRun &run) {
    return run.times.empty() ? 0.0 : run.times.front();
}

double replay_end(const ReplayRun &run) {
    if (run.times.empty()) {
        return run.paths.empty() ? 0.0 : static_cast<double>(run.paths.size() - 1);
    }
    return run.times.back();
}

void replay_locate(const ReplayRun &run, double t, int *k, double *s) {
    const int last = run.paths.empty() ? 0 : static_cast<int>(run.paths.size()) - 1;
    *k = last;
    *s = 0.0;
    if (last == 0 || !(t < replay_end(run))) { // also catches NaN
        return;
    }
    if (!(t > replay_start(run))) {
        *k = 0;
        return;
    }
    if (run.times.empty()) {
        *k = static_cast<int>(std::floor(t));
        *s = t - *k;
        return;
    }
    // The last frame whose time is <= t.
    const auto it = std::upper_bound(run.times.begin(), run.times.end(), t);
    *k = static_cast<int>(it - run.times.begin()) - 1;
    const double t0 = run.times[static_cast<std::size_t>(*k)],
                 t1 = run.times[static_cast<std::size_t>(*k) + 1];
    *s = t1 > t0 ? (t - t0) / (t1 - t0) : 0.0;
}

bool interpolate_frames(const Particles &a, const Particles &b, double ta, double tb, double s,
                        Particles *out) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a.body[i] != b.body[i]) {
            return false;
        }
    }
    Particles &o = *out;
    o.resize(a.size());
    const double h = tb - ta;
    const bool hermite = h > 0.0;
    // Hermite basis functions and their derivatives (per unit s).
    const double s2 = s * s, s3 = s2 * s;
    const double h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s;
    const double h01 = -2 * s3 + 3 * s2, h11 = s3 - s2;
    const double d00 = 6 * s2 - 6 * s, d10 = 3 * s2 - 4 * s + 1;
    const double d01 = -6 * s2 + 6 * s, d11 = 3 * s2 - 2 * s;
    auto pos = [&](double xa, double va, double xb, double vb) {
        return hermite ? h00 * xa + h10 * h * va + h01 * xb + h11 * h * vb : xa + s * (xb - xa);
    };
    auto vel = [&](double xa, double va, double xb, double vb) {
        return hermite ? (d00 * xa + d01 * xb) / h + d10 * va + d11 * vb : va + s * (vb - va);
    };
    for (std::size_t i = 0; i < a.size(); ++i) {
        o.x[i] = pos(a.x[i], a.vx[i], b.x[i], b.vx[i]);
        o.y[i] = pos(a.y[i], a.vy[i], b.y[i], b.vy[i]);
        o.z[i] = pos(a.z[i], a.vz[i], b.z[i], b.vz[i]);
        o.vx[i] = vel(a.x[i], a.vx[i], b.x[i], b.vx[i]);
        o.vy[i] = vel(a.y[i], a.vy[i], b.y[i], b.vy[i]);
        o.vz[i] = vel(a.z[i], a.vz[i], b.z[i], b.vz[i]);
        o.heat[i] = a.heat[i] + s * (b.heat[i] - a.heat[i]);
        o.mass[i] = a.mass[i];
        o.radius[i] = a.radius[i];
        o.body[i] = a.body[i];
        o.material[i] = a.material[i];
    }
    return true;
}

namespace {

ReplayShow read_checked(const ReplayRun &run, int k, FrameReader read, Particles *out,
                        std::string *error) {
    const std::string &path = run.paths[static_cast<std::size_t>(k)];
    if (!read(path.c_str(), *out)) {
        *error = "can't read '" + path + "' (not a frame of this format)";
        return REPLAY_READ_FAILED;
    }
    if (!frame_is_finite(*out)) {
        *error = "'" + path + "' holds NaN/Inf values (the run blew up there)";
        return REPLAY_NOT_FINITE;
    }
    return REPLAY_OK;
}

} // namespace

ReplayShow replay_cache_show(const ReplayRun &run, double t, ReplayCache *cache, FrameReader read,
                             Particles *out, int *k_shown, std::string *error) {
    int k = 0;
    double s = 0.0;
    replay_locate(run, t, &k, &s);
    *k_shown = k;
    ReplayCache &c = *cache;
    if (k == c.kb && c.ka != k) { // moved forward one frame: the later one is now the earlier
        std::swap(c.fa, c.fb);
        std::swap(c.ka, c.kb);
    }
    if (c.ka != k) {
        c.ka = -1;
        const ReplayShow r = read_checked(run, k, read, &c.fa, error);
        if (r != REPLAY_OK) {
            return r;
        }
        c.ka = k;
    }
    if (s <= 0.0) {
        *out = c.fa;
        return REPLAY_OK;
    }
    if (c.kb != k + 1) {
        c.kb = -1;
        const ReplayShow r = read_checked(run, k + 1, read, &c.fb, error);
        if (r != REPLAY_OK) {
            return r;
        }
        c.kb = k + 1;
    }
    // Without times there is no time gap to scale velocities by: ta = tb gives linear.
    const bool timed = !run.times.empty();
    const double ta = timed ? run.times[static_cast<std::size_t>(k)] : 0.0;
    const double tb = timed ? run.times[static_cast<std::size_t>(k + 1)] : 0.0;
    if (!interpolate_frames(c.fa, c.fb, ta, tb, s, out)) {
        *out = c.fa;
        *error = "'" + run.paths[static_cast<std::size_t>(k)] + "' and '" +
                 run.paths[static_cast<std::size_t>(k + 1)] +
                 "' don't match (different particle counts or bodies: frames from two runs?)";
        return REPLAY_MISMATCH;
    }
    return REPLAY_OK;
}

long export_frame_count(double t0, double t1, double speed, double fps) {
    if (!std::isfinite(t0) || !std::isfinite(t1) || !std::isfinite(speed) || !std::isfinite(fps) ||
        !(speed > 0.0) || !(fps > 0.0) || t1 < t0) {
        return -1;
    }
    const double intervals = std::fmax(0.0, std::ceil((t1 - t0) / speed * fps - 1e-9));
    if (!std::isfinite(intervals) || intervals >= 1e6) {
        return -1;
    }
    return static_cast<long>(intervals) + 1;
}

double export_time(long i, double t0, double t1, double speed, double fps) {
    return std::fmin(t1, t0 + static_cast<double>(i) * speed / fps);
}

bool frame_is_finite(const Particles &p) {
    for (std::size_t i = 0; i < p.size(); ++i) {
        if (!std::isfinite(p.x[i]) || !std::isfinite(p.y[i]) || !std::isfinite(p.z[i]) ||
            !std::isfinite(p.vx[i]) || !std::isfinite(p.vy[i]) || !std::isfinite(p.vz[i]) ||
            !std::isfinite(p.heat[i])) {
            return false;
        }
    }
    return true;
}
