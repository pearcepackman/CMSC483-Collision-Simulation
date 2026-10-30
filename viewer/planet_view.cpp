// planet_view: a live 3D window onto the simulation (#45). Same physics and setup
// flags as planet_sim; keys 1/2/3 switch the gravity backend mid-run (#48), so the
// speed of serial, OpenMP and the GPU can be seen side by side.
//
// One thread does both: each frame runs simulation steps until a time budget is used
// (always at least one step), then draws. A slow backend therefore makes the window
// slow too, which is the point of the demo.

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>
#include <raylib.h>
#include <rlgl.h>

#include "cli.hpp"
#include "config.hpp"
#include "gravity.hpp"
#include "sim.hpp"

namespace {

const char *viewer_usage() {
    return "usage: planet_view (--config <file> | --scenario orbit|figure8|cloud [--n N] "
           "[--seed S])\n"
           "                   [--backend serial|omp|hip] [--dt h] [--frames N]\n"
           "                   [--screenshot <file.png>] [--help]\n"
           "  --frames      quit after N frames (for scripted runs and CI); default: run "
           "until closed\n"
           "  --screenshot  save the last frame as a PNG when the viewer exits\n"
           "keys: 1/2/3 serial/OpenMP/HIP, space pause, r restart, [ ] step time budget,\n"
           "      drag to orbit, mouse wheel to zoom\n"
           "exit codes: 0 ok, 1 usage or setup error, 2 the simulation produced NaN/Inf\n";
}

// Everything the window shows and changes. Plain struct, one instance in main().
struct Viewer {
    CliOptions cli;
    Particles p;
    SimParams params;
    Backend backend = BACKEND_SERIAL;

    long long step = 0;
    double sim_time = 0.0;
    bool paused = false;
    bool blew_up = false;
    std::size_t bad_particle = 0;

    double budget_ms = 12.0;  // simulation time per frame before drawing
    double ms_per_step = 0.0; // rolling average
    double frame_ms = 0.0;    // rolling average of whole frames (raylib's GetFPS lags ~30)
    int steps_last_frame = 0;
    std::string notice;        // short message shown for a few seconds
    double notice_until = 0.0; // omp_get_wtime() when the notice disappears

    float yaw = 0.6f, pitch = 0.35f, dist = 5.0f; // orbit camera around the origin
};

void show_notice(Viewer &v, const std::string &text) {
    v.notice = text;
    v.notice_until = omp_get_wtime() + 3.0;
}

// The starting particles and settings from the command line, exactly as planet_sim
// builds them (config + relaxation, or a built-in scenario, then --dt). Built into
// locals first, so a failed restart leaves the running simulation untouched. Config
// planets are relaxed on the current backend, so a restart on serial can take a while.
bool setup(Viewer &v, std::string *error) {
    Particles p;
    SimParams params;
    int steps = 1, output_every = 1; // planet_sim-only settings, unused here
    if (!v.cli.config_path.empty()) {
        Config cfg;
        if (!load_config(v.cli.config_path.c_str(), cfg)) {
            *error = "couldn't load config '" + v.cli.config_path + "'";
            return false;
        }
        params = cfg.params;
        p = build_planets(cfg, v.backend);
    } else if (!build_scenario(v.cli.scenario.c_str(), v.cli.n, v.cli.seed, p, params)) {
        *error = "unknown scenario '" + v.cli.scenario + "' (orbit, figure8, cloud)";
        return false;
    }
    if (!apply_overrides(v.cli, false, &params, &steps, &output_every, error)) {
        return false;
    }
    v.p = std::move(p);
    v.params = params;
    v.step = 0;
    v.sim_time = 0.0;
    v.blew_up = false;
    v.ms_per_step = 0.0;
    v.frame_ms = 0.0;
    v.bad_particle = sim_init(v.p, v.params, v.backend);
    if (v.bad_particle != v.p.size()) {
        v.blew_up = true;
        v.paused = true;
    }
    // Frame the whole system: camera distance from the farthest particle (skipped for a
    // NaN/Inf start, which would put the camera at infinity).
    double r_max = 0.0;
    for (std::size_t i = 0; i < v.p.size() && !v.blew_up; ++i) {
        r_max = std::fmax(
            r_max, std::sqrt(v.p.x[i] * v.p.x[i] + v.p.y[i] * v.p.y[i] + v.p.z[i] * v.p.z[i]));
    }
    v.dist = static_cast<float>(std::fmax(3.0 * r_max, 1.0));
    return true;
}

// Switches the gravity backend if this build and machine can run it.
void switch_backend(Viewer &v, Backend b) {
    if (!backend_available(b)) {
        show_notice(v,
                    std::string(backend_name(b)) + " backend not built (use: cmake --preset demo)");
        return;
    }
#ifdef SIM_HAVE_HIP
    if (b == BACKEND_HIP) {
        std::string info;
        if (!hip_init(&info)) {
            show_notice(v, "hip: " + info);
            return;
        }
    }
#endif
    // No data to move: Particles live on the host and the HIP backend copies them every
    // call. (#30 keeps data on the GPU; switching will need a copy then.)
    v.backend = b;
    v.ms_per_step = 0.0; // the old backend's averages mean nothing now
    v.frame_ms = 0.0;
    show_notice(v, std::string("backend: ") + backend_name(b));
}

void handle_input(Viewer &v) {
    if (IsKeyPressed(KEY_ONE)) {
        switch_backend(v, BACKEND_SERIAL);
    }
    if (IsKeyPressed(KEY_TWO)) {
        switch_backend(v, BACKEND_OMP);
    }
    if (IsKeyPressed(KEY_THREE)) {
        switch_backend(v, BACKEND_HIP);
    }
    if (IsKeyPressed(KEY_SPACE) && !v.blew_up) {
        v.paused = !v.paused;
    }
    if (IsKeyPressed(KEY_R)) {
        std::string error;
        if (!setup(v, &error)) {
            show_notice(v, "restart failed: " + error);
        }
        v.paused = v.blew_up;
    }
    if (IsKeyPressed(KEY_LEFT_BRACKET)) {
        v.budget_ms = std::fmax(1.0, v.budget_ms / 1.5);
    }
    if (IsKeyPressed(KEY_RIGHT_BRACKET)) {
        v.budget_ms = std::fmin(200.0, v.budget_ms * 1.5);
    }
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 d = GetMouseDelta();
        v.yaw -= d.x * 0.006f;
        v.pitch = std::fmax(-1.5f, std::fmin(1.5f, v.pitch + d.y * 0.006f));
    }
    v.dist = std::fmax(0.01f, std::fmin(1e6f, v.dist * std::exp(-GetMouseWheelMove() * 0.1f)));
}

// Steps until the frame's time budget is used (at least one step).
void advance(Viewer &v) {
    v.steps_last_frame = 0;
    if (v.paused) {
        return;
    }
    const double start = omp_get_wtime();
    double elapsed = 0.0;
    do {
        v.bad_particle = sim_step(v.p, v.params, v.backend);
        ++v.step;
        ++v.steps_last_frame;
        v.sim_time += v.params.dt;
        elapsed = omp_get_wtime() - start;
        if (v.bad_particle != v.p.size()) {
            v.blew_up = true;
            v.paused = true;
            break;
        }
    } while (elapsed * 1000.0 < v.budget_ms);
    const double per_step = elapsed * 1000.0 / v.steps_last_frame;
    v.ms_per_step = v.ms_per_step == 0.0 ? per_step : 0.9 * v.ms_per_step + 0.1 * per_step;
}

Vector3 add(Vector3 a, Vector3 b) {
    return Vector3{a.x + b.x, a.y + b.y, a.z + b.z};
}
Vector3 scale(Vector3 a, float s) {
    return Vector3{a.x * s, a.y * s, a.z * s};
}

// Particles as small camera-facing squares (plain points; the glow look is #47).
void draw_particles(const Viewer &v, Vector3 right, Vector3 up) {
    const float size = v.dist * 0.0025f;
    const Vector3 r = scale(right, size), u = scale(up, size);
    const Color colors[] = {Color{255, 170, 90, 255}, Color{90, 190, 255, 255}};
    rlBegin(RL_QUADS);
    for (std::size_t i = 0; i < v.p.size(); ++i) {
        rlCheckRenderBatchLimit(4); // flush the batch if it's full
        const Color c = colors[v.p.body[i] & 1];
        rlColor4ub(c.r, c.g, c.b, c.a);
        const Vector3 pos{static_cast<float>(v.p.x[i]), static_cast<float>(v.p.y[i]),
                          static_cast<float>(v.p.z[i])};
        const Vector3 corners[] = {add(pos, add(scale(r, -1), scale(u, -1))),
                                   add(pos, add(r, scale(u, -1))), add(pos, add(r, u)),
                                   add(pos, add(scale(r, -1), u))};
        for (const Vector3 &q : corners) {
            rlVertex3f(q.x, q.y, q.z);
        }
    }
    rlEnd();
}

void draw_hud(const Viewer &v) {
    char threads[16] = "GPU"; // the hip backend computes gravity on the GPU
    if (v.backend != BACKEND_HIP) {
        std::snprintf(threads, sizeof threads, "%d",
                      v.backend == BACKEND_OMP ? omp_get_max_threads() : 1);
    }
    const double steps_per_s = v.ms_per_step > 0.0 ? 1000.0 / v.ms_per_step : 0.0;
    DrawRectangle(10, 10, 330, 178, Color{12, 14, 22, 200});
    int y = 18;
    const int line = 20;
    DrawText(TextFormat("backend   %s", backend_name(v.backend)), 20, y, 20,
             Color{255, 160, 70, 255});
    y += line + 4;
    DrawText(TextFormat("particles %zu", v.p.size()), 20, y, 18, RAYWHITE);
    y += line;
    DrawText(TextFormat("threads   %s", threads), 20, y, 18, RAYWHITE);
    y += line;
    DrawText(TextFormat("ms/step   %.2f   (%.0f steps/s)", v.ms_per_step, steps_per_s), 20, y, 18,
             RAYWHITE);
    y += line;
    DrawText(TextFormat("steps/frame %d   budget %.0f ms", v.steps_last_frame, v.budget_ms), 20, y,
             18, RAYWHITE);
    y += line;
    DrawText(TextFormat("fps       %.0f", v.frame_ms > 0.0 ? 1000.0 / v.frame_ms : 0.0), 20, y, 18,
             RAYWHITE);
    y += line;
    DrawText(TextFormat("sim time  %.4g   step %lld", v.sim_time, v.step), 20, y, 18, RAYWHITE);

    if (v.blew_up) {
        DrawText(TextFormat("simulation blew up (particle %zu is NaN/Inf) at step %lld: press r",
                            v.bad_particle, v.step),
                 20, 200, 20, RED);
    } else if (v.paused) {
        DrawText("paused (space)", 20, 200, 20, YELLOW);
    }
    if (omp_get_wtime() < v.notice_until) {
        DrawText(v.notice.c_str(), 20, GetScreenHeight() - 60, 20, Color{255, 200, 120, 255});
    }
    DrawText("1/2/3 serial/omp/hip   space pause   r restart   [ ] budget   drag orbit   wheel "
             "zoom",
             20, GetScreenHeight() - 30, 16, GRAY);
}

void draw(const Viewer &v) {
    const float cp = std::cos(v.pitch), sp = std::sin(v.pitch);
    const float cy = std::cos(v.yaw), sy = std::sin(v.yaw);
    Camera3D cam{};
    cam.position = Vector3{v.dist * cp * sy, v.dist * sp, v.dist * cp * cy};
    cam.target = Vector3{0.0f, 0.0f, 0.0f};
    cam.up = Vector3{0.0f, 1.0f, 0.0f};
    cam.fovy = 45.0f;
    cam.projection = CAMERA_PERSPECTIVE;
    // Screen-aligned axes for the particle squares.
    const Vector3 right{cy, 0.0f, -sy};
    const Vector3 up{-sp * sy, cp, -sp * cy};

    BeginDrawing();
    ClearBackground(Color{5, 6, 10, 255});
    BeginMode3D(cam);
    draw_particles(v, right, up);
    EndMode3D();
    draw_hud(v);
    EndDrawing();
}

// Pulls the viewer-only flags (`--frames N`, `--screenshot <file>`) out of argv, leaving
// the rest for parse_cli. Returns false on a bad value.
bool take_viewer_flags(int argc, char **argv, std::vector<const char *> *rest, long *frames,
                       std::string *screenshot, std::string *error) {
    for (int i = 0; i < argc; ++i) {
        const bool is_frames = std::strcmp(argv[i], "--frames") == 0;
        const bool is_shot = std::strcmp(argv[i], "--screenshot") == 0;
        if (!is_frames && !is_shot) {
            rest->push_back(argv[i]);
            continue;
        }
        const char *value = i + 1 < argc ? argv[i + 1] : "";
        ++i;
        if (is_shot) {
            // Checked now, not after the whole run: raylib picks the format by extension.
            const std::size_t len = std::strlen(value);
            if (len < 5 || std::strcmp(value + len - 4, ".png") != 0) {
                *error = std::string("--screenshot needs a .png file name, got '") + value + "'";
                return false;
            }
            *screenshot = value;
            continue;
        }
        char *end = nullptr;
        errno = 0;
        *frames = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || errno == ERANGE || *frames < 1 ||
            !(value[0] >= '0' && value[0] <= '9')) {
            *error = std::string("--frames needs a whole number >= 1, got '") + value + "'";
            return false;
        }
    }
    return true;
}

// The viewer's whole run, separated from main() so running out of memory for a huge --n
// is reported cleanly (as in planet_sim).
int run(int argc, char **argv) {
    Viewer v;
    std::vector<const char *> args;
    long frames = -1; // -1: until the window is closed
    std::string screenshot, error;
    if (!take_viewer_flags(argc, argv, &args, &frames, &screenshot, &error) ||
        !parse_cli(static_cast<int>(args.size()), args.data(), &v.cli, &error)) {
        std::fprintf(stderr, "planet_view: %s\n%s", error.c_str(), viewer_usage());
        return 1;
    }
    if (v.cli.help) {
        std::printf("%s", viewer_usage());
        return 0;
    }
    if (!v.cli.out_dir.empty() || v.cli.bench || v.cli.has_steps || v.cli.has_output_every) {
        std::fprintf(stderr,
                     "planet_view: --out, --bench, --steps and --output-every are planet_sim "
                     "flags; the viewer runs until you close it\n%s",
                     viewer_usage());
        return 1;
    }
    v.backend = v.cli.backend;
#ifdef SIM_HAVE_HIP
    if (v.backend == BACKEND_HIP) {
        std::string info;
        if (!hip_init(&info)) {
            std::fprintf(stderr, "planet_view: --backend hip: %s\n", info.c_str());
            return 1;
        }
    }
#endif
    if (!setup(v, &error)) {
        std::fprintf(stderr, "planet_view: %s\n", error.c_str());
        return 1;
    }

    SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
    InitWindow(1280, 800, "planet_view");
    SetTargetFPS(60);
    for (long frame = 0; !WindowShouldClose() && (frames < 0 || frame < frames); ++frame) {
        handle_input(v);
        advance(v);
        draw(v);
        const double ms = GetFrameTime() * 1000.0;
        v.frame_ms = v.frame_ms == 0.0 ? ms : 0.8 * v.frame_ms + 0.2 * ms;
    }
    if (!screenshot.empty()) {
        // Read back what's on screen now (the last frame drawn) and save it.
        Image shot = LoadImageFromScreen();
        const bool saved = ExportImage(shot, screenshot.c_str());
        UnloadImage(shot);
        if (!saved) {
            std::fprintf(stderr, "planet_view: couldn't save screenshot '%s'\n",
                         screenshot.c_str());
            CloseWindow();
            return 1;
        }
    }
    CloseWindow();
    if (v.blew_up) { // like planet_sim: a run that exploded must not look like success
        std::fprintf(stderr, "planet_view: particle %zu became NaN/Inf at step %lld\n",
                     v.bad_particle, v.step);
        return 2;
    }
    return 0;
}

} // namespace

// Exit codes: 0 closed normally (or after --frames), 1 usage or setup error,
// 2 the simulation produced NaN/Inf.
int main(int argc, char **argv) {
    try {
        return run(argc, argv);
    } catch (const std::bad_alloc &) {
        std::fprintf(stderr, "planet_view: out of memory (try a smaller --n)\n");
        return 1;
    }
}
