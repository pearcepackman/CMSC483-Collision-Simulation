// planet_view: a live 3D window onto the simulation (#45). Same physics and setup
// flags as planet_sim; keys 1/2/3 switch the gravity backend mid-run (#48), so the
// speed of serial, OpenMP and the GPU can be seen side by side.
//
// One thread does both: each frame runs simulation steps until a time budget is used
// (always at least one step), then draws. A slow backend therefore makes the window
// slow too, which is the point of the demo.
//
// --replay plays a recorded run (planet_sim --out) instead, with the same look, and
// --export renders that replay offscreen to PNGs for a video (#49).

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
#include "io.hpp"
#include "look.hpp"
#include "replay.hpp"
#include "sim.hpp"

namespace {

const char *viewer_usage() {
    return "usage: planet_view (--config <file> | --scenario orbit|figure8|cloud [--n N] "
           "[--seed S])\n"
           "                   [--backend serial|omp|hip] [--gpu-precision float|double]\n"
           "                   [--dt h] [--frames N] [--screenshot <file.png>]\n"
           "                   [--color heat|material|planet|speed|plain] [--no-bloom] [--trails]\n"
           "                   [--help]\n"
           "       planet_view --replay <run folder> [--speed S] [--frames N] [--screenshot ...]\n"
           "                   [--export <folder> [--fps F] [--size WxH] [--orbit DEG]]\n"
           "                   [--color ...] [--no-bloom] [--trails]\n"
           "  --frames      quit after N frames (for scripted runs and CI); default: run "
           "until closed\n"
           "  --screenshot  save the last frame as a PNG when the viewer exits\n"
           "  --color       starting color mode (default heat: material, glowing where hot)\n"
           "  --replay      play the frames planet_sim --out wrote, smoothly between saved "
           "frames\n"
           "  --speed       simulation time per second of replay or video (default: the whole "
           "run in\n"
           "                about 8 s, slower if frames are big to read)\n"
           "  --export      render the replay offscreen to <folder>/frame_NNNNN.png, no window "
           "needed;\n"
           "                the folder must be new or empty (scripts/make_video.py makes the "
           "video)\n"
           "  --fps, --size, --orbit  export frame rate (default 60), image size (default "
           "1920x1080),\n"
           "                and how far the camera swings around over the video (default 40 "
           "degrees)\n"
           "keys: 1/2/3 serial/OpenMP/HIP, p GPU float/double, c color, b bloom, t trails,\n"
           "      space pause, r restart, [ ] step time budget, drag to orbit, wheel to zoom\n"
           "replay keys: space pause, left/right saved frame, up/down speed, home start\n"
           "exit codes: 0 ok, 1 usage or setup error (or a replayed frame can't be read),\n"
           "            2 the simulation produced NaN/Inf (or a replayed frame holds NaN/Inf or\n"
           "            doesn't match the run)\n";
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
    bool camera_moved = false; // this frame: trails are cleared instead of smeared

    // The look (#47).
    ColorMode color = COLOR_HEAT;
    bool bloom = true;
    bool trails = false;
    LookScales scales{1.0, 0.5}; // fixed per scenario: set in setup()

    // Replay (#49): showing recorded frames instead of simulating.
    bool replaying = false;
    ReplayRun run;
    ReplayCache cache;   // the saved frames around replay_t
    int replay_k = 0;    // the earlier frame of the pair shown
    int replay_exit = 0; // set when a frame turns out broken: 2 NaN/Inf or mismatch, 1 unreadable
    std::string replay_error; // what was wrong with it, printed when the viewer exits
    double replay_t = 0.0;
    double speed = 1.0; // replay time per real second
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
    // The same setup planet_sim uses (tested there): overrides, then planets.
    if (!setup_run(v.cli, v.backend, false, &p, &params, &steps, &output_every, error)) {
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
    // Color scales come from the starting state, so a color means the same thing for the
    // whole run (and again after a restart).
    v.scales = look_scales(v.p);
    return true;
}

// Replay time of saved frame k (frame k itself without energy.csv times).
double frame_time(const ReplayRun &run, int k) {
    return run.times.empty() ? static_cast<double>(k) : run.times[static_cast<std::size_t>(k)];
}

// Puts the run at replay_t into v.p (replay_cache_show) and remembers the exit code a
// broken frame calls for: 2 for NaN/Inf or frames that don't match (as a simulation that
// blows up), 1 for a file that can't be read.
ReplayShow replay_show(Viewer &v, std::string *error) {
    const ReplayShow r =
        replay_cache_show(v.run, v.replay_t, &v.cache, read_frame, &v.p, &v.replay_k, error);
    if (r == REPLAY_NOT_FINITE || r == REPLAY_MISMATCH) {
        v.replay_exit = 2;
        v.replay_error = *error;
    } else if (r == REPLAY_READ_FAILED && v.replay_exit == 0) {
        v.replay_exit = 1;
        v.replay_error = *error;
    }
    return r;
}

// Opens a recorded run: frame 0 sets the color scales and the camera, as setup() does
// for a live run. speed_flag > 0 is --speed; otherwise the whole run plays in about 8 s,
// and for live replay slower if reading a frame is slow (big runs), so playback doesn't
// fall behind. Export isn't real time, so it always gets the 8 s default.
bool replay_setup(Viewer &v, const char *dir, double speed_flag, bool exporting,
                  std::string *error) {
    if (!replay_open(dir, &v.run, error)) {
        return false;
    }
    v.replaying = true;
    v.replay_t = replay_start(v.run);
    const double read_start = omp_get_wtime();
    if (replay_show(v, error) != REPLAY_OK) {
        return false;
    }
    const double read_seconds = std::fmax(omp_get_wtime() - read_start, 1e-4);
    v.scales = look_scales(v.p);
    double r_max = 0.0;
    for (std::size_t i = 0; i < v.p.size(); ++i) {
        r_max = std::fmax(
            r_max, std::sqrt(v.p.x[i] * v.p.x[i] + v.p.y[i] * v.p.y[i] + v.p.z[i] * v.p.z[i]));
    }
    v.dist = static_cast<float>(std::fmax(3.0 * r_max, 1.0));
    const double duration = replay_end(v.run) - replay_start(v.run);
    if (speed_flag > 0.0) {
        v.speed = speed_flag;
    } else if (duration > 0.0) {
        v.speed = duration / 8.0;
        if (!exporting) {
            // Reading frames may take at most half of each second: at most
            // 0.5 / read_seconds saved frames per second of replay.
            const double frames_per_time = static_cast<double>(v.run.paths.size() - 1) / duration;
            v.speed = std::fmin(v.speed, 0.5 / read_seconds / frames_per_time);
        }
    }
    if (!(v.speed > 0.0) || !std::isfinite(v.speed)) {
        v.speed = 1.0; // a one-frame run: nothing moves anyway
    }
    return true;
}

// Moves replay time on by the frame time (at most 1/30 s, so a slow frame read doesn't
// jump replay past the frames it just loaded) and shows that moment.
void replay_advance(Viewer &v) {
    v.steps_last_frame = 0;
    if (v.paused) {
        return;
    }
    v.replay_t += std::fmin(static_cast<double>(GetFrameTime()), 1.0 / 30.0) * v.speed;
    if (v.replay_t >= replay_end(v.run)) {
        v.replay_t = replay_end(v.run);
        v.paused = true; // the end: space starts again from the beginning
    }
    std::string error;
    if (replay_show(v, &error) != REPLAY_OK) {
        v.paused = true;
        show_notice(v, error);
    }
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

// Keys that only mean something in a replay: pause, scrub, speed, start.
void handle_replay_keys(Viewer &v) {
    if (IsKeyPressed(KEY_ONE) || IsKeyPressed(KEY_TWO) || IsKeyPressed(KEY_THREE) ||
        IsKeyPressed(KEY_P)) {
        show_notice(v, "replay: no simulation to switch (run planet_view --config ... for that)");
    }
    const double start = replay_start(v.run), end = replay_end(v.run);
    bool moved = false;
    if (IsKeyPressed(KEY_SPACE)) {
        v.paused = !v.paused;
        if (!v.paused && v.replay_t >= end) { // at the end: play again from the start
            v.replay_t = start;
            moved = true;
        }
    }
    if (IsKeyPressed(KEY_HOME) || IsKeyPressed(KEY_R)) {
        v.replay_t = start;
        moved = true;
    }
    if (IsKeyPressed(KEY_RIGHT) || IsKeyPressed(KEY_LEFT)) {
        // To the next or previous saved frame, and pause there.
        int k = 0;
        double s = 0.0;
        replay_locate(v.run, v.replay_t, &k, &s);
        const int last = static_cast<int>(v.run.paths.size()) - 1;
        if (IsKeyPressed(KEY_RIGHT)) {
            k = std::min(k + 1, last);
        } else if (s <= 0.0) { // on a saved frame: the one before (else this one)
            k = std::max(k - 1, 0);
        }
        v.replay_t = frame_time(v.run, k);
        v.paused = true;
        moved = true;
    }
    if (IsKeyPressed(KEY_UP)) {
        v.speed *= 1.5;
        show_notice(v, TextFormat("speed %.3g", v.speed));
    }
    if (IsKeyPressed(KEY_DOWN)) {
        v.speed /= 1.5;
        show_notice(v, TextFormat("speed %.3g", v.speed));
    }
    if (moved) {
        std::string error;
        if (replay_show(v, &error) != REPLAY_OK) {
            v.paused = true;
            show_notice(v, error);
        }
    }
}

void handle_input(Viewer &v) {
    if (v.replaying) {
        handle_replay_keys(v); // the look and camera keys below work in a replay too
    }
    if (!v.replaying && IsKeyPressed(KEY_ONE)) {
        switch_backend(v, BACKEND_SERIAL);
    }
    if (!v.replaying && IsKeyPressed(KEY_TWO)) {
        switch_backend(v, BACKEND_OMP);
    }
    if (!v.replaying && IsKeyPressed(KEY_THREE)) {
        switch_backend(v, BACKEND_HIP);
    }
#ifdef SIM_HAVE_HIP
    if (!v.replaying &&
        IsKeyPressed(KEY_P)) { // GPU precision, live: the float-vs-double trade-off on screen
        hip_set_double(!hip_uses_double());
        v.ms_per_step = 0.0;
        v.frame_ms = 0.0;
        show_notice(v, std::string("GPU precision: ") + (hip_uses_double() ? "double" : "float"));
    }
#endif
    if (!v.replaying && IsKeyPressed(KEY_SPACE) && !v.blew_up) {
        v.paused = !v.paused;
    }
    if (!v.replaying && IsKeyPressed(KEY_R)) {
        std::string error;
        if (!setup(v, &error)) {
            show_notice(v, "restart failed: " + error);
        }
        v.paused = v.blew_up;
    }
    if (IsKeyPressed(KEY_C)) {
        v.color = next_color_mode(v.color);
        show_notice(v, std::string("color: ") + color_mode_name(v.color));
    }
    if (IsKeyPressed(KEY_B)) {
        v.bloom = !v.bloom;
        show_notice(v, v.bloom ? "bloom on" : "bloom off");
    }
    if (IsKeyPressed(KEY_T)) {
        v.trails = !v.trails;
        show_notice(v, v.trails ? "trails on" : "trails off");
    }
    if (!v.replaying && IsKeyPressed(KEY_LEFT_BRACKET)) {
        v.budget_ms = std::fmax(1.0, v.budget_ms / 1.5);
    }
    if (!v.replaying && IsKeyPressed(KEY_RIGHT_BRACKET)) {
        v.budget_ms = std::fmin(200.0, v.budget_ms * 1.5);
    }
    v.camera_moved = false;
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 d = GetMouseDelta();
        v.yaw -= d.x * 0.006f;
        v.pitch = std::fmax(-1.5f, std::fmin(1.5f, v.pitch + d.y * 0.006f));
        v.camera_moved = v.camera_moved || d.x != 0.0f || d.y != 0.0f;
    }
    const float wheel = GetMouseWheelMove();
    v.dist = std::fmax(0.01f, std::fmin(1e6f, v.dist * std::exp(-wheel * 0.1f)));
    v.camera_moved = v.camera_moved || wheel != 0.0f;
}

// Steps until the frame's time budget is used (at least one step); in a replay, moves
// replay time on instead.
void advance(Viewer &v) {
    if (v.replaying) {
        replay_advance(v);
        return;
    }
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

Vector3 scale(Vector3 a, float s) {
    return Vector3{a.x * s, a.y * s, a.z * s};
}

// ---- Rendering (#47) ----
//
// Particles are glowing sprites added together (additive blending) into an offscreen
// scene texture with 16-bit float channels. An 8-bit texture would clip: the hundreds of
// sprites overlapping in a planet's core would saturate to white, and in heat mode white
// means white-hot. A final pass compresses brightness by each color's brightest channel,
// keeping its hue, so a bright cold core stays rock-colored and only real heat turns
// white. Hot particles also emit more light than cold ones (up to HOT_INTENSITY), so a
// lone white-hot speck of debris still reads as white. Bloom blurs the bright parts at
// half resolution and adds them back.

const float GLOW_FACTOR = 2.5f;   // a sprite's half-size, in particle radii
const float SPRITE_ALPHA = 0.35f; // how much light one cold sprite adds
const float HOT_INTENSITY = 3.0f; // ... and a white-hot one (heat mode)
const float SPRITE_GAIN = 4.0f;   // intensity = vertex alpha x this (alpha stops at 1)
const float MIN_PIXELS = 1.5f;    // smallest sprite half-size on screen
const float EXPOSURE = 0.9f;      // brightness compression in the composite pass
const float BLOOM_THRESHOLD = 1.0f;
const float BLOOM_STRENGTH = 0.7f;
const float TRAIL_SECONDS = 0.25f; // how fast trails fade (time to fall to 1/e)
const float FOVY_DEGREES = 45.0f;
const Color BACKGROUND = {5, 6, 10, 255};

// Bright-pass (optional) plus one direction of a 9-tap Gaussian blur.
const char *BLUR_FS = R"(#version 330
in vec2 fragTexCoord;
uniform sampler2D texture0;
uniform vec2 texel;
uniform vec2 dir;
uniform int bright;
uniform float threshold;
out vec4 finalColor;
vec3 pick(vec2 uv) {
    vec3 c = texture(texture0, uv).rgb;
    if (bright == 1) {
        float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
        c *= max(l - threshold, 0.0) / max(l, 1e-4);
    }
    return c;
}
void main() {
    float w[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec3 s = pick(fragTexCoord) * w[0];
    for (int i = 1; i < 5; ++i) {
        vec2 o = dir * texel * float(i) * 1.5;
        s += (pick(fragTexCoord + o) + pick(fragTexCoord - o)) * w[i];
    }
    finalColor = vec4(s, 1.0);
}
)";

// A glow sprite whose brightness can go past 1: the vertex color's alpha carries the
// intensity divided by SPRITE_GAIN. Drawn with plain additive blending (ONE, ONE).
const char *SPRITE_FS = R"(#version 330
in vec2 fragTexCoord;
in vec4 fragColor;
uniform sampler2D texture0;
uniform float gain;
out vec4 finalColor;
void main() {
    float a = texture(texture0, fragTexCoord).a;
    finalColor = vec4(fragColor.rgb * (fragColor.a * gain * a), 1.0);
}
)";

// Scene + bloom, then brightness compressed with the hue kept, over the background.
const char *COMPOSITE_FS = R"(#version 330
in vec2 fragTexCoord;
uniform sampler2D texture0;
uniform sampler2D bloom;
uniform float exposure;
uniform float strength;
uniform int raw;
uniform vec3 background;
out vec4 finalColor;
void main() {
    vec3 c = texture(texture0, fragTexCoord).rgb + strength * texture(bloom, fragTexCoord).rgb;
    if (raw == 0) {
        // Scale by the brightest channel, not overall brightness: the hue is kept exactly
        // and no channel can clip (by luminance, a strong orange clipped toward yellow).
        float peak = max(c.r, max(c.g, c.b));
        float m = 1.0 - exp(-exposure * peak);
        c = peak > 1e-6 ? c * (m / peak) : vec3(0.0);
    }
    finalColor = vec4(min(background + c, vec3(1.0)), 1.0);
}
)";

struct Renderer {
    int w = 0, h = 0; // render size in pixels (HiDPI aware)
    RenderTexture2D scene{}, bloom_a{}, bloom_b{};
    Texture2D glow{};
    Shader blur{}, composite{}, sprite{};
    bool shaders_ok = false;
    bool target_failed = false; // the float scene texture couldn't be made (see renderer_fit)
    int sprite_gain = -1;
    int blur_texel = -1, blur_dir = -1, blur_bright = -1, blur_threshold = -1;
    int comp_bloom = -1, comp_exposure = -1, comp_strength = -1, comp_raw = -1, comp_bg = -1;
};

// A framebuffer with one RGBA16F color texture and no depth buffer (sprites don't use it).
RenderTexture2D load_float_target(int w, int h) {
    RenderTexture2D t{};
    t.id = rlLoadFramebuffer();
    if (t.id == 0) {
        return t;
    }
    rlEnableFramebuffer(t.id);
    t.texture.id = rlLoadTexture(nullptr, w, h, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, 1);
    t.texture.width = w;
    t.texture.height = h;
    t.texture.format = PIXELFORMAT_UNCOMPRESSED_R16G16B16A16;
    t.texture.mipmaps = 1;
    rlFramebufferAttach(t.id, t.texture.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D,
                        0);
    if (!rlFramebufferComplete(t.id)) {
        rlDisableFramebuffer();
        UnloadRenderTexture(t);
        return RenderTexture2D{};
    }
    rlDisableFramebuffer();
    SetTextureFilter(t.texture, TEXTURE_FILTER_BILINEAR);
    return t;
}

void unload_target(RenderTexture2D &t) {
    if (t.id != 0) {
        UnloadRenderTexture(t);
    }
    t = RenderTexture2D{};
}

// A soft round glow: white, alpha falling smoothly from the center to 0 at the edge.
Texture2D make_glow_texture() {
    const int n = 64;
    std::vector<unsigned char> px(static_cast<std::size_t>(n * n * 4));
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            float dx = (static_cast<float>(x) + 0.5f) / n * 2.0f - 1.0f;
            float dy = (static_cast<float>(y) + 0.5f) / n * 2.0f - 1.0f;
            float r2 = dx * dx + dy * dy;
            float a = r2 >= 1.0f ? 0.0f : std::exp(-4.0f * r2) * (1.0f - r2);
            unsigned char *q = &px[static_cast<std::size_t>((y * n + x) * 4)];
            q[0] = q[1] = q[2] = 255;
            q[3] = static_cast<unsigned char>(a * 255.0f + 0.5f);
        }
    }
    Image img{px.data(), n, n, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
    Texture2D t = LoadTextureFromImage(img); // copies the pixels to the GPU
    SetTextureFilter(t, TEXTURE_FILTER_BILINEAR);
    return t;
}

bool shader_loaded(Shader sh) {
    return sh.id != 0 && sh.id != rlGetShaderIdDefault();
}

void renderer_init(Renderer &r) {
    r.glow = make_glow_texture();
    r.blur = LoadShaderFromMemory(nullptr, BLUR_FS);
    r.composite = LoadShaderFromMemory(nullptr, COMPOSITE_FS);
    r.sprite = LoadShaderFromMemory(nullptr, SPRITE_FS);
    r.shaders_ok = shader_loaded(r.blur) && shader_loaded(r.composite) && shader_loaded(r.sprite);
    if (r.shaders_ok) {
        r.sprite_gain = GetShaderLocation(r.sprite, "gain");
        r.blur_texel = GetShaderLocation(r.blur, "texel");
        r.blur_dir = GetShaderLocation(r.blur, "dir");
        r.blur_bright = GetShaderLocation(r.blur, "bright");
        r.blur_threshold = GetShaderLocation(r.blur, "threshold");
        r.comp_bloom = GetShaderLocation(r.composite, "bloom");
        r.comp_exposure = GetShaderLocation(r.composite, "exposure");
        r.comp_strength = GetShaderLocation(r.composite, "strength");
        r.comp_raw = GetShaderLocation(r.composite, "raw");
        r.comp_bg = GetShaderLocation(r.composite, "background");
    }
}

// (Re)creates the offscreen targets at the window's render size, or at w x h when given
// (export renders at its own size, not the window's). Returns false while the window is
// too small to draw into (e.g. minimized); the frame is then skipped. If the float scene
// texture can't be made at all (a driver without half-float render targets), it sets
// target_failed so the caller can stop with a message instead of showing black.
bool renderer_fit(Renderer &r, int forced_w = 0, int forced_h = 0) {
    const int w = forced_w > 0 ? forced_w : GetRenderWidth();
    const int h = forced_h > 0 ? forced_h : GetRenderHeight();
    if (w < 2 || h < 2) {
        return false;
    }
    if (w == r.w && h == r.h && r.scene.id != 0) {
        return true;
    }
    unload_target(r.scene);
    unload_target(r.bloom_a);
    unload_target(r.bloom_b);
    r.scene = load_float_target(w, h);
    r.bloom_a = load_float_target(std::max(1, w / 2), std::max(1, h / 2));
    r.bloom_b = load_float_target(std::max(1, w / 2), std::max(1, h / 2));
    r.w = w;
    r.h = h;
    if (r.scene.id == 0) {
        r.target_failed = true;
        return false;
    }
    BeginTextureMode(r.scene);
    ClearBackground(BLANK);
    EndTextureMode();
    return true;
}

void renderer_unload(Renderer &r) {
    unload_target(r.scene);
    unload_target(r.bloom_a);
    unload_target(r.bloom_b);
    UnloadTexture(r.glow);
    if (shader_loaded(r.blur)) {
        UnloadShader(r.blur);
    }
    if (shader_loaded(r.composite)) {
        UnloadShader(r.composite);
    }
    if (shader_loaded(r.sprite)) {
        UnloadShader(r.sprite);
    }
}

// The particles into the scene texture: glowing additive sprites, or opaque squares in
// plain mode. Sprites are sized from each particle's real radius (never under MIN_PIXELS).
void draw_particles(const Viewer &v, const Renderer &r, const Camera3D &cam, Vector3 right,
                    Vector3 up) {
    const bool plain = v.color == COLOR_PLAIN;
    const Vector3 fwd = scale(Vector3{cam.target.x - cam.position.x, cam.target.y - cam.position.y,
                                      cam.target.z - cam.position.z},
                              1.0f / v.dist);
    const float focal_px =
        0.5f * static_cast<float>(r.h) / std::tan(0.5f * FOVY_DEGREES * 3.14159265f / 180.0f);
    // No depth test: additive glow doesn't depend on draw order, and depth writes from
    // a sprite's transparent corners would cut dark squares into whatever is behind it.
    rlDisableDepthTest();
    rlDisableDepthMask();
    const bool boosted = !plain && r.shaders_ok; // the sprite shader: intensity past 1
    if (plain) {
        BeginBlendMode(BLEND_ALPHA);
        rlSetTexture(rlGetTextureIdDefault());
    } else if (boosted) {
        const float gain = SPRITE_GAIN;
        BeginShaderMode(r.sprite);
        SetShaderValue(r.sprite, r.sprite_gain, &gain, SHADER_UNIFORM_FLOAT);
        BeginBlendMode(BLEND_ADD_COLORS);
        rlSetTexture(r.glow.id);
    } else {
        BeginBlendMode(BLEND_ADDITIVE);
        rlSetTexture(r.glow.id);
    }
    rlBegin(RL_QUADS);
    for (std::size_t i = 0; i < v.p.size(); ++i) {
        rlCheckRenderBatchLimit(4); // flush the batch if it's full (keeps texture and mode)
        const Vector3 pos{static_cast<float>(v.p.x[i]), static_cast<float>(v.p.y[i]),
                          static_cast<float>(v.p.z[i])};
        const float depth = (pos.x - cam.position.x) * fwd.x + (pos.y - cam.position.y) * fwd.y +
                            (pos.z - cam.position.z) * fwd.z;
        const float half =
            sprite_half_size(v.p.radius[i], plain ? 1.0 : GLOW_FACTOR, depth, focal_px, MIN_PIXELS);
        const double speed =
            std::sqrt(v.p.vx[i] * v.p.vx[i] + v.p.vy[i] * v.p.vy[i] + v.p.vz[i] * v.p.vz[i]);
        const Rgb c = particle_color(v.color, v.p.body[i], v.p.material[i], v.p.mass[i], speed,
                                     v.p.heat[i], v.scales);
        float alpha = plain ? 1.0f : SPRITE_ALPHA;
        if (boosted) {
            float intensity = SPRITE_ALPHA;
            if (v.color == COLOR_HEAT) { // hot particles glow brighter, up to HOT_INTENSITY
                const double t = heat_glow(v.p.mass[i], v.p.heat[i], v.scales);
                intensity += static_cast<float>(t) * (HOT_INTENSITY - SPRITE_ALPHA);
            }
            alpha = intensity / SPRITE_GAIN;
        }
        rlColor4f(c.r, c.g, c.b, alpha);
        const Vector3 rr = scale(right, half), uu = scale(up, half);
        rlTexCoord2f(0.0f, 1.0f);
        rlVertex3f(pos.x - rr.x - uu.x, pos.y - rr.y - uu.y, pos.z - rr.z - uu.z);
        rlTexCoord2f(1.0f, 1.0f);
        rlVertex3f(pos.x + rr.x - uu.x, pos.y + rr.y - uu.y, pos.z + rr.z - uu.z);
        rlTexCoord2f(1.0f, 0.0f);
        rlVertex3f(pos.x + rr.x + uu.x, pos.y + rr.y + uu.y, pos.z + rr.z + uu.z);
        rlTexCoord2f(0.0f, 0.0f);
        rlVertex3f(pos.x - rr.x + uu.x, pos.y - rr.y + uu.y, pos.z - rr.z + uu.z);
    }
    rlEnd();
    rlSetTexture(0);
    EndBlendMode();
    if (boosted) {
        EndShaderMode();
    }
    rlEnableDepthMask();
}

// One blur pass from src into dst (dst's size), with the bright-pass on the first.
void blur_pass(const Renderer &r, const RenderTexture2D &src, const RenderTexture2D &dst,
               Vector2 dir, bool bright) {
    // Steps in units of the half-resolution bloom texture for both passes, so the blur
    // reaches as far vertically as horizontally (the first pass reads the full-size scene).
    const float texel[2] = {1.0f / static_cast<float>(dst.texture.width),
                            1.0f / static_cast<float>(dst.texture.height)};
    const int bright_flag = bright ? 1 : 0;
    const float threshold = BLOOM_THRESHOLD;
    BeginTextureMode(dst);
    ClearBackground(BLACK);
    BeginShaderMode(r.blur);
    SetShaderValue(r.blur, r.blur_texel, texel, SHADER_UNIFORM_VEC2);
    SetShaderValue(r.blur, r.blur_dir, &dir, SHADER_UNIFORM_VEC2);
    SetShaderValue(r.blur, r.blur_bright, &bright_flag, SHADER_UNIFORM_INT);
    SetShaderValue(r.blur, r.blur_threshold, &threshold, SHADER_UNIFORM_FLOAT);
    // Render textures are stored upside down; a negative source height flips them back.
    DrawTexturePro(src.texture,
                   Rectangle{0, 0, static_cast<float>(src.texture.width),
                             -static_cast<float>(src.texture.height)},
                   Rectangle{0, 0, static_cast<float>(dst.texture.width),
                             static_cast<float>(dst.texture.height)},
                   Vector2{0, 0}, 0.0f, WHITE);
    EndShaderMode();
    EndTextureMode();
}

void draw_replay_hud(const Viewer &v, const Renderer &r) {
    DrawRectangle(10, 10, 330, 158, Color{12, 14, 22, 200});
    int y = 18;
    const int line = 20;
    DrawText(TextFormat("replay    frame %d / %zu", v.replay_k + 1, v.run.paths.size()), 20, y, 20,
             Color{120, 200, 255, 255});
    y += line + 4;
    DrawText(TextFormat("particles %zu", v.p.size()), 20, y, 18, RAYWHITE);
    y += line;
    if (v.run.times.empty()) {
        DrawText("no energy.csv: steps between frames", 20, y, 18, RAYWHITE);
    } else {
        DrawText(TextFormat("sim time  %.4g of %.4g", v.replay_t, replay_end(v.run)), 20, y, 18,
                 RAYWHITE);
    }
    y += line;
    DrawText(TextFormat("speed     %.3g sim time / s", v.speed), 20, y, 18, RAYWHITE);
    y += line;
    DrawText(TextFormat("fps       %.0f", v.frame_ms > 0.0 ? 1000.0 / v.frame_ms : 0.0), 20, y, 18,
             RAYWHITE);
    y += line;
    DrawText(TextFormat("look      %s   bloom %s   trails %s", color_mode_name(v.color),
                        v.bloom && r.shaders_ok ? "on" : "off", v.trails ? "on" : "off"),
             20, y, 18, RAYWHITE);
    if (v.paused) {
        DrawText(v.replay_t >= replay_end(v.run) ? "end of the run (space plays again)"
                                                 : "paused (space)",
                 20, 180, 20, YELLOW);
    }
    if (omp_get_wtime() < v.notice_until) {
        DrawText(v.notice.c_str(), 20, GetScreenHeight() - 60, 20, Color{255, 200, 120, 255});
    }
    DrawText("space pause   left/right frame   up/down speed   home start   c color   b bloom   "
             "t trails   drag orbit   wheel zoom",
             20, GetScreenHeight() - 30, 16, GRAY);
}

void draw_hud(const Viewer &v, const Renderer &r) {
    if (v.replaying) {
        draw_replay_hud(v, r);
        return;
    }
    char threads[16] = "GPU"; // the hip backend computes gravity on the GPU
    if (v.backend != BACKEND_HIP) {
        std::snprintf(threads, sizeof threads, "%d",
                      v.backend == BACKEND_OMP ? omp_get_max_threads() : 1);
    }
    const double steps_per_s = v.ms_per_step > 0.0 ? 1000.0 / v.ms_per_step : 0.0;
    DrawRectangle(10, 10, 330, 198, Color{12, 14, 22, 200});
    int y = 18;
    const int line = 20;
    const char *precision = "double"; // the CPU backends always compute in double
#ifdef SIM_HAVE_HIP
    if (v.backend == BACKEND_HIP && !hip_uses_double()) {
        precision = "float";
    }
#endif
    DrawText(TextFormat("backend   %s (%s)", backend_name(v.backend), precision), 20, y, 20,
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
    y += line;
    DrawText(TextFormat("look      %s   bloom %s   trails %s", color_mode_name(v.color),
                        v.bloom && r.shaders_ok ? "on" : "off", v.trails ? "on" : "off"),
             20, y, 18, RAYWHITE);

    if (v.blew_up) {
        DrawText(TextFormat("simulation blew up (particle %zu is NaN/Inf) at step %lld: press r",
                            v.bad_particle, v.step),
                 20, 220, 20, RED);
    } else if (v.paused) {
        DrawText("paused (space)", 20, 220, 20, YELLOW);
    }
    if (omp_get_wtime() < v.notice_until) {
        DrawText(v.notice.c_str(), 20, GetScreenHeight() - 60, 20, Color{255, 200, 120, 255});
    }
    DrawText("1/2/3 serial/omp/hip   p float/double   c color   b bloom   t trails   space pause   "
             "r restart   [ ] budget   drag orbit   wheel zoom",
             20, GetScreenHeight() - 30, 16, GRAY);
}

// The particles and bloom into the renderer's offscreen textures (already fit to size).
// Returns whether bloom was drawn, for composite_to.
bool render_scene(const Viewer &v, Renderer &r) {
    const float cp = std::cos(v.pitch), sp = std::sin(v.pitch);
    const float cy = std::cos(v.yaw), sy = std::sin(v.yaw);
    Camera3D cam{};
    cam.position = Vector3{v.dist * cp * sy, v.dist * sp, v.dist * cp * cy};
    cam.target = Vector3{0.0f, 0.0f, 0.0f};
    cam.up = Vector3{0.0f, 1.0f, 0.0f};
    cam.fovy = FOVY_DEGREES;
    cam.projection = CAMERA_PERSPECTIVE;
    // Screen-aligned axes for the particle sprites.
    const Vector3 right{cy, 0.0f, -sy};
    const Vector3 up{-sp * sy, cp, -sp * cy};

    const bool plain = v.color == COLOR_PLAIN;
    BeginTextureMode(r.scene);
    if (v.trails && !plain && !v.camera_moved) {
        // Fade what's there instead of clearing it, by a fraction set from the frame time so
        // trails last the same on fast and slow machines. Float channels fade all the way
        // to black (8-bit ones would stall at a dim ghost).
        const float fade = 1.0f - std::exp(-GetFrameTime() / TRAIL_SECONDS);
        DrawRectangle(0, 0, r.w, r.h, Color{0, 0, 0, static_cast<unsigned char>(fade * 255.0f)});
    } else {
        ClearBackground(BLANK);
    }
    BeginMode3D(cam);
    draw_particles(v, r, cam, right, up);
    EndMode3D();
    EndTextureMode();

    const bool bloom = v.bloom && !plain && r.shaders_ok && r.bloom_a.id != 0 && r.bloom_b.id != 0;
    if (bloom) {
        blur_pass(r, r.scene, r.bloom_a, Vector2{1.0f, 0.0f}, true);
        blur_pass(r, r.bloom_a, r.bloom_b, Vector2{0.0f, 1.0f}, false);
    }
    return bloom;
}

// The scene through the tone-map pass (plus bloom) into dst of whatever is being drawn
// to: the screen, or export's image target.
void composite_to(const Viewer &v, const Renderer &r, bool bloom, Rectangle dst) {
    const bool plain = v.color == COLOR_PLAIN;
    const Rectangle src{0, 0, static_cast<float>(r.w), -static_cast<float>(r.h)};
    if (r.shaders_ok) {
        const float exposure = EXPOSURE, strength = bloom ? BLOOM_STRENGTH : 0.0f;
        const int raw = plain ? 1 : 0;
        const float bg[3] = {BACKGROUND.r / 255.0f, BACKGROUND.g / 255.0f, BACKGROUND.b / 255.0f};
        BeginShaderMode(r.composite);
        SetShaderValueTexture(r.composite, r.comp_bloom, r.bloom_b.texture);
        SetShaderValue(r.composite, r.comp_exposure, &exposure, SHADER_UNIFORM_FLOAT);
        SetShaderValue(r.composite, r.comp_strength, &strength, SHADER_UNIFORM_FLOAT);
        SetShaderValue(r.composite, r.comp_raw, &raw, SHADER_UNIFORM_INT);
        SetShaderValue(r.composite, r.comp_bg, bg, SHADER_UNIFORM_VEC3);
        DrawTexturePro(r.scene.texture, src, dst, Vector2{0, 0}, 0.0f, WHITE);
        EndShaderMode();
    } else {
        DrawTexturePro(r.scene.texture, src, dst, Vector2{0, 0}, 0.0f, WHITE);
    }
}

// One window frame: scene, composite to the screen, HUD.
void draw(const Viewer &v, Renderer &r) {
    if (!renderer_fit(r)) { // minimized: nothing to draw into
        BeginDrawing();
        EndDrawing();
        return;
    }
    const bool bloom = render_scene(v, r);
    BeginDrawing();
    ClearBackground(BACKGROUND);
    composite_to(v, r, bloom,
                 Rectangle{0, 0, static_cast<float>(GetScreenWidth()),
                           static_cast<float>(GetScreenHeight())});
    draw_hud(v, r);
    EndDrawing();
}

// The viewer's own flags (planet_sim doesn't know them).
struct ViewerFlags {
    long frames = -1; // -1: until the window is closed
    std::string screenshot;
    ColorMode color = COLOR_HEAT;
    bool bloom = true;
    bool trails = false;
    // Replay and export (#49).
    std::string replay, export_dir;
    double speed = 0.0; // 0: chosen from the run (replay_setup)
    double fps = 60.0;
    int width = 1920, height = 1080;
    double orbit_degrees = 40.0;
    bool has_fps = false, has_size = false, has_orbit = false, has_speed = false;
};

// A finite number with nothing after it, or false.
bool parse_number(const char *s, double *out) {
    char *end = nullptr;
    errno = 0;
    const double d = std::strtod(s, &end);
    if (end == s || *end != '\0' || errno == ERANGE || !std::isfinite(d)) {
        return false;
    }
    *out = d;
    return true;
}

// "WxH" with each side a whole number from 16 to 7680.
bool parse_size(const char *s, int *w, int *h) {
    char *end = nullptr;
    const long a = std::strtol(s, &end, 10);
    if (end == s || (*end != 'x' && *end != 'X') || !(s[0] >= '0' && s[0] <= '9')) {
        return false;
    }
    const char *rest = end + 1;
    const long b = std::strtol(rest, &end, 10);
    if (end == rest || *end != '\0' || !(rest[0] >= '0' && rest[0] <= '9') || a < 16 || a > 7680 ||
        b < 16 || b > 7680) {
        return false;
    }
    *w = static_cast<int>(a);
    *h = static_cast<int>(b);
    return true;
}

// Pulls the viewer-only flags out of argv, leaving the rest for parse_cli. Returns false
// on a bad value, before any window opens.
bool take_viewer_flags(int argc, char **argv, std::vector<const char *> *rest, ViewerFlags *f,
                       std::string *error) {
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-bloom") == 0) {
            f->bloom = false;
            continue;
        }
        if (std::strcmp(argv[i], "--trails") == 0) {
            f->trails = true;
            continue;
        }
        const bool is_frames = std::strcmp(argv[i], "--frames") == 0;
        const bool is_shot = std::strcmp(argv[i], "--screenshot") == 0;
        const bool is_color = std::strcmp(argv[i], "--color") == 0;
        const char *flag = argv[i];
        const bool is_replay_flag =
            std::strcmp(flag, "--replay") == 0 || std::strcmp(flag, "--export") == 0 ||
            std::strcmp(flag, "--speed") == 0 || std::strcmp(flag, "--fps") == 0 ||
            std::strcmp(flag, "--size") == 0 || std::strcmp(flag, "--orbit") == 0;
        if (!is_frames && !is_shot && !is_color && !is_replay_flag) {
            rest->push_back(argv[i]);
            continue;
        }
        const char *value = i + 1 < argc ? argv[i + 1] : "";
        ++i;
        if (is_replay_flag) {
            const std::string name = flag;
            double d = 0.0;
            if (name == "--replay" || name == "--export") {
                if (value[0] == '\0' || std::strncmp(value, "--", 2) == 0) {
                    *error = name + " needs a folder, got '" + value + "'";
                    return false;
                }
                (name == "--replay" ? f->replay : f->export_dir) = value;
            } else if (name == "--size") {
                if (!parse_size(value, &f->width, &f->height)) {
                    *error = std::string("--size needs WxH with each side 16 to 7680, e.g. "
                                         "1920x1080, got '") +
                             value + "'";
                    return false;
                }
                f->has_size = true;
            } else if (!parse_number(value, &d) ||
                       (name == "--orbit" ? std::fabs(d) > 3600.0
                                          : !(d > 0.0) || (name == "--fps" && d > 1000.0))) {
                *error = name +
                         (name == "--orbit" ? " needs degrees from -3600 to 3600"
                          : name == "--fps" ? " needs a number above 0, at most 1000"
                                            : " needs a number above 0") +
                         ", got '" + value + "'";
                return false;
            } else if (name == "--speed") {
                f->speed = d;
                f->has_speed = true;
            } else if (name == "--fps") {
                f->fps = d;
                f->has_fps = true;
            } else {
                f->orbit_degrees = d;
                f->has_orbit = true;
            }
            continue;
        }
        if (is_color) {
            if (!color_mode_from_name(value, &f->color)) {
                *error =
                    std::string("--color needs heat, material, planet, speed or plain, got '") +
                    value + "'";
                return false;
            }
            continue;
        }
        if (is_shot) {
            // Checked now, not after the whole run: raylib picks the format by extension.
            const std::size_t len = std::strlen(value);
            if (len < 5 || std::strcmp(value + len - 4, ".png") != 0) {
                *error = std::string("--screenshot needs a .png file name, got '") + value + "'";
                return false;
            }
            f->screenshot = value;
            continue;
        }
        char *end = nullptr;
        errno = 0;
        f->frames = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || errno == ERANGE || f->frames < 1 ||
            !(value[0] >= '0' && value[0] <= '9')) {
            *error = std::string("--frames needs a whole number >= 1, got '") + value + "'";
            return false;
        }
    }
    return true;
}

// Renders the replay offscreen to <export_dir>/frame_NNNNN.png: `count` images from the
// first frame's time to the last at the replay speed, the camera swinging round by
// --orbit degrees over the whole video. Returns the exit code: 0, 1 for a setup or write
// failure, 2 for a frame with NaN/Inf or frames that don't match.
int run_export(Viewer &v, Renderer &r, const ViewerFlags &f, long count) {
    if (!r.shaders_ok) {
        std::fprintf(stderr, "planet_view: the glow shaders failed to compile, so --export "
                             "can't render the look; stopping\n");
        return 1;
    }
    RenderTexture2D image = LoadRenderTexture(f.width, f.height); // 8-bit, what PNGs hold
    if (image.id == 0) {
        std::fprintf(stderr, "planet_view: couldn't make a %dx%d image to render into\n", f.width,
                     f.height);
        return 1;
    }
    const double t0 = replay_start(v.run), t1 = replay_end(v.run);
    const float yaw0 = v.yaw;
    const double orbit = f.orbit_degrees * 3.14159265358979323846 / 180.0;
    v.trails = false; // trails depend on the frame time, which export doesn't have
    const double started = omp_get_wtime();
    int code = 0;
    for (long i = 0; i < count; ++i) {
        v.replay_t = export_time(i, t0, t1, v.speed, f.fps);
        std::string error;
        const ReplayShow shown = replay_show(v, &error);
        if (shown != REPLAY_OK) {
            std::fprintf(stderr, "planet_view: %s\n", error.c_str());
            code = shown == REPLAY_READ_FAILED ? 1 : 2;
            break;
        }
        const double turn =
            count > 1 ? orbit * static_cast<double>(i) / static_cast<double>(count - 1) : 0.0;
        v.yaw = yaw0 + static_cast<float>(turn);
        if (!renderer_fit(r, f.width, f.height)) {
            std::fprintf(stderr, "planet_view: this graphics driver can't render to a 16-bit "
                                 "float texture, which the viewer needs; stopping\n");
            code = 1;
            break;
        }
        const bool bloom = render_scene(v, r);
        BeginTextureMode(image);
        ClearBackground(BACKGROUND);
        composite_to(v, r, bloom,
                     Rectangle{0, 0, static_cast<float>(f.width), static_cast<float>(f.height)});
        EndTextureMode();
        Image img = LoadImageFromTexture(image.texture);
        ImageFlipVertical(&img); // render textures are stored upside down
        char name[32];
        std::snprintf(name, sizeof name, "/frame_%05ld.png", i);
        const std::string path = f.export_dir + name;
        const bool saved = ExportImage(img, path.c_str());
        UnloadImage(img);
        if (!saved) {
            std::fprintf(stderr, "planet_view: couldn't write '%s'\n", path.c_str());
            code = 1;
            break;
        }
        if ((i + 1) * 10 / count != i * 10 / count) {
            std::printf("export: %ld / %ld images\n", i + 1, count);
            std::fflush(stdout);
        }
    }
    UnloadRenderTexture(image);
    if (code != 0) {
        std::fprintf(stderr,
                     "planet_view: export stopped; images written so far are in %s (remove it "
                     "or pick a new folder before exporting again)\n",
                     f.export_dir.c_str());
    }
    if (code == 0) {
        std::printf("wrote %ld image%s (%dx%d) to %s in %.1f s\n", count, count == 1 ? "" : "s",
                    f.width, f.height, f.export_dir.c_str(), omp_get_wtime() - started);
    }
    return code;
}

// Checks the replay and export flags and opens the run, before any window opens.
// Returns false with a message on any problem.
bool replay_from_flags(Viewer &v, const ViewerFlags &flags, const std::vector<const char *> &args,
                       long *export_count, std::string *error) {
    if (flags.export_dir.empty() && (flags.has_fps || flags.has_size || flags.has_orbit)) {
        *error = "--fps, --size and --orbit set up --export";
        return false;
    }
    if (!flags.export_dir.empty() && (flags.frames >= 0 || !flags.screenshot.empty())) {
        *error = "--frames and --screenshot are for the window; --export writes every image "
                 "itself";
        return false;
    }
    if (args.size() > 1) { // args[0] is the program name
        *error = std::string(args[1]) +
                 " doesn't apply to --replay (a replay shows recorded frames and runs no "
                 "simulation)";
        return false;
    }
    if (!flags.export_dir.empty()) {
        // Never overwrite an earlier export: the folder must be new or empty.
        std::error_code ec;
        const std::filesystem::path dir(flags.export_dir);
        if (std::filesystem::exists(dir, ec) &&
            (!std::filesystem::is_directory(dir, ec) || !std::filesystem::is_empty(dir, ec))) {
            *error = "--export folder '" + flags.export_dir +
                     "' already exists and isn't empty; give a new folder";
            return false;
        }
    }
    if (!replay_setup(v, flags.replay.c_str(), flags.speed, !flags.export_dir.empty(), error)) {
        return false;
    }
    if (!flags.export_dir.empty()) {
        *export_count =
            export_frame_count(replay_start(v.run), replay_end(v.run), v.speed, flags.fps);
        if (*export_count < 0) {
            *error = "that --speed and --fps would make over 1,000,000 images";
            return false;
        }
        std::error_code ec;
        std::filesystem::create_directories(flags.export_dir, ec);
        if (ec) {
            *error = "can't create '" + flags.export_dir + "': " + ec.message();
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
    ViewerFlags flags;
    std::string error;
    if (!take_viewer_flags(argc, argv, &args, &flags, &error)) {
        std::fprintf(stderr, "planet_view: %s\n%s", error.c_str(), viewer_usage());
        return 1;
    }
    for (std::size_t i = 1; i < args.size(); ++i) {
        if (std::strcmp(args[i], "--help") == 0 || std::strcmp(args[i], "-h") == 0) {
            std::printf("%s", viewer_usage());
            return 0;
        }
    }
    const bool replaying = !flags.replay.empty();
    if (!replaying && (!flags.export_dir.empty() || flags.has_speed || flags.has_fps ||
                       flags.has_size || flags.has_orbit)) {
        std::fprintf(stderr,
                     "planet_view: --export, --speed, --fps, --size and --orbit need --replay "
                     "<run folder>\n%s",
                     viewer_usage());
        return 1;
    }
    v.color = flags.color;
    v.bloom = flags.bloom;
    v.trails = flags.trails;
    long export_count = 0;
    if (replaying) {
        if (!replay_from_flags(v, flags, args, &export_count, &error)) {
            std::fprintf(stderr, "planet_view: %s\n", error.c_str());
            return 1;
        }
    } else {
        if (!parse_cli(static_cast<int>(args.size()), args.data(), &v.cli, &error)) {
            std::fprintf(stderr, "planet_view: %s\n%s", error.c_str(), viewer_usage());
            return 1;
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
        hip_set_double(v.cli.gpu_double);
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
    }

    const bool exporting = !flags.export_dir.empty();
    // Export draws only offscreen, so its window stays hidden, and raylib's per-image log
    // lines (two for every PNG) are turned down to warnings.
    if (exporting) {
        SetTraceLogLevel(LOG_WARNING);
    }
    SetConfigFlags(exporting ? FLAG_WINDOW_HIDDEN : (FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT));
    InitWindow(exporting ? 320 : 1280, exporting ? 200 : 800, "planet_view");
    SetTargetFPS(60);
    Renderer r;
    renderer_init(r);
    if (exporting) {
        const int code = run_export(v, r, flags, export_count);
        renderer_unload(r);
        CloseWindow();
        return code;
    }
    if (!r.shaders_ok) {
        // Not fatal: particles still draw, just without bloom and brightness compression.
        std::fprintf(stderr, "planet_view: glow shaders failed to compile; bloom disabled\n");
        show_notice(v, "bloom disabled: shaders failed to compile");
    }
    for (long frame = 0; !WindowShouldClose() && (flags.frames < 0 || frame < flags.frames);
         ++frame) {
        handle_input(v);
        advance(v);
        draw(v, r);
        if (r.target_failed) {
            std::fprintf(stderr, "planet_view: this graphics driver can't render to a 16-bit float "
                                 "texture, which the viewer needs; stopping\n");
            renderer_unload(r);
            CloseWindow();
            return 1;
        }
        const double ms = GetFrameTime() * 1000.0;
        v.frame_ms = v.frame_ms == 0.0 ? ms : 0.8 * v.frame_ms + 0.2 * ms;
    }
    if (!flags.screenshot.empty()) {
        // Read back what's on screen now (the last frame drawn) and save it.
        Image shot = LoadImageFromScreen();
        const bool saved = ExportImage(shot, flags.screenshot.c_str());
        UnloadImage(shot);
        if (!saved) {
            std::fprintf(stderr, "planet_view: couldn't save screenshot '%s'\n",
                         flags.screenshot.c_str());
            renderer_unload(r);
            CloseWindow();
            return 1;
        }
    }
    renderer_unload(r);
    CloseWindow();
    if (v.blew_up) { // like planet_sim: a run that exploded must not look like success
        std::fprintf(stderr, "planet_view: particle %zu became NaN/Inf at step %lld\n",
                     v.bad_particle, v.step);
        return 2;
    }
    if (v.replay_exit != 0) { // a replay that met a broken frame must not look like success
        std::fprintf(stderr, "planet_view: %s\n", v.replay_error.c_str());
        return v.replay_exit;
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
