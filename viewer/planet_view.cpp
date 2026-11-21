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
#include "look.hpp"
#include "sim.hpp"

namespace {

const char *viewer_usage() {
    return "usage: planet_view (--config <file> | --scenario orbit|figure8|cloud [--n N] "
           "[--seed S])\n"
           "                   [--backend serial|omp|hip] [--gpu-precision float|double]\n"
           "                   [--dt h] [--frames N] [--screenshot <file.png>]\n"
           "                   [--color heat|material|planet|speed|plain] [--no-bloom] [--trails]\n"
           "                   [--help]\n"
           "  --frames      quit after N frames (for scripted runs and CI); default: run "
           "until closed\n"
           "  --screenshot  save the last frame as a PNG when the viewer exits\n"
           "  --color       starting color mode (default heat: material, glowing where hot)\n"
           "keys: 1/2/3 serial/OpenMP/HIP, p GPU float/double, c color, b bloom, t trails,\n"
           "      space pause, r restart, [ ] step time budget, drag to orbit, wheel to zoom\n"
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
    bool camera_moved = false; // this frame: trails are cleared instead of smeared

    // The look (#47).
    ColorMode color = COLOR_HEAT;
    bool bloom = true;
    bool trails = false;
    LookScales scales{1.0, 0.5}; // fixed per scenario: set in setup()
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
#ifdef SIM_HAVE_HIP
    if (IsKeyPressed(KEY_P)) { // GPU precision, live: the float-vs-double trade-off on screen
        hip_set_double(!hip_uses_double());
        v.ms_per_step = 0.0;
        v.frame_ms = 0.0;
        show_notice(v, std::string("GPU precision: ") + (hip_uses_double() ? "double" : "float"));
    }
#endif
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
    if (IsKeyPressed(KEY_LEFT_BRACKET)) {
        v.budget_ms = std::fmax(1.0, v.budget_ms / 1.5);
    }
    if (IsKeyPressed(KEY_RIGHT_BRACKET)) {
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

// (Re)creates the offscreen targets at the window's render size. Returns false while the
// window is too small to draw into (e.g. minimized); the frame is then skipped. If the
// float scene texture can't be made at all (a driver without half-float render targets),
// it sets target_failed so the caller can stop with a message instead of showing black.
bool renderer_fit(Renderer &r) {
    const int w = GetRenderWidth(), h = GetRenderHeight();
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

void draw_hud(const Viewer &v, const Renderer &r) {
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

void draw(const Viewer &v, Renderer &r) {
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

    if (!renderer_fit(r)) { // minimized: nothing to draw into
        BeginDrawing();
        EndDrawing();
        return;
    }
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

    BeginDrawing();
    ClearBackground(BACKGROUND);
    const Rectangle src{0, 0, static_cast<float>(r.w), -static_cast<float>(r.h)};
    const Rectangle dst{0, 0, static_cast<float>(GetScreenWidth()),
                        static_cast<float>(GetScreenHeight())};
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
};

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
        if (!is_frames && !is_shot && !is_color) {
            rest->push_back(argv[i]);
            continue;
        }
        const char *value = i + 1 < argc ? argv[i + 1] : "";
        ++i;
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

// The viewer's whole run, separated from main() so running out of memory for a huge --n
// is reported cleanly (as in planet_sim).
int run(int argc, char **argv) {
    Viewer v;
    std::vector<const char *> args;
    ViewerFlags flags;
    std::string error;
    if (!take_viewer_flags(argc, argv, &args, &flags, &error) ||
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
    v.color = flags.color;
    v.bloom = flags.bloom;
    v.trails = flags.trails;
#ifdef SIM_HAVE_HIP
    hip_set_double(v.cli.gpu_double);
#endif
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
    Renderer r;
    renderer_init(r);
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
