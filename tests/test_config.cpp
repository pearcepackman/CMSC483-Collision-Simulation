#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <string>

#include <unistd.h>

#include "config.hpp"

namespace {

// A folder of our own inside gtest's temp directory: the process id and the test name
// keep two test runs (ctest -j, or dev and asan at once) from overwriting each
// other's files, as in test_io.cpp.
std::string test_dir() {
    const char *name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::path dir = std::filesystem::path(::testing::TempDir()) /
                                ("test_config_" + std::to_string(getpid())) / name;
    std::filesystem::create_directories(dir);
    return dir.string();
}

// Writes `text` to `name` in this test's folder and returns its path.
std::string write_cfg(const char *name, const std::string &text) {
    std::string path = test_dir() + "/" + name;
    FILE *f = std::fopen(path.c_str(), "wb");
    EXPECT_NE(f, nullptr) << path;
    if (f != nullptr) {
        std::fputs(text.c_str(), f);
        std::fclose(f);
    }
    return path;
}

// The smallest file that loads: output_every and one complete planet.
const char *kMinimal = "output_every = 10\n"
                       "planet0.n = 10\n"
                       "planet0.radius = 1\n"
                       "planet0.particle_radius = 0.1\n"
                       "planet0.mass = 1\n";

// Loads `text` and returns what load_config printed to stderr; *ok gets its result.
std::string load_text(const std::string &text, Config &cfg, bool *ok) {
    std::string path = write_cfg("test.cfg", text);
    testing::internal::CaptureStderr();
    *ok = load_config(path.c_str(), cfg);
    return testing::internal::GetCapturedStderr();
}

// Expects `text` to fail to load with a message containing `why`, leaving cfg as it was.
void expect_rejected(const std::string &text, const std::string &why) {
    SCOPED_TRACE(text);
    Config cfg;
    cfg.steps = 77; // sentinel: a failed load must not touch cfg
    bool ok = true;
    std::string err = load_text(text, cfg, &ok);
    EXPECT_FALSE(ok);
    EXPECT_NE(err.find(why), std::string::npos) << "stderr was: " << err;
    EXPECT_EQ(cfg.steps, 77);
    EXPECT_TRUE(cfg.planets.empty());
}

} // namespace

TEST(Config, SampleFileLoadsEveryKey) {
    const char *text = "# a comment line\n"
                       "\n"
                       "dt = 0.002\n"
                       "softening=0.05\n" // no spaces
                       "  contact = true   # trailing comment\n"
                       "contact_k = 20\n"
                       "contact_c = 0.05\n"
                       "steps = 5000\n"
                       "output_every = 25\n"
                       "relax_ke_threshold = 1e-4\n"
                       "relax_max_steps = 1500\n"
                       "planet0.n = 2000\n"
                       "planet0.radius = 1.0\n"
                       "planet0.particle_radius = 0.065\n"
                       "planet0.mass = 2.0\n"
                       "planet0.center = -2.5 0 0.25\n"
                       "planet0.velocity = 0.5\t-0.125 0\n" // tabs separate too
                       "planet0.seed = 7\n"
                       "planet1.mass = 0.5\n" // planet keys in any order
                       "planet1.n = 300\n"
                       "planet1.particle_radius = 0.04\n"
                       "planet1.radius = 0.6\n";
    Config cfg;
    bool ok = false;
    std::string err = load_text(text, cfg, &ok);
    ASSERT_TRUE(ok) << err;
    EXPECT_EQ(err, "");
    EXPECT_EQ(cfg.params.dt, 0.002);
    EXPECT_EQ(cfg.params.softening, 0.05);
    EXPECT_TRUE(cfg.params.contact_enabled);
    EXPECT_EQ(cfg.params.contact_k, 20.0);
    EXPECT_EQ(cfg.params.contact_c, 0.05);
    EXPECT_EQ(cfg.steps, 5000);
    EXPECT_EQ(cfg.output_every, 25);
    EXPECT_EQ(cfg.relax_ke_threshold, 1e-4);
    EXPECT_EQ(cfg.relax_max_steps, 1500);

    ASSERT_EQ(cfg.planets.size(), 2u);
    const PlanetSpec &a = cfg.planets[0];
    EXPECT_EQ(a.n, 2000);
    EXPECT_EQ(a.radius, 1.0);
    EXPECT_EQ(a.particle_radius, 0.065);
    EXPECT_EQ(a.mass, 2.0);
    EXPECT_EQ(a.center.x, -2.5);
    EXPECT_EQ(a.center.y, 0.0);
    EXPECT_EQ(a.center.z, 0.25);
    EXPECT_EQ(a.velocity.x, 0.5);
    EXPECT_EQ(a.velocity.y, -0.125);
    EXPECT_EQ(a.velocity.z, 0.0);
    EXPECT_EQ(a.seed, 7u);
    EXPECT_EQ(a.body, 0);
    const PlanetSpec &b = cfg.planets[1];
    EXPECT_EQ(b.n, 300);
    EXPECT_EQ(b.radius, 0.6);
    EXPECT_EQ(b.particle_radius, 0.04);
    EXPECT_EQ(b.mass, 0.5);
    EXPECT_EQ(b.center.x, 0.0); // not set: PlanetSpec defaults
    EXPECT_EQ(b.velocity.x, 0.0);
    EXPECT_EQ(b.seed, 0u);
    EXPECT_EQ(b.body, 1); // body id = planet number, for coloring frames
}

TEST(Config, MissingKeysKeepValuesAlreadyInCfg) {
    Config cfg;
    cfg.params.dt = 0.125;
    cfg.params.contact_k = 3.0;
    cfg.steps = 42;
    cfg.relax_max_steps = 9;
    bool ok = false;
    std::string err = load_text(kMinimal, cfg, &ok);
    ASSERT_TRUE(ok) << err;
    EXPECT_EQ(cfg.params.dt, 0.125);
    EXPECT_EQ(cfg.params.contact_k, 3.0);
    EXPECT_FALSE(cfg.params.contact_enabled);
    EXPECT_EQ(cfg.steps, 42);
    EXPECT_EQ(cfg.relax_max_steps, 9);
    EXPECT_EQ(cfg.output_every, 10);
    ASSERT_EQ(cfg.planets.size(), 1u);
}

TEST(Config, WindowsLineEndingsLoad) {
    Config cfg;
    bool ok = false;
    std::string err = load_text("output_every = 5\r\nsteps = 3\r\n"
                                "planet0.n = 1\r\nplanet0.radius = 1\r\n"
                                "planet0.particle_radius = 0.1\r\nplanet0.mass = 1\r\n",
                                cfg, &ok);
    ASSERT_TRUE(ok) << err;
    EXPECT_EQ(cfg.output_every, 5);
    EXPECT_EQ(cfg.steps, 3);
}

TEST(Config, MisspelledKeyFails) {
    // The error names the file's line number and the key.
    expect_rejected(std::string(kMinimal) + "stpes = 100\n", "test.cfg:6: unknown key 'stpes'");
    expect_rejected(std::string(kMinimal) + "planet0.raduis = 1\n", "unknown key 'planet0.raduis'");
    expect_rejected(std::string(kMinimal) + "DT = 0.1\n", "unknown key 'DT'"); // case-sensitive
    expect_rejected(std::string(kMinimal) + "planet.n = 1\n", "unknown key 'planet.n'");
    expect_rejected(std::string(kMinimal) + "planetx.n = 1\n", "unknown key 'planetx.n'");
}

TEST(Config, LineNumbersCountCommentsAndBlankLines) {
    // Real configs are mostly comments: the reported line must be the file's line.
    expect_rejected("# header comment\n"
                    "\n"
                    "output_every = 10 # trailing\n"
                    "   # indented comment\n"
                    "\n"
                    "stpes = 3\n",
                    "test.cfg:6: unknown key 'stpes'");
}

TEST(Config, ByteOrderMarkIsIgnored) {
    // Windows PowerShell 5.1's Out-File starts files with a UTF-8 byte order mark.
    Config cfg;
    bool ok = false;
    std::string err = load_text(std::string("\xEF\xBB\xBF") + kMinimal, cfg, &ok);
    EXPECT_TRUE(ok) << err;
    EXPECT_EQ(cfg.output_every, 10);
    ok = false;
    err = load_text(std::string("\xEF\xBB\xBF# comment first\n") + kMinimal, cfg, &ok);
    EXPECT_TRUE(ok) << err;
    // Only at the very start of the file: anywhere else it is part of a key.
    expect_rejected(std::string(kMinimal) + "\xEF\xBB\xBFsteps = 3\n", "unknown key");
}

TEST(Config, MalformedLinesFail) {
    expect_rejected(std::string(kMinimal) + "steps 100\n", ":6: expected 'key = value'");
    expect_rejected(std::string(kMinimal) + "= 100\n", "missing key before '='");
    expect_rejected(std::string(kMinimal) + "steps =\n", "missing value for 'steps'");
    expect_rejected(std::string(kMinimal) + "steps = # nothing\n", "missing value for 'steps'");
    expect_rejected(std::string(kMinimal) + "steps = 1\nsteps = 2\n", ":7: 'steps' is set twice");
    expect_rejected(std::string(kMinimal) + "planet0.n = 5\n", "'planet0.n' is set twice");
}

TEST(Config, BadValuesFail) {
    const char *bad[] = {
        "dt = fast",
        "dt = 0.1x",
        "dt = nan",
        "dt = inf",
        "dt = 1e999",
        "dt = 0.1 0.2",
        "steps = 1.5",
        "steps = 99999999999",
        "steps = 1e3",
        "contact = yes",
        "contact = 1",
        "contact = True",
        "planet0.seed = -1",
        "planet0.seed = 1e3",
        "planet0.center = 1 2",
        "planet0.center = 1 2 3 4",
        "planet0.center = 1,2,3",
        "planet0.center = 1-2-3", // would otherwise read as (1, -2, -3)
        "planet0.center = 1+2+3",
        "planet0.velocity = 1 nan 0",
        "planet0.seed = 4294967296", // UINT_MAX + 1 would wrap to seed 0
        "dt = 1e-400",               // underflows to 0
    };
    for (const char *line : bad) {
        std::string text = std::string(kMinimal) + line + "\n";
        // planet0.seed etc. are new keys here; only planet0.n/radius/... are in kMinimal.
        std::string key = std::string(line).substr(0, std::string(line).find(' '));
        expect_rejected(text, "bad value for '" + key + "'");
    }
}

TEST(Config, OutputEveryIsRequired) {
    // Without it every step would write a frame and compute the O(N^2) energy.
    expect_rejected("planet0.n = 10\nplanet0.radius = 1\nplanet0.particle_radius = 0.1\n"
                    "planet0.mass = 1\n",
                    "missing 'output_every'");
}

TEST(Config, PlanetsMustBeCompleteAndNumberedInOrder) {
    expect_rejected("output_every = 1\n", "no planets");
    // planet1 without planet0.
    expect_rejected("output_every = 1\nplanet1.n = 10\nplanet1.radius = 1\n"
                    "planet1.particle_radius = 0.1\nplanet1.mass = 1\n",
                    "planet0 is missing");
    // A gap: planet0 and planet2.
    expect_rejected(std::string(kMinimal) + "planet2.n = 10\nplanet2.radius = 1\n"
                                            "planet2.particle_radius = 0.1\nplanet2.mass = 1\n",
                    "planet1 is missing");
    // Each required key left out in turn.
    const char *required[] = {"n = 10", "radius = 1", "particle_radius = 0.1", "mass = 1"};
    for (int skip = 0; skip < 4; ++skip) {
        std::string text = "output_every = 1\n";
        for (int k = 0; k < 4; ++k) {
            if (k != skip) {
                text += std::string("planet0.") + required[k] + "\n";
            }
        }
        expect_rejected(text, "planet0 needs n, radius, particle_radius and mass");
    }
    // Only optional keys set still counts as a planet that is missing its required keys.
    expect_rejected(std::string(kMinimal) + "planet1.seed = 3\n",
                    "planet1 needs n, radius, particle_radius and mass");
    // "planet01" would be a second name for planet1.
    expect_rejected(std::string(kMinimal) + "planet01.n = 10\n", "no leading zeros");
    expect_rejected(std::string(kMinimal) + "planet1000.n = 10\n", "planet numbers are 0 to 999");
    expect_rejected(std::string(kMinimal) + "planet99999999999.n = 10\n",
                    "planet numbers are 0 to 999"); // too many digits for atoi
}

// Values make_planet would abort on are a clean load error that names the planet, so a
// typo in a config is exit 1 with a message, not a crash.
TEST(Config, PlanetsMakePlanetCantBuildFail) {
    const std::string base = "output_every = 1\nplanet0.n = 10\nplanet0.radius = 1\n";
    expect_rejected("output_every = 1\nplanet0.n = -5\nplanet0.radius = 1\n"
                    "planet0.particle_radius = 0.1\nplanet0.mass = 1\n",
                    "test.cfg: planet0: n must be >= 0");
    expect_rejected(base + "planet0.particle_radius = 0.1\nplanet0.mass = -1\n",
                    "planet0: radius, particle_radius and mass must be finite and >= 0");
    expect_rejected(base + "planet0.particle_radius = 0.1\nplanet0.mass = 0\n",
                    "planet0: mass must be > 0 when n > 0");
    expect_rejected(base + "planet0.particle_radius = 1\nplanet0.mass = 1\n",
                    "planet0: radius is too small for particle_radius");
    // 2,000 particles fit a unit ball up to particle_radius ~0.0688 (docs/notes.md).
    expect_rejected("output_every = 1\nplanet0.n = 2000\nplanet0.radius = 1\n"
                    "planet0.particle_radius = 0.07\nplanet0.mass = 2\n",
                    "planet0: too many particles for this radius and particle_radius");
    // The second planet is checked too, and named.
    expect_rejected(std::string(kMinimal) + "planet1.n = 10\nplanet1.radius = 1\n"
                                            "planet1.particle_radius = 0.1\nplanet1.mass = 0\n",
                    "planet1: mass must be > 0");
}

TEST(Config, BadRunSettingsFail) {
    const std::string m = kMinimal;
    expect_rejected(m + "softening = -0.05\n", "softening must be >= 0");
    expect_rejected(m + "contact = true\n", "contact = true needs contact_k > 0");
    expect_rejected(m + "contact = true\ncontact_k = -20\n", "contact = true needs contact_k > 0");
    expect_rejected(m + "contact_c = -0.05\n", "contact_c must be >= 0");
    expect_rejected(m + "relax_max_steps = -3\n", "relax_ke_threshold and relax_max_steps");
    expect_rejected(m + "relax_ke_threshold = -1\n", "relax_ke_threshold and relax_max_steps");
    // contact_k only matters with contact on, and 0 is fine for the others.
    Config cfg;
    bool ok = false;
    std::string err = load_text(m + "contact = false\ncontact_k = 0\ncontact_c = 0\n"
                                    "softening = 0\nrelax_max_steps = 0\n",
                                cfg, &ok);
    EXPECT_TRUE(ok) << err;
}

TEST(Config, MissingFileFails) {
    Config cfg;
    testing::internal::CaptureStderr();
    EXPECT_FALSE(load_config("no/such/file.cfg", cfg));
    std::string err = testing::internal::GetCapturedStderr();
    EXPECT_NE(err.find("no/such/file.cfg: can't open the file"), std::string::npos) << err;
    // A directory opens on Linux and reads as empty: say so, not "missing 'output_every'".
    testing::internal::CaptureStderr();
    EXPECT_FALSE(load_config(test_dir().c_str(), cfg));
    err = testing::internal::GetCapturedStderr();
    EXPECT_NE(err.find("not a regular file"), std::string::npos) << err;
}

// The real scenario files must keep loading as the format and the code change.
TEST(Config, ShippedConfigsLoad) {
    const char *files[] = {"head_on.cfg", "smoke.cfg"};
    for (const char *name : files) {
        SCOPED_TRACE(name);
        std::string path = std::string(SIM_SOURCE_DIR) + "/configs/" + name;
        Config cfg;
        testing::internal::CaptureStderr();
        bool ok = load_config(path.c_str(), cfg);
        std::string err = testing::internal::GetCapturedStderr();
        ASSERT_TRUE(ok) << err;
        EXPECT_EQ(cfg.planets.size(), 2u);
        EXPECT_TRUE(cfg.params.contact_enabled);
        EXPECT_GT(cfg.params.dt, 0.0);
        EXPECT_GT(cfg.steps, 0);
        EXPECT_GT(cfg.output_every, 1);
    }
}
