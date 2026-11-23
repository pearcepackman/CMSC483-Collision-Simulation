#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <string>

#include <unistd.h>

#include "io.hpp"
#include "replay.hpp"

// Replay and export without graphics (#49).

namespace {

std::string fresh_dir() {
    const char *name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::path dir = std::filesystem::path(::testing::TempDir()) /
                                ("test_replay_" + std::to_string(getpid())) / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir.string();
}

void write_text(const std::string &path, const std::string &text) {
    FILE *f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr) << path;
    std::fputs(text.c_str(), f);
    std::fclose(f);
}

// Writes frames with the given numbers (one particle each, x = the number).
void write_frames(const std::string &dir, std::initializer_list<int> numbers) {
    for (int n : numbers) {
        Particles p;
        p.resize(1);
        p.x[0] = n;
        ASSERT_TRUE(write_frame(p, dir.c_str(), n));
    }
}

std::string energy(std::initializer_list<std::pair<long, double>> rows) {
    std::string text = "step,t,kinetic,potential,contact,heat,total,px,py,pz\n";
    for (const auto &r : rows) {
        char line[128];
        std::snprintf(line, sizeof line, "%ld,%.17g,0,0,0,0,0,0,0,0\n", r.first, r.second);
        text += line;
    }
    return text;
}

ReplayRun run_with_times(std::initializer_list<double> times) {
    ReplayRun r;
    for (double t : times) {
        r.times.push_back(t);
        r.paths.push_back("frame");
    }
    return r;
}

// A frame reader that counts its reads, and fails for one file name when asked.
int g_reads = 0;
std::string g_fail_suffix;
bool counting_reader(const char *path, Particles &p) {
    ++g_reads;
    const std::string s = path;
    if (!g_fail_suffix.empty() && s.size() >= g_fail_suffix.size() &&
        s.compare(s.size() - g_fail_suffix.size(), g_fail_suffix.size(), g_fail_suffix) == 0) {
        return false;
    }
    return read_frame(path, p);
}

// A run of 4 frames at t = 0, 1, 2, 3; one particle at x = 10 k moving at 10 per unit time.
ReplayRun moving_run(const std::string &dir) {
    std::string text = "step,t,kinetic,potential,contact,heat,total,px,py,pz\n";
    for (int k = 0; k < 4; ++k) {
        Particles p;
        p.resize(1);
        p.x[0] = 10.0 * k;
        p.vx[0] = 10.0;
        EXPECT_TRUE(write_frame(p, dir.c_str(), k));
        text += std::to_string(k) + "," + std::to_string(k) + ",0,0,0,0,0,0,0,0\n";
    }
    write_text(dir + "/energy.csv", text);
    ReplayRun run;
    std::string error;
    EXPECT_TRUE(replay_open(dir.c_str(), &run, &error)) << error;
    EXPECT_EQ(run.times.size(), 4u);
    return run;
}

} // namespace

TEST(Replay, OpenListsFramesInNumberOrder) {
    const std::string dir = fresh_dir();
    write_frames(dir, {100, 9, 10});
    write_text(dir + "/notes.txt", "not a frame");
    write_text(dir + "/frame_abc.csv", "not a frame either");
    ReplayRun run;
    std::string error;
    ASSERT_TRUE(replay_open(dir.c_str(), &run, &error)) << error;
    ASSERT_EQ(run.paths.size(), 3u);
    EXPECT_NE(run.paths[0].find("frame_00009.csv"), std::string::npos);
    EXPECT_NE(run.paths[1].find("frame_00010.csv"), std::string::npos);
    EXPECT_NE(run.paths[2].find("frame_00100.csv"), std::string::npos);
    EXPECT_TRUE(run.times.empty()); // no energy.csv
    EXPECT_EQ(replay_start(run), 0.0);
    EXPECT_EQ(replay_end(run), 2.0);
}

TEST(Replay, OpenReadsTimesOnlyWhenTheyMatch) {
    const std::string dir = fresh_dir();
    write_frames(dir, {0, 1, 2});
    ReplayRun run;
    std::string error;
    // The final step is always written, so the last gap can be shorter (steps = 60,
    // output_every = 40).
    write_text(dir + "/energy.csv", energy({{0, 0.0}, {40, 0.04}, {60, 0.06}}));
    ASSERT_TRUE(replay_open(dir.c_str(), &run, &error));
    ASSERT_EQ(run.times.size(), 3u);
    EXPECT_EQ(run.times[2], 0.06);
    EXPECT_EQ(replay_end(run), 0.06);
    // One row too many, or steps not increasing: not trusted.
    write_text(dir + "/energy.csv", energy({{0, 0.0}, {40, 0.04}, {80, 0.08}, {120, 0.12}}));
    ASSERT_TRUE(replay_open(dir.c_str(), &run, &error));
    EXPECT_TRUE(run.times.empty());
    write_text(dir + "/energy.csv", energy({{0, 0.0}, {40, 0.04}, {40, 0.04}}));
    ASSERT_TRUE(replay_open(dir.c_str(), &run, &error));
    EXPECT_TRUE(run.times.empty());
}

TEST(Replay, OpenRejectsEmptyAndOldFrames) {
    const std::string dir = fresh_dir();
    ReplayRun run;
    std::string error;
    EXPECT_FALSE(replay_open(dir.c_str(), &run, &error));
    EXPECT_NE(error.find("no frame_*.csv"), std::string::npos) << error;
    EXPECT_FALSE(replay_open((dir + "/missing").c_str(), &run, &error));
    EXPECT_NE(error.find("not a folder"), std::string::npos) << error;
    write_text(dir + "/frame_00000.csv",
               "id,x,y,z,vx,vy,vz,mass,body,material,heat\n0,1,2,3,0,0,0,1,0,0,0\n");
    EXPECT_FALSE(replay_open(dir.c_str(), &run, &error));
    EXPECT_NE(error.find("rerun planet_sim"), std::string::npos) << error;
}

TEST(Replay, LocateFindsFrameAndFraction) {
    const ReplayRun run = run_with_times({0.0, 0.04, 0.08, 0.10});
    int k = -1;
    double s = -1.0;
    replay_locate(run, 0.0, &k, &s);
    EXPECT_EQ(k, 0);
    EXPECT_EQ(s, 0.0);
    replay_locate(run, 0.05, &k, &s);
    EXPECT_EQ(k, 1);
    EXPECT_NEAR(s, 0.25, 1e-12);
    replay_locate(run, 0.08, &k, &s); // exactly on a frame
    EXPECT_EQ(k, 2);
    EXPECT_NEAR(s, 0.0, 1e-12);
    replay_locate(run, 0.09, &k, &s); // the shorter last gap
    EXPECT_EQ(k, 2);
    EXPECT_NEAR(s, 0.5, 1e-12);
    replay_locate(run, 0.10, &k, &s); // the end: last frame, s = 0
    EXPECT_EQ(k, 3);
    EXPECT_EQ(s, 0.0);
    replay_locate(run, 5.0, &k, &s);
    EXPECT_EQ(k, 3);
    replay_locate(run, -1.0, &k, &s);
    EXPECT_EQ(k, 0);
    EXPECT_EQ(s, 0.0);
    replay_locate(run, std::nan(""), &k, &s);
    EXPECT_EQ(k, 3);
    EXPECT_EQ(s, 0.0);
}

TEST(Replay, LocateWithoutTimesCountsFrames) {
    ReplayRun run = run_with_times({0, 0, 0});
    run.times.clear();
    int k = -1;
    double s = -1.0;
    replay_locate(run, 1.25, &k, &s);
    EXPECT_EQ(k, 1);
    EXPECT_NEAR(s, 0.25, 1e-12);
}

TEST(Replay, LocateOneFrameRunIsStatic) {
    ReplayRun run;
    run.paths.push_back("frame");
    for (double t : {-1.0, 0.0, 0.5, 100.0}) {
        int k = -1;
        double s = -1.0;
        replay_locate(run, t, &k, &s);
        EXPECT_EQ(k, 0);
        EXPECT_EQ(s, 0.0);
    }
    EXPECT_EQ(replay_start(run), 0.0);
    EXPECT_EQ(replay_end(run), 0.0);
}

TEST(Replay, HermiteIsExactForConstantAcceleration) {
    // x(t) = x0 + v0 t + a t^2 / 2 on each axis, frames at t = 1 and t = 1.5.
    const double x0[3] = {0.5, -2.0, 3.0}, v0[3] = {1.0, 0.25, -4.0}, acc[3] = {-9.8, 2.0, 0.5};
    auto at = [&](double t, Particles *p) {
        p->resize(1);
        double *pos[3] = {&p->x[0], &p->y[0], &p->z[0]};
        double *vel[3] = {&p->vx[0], &p->vy[0], &p->vz[0]};
        for (int d = 0; d < 3; ++d) {
            *pos[d] = x0[d] + v0[d] * t + 0.5 * acc[d] * t * t;
            *vel[d] = v0[d] + acc[d] * t;
        }
    };
    Particles a, b, want, got;
    at(1.0, &a);
    at(1.5, &b);
    a.heat[0] = 2.0;
    b.heat[0] = 4.0;
    a.mass[0] = b.mass[0] = 0.5;
    a.radius[0] = 0.04;
    for (double s : {0.0, 0.1, 0.5, 0.9, 1.0}) {
        ASSERT_TRUE(interpolate_frames(a, b, 1.0, 1.5, s, &got));
        at(1.0 + 0.5 * s, &want);
        EXPECT_NEAR(got.x[0], want.x[0], 1e-12) << s;
        EXPECT_NEAR(got.y[0], want.y[0], 1e-12) << s;
        EXPECT_NEAR(got.z[0], want.z[0], 1e-12) << s;
        EXPECT_NEAR(got.vx[0], want.vx[0], 1e-12) << s; // velocity is the derivative
        EXPECT_NEAR(got.vy[0], want.vy[0], 1e-12) << s;
        EXPECT_NEAR(got.vz[0], want.vz[0], 1e-12) << s;
        EXPECT_NEAR(got.heat[0], 2.0 + 2.0 * s, 1e-12) << s; // heat: linear
        EXPECT_EQ(got.mass[0], 0.5);
        EXPECT_EQ(got.radius[0], 0.04);
    }
}

TEST(Replay, WithoutTimesInterpolationIsLinear) {
    Particles a, b, got;
    a.resize(1);
    b.resize(1);
    b.x[0] = 2.0;
    a.vx[0] = 100.0; // ignored for positions: no time gap to scale it by
    ASSERT_TRUE(interpolate_frames(a, b, 0.0, 0.0, 0.25, &got));
    EXPECT_NEAR(got.x[0], 0.5, 1e-12);
}

TEST(Replay, InterpolationRejectsFramesOfDifferentRuns) {
    Particles a, b, out;
    a.resize(2);
    b.resize(3);
    out.resize(1);
    out.x[0] = 7.0;
    EXPECT_FALSE(interpolate_frames(a, b, 0.0, 1.0, 0.5, &out));
    b.resize(2);
    b.body[1] = 1;
    EXPECT_FALSE(interpolate_frames(a, b, 0.0, 1.0, 0.5, &out));
    EXPECT_EQ(out.size(), 1u); // untouched
    EXPECT_EQ(out.x[0], 7.0);
}

TEST(Replay, ExportFrameCount) {
    // A ratio that isn't whole: 4 / 0.7 * 60 = 342.86 intervals, rounded up to 343, and the
    // last of the 344 images is clamped to the end of the run.
    EXPECT_EQ(export_frame_count(0.0, 4.0, 0.7, 60.0), 344);
    EXPECT_EQ(export_time(343, 0.0, 4.0, 0.7, 60.0), 4.0);
    EXPECT_LT(export_time(342, 0.0, 4.0, 0.7, 60.0), 4.0);
    // 4 time units at half speed and 60 fps: 8 s of video, 480 intervals, 481 images.
    EXPECT_EQ(export_frame_count(0.0, 4.0, 0.5, 60.0), 481);
    EXPECT_EQ(export_time(480, 0.0, 4.0, 0.5, 60.0), 4.0); // the last image is the end
    EXPECT_EQ(export_time(10000, 0.0, 4.0, 0.5, 60.0), 4.0);
    // Ratios that aren't exact in binary: 1.2 / 0.6 * 10 = 20 intervals.
    EXPECT_EQ(export_frame_count(0.0, 1.2, 0.6, 10.0), 21);
    EXPECT_EQ(export_frame_count(0.0, 0.3, 0.1, 1.0), 4);   // 0.3 / 0.1 = 2.9999999999999996
    EXPECT_EQ(export_frame_count(0.0, 0.7, 0.1, 1.0), 8);   // 0.7 / 0.1 = 6.999999999999999
    EXPECT_EQ(export_frame_count(0.0, 0.1, 0.3, 30.0), 11); // 0.1 / 0.3 * 30 = 10.000000000000002
    EXPECT_EQ(export_frame_count(2.0, 2.0, 1.0, 60.0), 1);  // one-frame or zero-length run
    EXPECT_EQ(export_frame_count(0.0, 1.0, 0.0, 60.0), -1);
    EXPECT_EQ(export_frame_count(0.0, 1.0, 1.0, -5.0), -1);
    EXPECT_EQ(export_frame_count(1.0, 0.0, 1.0, 60.0), -1);
    EXPECT_EQ(export_frame_count(0.0, std::nan(""), 1.0, 60.0), -1);
    EXPECT_EQ(export_frame_count(0.0, 1e9, 1.0, 60.0), -1); // over 1,000,000 images
    EXPECT_EQ(export_frame_count(0.0, 1.0, 1e-300, 60.0), -1);
}

TEST(Replay, FiniteCheck) {
    Particles p;
    p.resize(3);
    EXPECT_TRUE(frame_is_finite(p));
    p.vx[1] = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(frame_is_finite(p));
    p.vx[1] = 0.0;
    p.heat[2] = std::nan("");
    EXPECT_FALSE(frame_is_finite(p));
    p.heat[2] = 0.0;
    p.z[0] = std::nan("");
    EXPECT_FALSE(frame_is_finite(p));
}

TEST(Replay, CacheReadsEachFrameOnceGoingForward) {
    const ReplayRun run = moving_run(fresh_dir());
    ReplayCache cache;
    Particles out;
    int k = -1;
    std::string error;
    g_reads = 0;
    g_fail_suffix.clear();
    // Many moments from t = 0 to 3: each of the 4 frames is read exactly once.
    for (double t = 0.0; t <= 3.0; t += 0.125) {
        ASSERT_EQ(replay_cache_show(run, t, &cache, counting_reader, &out, &k, &error), REPLAY_OK)
            << error;
        EXPECT_NEAR(out.x[0], 10.0 * t, 1e-9) << t; // the right frames, in the right order
        EXPECT_EQ(k, std::min(static_cast<int>(t), 3)) << t;
    }
    EXPECT_EQ(g_reads, 4);
    // Back to the start: frames 0 and 1 are read again (only two are kept).
    ASSERT_EQ(replay_cache_show(run, 0.5, &cache, counting_reader, &out, &k, &error), REPLAY_OK);
    EXPECT_NEAR(out.x[0], 5.0, 1e-9);
    EXPECT_EQ(g_reads, 6);
    // Same moment again: nothing read.
    ASSERT_EQ(replay_cache_show(run, 0.5, &cache, counting_reader, &out, &k, &error), REPLAY_OK);
    EXPECT_EQ(g_reads, 6);
}

TEST(Replay, CacheRecoversFromAFailedRead) {
    const ReplayRun run = moving_run(fresh_dir());
    ReplayCache cache;
    Particles out;
    int k = -1;
    std::string error;
    g_fail_suffix.clear();
    ASSERT_EQ(replay_cache_show(run, 0.5, &cache, counting_reader, &out, &k, &error), REPLAY_OK);
    const double before = out.x[0];
    g_fail_suffix = "frame_00002.csv";
    EXPECT_EQ(replay_cache_show(run, 1.5, &cache, counting_reader, &out, &k, &error),
              REPLAY_READ_FAILED);
    EXPECT_NE(error.find("frame_00002.csv"), std::string::npos) << error;
    EXPECT_EQ(out.x[0], before); // the last good picture stays
    g_fail_suffix.clear();       // the file is fine again: the next call reads it
    ASSERT_EQ(replay_cache_show(run, 1.5, &cache, counting_reader, &out, &k, &error), REPLAY_OK);
    EXPECT_NEAR(out.x[0], 15.0, 1e-9);
}

TEST(Replay, CacheReportsBrokenFrames) {
    const std::string dir = fresh_dir();
    const ReplayRun run = moving_run(dir);
    ReplayCache cache;
    Particles out;
    int k = -1;
    std::string error;
    g_fail_suffix.clear();
    // Frame 2 with NaN: not finite.
    Particles bad;
    bad.resize(1);
    bad.x[0] = std::nan("");
    ASSERT_TRUE(write_frame(bad, dir.c_str(), 2));
    EXPECT_EQ(replay_cache_show(run, 2.0, &cache, counting_reader, &out, &k, &error),
              REPLAY_NOT_FINITE);
    EXPECT_NE(error.find("NaN/Inf"), std::string::npos) << error;
    // Frame 2 with two particles next to frame 1's one: they don't match, so frame 1 alone.
    bad.resize(2);
    bad.x[0] = bad.x[1] = 20.0;
    ASSERT_TRUE(write_frame(bad, dir.c_str(), 2));
    cache = ReplayCache{};
    EXPECT_EQ(replay_cache_show(run, 1.5, &cache, counting_reader, &out, &k, &error),
              REPLAY_MISMATCH);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out.x[0], 10.0);
    EXPECT_NE(error.find("don't match"), std::string::npos) << error;
}
