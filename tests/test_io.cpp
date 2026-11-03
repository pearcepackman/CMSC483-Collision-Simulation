#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>

#include <unistd.h>

#include "init.hpp"
#include "io.hpp"

namespace {

// A fresh, empty folder for the current test, inside GoogleTest's temp directory. The
// process id keeps two builds' test runs (e.g. dev and asan at once) apart.
std::string fresh_dir() {
    const char *name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::filesystem::path dir = std::filesystem::path(::testing::TempDir()) /
                                ("test_io_" + std::to_string(getpid())) / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir.string();
}

// Writes the bytes exactly (a std::string can hold NUL bytes, a C string can't).
void write_text(const std::string &path, const std::string &text) {
    FILE *f = std::fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr) << path;
    std::fwrite(text.data(), 1, text.size(), f);
    std::fclose(f);
}

std::string read_text(const std::string &path) {
    std::string text;
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return text;
    }
    char buf[4096];
    std::size_t got;
    while ((got = std::fread(buf, 1, sizeof buf, f)) > 0) {
        text.append(buf, got);
    }
    std::fclose(f);
    return text;
}

// Same bits, so -0.0 and 0.0 count as different.
bool same_bits(double a, double b) {
    return std::memcmp(&a, &b, sizeof a) == 0;
}

void expect_same_frame(const Particles &a, const Particles &b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_TRUE(same_bits(a.x[i], b.x[i])) << "x of particle " << i;
        EXPECT_TRUE(same_bits(a.y[i], b.y[i])) << "y of particle " << i;
        EXPECT_TRUE(same_bits(a.z[i], b.z[i])) << "z of particle " << i;
        EXPECT_EQ(a.body[i], b.body[i]) << "body of particle " << i;
    }
}

} // namespace

TEST(Io, WriteThenReadGivesSameValues) {
    const std::string dir = fresh_dir();
    Particles p;
    p.resize(6);
    const double values[] = {-0.56367743418916594,    1e-300, 1e300,
                             4.9406564584124654e-324, -0.0,   0.1};
    for (std::size_t i = 0; i < 6; ++i) {
        p.x[i] = values[i];
        p.y[i] = -values[5 - i];
        p.z[i] = values[i] * 3.0;
        p.body[i] = static_cast<int>(i % 3);
        p.vx[i] = 9.0; // not in the frame: must read back as zero
        p.mass[i] = 2.0;
    }
    ASSERT_TRUE(write_frame(p, dir.c_str(), 7));
    Particles q;
    ASSERT_TRUE(read_frame((dir + "/frame_00007.csv").c_str(), q));
    expect_same_frame(p, q);
    for (std::size_t i = 0; i < q.size(); ++i) {
        EXPECT_EQ(q.vx[i], 0.0);
        EXPECT_EQ(q.mass[i], 0.0);
        EXPECT_EQ(q.radius[i], 0.0);
    }

    // A real planet: 2,000 particles, two bodies.
    PlanetSpec s;
    s.n = 2000;
    s.radius = 1.0;
    s.particle_radius = 0.04;
    s.mass = 1.0;
    s.seed = 7;
    Particles planet;
    make_planet(planet, s);
    s.body = 1;
    s.center = Vec3{3.0, 0.0, 0.0};
    make_planet(planet, s);
    ASSERT_TRUE(write_frame(planet, dir.c_str(), 8));
    ASSERT_TRUE(read_frame((dir + "/frame_00008.csv").c_str(), q));
    expect_same_frame(planet, q);
}

TEST(Io, FileNameHasFiveDigits) {
    const std::string dir = fresh_dir();
    Particles p;
    p.resize(1);
    const int frames[] = {0, 42, 123456}; // past 99999 the number just gets wider
    const char *names[] = {"frame_00000.csv", "frame_00042.csv", "frame_123456.csv"};
    for (int k = 0; k < 3; ++k) {
        ASSERT_TRUE(write_frame(p, dir.c_str(), frames[k]));
        EXPECT_TRUE(std::filesystem::exists(dir + "/" + names[k])) << names[k];
    }
}

TEST(Io, FileStartsWithHeader) {
    const std::string dir = fresh_dir();
    Particles p;
    p.resize(2);
    p.x[1] = 0.5;
    p.body[1] = 1;
    ASSERT_TRUE(write_frame(p, dir.c_str(), 0));
    // \n line endings on every OS (no \r), so Python and diff see the same file.
    EXPECT_EQ(read_text(dir + "/frame_00000.csv"), "id,x,y,z,body\n0,0,0,0,0\n1,0.5,0,0,1\n");
}

TEST(Io, NoParticlesWritesHeaderOnly) {
    const std::string dir = fresh_dir();
    Particles p;
    ASSERT_TRUE(write_frame(p, dir.c_str(), 0));
    EXPECT_EQ(read_text(dir + "/frame_00000.csv"), "id,x,y,z,body\n");
    Particles q;
    q.resize(4);
    ASSERT_TRUE(read_frame((dir + "/frame_00000.csv").c_str(), q));
    EXPECT_EQ(q.size(), 0u);
}

// A run that blows up writes its last frame with NaN/inf; it should still be readable.
TEST(Io, NonFiniteValuesRoundTrip) {
    const std::string dir = fresh_dir();
    Particles p;
    p.resize(1);
    p.x[0] = std::numeric_limits<double>::quiet_NaN();
    p.y[0] = std::numeric_limits<double>::infinity();
    p.z[0] = -std::numeric_limits<double>::infinity();
    ASSERT_TRUE(write_frame(p, dir.c_str(), 0));
    Particles q;
    ASSERT_TRUE(read_frame((dir + "/frame_00000.csv").c_str(), q));
    ASSERT_EQ(q.size(), 1u);
    EXPECT_TRUE(std::isnan(q.x[0]));
    EXPECT_EQ(q.y[0], std::numeric_limits<double>::infinity());
    EXPECT_EQ(q.z[0], -std::numeric_limits<double>::infinity());
}

TEST(Io, WriteFailsCleanly) {
    const std::string dir = fresh_dir();
    Particles p;
    p.resize(1);
    EXPECT_FALSE(write_frame(p, (dir + "/no_such_folder").c_str(), 0));
    EXPECT_FALSE(write_frame(p, nullptr, 0));
    EXPECT_FALSE(write_frame(p, dir.c_str(), -1));
    EXPECT_TRUE(std::filesystem::is_empty(dir)); // these fail before creating any file
}

// A full disk must make write_frame return false, not leave a truncated frame that
// looks fine. Linux's /dev/full fails every write; a small frame only fails when
// fclose flushes it, a big one already fails while writing (ferror).
TEST(Io, WriteReportsDiskFull) {
    if (!std::filesystem::exists("/dev/full")) {
        GTEST_SKIP() << "no /dev/full on this OS";
    }
    const std::string dir = fresh_dir();
    std::filesystem::create_symlink("/dev/full", dir + "/frame_00000.csv");
    std::filesystem::create_symlink("/dev/full", dir + "/frame_00001.csv");
    Particles small, big;
    small.resize(3);
    big.resize(200000);
    EXPECT_FALSE(write_frame(small, dir.c_str(), 0));
    EXPECT_FALSE(write_frame(big, dir.c_str(), 1));
}

TEST(Io, ReadRejectsMalformed) {
    const std::string dir = fresh_dir();
    const std::string path = dir + "/bad.csv";
    // One 264-character row (its body field has 247 leading zeros, then extra columns).
    // Cut at the 256-byte line buffer it would look like two valid rows, so this checks
    // the reader rejects overlong lines instead of splitting them.
    const std::string long_line = "id,x,y,z,body\n0,1,2,3," + std::string(247, '0') + "1,1,2,3,0\n";
    const std::string nul = std::string(1, '\0');
    const std::string bad[] = {
        "",                                      // empty file
        "id,x,y,z,mass\n0,1,2,3,0\n",            // wrong header, valid rows
        "id,x,y,z,body\n0,1,2,3\n",              // missing column
        "id,x,y,z,body\n0,1,2,3,0,9\n",          // extra column
        "id,x,y,z,body\n0,1,abc,3,0\n",          // not a number
        "id,x,y,z,body\n0,1,2,3,0.5\n",          // body is not a whole number
        "id,x,y,z,body\n0,1,2,3,99999999999\n",  // body doesn't fit in an int
        "id,x,y,z,body\n0,1,2,3,0\n2,1,2,3,0\n", // ids skip 1
        "id,x,y,z,body\n1,1,2,3,0\n",            // ids don't start at 0
        "id,x,y,z,body\n\n0,1,2,3,0\n",          // blank line before the last row
        "id,x,y,z,body\n0,1,2,3,0\n\n\n",        // more than one blank line at the end
        long_line,                               // longer than any real row
        "id,x,y,z,body\n" + nul + "0,1,2,3,0\n", // NUL byte at the start of a line
        "id,x,y,z,body\n0,1" + nul + ",2,3,0\n", // NUL byte inside a line
    };
    for (const std::string &text : bad) {
        write_text(path, text);
        Particles p;
        p.resize(3);
        p.x[0] = 7.0;
        EXPECT_FALSE(read_frame(path.c_str(), p)) << "file:\n" << text;
        EXPECT_EQ(p.size(), 3u) << "p must be untouched on failure";
        EXPECT_EQ(p.x[0], 7.0);
    }
    Particles p;
    EXPECT_FALSE(read_frame((dir + "/missing.csv").c_str(), p));
    EXPECT_FALSE(read_frame(nullptr, p));
}

// Files a person might reasonably produce by hand or with an editor still read.
TEST(Io, ReadAcceptsEditorEndings) {
    const std::string dir = fresh_dir();
    const std::string path = dir + "/edited.csv";
    const char *good[] = {
        "id,x,y,z,body\n0,1.5,-2,3,1\n1,0,0,0,0",     // no newline after the last row
        "id,x,y,z,body\n0,1.5,-2,3,1\n1,0,0,0,0\n\n", // one blank line at the end
    };
    for (const char *text : good) {
        write_text(path, text);
        Particles p;
        ASSERT_TRUE(read_frame(path.c_str(), p)) << "file:\n" << text;
        ASSERT_EQ(p.size(), 2u);
        EXPECT_EQ(p.x[0], 1.5);
        EXPECT_EQ(p.body[0], 1);
    }
}

// Someone may open a frame in a Windows editor and save it with \r\n line endings.
TEST(Io, ReadAcceptsCrlf) {
    const std::string dir = fresh_dir();
    const std::string path = dir + "/crlf.csv";
    write_text(path, "id,x,y,z,body\r\n0,1.5,-2,3,1\r\n1,0,0,0,0\r\n");
    Particles p;
    ASSERT_TRUE(read_frame(path.c_str(), p));
    ASSERT_EQ(p.size(), 2u);
    EXPECT_EQ(p.x[0], 1.5);
    EXPECT_EQ(p.y[0], -2.0);
    EXPECT_EQ(p.z[0], 3.0);
    EXPECT_EQ(p.body[0], 1);
}
