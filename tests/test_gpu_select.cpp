#include <gtest/gtest.h>

#include "gravity.hpp"

// GPU choice (#24 review). Pure logic, so it runs in every build without a GPU. The dev
// PC lists two devices: the RX 9060 XT (gfx1200, many compute units) and the Ryzen's
// integrated GPU (gfx1036, 1 unit). Choosing a GPU the kernel wasn't compiled for makes
// the HIP runtime abort on the first launch, so it must never be chosen.

TEST(PickGpu, MostComputeUnitsWinsWhateverTheOrder) {
    const int units[] = {1, 16};
    const char *const archs[] = {"gfx1036", "gfx1200"}; // integrated GPU listed first
    EXPECT_EQ(pick_gpu(2, units, archs, "gfx1200,gfx1036"), 1);

    const int units2[] = {16, 1};
    const char *const archs2[] = {"gfx1200", "gfx1036"};
    EXPECT_EQ(pick_gpu(2, units2, archs2, "gfx1200,gfx1036"), 0);
}

TEST(PickGpu, SkipsGpusTheKernelWasNotBuiltFor) {
    const int units[] = {64, 16};
    const char *const archs[] = {"gfx942", "gfx1200"}; // bigger, but no code for it
    EXPECT_EQ(pick_gpu(2, units, archs, "gfx1200,gfx1036"), 1);

    const int units2[] = {1};
    const char *const archs2[] = {"gfx1036"};
    EXPECT_EQ(pick_gpu(1, units2, archs2, "gfx1200"), -1); // only an unsupported GPU
}

TEST(PickGpu, NoDevicesMeansNone) {
    EXPECT_EQ(pick_gpu(0, nullptr, nullptr, "gfx1200"), -1);
}

TEST(PickGpu, ArchNamesMatchExactlyIgnoringFeatureFlags) {
    const int units[] = {16};
    const char *const flagged[] = {"gfx1200:xnack-"};
    EXPECT_EQ(pick_gpu(1, units, flagged, "gfx1200"), 0);

    // A name that is only a prefix (or extension) of a built one doesn't count.
    const char *const shorter[] = {"gfx120"};
    EXPECT_EQ(pick_gpu(1, units, shorter, "gfx1200"), -1);
    const char *const longer[] = {"gfx12001"};
    EXPECT_EQ(pick_gpu(1, units, longer, "gfx1200"), -1);
    EXPECT_EQ(pick_gpu(1, units, flagged, "gfx1036,gfx1200"), 0); // last in the list
    EXPECT_EQ(pick_gpu(1, units, flagged, ""), -1);
}

TEST(PickGpu, TiesGoToTheFirstListed) {
    const int units[] = {16, 16};
    const char *const archs[] = {"gfx1200", "gfx1200"};
    EXPECT_EQ(pick_gpu(2, units, archs, "gfx1200"), 0);
}
