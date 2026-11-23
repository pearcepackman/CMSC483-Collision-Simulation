"""Tests for analyze_impact.py. Run with: python -m unittest discover scripts"""

import contextlib
import io
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_impact as ai  # noqa: E402
from render import read_frame  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = "id,x,y,z,vx,vy,vz,mass,body,material,heat\n"


def ball(n, radius, seed, center=(0, 0, 0)):
    """n random points spread uniformly through a ball."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    r = radius * rng.random(n) ** (1.0 / 3.0)
    return d * r[:, None] + np.asarray(center, float)


def planet(n=500, radius=1.0, mass=1.0, seed=1, core_ratio=1.0, core_fraction=0.55):
    """A frame dict for a planet at rest: uniform positions, and, with core_ratio > 1,
    particles inside core_fraction * radius are iron and core_ratio x heavier (so the
    core is that much denser, as in planet_sim's layered planets)."""
    x = ball(n, radius, seed)
    inner = np.linalg.norm(x, axis=1) < core_fraction * radius
    m = np.where(inner, core_ratio, 1.0)
    m *= mass / m.sum()
    material = np.where(inner & (core_ratio > 1.0), ai.MATERIAL_IRON, ai.MATERIAL_ROCK)
    return {"x": x, "v": np.zeros((n, 3)), "mass": m, "body": np.zeros(n, int),
            "material": material.astype(int), "heat": np.zeros(n)}


def add(frame, x, v, mass, body=1, material=ai.MATERIAL_ROCK):
    """frame plus one more particle."""
    return {"x": np.vstack([frame["x"], x]), "v": np.vstack([frame["v"], v]),
            "mass": np.append(frame["mass"], mass), "body": np.append(frame["body"], body),
            "material": np.append(frame["material"], material),
            "heat": np.append(frame["heat"], 0.0)}


def write_frame(folder, number, frame):
    path = os.path.join(folder, "frame_%05d.csv" % number)
    with open(path, "w", newline="\n") as f:
        f.write(HEADER)
        for i in range(len(frame["mass"])):
            x, v = frame["x"][i], frame["v"][i]
            f.write("%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,0\n" % (
                i, x[0], x[1], x[2], v[0], v[1], v[2], frame["mass"][i], frame["body"][i],
                frame["material"][i]))
    return path


def run_main(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = ai.main(args)
    return code, out.getvalue(), err.getvalue()


class AnalyzeImpactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_planet_is_remnant_and_circular_orbit_is_disk(self):
        p = planet()
        v_circ = np.sqrt(1.0 / 3.0)
        f = add(p, [3.0, 0, 0], [0, v_circ, 0], 1e-4)
        result = ai.analyze(f)
        rem = result["remnant"]
        self.assertAlmostEqual(rem["M"], 1.0, delta=0.05)
        self.assertAlmostEqual(rem["R"], 1.0, delta=0.1)
        self.assertTrue(np.all(result["classes"][:500] == ai.REMNANT))
        self.assertEqual(result["classes"][500], ai.DISK)

    def test_escaping_and_reimpacting(self):
        p = planet()
        v_esc = np.sqrt(2.0 / 3.0)
        f = add(p, [3.0, 0, 0], [0, 1.5 * v_esc, 0], 1e-4)
        f = add(f, [0, 3.0, 0], [0.1 * np.sqrt(1.0 / 3.0), 0, 0], 1e-4)  # falls nearly straight in
        f = add(f, [0, 0, 4.0], [0, 0, -3.0], 1e-4)  # unbound, coming straight in: will hit
        f = add(f, [0, 0, -4.0], [0, 0, -3.0], 1e-4)  # unbound, straight out: escaping
        f = add(f, [4.0, 4.0, 0], [-3.0, 0, 0], 1e-4)  # unbound, approaching, but misses
        cls = ai.analyze(f)["classes"]
        self.assertEqual(cls[500], ai.ESCAPING)
        self.assertEqual(cls[501], ai.REIMPACTING)
        self.assertEqual(cls[502], ai.REIMPACTING)
        self.assertEqual(cls[503], ai.ESCAPING)
        self.assertEqual(cls[504], ai.ESCAPING)

    def test_moving_remnant_gives_same_answer(self):
        p = planet()
        f = add(p, [3.0, 0, 0], [0, np.sqrt(1.0 / 3.0), 0], 1e-4)
        f = add(f, [-3.0, 0, 0], [0, 1.2, 0], 1e-4)
        moved = dict(f)
        moved["x"] = f["x"] + [10.0, -4.0, 2.0]
        moved["v"] = f["v"] + [5.0, 0.0, -1.0]
        a, b = ai.analyze(f), ai.analyze(moved)
        np.testing.assert_array_equal(a["classes"], b["classes"])
        self.assertAlmostEqual(a["remnant"]["M"], b["remnant"]["M"], places=12)

    def test_split_table_cells_and_sums(self):
        cls = np.array([ai.REMNANT, ai.REMNANT, ai.DISK, ai.ESCAPING, ai.ESCAPING])
        mass = np.array([1.0, 2.0, 0.5, 0.25, 0.125])
        material = np.array([1, 0, 1, 0, 1])
        body = np.array([0, 1, 1, 0, 3])
        t = ai.split_table(cls, mass, material, body)
        self.assertEqual(t["columns"], ["total", "iron", "rock", "target (body 0)",
                                        "impactor (body 1)", "other bodies"])
        self.assertEqual(t["rows"]["remnant"], [3.0, 1.0, 2.0, 1.0, 2.0, 0.0])
        self.assertEqual(t["rows"]["disk"], [0.5, 0.5, 0.0, 0.0, 0.5, 0.0])
        self.assertEqual(t["rows"]["reimpacting"], [0.0] * 6)
        self.assertEqual(t["rows"]["escaping"], [0.375, 0.125, 0.25, 0.25, 0.0, 0.125])
        for row in t["rows"].values():
            self.assertAlmostEqual(row[1] + row[2], row[0])  # iron + rock
            self.assertAlmostEqual(sum(row[3:]), row[0])  # every body
        # A single-planet run has no impactor column.
        one = ai.split_table(cls, mass, material, np.zeros(5, int))
        self.assertEqual(one["columns"], ["total", "iron", "rock", "target (body 0)"])

    def test_time_series_uses_energy_times_or_frame_numbers(self):
        p = planet(n=200)
        for k in range(3):
            write_frame(self.dir, k, p)
        prefix = os.path.join(self.dir, "out")
        code, out, _ = run_main([self.dir, "--every", "2", "-o", prefix])
        self.assertEqual(code, 0, out)
        with open(prefix + ".csv") as f:
            lines = f.read().splitlines()
        self.assertEqual(lines[0], "frame,t,remnant,disk,reimpacting,escaping,R,moon_estimate")
        self.assertEqual([line.split(",")[:2] for line in lines[1:]],
                         [["0", "0"], ["2", "2"]])  # frame numbers (no energy.csv), last kept
        self.assertTrue(os.path.getsize(prefix + ".png") > 0)
        with open(os.path.join(self.dir, "energy.csv"), "w") as f:
            f.write("step,t,kinetic,potential,contact,heat,total,px,py,pz\n")
            for k in range(3):
                f.write(f"{40 * k},{0.04 * k},0,0,0,0,0,0,0,0\n")
        run_main([self.dir, "--every", "1", "-o", prefix])
        with open(prefix + ".csv") as f:
            times = [line.split(",")[1] for line in f.read().splitlines()[1:]]
        self.assertEqual(times, ["0", "0.04", "0.08"])
        # planet_sim always writes the final step, so the last gap can be shorter
        # (steps = 60 with output_every = 40): still one row per frame, still used.
        with open(os.path.join(self.dir, "energy.csv"), "w") as f:
            f.write("step,t,kinetic,potential,contact,heat,total,px,py,pz\n")
            for step in (0, 40, 60):
                f.write(f"{step},{0.001 * step},0,0,0,0,0,0,0,0\n")
        self.assertEqual(ai.frame_times(self.dir, [0, 1, 2], 3), [0.0, 0.04, 0.06])
        with open(os.path.join(self.dir, "energy.csv"), "a") as f:
            f.write("120,0.12,0,0,0,0,0,0,0,0\n")  # 4 rows for 3 frames: don't trust it
        self.assertIsNone(ai.frame_times(self.dir, [0, 1, 2], 3))

    def test_time_series_matches_single_frames(self):
        # Each frame is analyzed on its own: a planet that moves between frames, with a
        # small clump left where it was, gives the same answer through --every as alone.
        p = planet(n=400)
        frames = []
        for k in range(3):
            f = {key: val.copy() for key, val in p.items()}
            f["x"] = f["x"] + [3.0 * k, 0, 0]
            f["v"][:] = [30.0, 0, 0]
            f = add(f, [0.0, 0, 0], [0, 0, 0], 1e-4)  # debris at the old center
            frames.append(f)
            write_frame(self.dir, k, f)
        prefix = os.path.join(self.dir, "out")
        code, out, err = run_main([self.dir, "--every", "1", "-o", prefix])
        self.assertEqual(code, 0, err)
        with open(prefix + ".csv") as f:
            last = f.read().splitlines()[-1].split(",")
        alone = ai.analyze(read_frame(os.path.join(self.dir, "frame_00002.csv")))
        self.assertAlmostEqual(float(last[2]), alone["remnant"]["M"] / alone["total_mass"],
                               places=5)
        self.assertAlmostEqual(alone["remnant"]["M"], 1.0, places=12)  # debris not forced in

    def test_far_debris_doesnt_change_the_remnant(self):
        # One particle flung far away must not change the remnant (review: a softening
        # set from the frame's width let a particle at x = 2000 double R).
        f = add(planet(), [3.0, 0, 0], [0, np.sqrt(1.0 / 3.0), 0], 1e-4)
        near = ai.analyze(f)["remnant"]
        far = ai.analyze(add(f, [1e4, 0, 0], [1.0, 0, 0], 1e-4))["remnant"]
        self.assertEqual(near["M"], far["M"])
        self.assertAlmostEqual(near["R"], far["R"], places=12)

    def test_heavy_lone_particle_is_not_the_seed(self):
        # A particle 25x heavier than the planet's, alone at x = 5. Counting its own
        # term m / eps in its potential would make it the deepest point, and the
        # "remnant" would be that one particle.
        # (The iteration would recover from that seed here; the seed itself is checked.)
        f = add(planet(), [5.0, 0, 0], [0, 0, 0], 0.05)
        eps = 0.01 * ai.length_scale(f["x"])
        self.assertLess(ai.potential_min_index(f["x"], f["mass"], eps), 500)
        self.assertAlmostEqual(ai.analyze(f)["remnant"]["M"], 1.0, delta=0.05)

    def test_unbound_paths_bent_by_gravity(self):
        # Coming in from x = 10 at speed 2 (unbound), with impact parameter b. Gravity
        # bends the path: q = h^2 / (M (1 + e)). b = 0.9 gives q = 0.70 and b = 1.1,
        # which a straight line would miss, gives q = 0.90: both hit (R is about 1).
        # b = 1.6 gives q = 1.40: misses. Leaving e out (q = h^2 / M) would call
        # b = 0.9 a miss.
        f = planet()
        for b in (0.9, 1.1, 1.6):
            f = add(f, [10.0, b, 0], [-2.0, 0, 0], 1e-6)
        result = ai.analyze(f)
        self.assertLess(result["remnant"]["R"], 1.05)
        self.assertEqual(result["classes"][500:].tolist(),
                         [ai.REIMPACTING, ai.REIMPACTING, ai.ESCAPING])

    def test_moon_estimate(self):
        # 1.9 * 1 / sqrt(1 * 2.9) - 1.1 * 0.1 - 1.9 * 0.05 * 0.1
        self.assertAlmostEqual(ai.moon_mass_estimate(1.0, 0.1, 1.0, 1.0),
                               1.9 / np.sqrt(2.9) - 0.11 - 0.0095, places=12)
        self.assertEqual(ai.moon_mass_estimate(0.0, 0.1, 1.0, 1.0), 0.0)  # clamps
        self.assertEqual(ai.moon_mass_estimate(1.0, 0.0, 1.0, 1.0), 0.0)  # no disk

    def test_bad_input_exits_1(self):
        code, _, err = run_main([self.dir])
        self.assertEqual(code, 1)
        self.assertIn("no frame_*.csv", err)
        p = planet(n=50)
        p["x"][3, 1] = np.nan
        write_frame(self.dir, 0, p)
        code, _, err = run_main([self.dir])
        self.assertEqual(code, 1)
        self.assertIn("non-finite x", err)
        code, _, err = run_main([os.path.join(self.dir, "missing")])
        self.assertEqual(code, 1)
        code, _, err = run_main([self.dir, "--frame", "7"])
        self.assertEqual(code, 1)
        self.assertIn("no frame 7", err)

    def test_flip_between_two_sets_settles(self):
        # Frame 26 of the smash run flipped one particle in and out of the remnant every
        # round, forever. Forcing the same flip: the surface radius alternates between
        # two values, so the member set alternates between two sets. It must settle on
        # the larger one instead of running out of rounds.
        f = planet(n=300)
        r = np.linalg.norm(f["x"], axis=1)
        cut = np.sort(r)[-2]  # alternate between all 300 and all but the outermost
        real, calls = ai.surface_radius, [0]

        def flipping(rr, mass, eps):
            calls[0] += 1
            return 2.0 if calls[0] % 2 else cut * 0.999999
        try:
            ai.surface_radius = flipping
            result = ai.find_remnant(f["x"], f["v"], f["mass"])
        finally:
            ai.surface_radius = real
        self.assertTrue(result["converged"])
        self.assertLess(result["rounds"], 10)
        self.assertEqual(int(result["members"].sum()), 300)  # the larger set

    def test_frame_choice_and_usage_errors(self):
        write_frame(self.dir, 0, planet(n=50))
        write_frame(self.dir, 1, add(planet(n=50), [3.0, 0, 0], [0, 0, 0], 1.0))
        code, out, _ = run_main([self.dir, "--frame", "0"])
        self.assertEqual(code, 0)
        self.assertIn("frame_00000.csv: 50 particles", out)
        code, out, _ = run_main([self.dir])
        self.assertIn("frame_00001.csv: 51 particles", out)  # the last by default
        for bad in (["--every", "0"], ["--every", "1", "--frame", "0"], ["-o", "x"]):
            with self.assertRaises(SystemExit) as e:
                run_main([self.dir] + bad)
            self.assertEqual(e.exception.code, 2, bad)

    def test_warns_when_remnant_does_not_settle(self):
        write_frame(self.dir, 0, add(planet(n=200), [3.0, 0, 0], [0, 0.3, 0], 0.5))
        real = ai.find_remnant
        try:
            ai.find_remnant = lambda x, v, m: real(x, v, m, max_rounds=0)
            code, _, err = run_main([self.dir, "--every", "1", "-o",
                                     os.path.join(self.dir, "o")])
        finally:
            ai.find_remnant = real
        self.assertEqual(code, 0)
        self.assertIn("did not settle", err)

    def test_min_remnant(self):
        p = planet(n=200)
        f = add(p, [5.0, 0, 0], [0, 10.0, 0], 1.0)  # half the mass escapes
        write_frame(self.dir, 0, f)
        self.assertEqual(run_main([self.dir, "--min-remnant", "0.4"])[0], 0)
        code, _, err = run_main([self.dir, "--min-remnant", "0.6"])
        self.assertEqual(code, 1)
        self.assertIn("under --min-remnant", err)

    def test_tiny_frames(self):
        one = planet(n=1)
        result = ai.analyze(one)
        self.assertEqual(result["classes"].tolist(), [ai.REMNANT])
        self.assertAlmostEqual(result["remnant"]["M"], 1.0)
        self.assertTrue(np.isfinite(result["remnant"]["R"]))
        small = planet(n=20, radius=0.1)
        self.assertTrue(np.all(ai.analyze(small)["classes"] == ai.REMNANT))

    def test_two_clumps_picks_one(self):
        a = planet(n=300, seed=1)
        b = planet(n=300, seed=2)
        b["x"] = b["x"] + [50.0, 0, 0]
        b["body"][:] = 1
        both = {k: np.concatenate([a[k], b[k]]) for k in a}
        rem = ai.analyze(both)["remnant"]
        self.assertAlmostEqual(rem["M"], 1.0, delta=0.05)  # one clump, not both
        dist = min(np.linalg.norm(rem["center"]), np.linalg.norm(rem["center"] - [50, 0, 0]))
        self.assertLess(dist, 0.1)

    def test_layered_planet_keeps_its_mantle(self):
        # The review's failure case: a uniform-sphere radius rule shrank a cored planet to
        # its iron core. A 2.5x denser core inside 0.55 R, as in configs/smash.cfg.
        p = planet(n=2000, radius=1.0, mass=2.0, core_ratio=2.5)
        rem = ai.analyze(p)["remnant"]
        iron_true = p["mass"][p["material"] == ai.MATERIAL_IRON].sum() / 2.0
        iron = p["mass"][rem["members"] & (p["material"] == ai.MATERIAL_IRON)].sum() / rem["M"]
        self.assertAlmostEqual(rem["M"], 2.0, delta=0.1)
        self.assertAlmostEqual(rem["R"], 1.0, delta=0.1)
        self.assertAlmostEqual(iron, iron_true, delta=0.02)

    def test_real_relaxed_planet_is_all_remnant(self):
        # One layered planet relaxed by planet_sim (testdata/.../planet.cfg, 1 step).
        frame = read_frame(os.path.join(HERE, "testdata", "relaxed_layered_500",
                                        "frame_00000.csv"))
        result = ai.analyze(frame)
        self.assertAlmostEqual(result["remnant"]["M"] / result["total_mass"], 1.0, delta=0.01)
        iron = frame["mass"][frame["material"] == ai.MATERIAL_IRON].sum() / frame["mass"].sum()
        self.assertGreater(iron, 0.25)  # it really is layered

    def test_thin_disk_stays_disk(self):
        p = planet(n=1000)
        rng = np.random.default_rng(7)
        f = p
        for _ in range(50):
            r = rng.uniform(2.0, 4.0)
            a = rng.uniform(0, 2 * np.pi)
            v = np.sqrt(1.0 / r)
            f = add(f, [r * np.cos(a), r * np.sin(a), 0], [-v * np.sin(a), v * np.cos(a), 0],
                    1e-5)
        result = ai.analyze(f)
        self.assertTrue(np.all(result["classes"][1000:] == ai.DISK))
        self.assertLess(result["remnant"]["R"], 1.1)
        self.assertAlmostEqual(result["M_disk"], 50 * 1e-5, places=12)
        self.assertGreater(np.linalg.norm(result["L_disk"]), 0.0)
        # In the disk's plane: L points along z (not exactly, as the random planet's
        # center of mass is a little off the origin).
        self.assertGreater(result["L_disk"][2] / np.linalg.norm(result["L_disk"]), 0.999)

    def test_report_prints(self):
        f = add(planet(), [3.0, 0, 0], [0, np.sqrt(1.0 / 3.0), 0], 1e-4)
        out = io.StringIO()
        ai.print_report(ai.analyze(f), "x", out=out)
        text = out.getvalue()
        for word in ("remnant", "disk", "reimpacting", "escaping", "predicted moon mass",
                     "not a measurement"):
            self.assertIn(word, text)


if __name__ == "__main__":
    unittest.main()
