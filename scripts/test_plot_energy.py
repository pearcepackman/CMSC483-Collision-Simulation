"""Tests for plot_energy.py. Run with: python -m unittest discover scripts"""

import contextlib
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_energy  # noqa: E402

HEADER = "step,t,kinetic,potential,heat,total,px,py,pz\n"


def row(step, t, ke, pe, px=0.0, heat=0.0):
    """One line as planet_sim's write_energy_row() writes it (total = ke + pe + heat)."""
    return "%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,0,0\n" % (step, t, ke, pe, heat,
                                                             ke + pe + heat, px)


class PlotEnergyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, text):
        path = os.path.join(self.dir, name)
        with open(path, "w", newline="\n") as f:
            f.write(text)
        return path

    def run_main(self, argv):
        """main() with stdout and stderr captured. Returns (exit code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = plot_energy.main(argv)
            except SystemExit as e:  # argparse errors
                code = e.code
        return code, out.getvalue(), err.getvalue()

    # A run whose total energy only falls: -1.0, -1.1, -1.25.
    def falling(self):
        return self.write("falling.csv", HEADER + row(0, 0.0, 0.5, -1.5) + row(10, 0.02, 0.6, -1.7)
                          + row(20, 0.04, 0.25, -1.5))

    def test_loads_log(self):
        e = plot_energy.load_energy(self.falling())
        self.assertEqual(e["step"].tolist(), [0, 10, 20])
        self.assertEqual(e["t"].tolist(), [0.0, 0.02, 0.04])
        self.assertEqual(e["kinetic"].tolist(), [0.5, 0.6, 0.25])
        self.assertEqual(e["total"].tolist(), [-1.0, -1.1, -1.25])
        self.assertEqual(set(e), set(plot_energy.COLUMNS))

    def test_total_includes_heat(self):
        # Kinetic energy turns into heat: -1.0, then -1.0 again with 0.25 as heat (exact in binary).
        path = self.write("heat.csv", HEADER + row(0, 0, 0.5, -1.5) + row(1, 1, 0.25, -1.5, heat=0.25))
        e = plot_energy.load_energy(path)
        self.assertEqual(e["heat"].tolist(), [0.0, 0.25])
        s = plot_energy.summarize(e)
        self.assertAlmostEqual(s["change"], 0.0, places=15)
        self.assertIsNone(s["rise_step"])

    def test_loads_crlf_and_endings(self):
        body = row(0, 0.0, 0.5, -1.5) + row(5, 0.01, 0.4, -1.5)
        good = {
            "crlf.csv": (HEADER + body).replace("\n", "\r\n"),
            "no_final_newline.csv": (HEADER + body).rstrip("\n"),
            "one_blank_line_at_end.csv": HEADER + body + "\n",
        }
        for name, text in good.items():
            e = plot_energy.load_energy(self.write(name, text))
            self.assertEqual(e["step"].tolist(), [0, 5], name)

    def test_rejects_malformed(self):
        good = row(0, 0.0, 0.5, -1.5)
        bad = {
            "empty.csv": "",
            "header_only.csv": HEADER,
            "wrong_header.csv": "step,t,kinetic,potential,total\n" + good,
            # A log from before the heat column (#46): its header no longer matches.
            "before_heat.csv": "step,t,kinetic,potential,total,px,py,pz\n0,0,0.5,-1.5,-1,0,0,0\n",
            "columns.csv": HEADER + "0,0,0.5,-1.5,-1.0,0,0,0\n",
            "blank_middle.csv": HEADER + good + "\n" + row(5, 0.01, 0.4, -1.5),
            "not_a_number.csv": HEADER + "0,0,fast,-1.5,0,-1,0,0,0\n",
            "comment.csv": HEADER + "# note\n" + good,
            "steps_go_down.csv": HEADER + row(5, 0.01, 0.4, -1.5) + good,
            "steps_repeat.csv": HEADER + good + good,
            "fractional_step.csv": HEADER + "0.5,0,0.5,-1.5,0,-1,0,0,0\n",
        }
        for name, text in bad.items():
            with self.assertRaises(ValueError, msg=name) as cm:
                plot_energy.load_energy(self.write(name, text))
            self.assertIn(name, str(cm.exception))  # the message names the file

    def test_summary_of_falling_energy(self):
        s = plot_energy.summarize(plot_energy.load_energy(self.falling()))
        self.assertEqual(s["e0"], -1.0)
        self.assertEqual(s["e1"], -1.25)
        self.assertAlmostEqual(s["change"], -0.25, places=15)
        self.assertEqual(s["max_rise"], 0.0)
        self.assertIsNone(s["rise_step"])
        self.assertTrue(s["finite"])

    def test_summary_finds_the_largest_rise(self):
        # total: -1.0, -0.99 (+0.01), -1.2, -1.15 (+0.05), -1.3
        path = self.write("rises.csv", HEADER + row(0, 0, 0.5, -1.5) + row(1, 1, 0.51, -1.5)
                          + row(2, 2, 0.3, -1.5) + row(3, 3, 0.35, -1.5) + row(4, 4, 0.2, -1.5))
        s = plot_energy.summarize(plot_energy.load_energy(path))
        self.assertAlmostEqual(s["max_rise"], 0.05, places=12)
        self.assertEqual(s["rise_step"], 3)
        self.assertAlmostEqual(s["change"], -0.3, places=12)

    def test_summary_flags_nan(self):
        path = self.write("nan.csv", HEADER + row(0, 0, 0.5, -1.5) + "1,1,nan,-1.5,0,nan,0,0,0\n")
        s = plot_energy.summarize(plot_energy.load_energy(path))
        self.assertFalse(s["finite"])
        self.assertEqual(s["max_rise"], 0.0)  # a NaN isn't a rise; the run is reported as blown up

    def test_single_row(self):
        path = self.write("one.csv", HEADER + row(0, 0, 0.5, -1.5))
        s = plot_energy.summarize(plot_energy.load_energy(path))
        self.assertEqual(s["change"], 0.0)
        self.assertIsNone(s["rise_step"])

    def test_spread_labels(self):
        # Far apart already: unchanged.
        self.assertEqual(plot_energy.spread([0.0, 50.0, 100.0], 10.0), [0.0, 50.0, 100.0])
        # Two equal values (potential and total at rest): pushed gap apart around the
        # same center, order kept, the far one untouched.
        self.assertEqual(plot_energy.spread([-80.0, 5.0, 5.0], 10.0), [-80.0, 0.0, 10.0])
        # A push that runs into the next value takes it along: 0, 0, 12 -> 0-centered
        # pair overlaps 12, so all three spread around their mean 4.
        self.assertEqual(plot_energy.spread([0.0, 0.0, 12.0], 10.0), [-6.0, 4.0, 14.0])
        # Input order is kept in the output.
        out = plot_energy.spread([30.0, 0.0, 2.0], 10.0)
        self.assertLess(out[1], out[2])
        self.assertLess(out[2], out[0])
        self.assertGreaterEqual(out[2] - out[1], 10.0 - 1e-12)
        self.assertGreaterEqual(out[0] - out[2], 10.0 - 1e-12)

    def test_end_labels_dont_overlap(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for dpi in (72, 100, 200):  # the gap is in points, whatever the resolution
            fig, ax = plt.subplots(dpi=dpi)
            ax.set_xlim(0, 1)
            ax.set_ylim(-10, 10)
            plot_energy.label_line_ends(ax, 1.0, {"kinetic": 8.0, "potential": -5.0,
                                                  "total": -5.0})
            dy = {a.get_text(): a.xyann[1] for a in ax.texts}  # offsets in points
            plt.close(fig)
            self.assertEqual(set(dy), {"kinetic", "potential", "total"})
            self.assertAlmostEqual(dy["kinetic"], 0.0, msg=dpi)  # far away: not moved
            self.assertAlmostEqual(dy["total"] - dy["potential"], 11.0, msg=dpi)
            self.assertAlmostEqual(dy["total"] + dy["potential"], 0.0, msg=dpi)  # centered

    def test_main_writes_png_and_summary(self):
        png = os.path.join(self.dir, "energy.png")
        code, out, err = self.run_main([self.falling(), "-o", png, "--title", "test run"])
        self.assertEqual(code, 0, err)
        with open(png, "rb") as f:
            self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")
        self.assertIn("3 rows", out)
        self.assertIn("-1 -> -1.25", out)
        self.assertIn("total energy never rose", out)

    def test_max_rise(self):
        path = self.write("rise.csv", HEADER + row(0, 0, 0.5, -1.5) + row(1, 1, 0.52, -1.5))
        png = os.path.join(self.dir, "e.png")
        code, out, err = self.run_main([path, "-o", png, "--max-rise", "0.01"])
        self.assertEqual(code, 3)
        self.assertIn("rose by 0.02, more than --max-rise 0.01", err)
        self.assertIn("largest rise of total energy: 0.02 (ending at step 1)", out)
        code, _, err = self.run_main([path, "-o", png, "--max-rise", "0.05"])
        self.assertEqual(code, 0, err)
        code, _, err = self.run_main([self.falling(), "-o", png, "--max-rise", "0"])
        self.assertEqual(code, 0, err)  # never rose: even 0 passes
        # A rise exactly equal to the limit passes (-1.0 -> -0.75, both exact in binary).
        exact = self.write("exact.csv", HEADER + row(0, 0, 0.5, -1.5) + row(1, 1, 0.75, -1.5))
        code, _, err = self.run_main([exact, "-o", png, "--max-rise", "0.25"])
        self.assertEqual(code, 0, err)
        code, _, _ = self.run_main([exact, "-o", png, "--max-rise", "0.2499999"])
        self.assertEqual(code, 3)

    def test_main_errors(self):
        png = os.path.join(self.dir, "e.png")
        code, _, err = self.run_main([os.path.join(self.dir, "missing.csv"), "-o", png])
        self.assertEqual(code, 1)
        self.assertIn("missing.csv", err)
        code, _, err = self.run_main([self.falling(), "-o", os.path.join(self.dir, "no", "e.png")])
        self.assertEqual(code, 1)
        self.assertIn("doesn't exist", err)
        # An image type matplotlib can't write: one clean line, no traceback.
        code, _, err = self.run_main([self.falling(), "-o", os.path.join(self.dir, "e.xyz")])
        self.assertEqual(code, 1)
        self.assertIn("writing", err)
        self.assertIn("e.xyz", err)
        self.assertNotIn("Traceback", err)
        nan = self.write("nan.csv", HEADER + row(0, 0, 0.5, -1.5) + "1,1,nan,-1.5,0,nan,0,0,0\n")
        code, _, err = self.run_main([nan, "-o", png])
        self.assertEqual(code, 1)
        self.assertIn("blew up", err)
        for bad in ["-1", "nan", "inf"]:
            code, _, _ = self.run_main([self.falling(), "-o", png, "--max-rise", bad])
            self.assertEqual(code, 2, bad)


if __name__ == "__main__":
    unittest.main()
