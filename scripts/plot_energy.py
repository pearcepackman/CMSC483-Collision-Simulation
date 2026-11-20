"""Energy plot of a run: planet_sim's energy.csv -> PNG, plus a summary on stdout.

    python scripts/plot_energy.py frames/run1/energy.csv -o energy.png [--title T]
                                  [--max-rise X] [--max-drift Y]

Plots kinetic, potential, contact, heat and total energy against time, and under it the
change in total energy since the start, on its own scale, so small rises are visible.
Prints the total change, the largest rise of the total between two rows, and the
furthest the total moved from its starting value, up or down. With --max-rise X it exits
3 if a rise between two rows is above X; with --max-drift Y it exits 3 if the total ever
moves more than Y away from its start (which catches a slow climb made of small rises,
and energy that goes missing, e.g. heat that isn't booked). That lets a script or CI
job fail a run whose energy moved by more than the expected timestep error. Exit codes: 0 ok, 1 bad input, 2 bad command line, 3 a limit was exceeded. Needs
scripts/requirements.txt (numpy, matplotlib).

energy.csv's total is kinetic + gravitational + contact (energy held in compressed
contact springs) + heat (the energy contact damping has turned into heat), so it should
stay nearly flat. What's left is a small error that appears as contacts begin and
shrinks roughly in proportion to dt (docs/notes.md, #59).
"""

import argparse
import os
import sys

import numpy as np

# What planet_sim's write_energy_header() writes (src/sim.cpp).
HEADER = "step,t,kinetic,potential,contact,heat,total,px,py,pz"
COLUMNS = HEADER.split(",")
ENERGIES = ("kinetic", "potential", "contact", "heat", "total")
# The header before the contact column (#59). Logs with it get a "rerun" message.
OLD_HEADER = "step,t,kinetic,potential,heat,total,px,py,pz"

# Blue, orange, purple, yellow and green, picked to stay distinguishable with the common
# kinds of color blindness. Each keeps its color from before contact and heat were
# added. The yellow and green are under 3:1 contrast on white, so every line is also
# named by a text label at its end, and the labels replace a legend.
COLORS = {"kinetic": "#2a78d6", "potential": "#eb6834", "contact": "#8a5cc7", "heat": "#eda100",
          "total": "#1baf7a"}
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def load_energy(path):
    """Reads energy.csv. Returns a dict of numpy arrays, one per column.
    Raises ValueError, naming the file, if it isn't in write_energy_row()'s format."""
    with open(path, newline="") as f:
        header = f.readline().rstrip("\r\n")
        rows = f.read().splitlines()  # handles \n and \r\n
    if header == OLD_HEADER:
        raise ValueError(f"{path}: written by planet_sim before the contact column was added "
                         f"(#59); rerun planet_sim to get a log this script can read")
    if header != HEADER:
        raise ValueError(f"{path}: expected header '{HEADER}', got '{header}'")
    if rows and rows[-1] == "":
        rows.pop()  # one blank line at the end is fine
    if "" in rows:
        raise ValueError(f"{path}: blank line in the middle of the log")
    if not rows:
        raise ValueError(f"{path}: no rows after the header")
    try:
        # comments=None: a '#' line is an error, not something to skip.
        data = np.loadtxt(rows, delimiter=",", ndmin=2, comments=None)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
    if data.shape[1] != len(COLUMNS):
        raise ValueError(f"{path}: expected {len(COLUMNS)} columns, got {data.shape[1]}")
    out = {name: data[:, k] for k, name in enumerate(COLUMNS)}
    steps = out["step"]
    if not np.all(steps == np.round(steps)) or np.any(np.diff(steps) <= 0):
        raise ValueError(f"{path}: steps must be whole numbers that only go up")
    return out


def summarize(e):
    """Total energy change over the run, the largest rise between two rows, and the
    furthest the total moved from its start.
    Returns a dict: e0, e1, change, max_rise (0 if the total never went up),
    rise_step (the row where that rise ended, or None), max_drift (largest
    |total - start|), finite (no NaN/Inf anywhere)."""
    total = e["total"]
    finite = bool(np.all(np.isfinite(np.column_stack([e[c] for c in COLUMNS]))))
    rises = np.diff(total)
    max_rise, rise_step = 0.0, None
    if len(rises) > 0 and finite and np.max(rises) > 0.0:
        k = int(np.argmax(rises))
        max_rise, rise_step = float(rises[k]), int(e["step"][k + 1])
    max_drift = float(np.max(np.abs(total - total[0]))) if finite else 0.0
    return {"e0": float(total[0]), "e1": float(total[-1]), "change": float(total[-1] - total[0]),
            "max_rise": max_rise, "rise_step": rise_step, "max_drift": max_drift,
            "finite": finite}


def spread(ys, gap):
    """Moves values apart, keeping their order, until neighbors are at least gap apart.
    Values that are already far enough apart don't move; a cluster that has to spread
    stays centered on where its values were. Returns them in the input order."""
    order = sorted(range(len(ys)), key=lambda k: ys[k])
    groups = []  # [first value of the group after spreading, original values]
    for k in order:
        groups.append([ys[k], [ys[k]]])
        # Merge with the group below while the two overlap, then recenter the merged group.
        while len(groups) > 1:
            lo_start, lo_vals = groups[-2]
            hi_start, hi_vals = groups[-1]
            if lo_start + gap * len(lo_vals) <= hi_start:
                break
            vals = lo_vals + hi_vals
            center = sum(vals) / len(vals)
            groups[-2:] = [[center - gap * (len(vals) - 1) / 2.0, vals]]
    spread_sorted = [start + gap * j for start, vals in groups for j in range(len(vals))]
    result = [0.0] * len(ys)
    for k, idx in enumerate(order):
        result[idx] = spread_sorted[k]
    return result


def label_line_ends(ax, t_end, ends):
    """Text label at the end of each line, in text color (the line carries the identity).
    Lines that end close together (potential and total, once the planets are at rest)
    get their labels pushed apart so they don't print on top of each other."""
    names = list(ends)
    to_screen = ax.transData.transform
    ys = [to_screen((t_end, ends[n]))[1] for n in names]  # pixels
    gap = 11 * ax.figure.dpi / 72.0  # 11 points
    for name, y, y_label in zip(names, ys, spread(ys, gap)):
        ax.annotate(name, (t_end, ends[name]), xytext=(6, (y_label - y) * 72.0 / ax.figure.dpi),
                    textcoords="offset points", va="center", color=TEXT, fontsize=9)


def plot(e, out, title=None):
    """Two stacked panels sharing the time axis: the energies, then the change in total
    energy since the start."""
    import matplotlib

    matplotlib.use("Agg")  # no window needed, so this runs over SSH and in CI
    import matplotlib.pyplot as plt

    t = e["t"]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6), dpi=100, sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    for ax in (top, bottom):
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(TEXT_SECONDARY)
        ax.tick_params(colors=TEXT_SECONDARY)

    names = ENERGIES
    for name in names:
        top.plot(t, e[name], color=COLORS[name], linewidth=2, label=name)
    top.set_ylabel("energy (G = 1)", color=TEXT)
    top.set_title(title if title else "Energy", color=TEXT, loc="left")

    bottom.plot(t, e["total"] - e["total"][0], color=COLORS["total"], linewidth=2)
    bottom.axhline(0.0, color=TEXT_SECONDARY, linewidth=0.8)
    bottom.set_ylabel("total - start", color=TEXT)
    bottom.set_xlabel("time t", color=TEXT)

    fig.tight_layout()
    fig.subplots_adjust(right=0.88)  # room for the end labels
    label_line_ends(top, t[-1], {name: e[name][-1] for name in names})
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main(argv):
    parser = argparse.ArgumentParser(description="Plot planet_sim's energy.csv.")
    parser.add_argument("log", help="energy.csv written by planet_sim --out")
    parser.add_argument("-o", "--out", required=True, help="output image, e.g. energy.png")
    parser.add_argument("--title", help="plot title (default 'Energy')")
    parser.add_argument("--max-rise", type=float,
                        help="exit 3 if total energy rises by more than this between two rows")
    parser.add_argument("--max-drift", type=float,
                        help="exit 3 if total energy ever moves more than this from its start")
    args = parser.parse_args(argv)
    for flag, value in (("--max-rise", args.max_rise), ("--max-drift", args.max_drift)):
        if value is not None and not (value >= 0.0 and np.isfinite(value)):
            parser.error(f"{flag} needs a finite number >= 0")  # exits 2
    if not os.path.isdir(os.path.dirname(os.path.abspath(args.out))):
        print(f"plot_energy.py: the folder for '{args.out}' doesn't exist", file=sys.stderr)
        return 1
    try:
        e = load_energy(args.log)
    except (OSError, ValueError) as err:
        print(f"plot_energy.py: {err}", file=sys.stderr)
        return 1
    s = summarize(e)
    try:
        plot(e, args.out, title=args.title)
    except (OSError, ValueError) as err:  # ValueError: an image type matplotlib can't write
        print(f"plot_energy.py: writing '{args.out}' failed: {err}", file=sys.stderr)
        return 1
    rel = s["change"] / abs(s["e0"]) if s["e0"] != 0.0 else float("nan")
    print(f"wrote {args.out}: {len(e['step'])} rows, total energy {s['e0']:.6g} -> {s['e1']:.6g} "
          f"(change {s['change']:.3g}, {rel:.3g} of the start)")
    if not s["finite"]:
        print("plot_energy.py: the log has NaN/Inf values (the run blew up)", file=sys.stderr)
        return 1
    if s["rise_step"] is None:
        print("total energy never rose")
    else:
        print(f"largest rise of total energy: {s['max_rise']:.3g} (ending at step {s['rise_step']})")
    print(f"furthest from the start: {s['max_drift']:.3g}")
    code = 0
    if args.max_rise is not None and s["max_rise"] > args.max_rise:
        print(f"plot_energy.py: total energy rose by {s['max_rise']:.3g}, more than "
              f"--max-rise {args.max_rise:g}", file=sys.stderr)
        code = 3
    if args.max_drift is not None and s["max_drift"] > args.max_drift:
        print(f"plot_energy.py: total energy moved {s['max_drift']:.3g} from its start, more than "
              f"--max-drift {args.max_drift:g}", file=sys.stderr)
        code = 3
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
