"""Where did the mass go? Impact analysis of a planet_sim run (#67).

    python scripts/analyze_impact.py frames/run1 [--frame N] [--every K] [-o OUT] [--min-remnant F]

Reads one frame (the last by default) and sorts every particle into:
  remnant      the largest body left after the impact
  disk         bound to the remnant, on an orbit that stays outside its surface
               (a debris disk, the material a moon could form from)
  reimpacting  on a path that hits the remnant: bound with an orbit that dips inside
               its surface (falls back), or unbound but still on its way in on a
               path that hits (before the impact, the whole impactor is here)
  escaping     moving faster than the remnant's escape speed, on a path that misses
and prints the mass in each, split by material (iron/rock) and by the planet it
started in, plus the disk's angular momentum and a predicted moon mass.

--every K analyzes every K-th frame and writes OUT.csv and OUT.png (mass fractions
against time). --min-remnant F exits 1 if the remnant holds less than fraction F of
the mass (used by CI).

Method (the usual one in giant-impact papers, e.g. Canup 2004):
  1. Seed the remnant's center at the particle with the lowest gravitational potential.
  2. Its surface R is where the density, walking outward, drops below 10% of the
     density of the nearest 10% of particles.
  3. Members are particles within R that are bound to a point mass M at the center.
     Recompute M, center, velocity and R from them until the members stop changing.
  4. Every other particle follows a two-body orbit around the remnant (a point mass):
     a path whose periapsis is inside R hits (reimpacting) if the particle is bound, or
     if it is unbound but still approaching. Other unbound particles are escaping, and
     other bound ones are disk.
The point-mass orbit ignores the remnant's shape and spin; the answer is for the frame
analyzed, so use a late frame (the --every plot shows when the split settles).

The seed's potential is computed directly against every particle for up to 4,096
candidate particles (all of them in smaller frames, an evenly spaced sample in larger
ones), in chunks of 256 rows: about 1 s per frame at 10k particles and 2 s at 50k. Each
frame is analyzed on its own, so --every gives the same answer for a frame as analyzing
it alone. Units: G = 1, as in planet_sim. Exit codes: 0 ok, 1 bad input
or a failed --min-remnant, 2 bad command line.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import list_frames, read_frame  # noqa: E402

REMNANT, DISK, REIMPACTING, ESCAPING = 0, 1, 2, 3
CLASS_NAMES = ["remnant", "disk", "reimpacting", "escaping"]
MATERIAL_ROCK, MATERIAL_IRON = 0, 1

SURFACE_DENSITY_FRACTION = 0.1  # the surface is where density drops below this x inner
ROCHE_FACTOR = 2.9  # Roche limit for rock, in remnant radii (Ida et al. 1997)
DISK_ESCAPE_FRACTION = 0.05  # disk mass lost while a moon forms (Ida et al. 1997)
CHUNK = 256  # rows per block in the potential
MAX_CANDIDATES = 4096  # particles whose potential is computed when picking the seed


def length_scale(x):
    """A size for the main body that far-flung debris can't change: the median distance
    of the particles from their per-axis median position. Used for the softening."""
    if len(x) < 2:
        return 1.0
    d = float(np.median(np.linalg.norm(x - np.median(x, axis=0), axis=1)))
    return d if d > 0 else 1.0


def potential_min_index(x, mass, eps):
    """Index of the particle with the lowest gravitational potential
    phi_i = -sum_{j != i} m_j / sqrt(|x_i - x_j|^2 + eps^2), among up to MAX_CANDIDATES
    evenly spaced particles (every particle in frames that size or smaller). A sample is
    enough: it is only the seed, and the remnant's deep interior holds many candidates.
    A particle's own term is left out, or the heaviest particles (iron) would win
    wherever they are."""
    n = len(x)
    candidates = np.arange(n) if n <= MAX_CANDIDATES else \
        np.linspace(0, n - 1, MAX_CANDIDATES).astype(int)
    best_phi, best_i = np.inf, 0
    for s in range(0, len(candidates), CHUNK):
        rows = candidates[s:s + CHUNK]
        r2 = np.full((len(rows), n), eps * eps)
        for a in range(3):  # one axis at a time: no (rows, n, 3) temporary
            r2 += (x[rows, a, None] - x[None, :, a]) ** 2
        phi = -(mass[None, :] / np.sqrt(r2)).sum(axis=1) + mass[rows] / eps  # drop self
        i = int(np.argmin(phi))
        if phi[i] < best_phi:
            best_phi, best_i = phi[i], int(rows[i])
    return best_i


def surface_radius(r, mass, eps):
    """The remnant's radius from particle distances r to its center.

    The inner density is the mean density of the nearest 10% of particles (at least 16).
    Walking outward particle by particle, take the shell from that particle's distance
    to 2 mean particle spacings further out; the first shell whose density is under 10%
    of the inner density holds the surface, and the radius is its farthest particle.

    The shells have a fixed thickness, not a fixed particle count, because relaxed
    planets sit on a lattice: a window of a few particles can fall in the gap between
    two lattice layers and look empty. Starting from the nearest 10%, not the half-mass
    radius, matters when the remnant holds under half the mass (with two equal clumps the
    half-mass radius is the first clump's edge). Under 32 particles there is no profile
    to read, so it is the farthest one: in a frame that small, any bound particle counts
    as remnant however far out it is."""
    n = len(r)
    if n == 0:
        return 0.0
    order = np.argsort(r)
    r, mass = r[order], mass[order]
    if n < 32:
        return float(r[-1])
    cum = np.concatenate([[0.0], np.cumsum(mass)])  # cum[k] = mass of particles 0..k-1
    i0 = min(n - 1, max(16, n // 10))
    r_inner = max(float(r[i0]), eps)
    inner_volume = 4.0 / 3.0 * np.pi * r_inner**3
    rho_inner = cum[i0 + 1] / inner_volume
    spacing = (inner_volume / (i0 + 1)) ** (1.0 / 3.0)
    lo = r[i0:]
    hi = lo + 2.0 * spacing
    end = np.searchsorted(r, hi, side="right")  # particles k .. end-1 are in shell k
    shell_mass = cum[end] - cum[np.arange(i0, n)]
    shell_volume = 4.0 / 3.0 * np.pi * (hi**3 - lo**3)
    thin = np.nonzero(shell_mass / shell_volume < SURFACE_DENSITY_FRACTION * rho_inner)[0]
    if len(thin) == 0:
        return float(r[-1])
    return float(r[end[thin[0]] - 1])


def mass_center(x, v, mass, mask):
    m = mass[mask]
    M = float(m.sum())
    return M, (m[:, None] * x[mask]).sum(axis=0) / M, (m[:, None] * v[mask]).sum(axis=0) / M


def find_remnant(x, v, mass, max_rounds=100):
    """The largest remnant, seeded at the lowest-potential particle. Returns a dict:
    center, velocity, M, R, members (bool mask), rounds, converged, eps."""
    n = len(x)
    # Softening, only to avoid dividing by zero: 1% of the main body's size, which
    # debris flying off to large distances doesn't change (found in review: a frame-
    # width-based value let one particle at x = 2000 double R).
    eps = 0.01 * length_scale(x)
    seed = potential_min_index(x, mass, eps)
    d = np.linalg.norm(x - x[seed], axis=1)
    near = np.argsort(d)[: max(1, n // 100)]
    center = x[seed].copy()
    velocity = (mass[near, None] * v[near]).sum(axis=0) / mass[near].sum()
    # First round: everything within the surface around the seed.
    r = np.linalg.norm(x - center, axis=1)
    members = r <= surface_radius(r, mass, eps)
    members[seed] = True
    converged = False
    before = None  # the member set two rounds back
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        M, center, velocity = mass_center(x, v, mass, members)
        r = np.linalg.norm(x - center, axis=1)
        R = surface_radius(r, mass, eps)
        energy = 0.5 * ((v - velocity) ** 2).sum(axis=1) - M / np.maximum(r, eps)
        new = (r <= R) & (energy < 0.0)
        if not new.any():  # never empty: keep the particle nearest the center
            new[int(np.argmin(r))] = True
        if np.array_equal(new, members):
            converged = True
            break
        if before is not None and np.array_equal(new, before):
            # Flipping between two sets: a particle or two right at the edge goes in
            # and out each round (seen mid-impact in the smash, 1 particle of 10,000).
            # Settle on the larger set.
            members = members | new
            converged = True
            break
        before, members = members, new
    M, center, velocity = mass_center(x, v, mass, members)
    r = np.linalg.norm(x - center, axis=1)
    R = surface_radius(r, mass, eps)
    return {"center": center, "velocity": velocity, "M": M, "R": R, "members": members,
            "rounds": rounds, "converged": converged, "eps": eps}


def orbit_class(x, v, mass, remnant):
    """Class code per particle (REMNANT, DISK, REIMPACTING, ESCAPING): two-body orbits
    around the remnant as a point mass M at its center."""
    M, R, eps = remnant["M"], remnant["R"], remnant["eps"]
    rel_x = x - remnant["center"]
    rel_v = v - remnant["velocity"]
    r = np.maximum(np.linalg.norm(rel_x, axis=1), eps)
    energy = 0.5 * (rel_v**2).sum(axis=1) - M / r
    h = np.linalg.norm(np.cross(rel_x, rel_v), axis=1)
    e = np.sqrt(np.maximum(0.0, 1.0 + 2.0 * energy * h**2 / M**2))
    periapsis = h**2 / (M * (1.0 + e))
    hits = periapsis <= R
    approaching = (rel_x * rel_v).sum(axis=1) < 0.0
    cls = np.where(energy >= 0.0,
                   np.where(hits & approaching, REIMPACTING, ESCAPING),
                   np.where(hits, REIMPACTING, DISK))
    cls[remnant["members"]] = REMNANT
    return cls


def split_table(cls, mass, material, body):
    """Mass per class, overall and split by material and by starting planet.
    Returns {"columns": [...], "rows": {class name: [mass per column]}}. Columns:
    total, iron, rock, then body 0, body 1 for each body id that appears (ids above 1
    are added up as "other bodies")."""
    columns = ["total", "iron", "rock"]
    selectors = [np.ones(len(mass), bool), material == MATERIAL_IRON, material != MATERIAL_IRON]
    names = {0: "target (body 0)", 1: "impactor (body 1)"}
    for b in (0, 1):
        if np.any(body == b):
            columns.append(names[b])
            selectors.append(body == b)
    if np.any((body > 1) | (body < 0)):
        columns.append("other bodies")
        selectors.append((body > 1) | (body < 0))
    rows = {}
    for k, name in enumerate(CLASS_NAMES):
        in_class = cls == k
        rows[name] = [float(mass[in_class & sel].sum()) for sel in selectors]
    return {"columns": columns, "rows": rows}


def disk_angular_momentum(x, v, mass, cls, remnant):
    """Angular momentum vector of the disk particles about the remnant's center, in the
    remnant's frame."""
    disk = cls == DISK
    rel_x = x[disk] - remnant["center"]
    rel_v = v[disk] - remnant["velocity"]
    return (mass[disk, None] * np.cross(rel_x, rel_v)).sum(axis=0)


def moon_mass_estimate(L, M_disk, M, R):
    """Predicted moon mass from a disk (Ida et al. 1997, as used by Canup and Asphaug
    2001): M_moon = 1.9 L / sqrt(G M a_R) - 1.1 M_disk - 1.9 M_esc, with a_R = 2.9 R the
    Roche limit and M_esc = 0.05 M_disk, so 1.9 L / sqrt(M a_R) - 1.195 M_disk.
    A fitted estimate, not a measurement. Never below 0."""
    if M <= 0 or R <= 0 or M_disk <= 0:
        return 0.0
    a_roche = ROCHE_FACTOR * R
    moon = 1.9 * L / np.sqrt(M * a_roche) - 1.1 * M_disk - 1.9 * DISK_ESCAPE_FRACTION * M_disk
    return max(0.0, float(moon))


def check_finite(frame, path):
    for key in ("x", "v", "mass"):
        if not np.all(np.isfinite(frame[key])):
            raise ValueError(f"{path}: non-finite {key} (did the run blow up?)")
    if len(frame["mass"]) == 0:
        raise ValueError(f"{path}: no particles")
    if np.any(frame["mass"] <= 0):
        raise ValueError(f"{path}: a particle has mass <= 0")


def analyze(frame):
    """Everything about one frame: remnant, classes, table, disk L, moon estimate."""
    x, v, mass = frame["x"], frame["v"], frame["mass"]
    remnant = find_remnant(x, v, mass)
    cls = orbit_class(x, v, mass, remnant)
    table = split_table(cls, mass, frame["material"], frame["body"])
    L = disk_angular_momentum(x, v, mass, cls, remnant)
    M_disk = float(mass[cls == DISK].sum())
    return {"remnant": remnant, "classes": cls, "table": table, "L_disk": L,
            "M_disk": M_disk, "total_mass": float(mass.sum()),
            "moon": moon_mass_estimate(float(np.linalg.norm(L)), M_disk, remnant["M"],
                                       remnant["R"])}


def print_report(result, path, out=None):
    out = sys.stdout if out is None else out
    rem, table, total = result["remnant"], result["table"], result["total_mass"]
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"{path}: {len(result['classes'])} particles, total mass {total:.6g}")
    p(f"remnant: M = {rem['M']:.6g} ({rem['M'] / total:.1%} of the mass), R = {rem['R']:.4g}"
      f"{'' if rem['converged'] else '  (did not settle in ' + str(rem['rounds']) + ' rounds)'}")
    widths = [max(12, len(c) + 2) for c in table["columns"]]
    p("")
    p(f"{'mass':<13}" + "".join(f"{c:>{w}}" for c, w in zip(table["columns"], widths))
      + f"{'of total':>10}")
    for name in CLASS_NAMES:
        row = table["rows"][name]
        p(f"{name:<13}" + "".join(f"{m:>{w}.5g}" for m, w in zip(row, widths))
          + f"{row[0] / total:>10.1%}")
    for name in CLASS_NAMES:
        row = table["rows"][name]
        if row[0] > 0:
            iron = row[1] / row[0]
            p(f"  {name}: {iron:.1%} iron")
    L = result["L_disk"]
    p("")
    p(f"disk angular momentum: |L| = {np.linalg.norm(L):.5g}  (L = {L[0]:.4g}, {L[1]:.4g}, "
      f"{L[2]:.4g}), about the remnant's center")
    a_roche = ROCHE_FACTOR * rem["R"]
    p(f"predicted moon mass: {result['moon']:.5g} "
      f"({result['moon'] / rem['M'] if rem['M'] > 0 else 0.0:.2%} of the remnant)")
    p(f"  = max(0, 1.9 |L| / sqrt(M a_R) - 1.195 M_disk), a_R = 2.9 R = {a_roche:.4g}, "
      f"M_disk = {result['M_disk']:.5g}")
    p("  a fitted estimate (Ida et al. 1997, Canup and Asphaug 2001), not a measurement")


def frame_times(run_dir, frame_numbers, n_frames):
    """t for each frame number from the run's energy.csv, if it has one row per frame
    with steps increasing. planet_sim writes a frame and an energy row in the same
    output call, so row k is frame k; the last gap can be shorter (the final step is
    always written). Otherwise None."""
    path = os.path.join(run_dir, "energy.csv")
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError):
        return None
    if data.shape[0] != n_frames or data.shape[1] < 2:
        return None
    if np.any(np.diff(data[:, 0]) <= 0):
        return None
    return [float(data[k, 1]) for k in frame_numbers]


def time_series(run_dir, paths, every, out_prefix):
    """Analyzes every every-th frame (always including the last) and writes
    out_prefix.csv and out_prefix.png. Returns the last frame's result."""
    picked = list(range(0, len(paths), every))
    if picked[-1] != len(paths) - 1:
        picked.append(len(paths) - 1)
    times = frame_times(run_dir, picked, len(paths))
    import matplotlib  # before the slow part, so a missing package fails at once

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, result = [], None
    for k in picked:
        frame = read_frame(paths[k])
        check_finite(frame, paths[k])
        result = analyze(frame)
        rem = result["remnant"]
        if not rem["converged"]:
            print(f"analyze_impact.py: warning: {os.path.basename(paths[k])}: the remnant "
                  f"did not settle in {rem['rounds']} rounds; using the last one",
                  file=sys.stderr)
        total = result["total_mass"]
        fractions = [result["table"]["rows"][name][0] / total for name in CLASS_NAMES]
        rows.append([k, times[picked.index(k)] if times else float(k)] + fractions
                    + [rem["R"], result["moon"]])
    with open(out_prefix + ".csv", "w", newline="\n") as f:
        f.write("frame,t,remnant,disk,reimpacting,escaping,R,moon_estimate\n")
        for row in rows:
            f.write("%d,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n" % tuple(row))

    data = np.array(rows)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    for j, name in enumerate(CLASS_NAMES):
        ax.plot(data[:, 1], data[:, 2 + j], label=name)
    ax.set_xlabel("t (simulation units)" if times else "frame number (no matching energy.csv)")
    ax.set_ylabel("fraction of the mass")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(f"Where the mass goes: {os.path.basename(os.path.normpath(run_dir))}")
    fig.tight_layout()
    fig.savefig(out_prefix + ".png")
    plt.close(fig)
    return result


def main(argv):
    parser = argparse.ArgumentParser(description="Impact analysis of planet_sim frames.")
    parser.add_argument("run", help="folder written by planet_sim --out")
    parser.add_argument("--frame", type=int, help="frame number to analyze (default: the last)")
    parser.add_argument("--every", type=int, help="analyze every k-th frame; writes OUT.csv/png")
    parser.add_argument("-o", "--out", help="output prefix for --every (default: RUN/impact)")
    parser.add_argument("--min-remnant", type=float,
                        help="exit 1 if the remnant holds less than this fraction of the mass")
    args = parser.parse_args(argv)
    if args.every is not None and args.every < 1:
        parser.error("--every needs a whole number >= 1")
    if args.every is not None and args.frame is not None:
        parser.error("--frame and --every can't be used together")
    if args.out is not None and args.every is None:
        parser.error("-o only applies with --every (the report itself goes to the screen)")
    if not os.path.isdir(args.run):
        print(f"analyze_impact.py: '{args.run}' is not a folder", file=sys.stderr)
        return 1
    paths = list_frames(args.run)
    if not paths:
        print(f"analyze_impact.py: no frame_*.csv files in '{args.run}'", file=sys.stderr)
        return 1
    try:
        if args.every is not None:
            prefix = args.out or os.path.join(args.run, "impact")
            if not os.path.isdir(os.path.dirname(os.path.abspath(prefix))):
                print(f"analyze_impact.py: the folder for '{prefix}' doesn't exist",
                      file=sys.stderr)
                return 1
            result = time_series(args.run, paths, args.every, prefix)
            path = paths[-1]
            print(f"wrote {prefix}.csv and {prefix}.png")
        else:
            if args.frame is None:
                path = paths[-1]
            else:
                wanted = os.path.join(args.run, "frame_%05d.csv" % args.frame)
                matches = [p for p in paths if os.path.samefile(p, wanted)] \
                    if os.path.exists(wanted) else []
                if not matches:
                    print(f"analyze_impact.py: no frame {args.frame} in '{args.run}'",
                          file=sys.stderr)
                    return 1
                path = matches[0]
            frame = read_frame(path)
            check_finite(frame, path)
            result = analyze(frame)
    except ImportError as e:
        print(f"analyze_impact.py: --every needs matplotlib ({e}); "
              f"pip install -r scripts/requirements.txt", file=sys.stderr)
        return 1
    except (OSError, ValueError) as e:
        print(f"analyze_impact.py: {e}", file=sys.stderr)
        return 1
    print_report(result, path)
    if args.min_remnant is not None:
        fraction = result["remnant"]["M"] / result["total_mass"]
        if fraction < args.min_remnant:
            print(f"analyze_impact.py: the remnant holds {fraction:.1%} of the mass, "
                  f"under --min-remnant {args.min_remnant:.1%}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
