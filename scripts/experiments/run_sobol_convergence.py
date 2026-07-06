"""rehearsal/run_sobol_convergence.py — adaptive Sobol convergence + production in one.

Runs the Sobol evaluation at an increasing sequence of sample sizes N, and after EACH N
checks the mechanical convergence criterion (Sarrazin et al. 2016) on the two reported
descriptors (log10 mean, argmax). It STOPS at the first N that converges — so it never runs a needlessly large N —
and PROMOTES that point to the production Figure 6. Two birds, one stone: the converged
convergence point IS the production sensitivity analysis (no separate 1e6 run).

Each N is evaluated by run_sobol_production.py into rehearsal/sobol/conv/N<N>/ and is
RESUMABLE, so interrupting/re-running is safe and already-computed points are skipped.

Run:
    .venv/bin/python rehearsal/run_sobol_convergence.py                  # CI/stab = 0.05
    .venv/bin/python rehearsal/run_sobol_convergence.py --ci 0.05 --stab 0.05 --workers 8
    .venv/bin/python rehearsal/run_sobol_convergence.py --n-list 256,512,1024,2048,4096,8192,16384,32768
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fig06b_convergence import CONV, convergence_frame, load_tables

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_sobol_production.py"
FREEZE = HERE / "freeze_sobol_indices.py"             # raw run -> products/sobol_indices.csv
FIG6 = HERE.parent / "figures" / "fig06_sobol.py"     # the display (now in scripts/figures/)
FIG6B = HERE / "fig06b_convergence.py"


def run_point(n: int, workers: int, batch: int) -> None:
    cmd = [sys.executable, str(RUNNER), "--n", str(n), "--out-subdir", f"conv/N{n}",
           "--workers", str(workers), "--batch-size", str(batch)]
    print(f"\n=================== evaluating N={n:,} (resumable) ===================", flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", default="256,512,1024,2048,4096,8192,16384,32768",
                    help="increasing sample sizes to try, comma-separated")
    ap.add_argument("--ci", type=float, default=0.05)
    ap.add_argument("--stab", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1500)
    args = ap.parse_args()
    n_list = [int(x) for x in args.n_list.split(",") if x.strip()]

    chosen = None
    for n in n_list:
        run_point(n, args.workers, args.batch_size)
        conv = convergence_frame(load_tables(), args.ci, args.stab)
        row = conv[conv.N == n]
        print("\n--- convergence so far (reported descriptors) ---", flush=True)
        print(conv.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
        if len(row) and bool(row.iloc[0].converged):
            chosen = n
            print(f"\n>>> CONVERGED at N={n:,} ({int(row.iloc[0].total_sims):,} simulations). "
                  f"Stopping — no larger N needed.", flush=True)
            break
        print(f"\n... N={n:,} not yet converged, continuing to next N.", flush=True)

    # convergence plot over whatever points were run
    subprocess.run([sys.executable, str(FIG6B), "--ci", str(args.ci), "--stab", str(args.stab)], check=False)

    if chosen is None:
        print("\nNOT converged within the provided --n-list. Extend it and re-run "
              "(already-computed points are reused).", flush=True)
        return

    # PROMOTE the converged point: freeze its indices -> products/sobol_indices.csv, then plot Figure 6
    print(f"\n=================== promoting N={chosen:,} to production Figure 6 ===================", flush=True)
    subprocess.run([sys.executable, str(FREEZE), "--subdir", f"conv/N{chosen}"], check=True)
    subprocess.run([sys.executable, str(FIG6)], check=True)
    (CONV / "CHOSEN.json").write_text(json.dumps({
        "chosen_N": chosen, "production_dir": f"results_raw/sobol/conv/N{chosen}",
        "ci_threshold": args.ci, "stab_threshold": args.stab,
        "figure": "figures/Figure_6.{pdf,png}",
        "indices_table": "products/sobol_indices.csv",
    }, indent=2))
    print(f"\nPRODUCTION N={chosen:,} -> products/sobol_indices.csv + figures/Figure_6.{{pdf,png}}", flush=True)
    print(f"Pointer written: {CONV / 'CHOSEN.json'}", flush=True)


if __name__ == "__main__":
    main()
