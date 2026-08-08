"""Entry point of the sensitivity analysis: find the sample size that converges, and publish it.

Calls run_sobol_production.py at increasing sample sizes N and, after each one, applies the
convergence criterion of Sarrazin et al. (2016) to the two descriptors the paper reports, the
base-10 logarithm of the mean biomass and the day of year of the maximum. The first size that
satisfies the criterion stops the sequence and becomes the published analysis, so no larger sample
is run and no separate production run is needed.

Each size is computed and stored under its own directory, and is resumable, so an interrupted run
resumes and skips the sizes already computed.

Inputs : data/stations.zarr, parameters.yaml
Output : products/sobol_indices.csv, the table Figure 6 reads
Run    : .venv/bin/python scripts/experiments/run_sobol_convergence.py
         .venv/bin/python scripts/experiments/run_sobol_convergence.py --ci 0.05 --stab 0.05 --workers 8
         .venv/bin/python scripts/experiments/run_sobol_convergence.py --n-list 256,512,1024,2048,4096,8192
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
FIG6 = HERE.parent / "figures" / "fig06_sobol.py"
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
            print(f"\nConverged at N={n:,} ({int(row.iloc[0].total_sims):,} simulations). "
                  f"Stopping, no larger N needed.", flush=True)
            break
        print(f"\n... N={n:,} not yet converged, continuing to next N.", flush=True)

    # convergence plot over whatever points were run
    subprocess.run([sys.executable, str(FIG6B), "--ci", str(args.ci), "--stab", str(args.stab)], check=False)

    if chosen is None:
        print("\nNot converged within the provided --n-list. Extend it and re-run, "
              "the sizes already computed are reused.", flush=True)
        return

    # Publish the converged point: freeze its indices, then draw Figure 6.
    print(f"\n=================== promoting N={chosen:,} to Figure 6 ===================", flush=True)
    subprocess.run([sys.executable, str(FREEZE), "--subdir", f"conv/N{chosen}"], check=True)
    subprocess.run([sys.executable, str(FIG6)], check=True)
    (CONV / "CHOSEN.json").write_text(json.dumps({
        "chosen_N": chosen, "production_dir": f"results_raw/sobol/conv/N{chosen}",
        "ci_threshold": args.ci, "stab_threshold": args.stab,
        "figure": "figures/Figure_6.{pdf,png}",
        "indices_table": "products/sobol_indices.csv",
    }, indent=2))
    print(f"\nPublished N={chosen:,} -> products/sobol_indices.csv and figures/Figure_6.{{pdf,png}}", flush=True)
    print(f"Pointer written: {CONV / 'CHOSEN.json'}", flush=True)


if __name__ == "__main__":
    main()
