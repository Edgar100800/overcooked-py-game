"""Launcher de entrenamientos en paralelo (PLAN.md §16.3).

Dos modos:
  - SLURM (default): submitea el job array sbatch/train/run_train_ppo.sh (una tarea
    por linea de training/jobs.txt) y muestra comandos de monitoreo.
  - --local: corre los jobs en CPU en la maquina actual (para probar sin cola),
    respetando --max-parallel para no saturar los cores.

Uso:
  python -m training.launch_parallel                       # sbatch array (jobs.txt)
  python -m training.launch_parallel --array 0-1           # subconjunto
  python -m training.launch_parallel --gres shard:a100_1g.5gb:1   # shards MPS
  python -m training.launch_parallel --local --max-parallel 4 --timesteps 200000
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JOBS = REPO / "training" / "jobs.txt"
SBATCH = REPO / "sbatch" / "train" / "run_train_ppo.sh"


def read_jobs():
    out = []
    for ln in JOBS.read_text().splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = [p.strip() for p in s.split("|")]
        while len(parts) < 5:
            parts.append("")
        layout, lf, seed, ts, extra = parts[:5]
        out.append({"layout": layout, "layout_file": None if lf in ("-", "") else lf,
                    "seed": seed, "timesteps": ts, "extra": extra})
    return out


def submit_slurm(array: str | None, gres: str | None):
    cmd = ["sbatch"]
    if array:
        cmd += [f"--array={array}"]
    if gres:
        cmd += [f"--gres={gres}"]
    cmd += [str(SBATCH)]
    print("[launch] ", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip())
    print("\nMonitoreo:")
    print("  squeue -u $USER -o '%.10i %.12j %.8T %.20b %R'")
    print("  tail -f logs/ppo-*_*.out")
    print("  al terminar: python -m evaluation.run_gate --gate G7 --layout <L>  (habilita el mejor)")


def run_local(jobs, max_parallel, timesteps_override):
    import os
    procs = []
    logdir = REPO / "logs"; logdir.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONNOUSERSITE="1")
    py = str(REPO / ".venv" / "bin" / "python")

    def launch(j, idx):
        key = j["layout"] if not j["layout_file"] else Path(j["layout_file"]).stem
        out = REPO / "models" / key / f"seed{j['seed']}_local"
        ts = str(timesteps_override or j["timesteps"])
        args = [py, "-m", "training.train_ppo", "--seed", j["seed"], "--timesteps", ts,
                "--out", str(out), "--device", "cpu", "--obs", "lossless_grid",
                "--n-envs", "4"]
        args += (["--layout-file", j["layout_file"]] if j["layout_file"]
                 else ["--layout", j["layout"]])
        if j["extra"]:
            args += j["extra"].split()
        log = open(logdir / f"local_ppo_{idx}.log", "w")
        print(f"[launch-local] job{idx} {key} seed{j['seed']} steps{ts} -> {out}")
        return subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env)

    i = 0
    for idx, j in enumerate(jobs):
        while len([p for p in procs if p.poll() is None]) >= max_parallel:
            for p in procs:
                try:
                    p.wait(timeout=2); break
                except subprocess.TimeoutExpired:
                    pass
        procs.append(launch(j, idx))
    for p in procs:
        p.wait()
    print("[launch-local] todos los jobs terminaron.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--array", default=None, help="rango del array SLURM, p.ej. 0-6")
    ap.add_argument("--gres", default=None, help="override GRES, p.ej. shard:a100_1g.5gb:1")
    ap.add_argument("--local", action="store_true", help="correr en CPU local (sin SLURM)")
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--timesteps", type=int, default=None, help="override de timesteps")
    args = ap.parse_args()

    jobs = read_jobs()
    print(f"[launch] {len(jobs)} jobs en {JOBS}")
    if args.local:
        run_local(jobs, args.max_parallel, args.timesteps)
    else:
        submit_slurm(args.array, args.gres)


if __name__ == "__main__":
    main()
