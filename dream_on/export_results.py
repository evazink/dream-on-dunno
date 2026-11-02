"""
Aggregate all runs in runs/ into tables and figures for the report.

python dream_on/export_results.py --runs runs --out results
Outputs: results/summary.csv, results/summary.md, results/learning_curves.png,
         results/per_run.csv, (optional) results/diffusion_loss.png
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = {
    "ppo_none_sparse": "PPO, state only",
    "sac_none_sparse": "SAC, state only",
    "ppo_current_sparse": "PPO, state + current frame",
    "ppo_diffusion_sparse": "PPO, state + diffusion subgoal (sparse reward)",
    "ppo_diffusion_l2": "Dream On: PPO + subgoal + L2 reward",
    "ppo_diffusion_ssim": "Dream On: PPO + subgoal + SSIM reward",
}


def ma(x, w=3):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def per_run(run: Path, thr: float):
    d = np.load(run / "evaluations.npz")
    ts, res = d["timesteps"], d["results"]  # results: [n_evals, n_episodes] sparse returns
    succ = res.mean(axis=1).clip(0, 1)
    sm = ma(succ)
    ts_sm = ts[len(ts) - len(sm):]
    hit = np.where(sm >= thr)[0]
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    return dict(
        ts=ts, succ=succ,
        steps_to_thr=int(ts_sm[hit[0]]) if len(hit) else None,
        final=float(succ[-5:].mean()),
        auc=float(np.trapezoid(succ, ts) / (ts[-1] - ts[0])) if len(ts) > 1 else float(succ.mean()),
        wall_min=round(meta.get("wall_clock_sec", float("nan")) / 60, 1),
        done=(run / "DONE").exists(),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    p.add_argument("--thr", type=float, default=0.8, help="success threshold for steps-to-threshold")
    p.add_argument("--diffusion_tb", default="data/diffusion/tb", help="diffusion TensorBoard dir (optional)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    rows = []
    for run in sorted(Path(a.runs).glob("*/evaluations.npz")):
        run = run.parent
        cond = re.sub(r"_s\d+$", "", run.name)
        r = per_run(run, a.thr)
        groups[cond].append(r)
        rows.append(dict(run=run.name, condition=cond, steps_to_thr=r["steps_to_thr"], final_success=r["final"],
                         auc=round(r["auc"], 4), wall_min=r["wall_min"], finished=r["done"]))
    if not rows:
        raise SystemExit(f"no runs with evaluations.npz under {a.runs}")
    with open(out / "per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summ = []
    for cond, rs in groups.items():
        stt = [r["steps_to_thr"] for r in rs if r["steps_to_thr"] is not None]
        summ.append(dict(
            condition=cond, label=LABELS.get(re.sub(r"_lam[\d.]+", "", cond), cond), seeds=len(rs),
            reached_thr=f"{len(stt)}/{len(rs)}",
            steps_to_thr_mean=round(float(np.mean(stt))) if stt else None,
            steps_to_thr_std=round(float(np.std(stt))) if len(stt) > 1 else None,
            final_success_mean=round(float(np.mean([r["final"] for r in rs])), 3),
            final_success_std=round(float(np.std([r["final"] for r in rs])), 3),
            auc_mean=round(float(np.mean([r["auc"] for r in rs])), 3),
            auc_std=round(float(np.std([r["auc"] for r in rs])), 3),
            wall_min_mean=round(float(np.nanmean([r["wall_min"] for r in rs])), 1),
        ))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    with open(out / "summary.md", "w") as f:
        f.write(f"| Condition | Seeds | Reached {a.thr:.0%} | Steps to {a.thr:.0%} (mean ± std) | "
                f"Final success | AUC | Wall-clock, min |\n|---|---|---|---|---|---|---|\n")
        for s in summ:
            stt = "—" if s["steps_to_thr_mean"] is None else (
                f"{s['steps_to_thr_mean']:,}" + (f" ± {s['steps_to_thr_std']:,}" if s["steps_to_thr_std"] is not None else ""))
            f.write(f"| {s['label']} | {s['seeds']} | {s['reached_thr']} | {stt} | "
                    f"{s['final_success_mean']:.2f} ± {s['final_success_std']:.2f} | "
                    f"{s['auc_mean']:.3f} ± {s['auc_std']:.3f} | {s['wall_min_mean']} |\n")
    print((out / "summary.md").read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for cond, rs in groups.items():
        n = min(len(r["ts"]) for r in rs)
        ts = rs[0]["ts"][:n]
        Y = np.stack([r["succ"][:n] for r in rs])
        Ys = np.stack([np.concatenate([y[:2], ma(y)]) if len(y) >= 3 else y for y in Y])
        m = Ys.mean(0)
        label = LABELS.get(re.sub(r"_lam[\d.]+", "", cond), cond) + f" (n={len(rs)})"
        line, = ax.plot(ts, m, lw=2, label=label)
        if len(rs) > 1:
            ax.fill_between(ts, Ys.min(0), Ys.max(0), color=line.get_color(), alpha=0.15, lw=0)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("evaluation success rate (sparse reward)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(out / "learning_curves.png", dpi=200)
    print(f"saved {out}/learning_curves.png")

    tb = Path(a.diffusion_tb)
    if tb.exists():
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ev = [e for e in tb.rglob("events.out.tfevents.*")]
            xs, ys = [], []
            for e in sorted(ev):
                acc = EventAccumulator(str(e))
                acc.Reload()
                if "Loss/Epoch_Avg" in acc.Tags().get("scalars", []):
                    for s in acc.Scalars("Loss/Epoch_Avg"):
                        xs.append(s.step + 1)
                        ys.append(s.value)
            if xs:
                order = np.argsort(xs)
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(np.array(xs)[order], np.array(ys)[order], lw=2)
                ax.set_yscale("log")
                ax.set_xlabel("epoch")
                ax.set_ylabel("noise-prediction MSE")
                ax.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(out / "diffusion_loss.png", dpi=200)
                with open(out / "diffusion_loss.csv", "w") as f:
                    f.write("epoch,loss\n" + "".join(f"{xs[i]},{ys[i]}\n" for i in order))
                print(f"saved {out}/diffusion_loss.png")
        except Exception as e:
            print(f"diffusion loss export skipped: {e}")


if __name__ == "__main__":
    main()
