"""
Quantitative evaluation of the subgoal generator (research question 1).

Builds (I_t, I_{t+k}) pairs by replaying episodes of a Minari dataset that was NOT used
for diffusion training (default: D4RL/kitchen/mixed-v2), then compares predictors:
  * copy      : predict I_{t+k} = I_t (trivial baseline)
  * legacy    : original single-pass inference from src/diffusion/inference.py
  * ddim-N    : proper DDIM sampling with N steps
Metrics vs. the true future frame: RMSE, SSIM, and RMSE on the "moving" pixels only
(|I_{t+k} - I_t| > 0.05), which is where a useful subgoal must differ from the present.

python dream_on/eval_diffusion.py --ckpt data/diffusion/ckpt/best_model.pth
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dream_on.common import LegacyPredictor, SubgoalPredictor, preprocess_frame, rmse, ssim  # noqa: E402


def build_pairs(dataset_name, n_episodes, k, stride, seed):
    import minari
    ds = minari.load_dataset(dataset_name, download=True)
    env = ds.recover_environment(render_mode="rgb_array")
    pairs = []
    ids = list(range(min(n_episodes, ds.total_episodes)))
    for ep in ds.iterate_episodes(episode_indices=ids):
        env.reset(seed=seed)
        frames = []
        for act in ep.actions:
            frames.append(preprocess_frame(env.render()))
            _, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
        for t in range(0, len(frames) - k, stride):
            pairs.append((frames[t], frames[t + k]))
    env.close()
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="data/diffusion/ckpt/best_model.pth")
    p.add_argument("--dataset", default="D4RL/kitchen/mixed-v2")
    p.add_argument("--pairs_pkl", default="", help="use a cached pairs file from dataloader.py instead")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--max_pairs", type=int, default=300)
    p.add_argument("--ddim", type=str, default="5,10,20,50")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/diffusion_eval")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.pairs_pkl:
        import pickle
        raw = pickle.load(open(a.pairs_pkl, "rb"))
        pairs = [(c, f) for c, f, *_ in raw][:: a.stride]
        src = a.pairs_pkl
    else:
        pairs = build_pairs(a.dataset, a.episodes, a.k, a.stride, a.seed)
        src = a.dataset
    rng = np.random.default_rng(a.seed)
    if len(pairs) > a.max_pairs:
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), a.max_pairs, replace=False))]
    cur = torch.stack([c for c, _ in pairs])
    fut = torch.stack([f for _, f in pairs])
    print(f"{len(pairs)} pairs from {src}")

    preds = {"copy": cur.clone()}
    timing = {"copy": 0.0}
    leg = LegacyPredictor(a.ckpt, device=a.device)
    t0 = time.time()
    preds["legacy"] = torch.cat([leg.predict(cur[i:i + 16]) for i in range(0, len(cur), 16)])
    timing["legacy"] = (time.time() - t0) / len(cur)
    sampler = SubgoalPredictor(a.ckpt, device=a.device, seed=a.seed)
    for n in [int(x) for x in a.ddim.split(",")]:
        sampler.set_steps(n)
        t0 = time.time()
        preds[f"ddim-{n}"] = torch.cat([sampler.predict(cur[i:i + 16]) for i in range(0, len(cur), 16)])
        timing[f"ddim-{n}"] = (time.time() - t0) / len(cur)

    moving = (fut - cur).abs() > 0.05
    rows = {}
    for name, pr in preds.items():
        r = rmse(pr, fut).squeeze(1)
        s = ssim(pr, fut)
        mv = torch.stack([((pr[i] - fut[i])[moving[i]] ** 2).mean().sqrt() if moving[i].any()
                          else torch.tensor(float("nan")) for i in range(len(pr))])
        rows[name] = dict(rmse=float(r.mean()), rmse_std=float(r.std()), ssim=float(s.mean()),
                          ssim_std=float(s.std()), rmse_moving=float(np.nanmean(mv.numpy())),
                          sec_per_sample=round(timing[name], 4))
        print(f"{name:10s} RMSE {rows[name]['rmse']:.4f}  SSIM {rows[name]['ssim']:.4f}  "
              f"RMSE(moving) {rows[name]['rmse_moving']:.4f}  {timing[name]*1000:.1f} ms/sample")
    meta = dict(source=src, n_pairs=len(pairs), k=a.k, frac_moving_pixels=float(moving.float().mean()),
                ckpt=a.ckpt, device=str(sampler.device))
    json.dump(dict(meta=meta, results=rows), open(out / "diffusion_eval.json", "w"), indent=2)

    # qualitative grid
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["copy", "legacy", "ddim-10"] if "ddim-10" in preds else list(preds)[:3]
    idx = np.linspace(0, len(pairs) - 1, 4).astype(int)
    cols = ["current $I_t$", "true $I_{t+k}$"] + show[1:]
    fig, ax = plt.subplots(len(idx), len(cols), figsize=(2.2 * len(cols), 2.2 * len(idx)))
    for r_, i in enumerate(idx):
        ims = [cur[i], fut[i]] + [preds[n][i] for n in show[1:]]
        for c_, im in enumerate(ims):
            ax[r_, c_].imshow(im[0].numpy(), cmap="gray", vmin=-1, vmax=1)
            ax[r_, c_].axis("off")
            if r_ == 0:
                ax[r_, c_].set_title(cols[c_], fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "diffusion_examples.png", dpi=150)
    print(f"saved {out}/diffusion_eval.json and diffusion_examples.png")


if __name__ == "__main__":
    main()
