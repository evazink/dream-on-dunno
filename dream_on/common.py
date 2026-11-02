"""
Dream On v2 — shared utilities.

* preprocess_frame   : identical preprocessing to src/data_utils/dataloader.py
                       (480x480 RGB -> bilinear 120x120 -> grayscale -> [-1, 1])
* SubgoalPredictor   : loads a ConditionalUNet checkpoint trained by src/diffusion/train.py
                       and generates the k-step-ahead frame with proper DDIM sampling.
* LegacyPredictor    : re-implementation of the original src/diffusion/inference.py
                       (single UNet pass at t=0 on [I, I] in [0,1]); kept only to
                       quantify how far its output is from a real sample.
* ssim / rmse        : image distances used for reward shaping and evaluation.
* DreamOnWrapper     : FrankaKitchen wrapper: state (+ image) observation, optional
                       diffusion subgoal, optional dense pixel-space shaping reward.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.diffusion.model import ConditionalUNet  # noqa: E402

IMG = 120


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Preprocessing (must match the diffusion training data exactly)
# --------------------------------------------------------------------------- #
def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 render -> (1, 120, 120) float tensor in [-1, 1]."""
    x = torch.from_numpy(np.ascontiguousarray(frame)).float().permute(2, 0, 1).unsqueeze(0)
    if x.shape[-1] != IMG or x.shape[-2] != IMG:
        x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False)
    x = TF.rgb_to_grayscale(x)  # [1, 1, H, W], still 0..255
    x = x / 127.5 - 1.0
    return x[0]


# --------------------------------------------------------------------------- #
# Image distances (inputs in [-1, 1], shape [..., H, W])
# --------------------------------------------------------------------------- #
def rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Length-normalised L2 distance (root-mean-square error over pixels)."""
    return torch.sqrt(torch.mean((a - b) ** 2, dim=(-2, -1)))


def _gauss_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    g = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]  # [1, 1, size, size]


def ssim(a: torch.Tensor, b: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Mean SSIM (Wang et al., 2004) for single-channel images in [-1, 1]."""
    if a.dim() == 2:
        a, b = a[None, None], b[None, None]
    elif a.dim() == 3:
        a, b = a[:, None], b[:, None]
    w = _gauss_window(device=a.device)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a = F.conv2d(a, w)
    mu_b = F.conv2d(b, w)
    s_aa = F.conv2d(a * a, w) - mu_a ** 2
    s_bb = F.conv2d(b * b, w) - mu_b ** 2
    s_ab = F.conv2d(a * b, w) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * s_ab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2)
    return (num / den).mean(dim=(-3, -2, -1))


# --------------------------------------------------------------------------- #
# Diffusion predictors
# --------------------------------------------------------------------------- #
def _load_unet(ckpt_path: str, device: torch.device) -> tuple[ConditionalUNet, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["model_config"]
    model = ConditionalUNet(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    targs = ckpt.get("training_args", {}) or {}
    return model, targs


class SubgoalPredictor:
    """Conditional DDIM sampler for the k-step-ahead frame (Song et al., 2021, eta=0)."""

    def __init__(self, ckpt_path: str, ddim_steps: int = 10, device: str = "auto", seed: int = 0):
        self.device = pick_device(device)
        self.model, targs = _load_unet(ckpt_path, self.device)
        T = int(targs.get("diffusion_timesteps", 1000))
        b0 = float(targs.get("beta_start", 1e-4))
        b1 = float(targs.get("beta_end", 0.02))
        betas = torch.linspace(b0, b1, T, dtype=torch.float64)
        self.alpha_bar = torch.cumprod(1.0 - betas, dim=0).float().to(self.device)
        self.T = T
        self.set_steps(ddim_steps)
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.k_step_future = int(targs.get("k_step_future", -1))

    def set_steps(self, n: int) -> None:
        self.ddim_steps = n
        self.timesteps = torch.linspace(self.T - 1, 0, n).round().long().tolist()

    @torch.no_grad()
    def predict(self, cond: torch.Tensor) -> torch.Tensor:
        """cond: [B, 1, 120, 120] or [1, 120, 120] in [-1, 1] -> predicted future frame, same shape."""
        squeeze = cond.dim() == 3
        if squeeze:
            cond = cond[None]
        cond = cond.to(self.device)
        x = torch.randn(cond.shape, generator=self.gen).to(self.device)
        ts = self.timesteps
        x0 = x
        for i, t in enumerate(ts):
            ab_t = self.alpha_bar[t]
            ab_prev = self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=self.device)
            t_vec = torch.full((cond.shape[0],), t, device=self.device, dtype=torch.long)
            eps = self.model(torch.cat([x, cond], dim=1), t_vec)
            x0 = ((x - torch.sqrt(1 - ab_t) * eps) / torch.sqrt(ab_t)).clamp(-1, 1)
            x = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * eps
        out = x0.cpu()
        return out[0] if squeeze else out


class LegacyPredictor:
    """Exact behaviour of the original src/diffusion/inference.py::Inference.predict."""

    def __init__(self, ckpt_path: str, device: str = "auto"):
        self.device = pick_device(device)
        self.model, _ = _load_unet(ckpt_path, self.device)

    @torch.no_grad()
    def predict(self, cond: torch.Tensor) -> torch.Tensor:
        squeeze = cond.dim() == 3
        if squeeze:
            cond = cond[None]
        # original: uint8 grayscale -> /255 -> duplicate channel -> t=0 -> output*255 -> uint8
        u8 = ((cond + 1.0) * 127.5).round().clamp(0, 255)
        x = (u8 / 255.0).to(self.device)
        x = torch.cat([x, x], dim=1)
        t = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)
        out = self.model(x, t).cpu()
        out_u8 = (out * 255.0).clamp(0, 255).to(torch.uint8).float()
        out = out_u8 / 127.5 - 1.0
        return out[0] if squeeze else out


# --------------------------------------------------------------------------- #
# Environment wrapper
# --------------------------------------------------------------------------- #
def _flatten_state(obs: dict) -> np.ndarray:
    parts = [np.asarray(obs["observation"], dtype=np.float32).ravel()]
    for key in ("achieved_goal", "desired_goal"):
        sub = obs[key]
        for k in sorted(sub.keys()):
            parts.append(np.asarray(sub[k], dtype=np.float32).ravel())
    return np.concatenate(parts)


class DreamOnWrapper(gym.Wrapper):
    """
    image_mode : 'none'      -> state only (59 + goal dims)
                 'current'   -> state + current 120x120 frame
                 'diffusion' -> state + diffusion-predicted subgoal frame G_t
    reward_mode: 'sparse'    -> environment reward (1 on task completion)
                 'l2'        -> sparse - lam * RMSE(I_{t+1}, G_t)
                 'ssim'      -> sparse - lam * (1 - SSIM(I_{t+1}, G_t))
    goal_every : the subgoal is re-sampled every `goal_every` env steps
                 (= horizon k the diffusion model was trained for) and held fixed in between.
    """

    def __init__(self, env, image_mode="none", reward_mode="sparse", lam=0.1,
                 goal_every=5, predictor: SubgoalPredictor | None = None):
        super().__init__(env)
        assert image_mode in ("none", "current", "diffusion")
        assert reward_mode in ("sparse", "l2", "ssim")
        if reward_mode != "sparse":
            assert image_mode == "diffusion", "dense pixel reward needs a diffusion subgoal"
        if image_mode == "diffusion":
            assert predictor is not None
        self.image_mode, self.reward_mode = image_mode, reward_mode
        self.lam, self.goal_every, self.predictor = lam, goal_every, predictor
        self.tasks = list(env.unwrapped.tasks_to_complete) if hasattr(env.unwrapped, "tasks_to_complete") else None
        obs, _ = env.reset(seed=0)
        state_dim = _flatten_state(obs).shape[0]
        self.img_dim = IMG * IMG if image_mode != "none" else 0
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (state_dim + self.img_dim,), np.float32)
        self._goal = None
        self._t = 0
        self._ep_sparse = 0.0

    # -- helpers --
    def _frame(self) -> torch.Tensor:
        return preprocess_frame(self.env.render())

    def _obs(self, obs: dict, img: torch.Tensor | None) -> np.ndarray:
        s = _flatten_state(obs)
        if self.image_mode == "none":
            return s
        vis = self._goal if self.image_mode == "diffusion" else img
        return np.concatenate([s, vis.numpy().ravel().astype(np.float32)])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._t, self._ep_sparse = 0, 0.0
        img = self._frame() if self.image_mode != "none" else None
        if self.image_mode == "diffusion":
            self._goal = self.predictor.predict(img)
        return self._obs(obs, img), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._t += 1
        self._ep_sparse += float(r)
        img = self._frame() if self.image_mode != "none" else None
        shaping = 0.0
        if self.image_mode == "diffusion":
            if self.reward_mode == "l2":
                d = float(rmse(img, self._goal))
                shaping = -self.lam * d
                info["goal_dist"] = d
            elif self.reward_mode == "ssim":
                d = 1.0 - float(ssim(img, self._goal))
                shaping = -self.lam * d
                info["goal_dist"] = d
            if self._t % self.goal_every == 0:
                self._goal = self.predictor.predict(img)
        info["sparse_reward"] = float(r)
        info["shaping"] = shaping
        done_tasks = info.get("episode_task_completions", [])
        if self.tasks is not None:
            info["is_success"] = float(len(done_tasks) >= len(self.tasks))
        return self._obs(obs, img), float(r) + shaping, term, trunc, info

    def goal_image(self):
        return self._goal
