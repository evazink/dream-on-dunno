| Condition | Seeds | Reached 80% | Steps to 80% (mean ± std) | Final success | AUC | Wall-clock, min |
|---|---|---|---|---|---|---|
| PPO, state + current frame | 1 | 1/1 | 58,000 | 0.88 ± 0.00 | 0.529 ± 0.000 | 120.3 |
| Dream On: PPO + subgoal + L2 reward | 3 | 2/3 | 80,000 ± 14,000 | 0.53 ± 0.30 | 0.276 ± 0.157 | 196.5 |
| PPO, state + diffusion subgoal (sparse reward) | 1 | 0/1 | — | 0.12 ± 0.00 | 0.024 ± 0.000 | 288.5 |
| Dream On: PPO + subgoal + SSIM reward | 3 | 0/3 | — | 0.09 ± 0.13 | 0.033 ± 0.024 | 339.6 |
| PPO, state only | 8 | 4/8 | 36,500 ± 4,975 | 0.50 ± 0.50 | 0.399 ± 0.400 | 5.7 |
| SAC, state only | 1 | 0/1 | — | 0.00 ± 0.00 | 0.016 ± 0.000 | 20.8 |
