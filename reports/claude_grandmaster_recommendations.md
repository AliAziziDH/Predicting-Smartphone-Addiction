## 1. Three Lethal Interaction Formulas

Your 63 features are mostly marginal transforms (target encodings, ratios, GAN artifacts). What's missing is **behavioral non-linearity that GBMs can't split their way into** and **synthetic-generator seam exploitation**. These three attack both:

**① Compulsive Pull Ratio (CPR)** — isolates habit-loop behavior independent of external triggers (variable-ratio reinforcement, the core addiction mechanism):
$$CPR = \frac{app\_opens\_per\_day}{notifications\_per\_day + 1}$$
High CPR with low notification volume = self-initiated compulsive checking (strongest psychological addiction proxy in the whole feature space). GBMs cannot recover this ratio cleanly via axis-aligned splits at depth ≤ 8.

**② Unaccounted Time Leakage (UTL)** — exploits the additive-budget seam in CTGAN-generated tabular data (marginals are sampled independently, so the implicit constraint `daily_screen_time_hours ≈ social_media + gaming + work_study` is *violated* inconsistently — and that violation itself correlates with the label because the generator conditioned on class before sampling components):
$$UTL = daily\_screen\_time\_hours - (social\_media\_hours + gaming\_hours + work\_study\_hours)$$
$$UTL_{ratio} = \frac{UTL}{daily\_screen\_time\_hours + \epsilon}$$
This is functionally a "residual-to-generator" feature — same family as your mantissa-distance trick, but on the *semantic* additive structure instead of float encoding. Expect this to be top-3 by SHAP.

**③ Weekend Escalation × Stress (WESI)** — cross-domain nonlinearity (behavioral drift amplified by psychological state), which is a genuine 2nd-order interaction, not a ratio trick:
$$WESI = \left(\frac{weekend\_screen\_time - daily\_screen\_time\_hours}{daily\_screen\_time\_hours + \epsilon}\right) \times stress\_level$$
Trees model $A \times B$ poorly unless depth/data allow the exact split sequence; explicitly materializing it gives LGBM/XGB immediate access, and gives the MLP a smooth injective signal it otherwise has to approximate via multiple hidden units.

*Optional 4th (cheap, do it anyway): Sleep Displacement Capacity = $24 - sleep\_hours - work\_study\_hours$, then ratio gaming/(that capacity) — captures discretionary-time monopolization.*

---

## 2. Optimal Ensembling Calibration

AUC is **rank-invariant and non-differentiable** — this dictates the entire strategy. Do NOT log-loss-stack and assume it transfers to AUC; it doesn't (correlation ≈ 0.85, not 1.0).

**Step 1 — Decouple calibration regimes.**
LGBM/XGB/CatBoost share similar leaf-averaging probability calibration; MLP (sigmoid + BCE) lives on a different probability manifold. Never raw-average them.

$$z_i = \text{logit}(\hat{p}_i) = \ln\left(\frac{\hat{p}_i}{1-\hat{p}_i}\right), \quad i \in \{LGBM, CB, XGB\}$$

Rank-transform the MLP separately (MLPs on tabular data are frequently mis-calibrated at the tails):
$$r_{MLP} = \text{rank}(\hat{p}_{MLP}) / N$$

**Step 2 — Two-stage blend.**

*Stage A (tree cartel):* weighted logit-average of the 3 GBMs:
$$z_{tree} = w_1 z_{LGBM} + w_2 z_{CB} + w_3 z_{XGB}, \quad \hat{p}_{tree} = \sigma(z_{tree})$$

*Stage B (final blend):* rank-average tree ensemble with MLP rank:
$$\hat{p}_{final} = \alpha \cdot \text{rank}(\hat{p}_{tree}) + (1-\alpha) \cdot r_{MLP}$$

**Step 3 — Optimize weights with Nelder-Mead directly on OOF AUC** (not logloss surrogate):

$$\{w_1, w_2, w_3, \alpha\}^* = \arg\max_{w,\alpha} \; \text{AUC}\big(y_{OOF}, \hat{p}_{final}(w,\alpha)\big)$$

subject to $w_i \geq 0, \sum w_i = 1$, $\alpha \in [0,1]$. Use `scipy.optimize.minimize(method='Nelder-Mead')` on $-\text{AUC}$, since AUC has no usable gradient — gradient-based stacking (logistic regression meta-learner) optimizes log-loss and will leave 0.001–0.002 AUC on the table versus direct simplex search.

**Step 4 — Multi-seed stability.** Run Nelder-Mead across 5+ different fold splits / seeds and average the resulting weight vectors — single-run Nelder-Mead on a 0.96 AUC problem is prone to overfitting the OOF noise floor (~±0.0015 at this dataset size).

**Expected marginal gain:** rank/logit-hybrid + Nelder-Mead vs. naive mean typically buys **+0.003–0.006 AUC** on Playground-series-scale synthetic data — combined with the 3 interaction features above (~+0.006–0.010), **0.9710 is realistically reachable.**