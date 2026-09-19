# Next-Step Research Plan: Training-Free Refinement for Few-Step Diffusion Sampling

## 0. Current Status

The current sampler uses

\[
r_k = v_k - v_{k-1}
\]

as a numerical refinement proposal, followed by a consistency-based feasibility constraint.

The current constrained sampler approximately solves

\[
\min_{\delta_k}\|\delta_k-r_k\|_2^2
\]

subject to

\[
\nabla S_k(x_k)^\top \delta_k \le 0,
\]

where

\[
S_k(x_k)
=
\frac12
\left\|
\hat x_0^{(k)}
-
\operatorname{sg}(\hat x_0^{(k-1)})
\right\|^2.
\]

A finite trust-region check is then applied:

\[
S_k(x_k+\eta_k\delta_k)
\le
S_k(x_k)+\tau.
\]

If violated,

\[
\eta_k \leftarrow \beta\eta_k.
\]

Current observations:

1. Naive velocity-difference refinement can strongly increase \(x_0\)-consistency error.
2. VJP-based projection removes the first-order consistency-increasing component.
3. Finite-step checking is still necessary because first-order feasibility does not guarantee finite-step feasibility.
4. Trust-region backtracking sometimes shrinks or fully rejects a correction.
5. The current trajectory subspace is redundant because
   \[
   r_k=v_k-v_{k-1}\in\operatorname{span}\{v_k,v_{k-1}\}.
   \]
6. We have not yet established that preserving \(x_0\)-consistency improves final image quality.

The next goal is **not to add more machinery**. The next goal is to verify which parts of the current method actually matter.

---

# 1. Main Research Question

> Can additional inference-time trajectory correction improve a few-step distilled sampler while preventing the correction from entering locally unreliable regions of the student dynamics?

Working hypothesis:

\[
\boxed{
\text{useful refinement}
+
\text{local reliability constraint}
>
\text{naive refinement}
}
\]

The first reliability proxy being tested is \(x_0\)-prediction consistency.

---

# 2. Phase I — Establish the Core Baselines

Use exactly the same prompts, seeds, checkpoint, native timestep schedule, CFG, resolution, scheduler, and precision.

Compare four methods.

## A. Native Sampler

\[
x_{k+1}=\Phi_\theta(x_k).
\]

Purpose: main quality and latency baseline.

## B. Naive Velocity Refinement

\[
r_k=v_k-v_{k-1},
\]

\[
v_k^{\text{ref}}=v_k+\lambda_k r_k.
\]

No geometric constraint and no trust region.

Purpose: measure how much naive numerical extrapolation helps or hurts.

## C. First-Order Constrained Refinement

Use

\[
g_k=\nabla_{x_k}S_k(x_k).
\]

Solve

\[
\min_\delta \frac12\|\delta-r_k\|^2
\]

subject to

\[
g_k^\top\delta\le0.
\]

Closed form:

\[
\delta_k=
\begin{cases}
r_k,&g_k^\top r_k\le0,\\[2mm]
r_k-\dfrac{g_k^\top r_k}{\|g_k\|^2+\epsilon}g_k,&g_k^\top r_k>0.
\end{cases}
\]

Purpose: isolate the contribution of the first-order projection.

## D. Constrained Refinement + Trust Region

After projection, verify

\[
S_k(x_k+\eta_k\delta_k)
\le
S_k(x_k)+\tau.
\]

If violated:

\[
\eta_k\leftarrow\beta\eta_k.
\]

Purpose: isolate the contribution of finite-step feasibility.

---

# 3. Phase I Experiment Grid

Do not use a hand-written prompt set for the main benchmark.

Use an established prompt source so the evaluation is reproducible and comparable.

## Preferred Prompt Sources

### Option A — PickScore / Pick-a-Pic Prompt Set

Use prompts sampled from the Pick-a-Pic / PickScore evaluation ecosystem.

Advantages:

- directly aligned with the initial reward model, PickScore;
- prompts reflect realistic user preferences and text-to-image requests;
- convenient for preference-oriented evaluation;
- suitable for comparing small quality differences between sampling methods.

This should be the **default prompt source for the first large-scale benchmark**.

Recommended protocol:

- sample a fixed subset once;
- save the exact prompt list;
- use the same prompt subset for every method;
- avoid selecting prompts based on observed model performance.

Initial scale:

- debugging: 10–20 prompts;
- pilot benchmark: 100 prompts;
- stronger benchmark: 300–500 prompts.

Use multiple seeds per prompt:

\[
N_{\text{seed}} = 4
\]

initially.

Thus, for 100 prompts:

\[
100\times4=400
\]

samples per method.

---

### Option B — DrawBench

Use DrawBench as a complementary benchmark because it contains challenging compositional and linguistic cases.

Useful categories include:

- counting;
- spatial relations;
- color binding;
- compositional prompts;
- rare words;
- text rendering;
- complex descriptions.

DrawBench should be treated primarily as a **hard qualitative / compositional benchmark**, not necessarily as the only large-scale quantitative prompt source.

Recommended usage:

1. PickScore/Pick-a-Pic prompts for the primary quantitative benchmark.
2. DrawBench for stress testing and qualitative figures.

---

## Fixed Prompt Splits

Create explicit files such as:

```text
prompts/
  pickscore_debug.txt
  pickscore_eval_100.txt
  pickscore_eval_500.txt
  drawbench.txt
```

Do not regenerate the subset between experiments.

Store:

- prompt text;
- prompt ID if available;
- benchmark source;
- category if available.

This prevents accidental prompt-selection bias.

---

# 4. Metrics to Log Per Sampling Step


## 4.1 Proposal Magnitude

\[
\rho_r=\frac{\|r_k\|}{\|v_k\|}.
\]

## 4.2 Applied Correction Magnitude

\[
\rho_\delta=\frac{\|\lambda_k\delta_k\|}{\|v_k\|}.
\]

## 4.3 State-Space Correction

Log

\[
\|\Delta x_k^{\text{corr}}\|
\]

and

\[
\rho_x
=
\frac{\|\Delta x_k^{\text{corr}}\|}
{\|\Delta x_k^{\text{native}}\|}.
\]

## 4.4 Pre-Projection Alignment

\[
c_k^{\text{pre}}
=
\frac{r_k^\top g_k}{\|r_k\|\|g_k\|}.
\]

## 4.5 Post-Projection Alignment

\[
c_k^{\text{post}}
=
\frac{\delta_k^\top g_k}{\|\delta_k\|\|g_k\|}.
\]

For an active projection,

\[
c_k^{\text{post}}\approx0.
\]

## 4.6 Constraint Retention

\[
\rho_{\text{constraint}}
=
\frac{\|\delta_k\|}{\|r_k\|}.
\]

## 4.7 Consistency Before/After

\[
S_k^{\text{before}}=S_k(x_k),
\]

\[
S_k^{\text{after}}
=
S_k(x_k+\Delta x_k^{\text{corr}}).
\]

Then

\[
\rho_S
=
\frac{S_k^{\text{after}}}{S_k^{\text{before}}}.
\]

## 4.8 Trust-Region Statistics

Log:

- number of shrinks;
- final trust-region scale;
- rejection indicator.

Aggregate:

\[
P_{\text{reject}}
=
\frac{\#\text{rejected corrections}}
{\#\text{active corrections}}.
\]

---

# 5. Final-Image Evaluation

The evaluation stack should be implemented incrementally.

## Stage 1 — Primary Reward Model: PickScore

Use **PickScore first** as the main automated quality/preference metric.

For every generated image \(x\) and prompt \(c\), compute

\[
R_{\text{Pick}}(x,c).
\]

For each prompt and seed, compare refined sampling against the native sampler:

\[
\Delta R_{\text{Pick}}
=
R_{\text{Pick}}(x_{\text{refined}},c)
-
R_{\text{Pick}}(x_{\text{native}},c).
\]

Report:

- mean PickScore;
- median PickScore;
- mean paired improvement;
- fraction of samples improved;
- fraction degraded;
- confidence interval over prompts/seeds.

The **paired comparison** is especially important because every method uses the same prompt and seed.

Primary metric:

\[
\boxed{
\Delta R_{\text{Pick}}
}
\]

rather than only absolute reward.

---

## Stage 2 — Prepare the Evaluation Interface for Additional Metrics

The evaluation code should expose a common interface:

```python
scores = evaluator(
    images=images,
    prompts=prompts,
    metrics=[
        "pickscore",
        "hpsv2",
        "clipscore",
        "aesthetic",
    ],
)
```

Initially, only `pickscore` needs to be fully enabled.

The remaining metrics should be easy to activate without changing the generation pipeline.

---

## HPSv2

Prepare support for HPSv2 as a second human-preference-oriented reward model.

Use it to answer:

> Does an improvement measured by PickScore transfer to another independently trained preference model?

Eventually report

\[
\Delta R_{\text{HPSv2}}
=
R_{\text{HPSv2}}(x_{\text{refined}},c)
-
R_{\text{HPSv2}}(x_{\text{native}},c).
\]

Agreement between PickScore and HPSv2 will make the evaluation substantially stronger.

---

## CLIPScore

Prepare support for CLIPScore as a prompt-alignment metric.

Its role is different from preference rewards.

Use it mainly to detect whether refinement changes semantic alignment:

\[
\Delta R_{\text{CLIP}}
=
R_{\text{CLIP}}(x_{\text{refined}},c)
-
R_{\text{CLIP}}(x_{\text{native}},c).
\]

Interpretation:

- PickScore/HPSv2: overall human preference;
- CLIPScore: text-image semantic alignment.

Do not treat CLIPScore as the sole image-quality metric.

---

## Aesthetic Score

Prepare an aesthetic predictor for prompt-independent visual quality.

Compute

\[
R_{\text{aes}}(x).
\]

This helps distinguish two possible effects:

\[
\text{semantic improvement}
\]

from

\[
\text{visual / aesthetic improvement}.
\]

Eventually report both

\[
\Delta R_{\text{aes}}
\]

and prompt-conditioned reward changes.

---

## Recommended Metric Priority

Implement and validate metrics in this order:

1. **PickScore**
2. **HPSv2**
3. **CLIPScore**
4. **Aesthetic score**

The first experimental conclusions should rely mainly on PickScore.

Do not delay the first benchmark waiting for all reward models to be integrated.

---

## Diversity Metrics

Keep diversity evaluation separate from reward metrics.

For the same prompt across multiple seeds, prepare:

- pairwise LPIPS;
- pairwise DINO feature distance;
- CLIP image embedding distance;
- feature covariance effective rank.

For \(N\) samples from one prompt,

\[
D_{\text{pair}}
=
\frac{2}{N(N-1)}
\sum_{i<j}
d(x_i,x_j).
\]

The main question is:

> Does refinement improve reward while preserving seed-level diversity?

---

## Distribution-Level Metrics

These are lower priority for the current stage.

Later, for larger experiments, consider:

- FID;
- DINO feature statistics;
- precision / recall for generative models.

Do not make FID a blocking requirement for the first method-validation cycle.

---

# 6. Critical Correlation Experiment


For each corrected sample define

\[
\Delta S=S_{\text{after}}-S_{\text{before}},
\]

and for a quality score \(Q\),

\[
\Delta Q=Q_{\text{refined}}-Q_{\text{native}}.
\]

Measure

\[
\operatorname{corr}(\Delta S,\Delta Q).
\]

Questions:

1. Does increasing consistency error correlate with worse image quality?
2. Does preserving/reducing consistency correlate with less degradation?
3. Does lower consistency actually predict improvement, or merely stability?

### Outcome A

\[
\Delta S>0\Rightarrow\Delta Q<0
\]

consistently.

Then \(x_0\)-consistency is a useful reliability signal.

### Outcome B

Consistency predicts catastrophic degradation but not quality improvement.

Then treat \(S_k\) as a **safety constraint**, not an optimization objective.

### Outcome C

No meaningful correlation.

Then replace the reliability field.

---

# 7. Phase II — Strength Sweep

Sweep

\[
\lambda\in\{0.025,0.05,0.10,0.15,0.20,0.30\}.
\]

Measure:

- quality;
- consistency ratio;
- correction retention;
- rejection rate;
- diversity.

Expected tradeoff:

\[
\lambda\uparrow
\Rightarrow
\text{more potential refinement}
\]

but also

\[
\lambda\uparrow
\Rightarrow
\text{more constraint activation/rejection}.
\]

---

# 8. Phase III — Step-Dependent Strength

Test schedules such as

\[
[0,0.20,0.15,0.05],
\]

\[
[0,0.15,0.10,0.05],
\]

\[
[0,0.20,0.10,0].
\]

A more principled schedule is

\[
\lambda_k
=
\frac{\lambda_0}
{1+\alpha|\Delta\sigma_k|}.
\]

Hypothesis:

> late coarse steps require more conservative correction because a small velocity correction can induce a large state displacement.

---

# 9. Phase IV — Verify Whether Trust Region Is Necessary

Compare

\[
\text{projection only}
\]

versus

\[
\text{projection + finite check}.
\]

Report

\[
P_{\text{FO-fail}}
=
P\left[
S(x+\delta)>S(x)
\mid
g^\top\delta\le0
\right].
\]

If this probability is significant, the trust-region mechanism is justified.

---

# 10. Phase V — Remove the Redundant Subspace Component

The current subspace

\[
U_k=\operatorname{span}\{v_k,v_{k-1}\}
\]

does not constrain

\[
r_k=v_k-v_{k-1}.
\]

Therefore:

## Recommended option

Temporarily simplify the method to

\[
\boxed{
\text{proposal}
+
\text{reliability projection}
+
\text{trust region}
}
\]

until the core idea is validated.

Only later test genuine model-supported subspaces:

- hidden-state PCA;
- Jacobian singular vectors;
- final-head span;
- attention feature subspace;
- empirical native-state covariance;
- effective-rank-adaptive basis.

---

# 11. Phase VI — Test Alternative Refinement Proposals

The framework should accept arbitrary proposal \(r_k\).

Test:

## Velocity Difference

\[
r_k=v_k-v_{k-1}.
\]

## Normalized Velocity Difference

\[
r_k
=
\frac{v_k}{\|v_k\|}
-
\frac{v_{k-1}}{\|v_{k-1}\|}.
\]

## Secant / Curvature Estimate

\[
r_k
=
\frac{v_k-v_{k-1}}
{t_k-t_{k-1}}.
\]

## Momentum-Style Extrapolation

\[
v_k^{\text{ref}}
=
v_k+\beta_k(v_k-v_{k-1}).
\]

## \(x_0\)-Prediction Extrapolation

Construct the proposal in clean-prediction space and map it back to the sampler parameterization.

General target:

\[
\boxed{
r_k\rightarrow\operatorname{SafeRefine}(r_k)
}
\]

rather than tying the method to one proposal.

---

# 12. Phase VII — Test Alternative Reliability Fields

Only after benchmarking the current method.

## A. Current \(x_0\)-Consistency

\[
S_k^{x_0}
=
\frac12
\left\|
\hat x_0^{(k)}
-
\operatorname{sg}(\hat x_0^{(k-1)})
\right\|^2.
\]

## B. Velocity Sensitivity

\[
S_k^v
=
\|J_{v_\theta}(x_k,t_k)\|_F^2.
\]

## C. \(x_0\)-Jacobian Sensitivity

\[
S_k^J
=
\|J_{\hat x_0}(x_k,t_k)\|_F^2.
\]

Use Hutchinson:

\[
\|J\|_F^2
=
\mathbb E_z\|J^\top z\|^2.
\]

## D. Hidden-State Sensitivity

For hidden representation \(h_k\),

\[
S_k^h
=
\|h_k-h_{k-1}\|^2.
\]

---

# 13. Phase VIII — Separate Proposal and Reliability Signals

Design principle:

\[
\boxed{
\text{proposal signal}
\neq
\text{reliability signal}
}
\]

The proposal answers:

> In what direction do we want to correct the sampler?

The reliability field answers:

> Which part of that correction is locally safe?

Avoid constructing both from exactly the same quantity unless there is a strong justification.

---

# 14. Phase IX — Evaluate Multiple NFEs

## 4-Step

Primary development target.

## 2-Step

Test after 4-step is stable.

## 1-Step

Do not target yet. Velocity-history refinement is impossible and requires a different source:

- internal hidden-state geometry;
- multiple block features;
- Jacobian directions;
- local model probing.

---

# 15. Phase X — Diversity Check

For each prompt, sample many seeds.

Compare Native vs Refined.

Measure

\[
\Delta D
=
D_{\text{refined}}-D_{\text{native}}.
\]

If

\[
\Delta D\ll0,
\]

the method may only be contracting modes.

---

# 16. Phase XI — Computational Cost

Track:

- forward NFE;
- VJP/backward count;
- trust-region extra evaluations;
- latency;
- peak VRAM.

Compare against compute-matched baselines.

Important question:

> Is constrained refinement better than spending the same extra compute on a simpler strategy?

---

# 17. Mandatory Fair Baselines

At minimum compare against:

1. native distilled sampler;
2. naive velocity extrapolation;
3. stronger native sampler / extra NFE if available;
4. Heun-like correction when applicable;
5. compute-matched resampling or Best-of-N if feasible.

---

# 18. Recommended Immediate Experiment Order

## Experiment 1

Run:

- Native
- Naive
- Projection-only
- Projection + Trust Region

on 10 prompts × 4 seeds.

Goal: verify implementation and diagnostics.

## Experiment 2

Run the first quantitative benchmark on a **fixed PickScore/Pick-a-Pic prompt subset**.

Recommended first scale:

- 100 prompts;
- 4 seeds per prompt.

Primary metric:

- PickScore.

Also log:

- consistency statistics;
- correction retention;
- trust-region rejection rate;
- latency / VRAM.

Goal:

> determine whether constrained refinement produces a positive paired PickScore improvement over the native sampler.

Do not block this experiment on HPSv2, CLIPScore, or aesthetic integration.

## Experiment 3

Correlation:

\[
\Delta S
\quad\text{vs}\quad
\Delta Q.
\]

Goal: validate \(x_0\)-consistency as a reliability signal.

## Experiment 4

Strength sweep.

Goal: identify quality / constraint / rejection tradeoff.

## Experiment 5

Step-dependent schedule.

Goal: determine whether late-step correction should be weaker.

## Experiment 6

Replace the reliability field only if \(x_0\)-consistency is weakly correlated with quality or only acts as a failure detector.

---

# 19. Decision Tree

## If constrained refinement improves quality

Continue with:

- strength scheduling;
- alternative proposals;
- better reliability fields;
- 2-step generalization.

## If constrained refinement preserves quality but does not improve it

Interpret the reliability field as a safety filter.

Search for a better refinement proposal.

## If naive refinement improves quality more than constrained refinement

Relax the constraint:

\[
g^\top\delta
\le
\epsilon_{\text{relax}}
\]

or allow

\[
S_{\text{after}}
\le
(1+\tau)S_{\text{before}}.
\]

## If consistency has no relation to quality

Replace \(S_k\).

---

# 20. Possible Relaxed Constraint

Sweep

\[
S_{\text{after}}
\le
(1+\tau)S_{\text{before}}
\]

with

\[
\tau
\in
\{0,0.01,0.025,0.05,0.10\}.
\]

This gives a quality–reliability frontier instead of forcing perfect non-increase.

---

# 21. Core Tables for the Paper

## Table 1 — Main Results

| Method | NFE / Cost | ImageReward | HPSv2 | CLIPScore | Diversity | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Native | | | | | | |
| Naive Refinement | | | | | | |
| Projection | | | | | | |
| Projection + TR | | | | | | |

## Table 2 — Trajectory Diagnostics

| Method | Consistency Ratio | Constraint Retention | Reject Rate | Correction Norm |
|---|---:|---:|---:|---:|
| Naive | | | | |
| Projection | | | | |
| Projection + TR | | | | |

## Table 3 — Reliability Ablation

| Reliability Field | Quality | Consistency | Reject Rate | Cost |
|---|---:|---:|---:|---:|
| \(x_0\) consistency | | | | |
| Velocity sensitivity | | | | |
| \(x_0\)-Jacobian | | | | |
| Hidden consistency | | | | |

---

# 22. Main Figures

## Figure 1 — Method Diagram

\[
\text{native state}
\rightarrow
\text{proposal}
\rightarrow
\text{VJP projection}
\rightarrow
\text{finite check}
\rightarrow
\text{accept/shrink/reject}
\]

## Figure 2 — Consistency Across Steps

Plot \(S_k\) for:

- native;
- naive;
- constrained.

## Figure 3 — Quality vs Consistency Drift

Scatter:

\[
\Delta S
\]

against

\[
\Delta Q.
\]

## Figure 4 — Strength Tradeoff

x-axis:

\[
\lambda.
\]

y-axes:

- quality;
- rejection rate;
- diversity;
- consistency ratio.

---

# 23. What Not to Do Yet

Do not add all of the following before the basic experiments are complete:

- Fisher information machinery;
- large PCA subspaces;
- random projection;
- effective-rank adaptation;
- external reward guidance;
- 1-step models;
- training;
- additional learned modules.

First answer:

\[
\boxed{
\text{Does constrained refinement actually improve the final samples?}
}
\]

---

# 24. Short-Term Milestone

A successful first milestone is:

> On a 4-step distilled model, naive trajectory extrapolation produces measurable consistency drift and quality degradation for a subset of samples, while VJP-based constrained refinement with finite trust-region verification preserves local consistency and yields a statistically measurable improvement over the native sampler or over naive refinement at comparable compute.

---

# 25. Current Working Name

## Consistency-Constrained Sampling Refinement (CCSR)

Core algorithm:

\[
r_k=v_k-v_{k-1},
\]

\[
g_k=\nabla_{x_k}S_k,
\]

\[
\delta_k
=
\operatorname{Proj}_{g_k^\top\delta\le0}(r_k),
\]

followed by

\[
S_k(x_k+\eta_k\delta_k)
\le
S_k(x_k)+\tau.
\]

Avoid calling it “manifold-preserving” until the reliability field is shown to correlate with actual off-manifold behavior.

---

# 26. Immediate TODO Checklist

- [ ] Finalize the four core baselines.
- [ ] Verify identical native outputs across experiment modes when correction is disabled.
- [ ] Add automatic aggregate logging.
- [ ] Download / prepare fixed PickScore or Pick-a-Pic prompt subsets.
- [ ] Prepare DrawBench prompts as a complementary stress-test set.
- [ ] Save immutable prompt files with source metadata.
- [ ] Run 10–20 prompt debugging benchmark.
- [ ] Run 100-prompt × 4-seed PickScore benchmark.
- [ ] Implement PickScore as the primary reward model.
- [ ] Prepare pluggable evaluator interface for HPSv2.
- [ ] Prepare pluggable evaluator interface for CLIPScore.
- [ ] Prepare pluggable evaluator interface for aesthetic score.
- [ ] Add diversity metrics after the first PickScore benchmark.
- [ ] Plot consistency ratio by timestep.
- [ ] Plot correction retention by timestep.
- [ ] Measure trust-region rejection rate.
- [ ] Correlate \(\Delta S\) with \(\Delta Q\).
- [ ] Sweep \(\lambda\).
- [ ] Sweep trust-region tolerance \(\tau\).
- [ ] Compare constant vs step-dependent strength.
- [ ] Decide whether \(x_0\)-consistency is a constraint, objective, or unsuitable proxy.
- [ ] Only then test alternative reliability fields.
- [ ] Only then test alternative low-rank subspaces.
- [ ] Only then move from 4-step to 2-step.
