# Training-Free Manifold-Preserving Sampling Refinement for Few-Step Diffusion Models

## 1. Motivation

Few-step and one-step distilled diffusion models are attractive because of their low inference cost, but they have a critical weakness:

> **small arbitrary corrections in latent/state space can easily move the trajectory off the student model's native state distribution.**

For a standard few-step sampler,

\[
x_{t_{k+1}} = \Phi_\theta(x_{t_k}, t_k, t_{k+1}, c),
\]

a naive refinement of the form

\[
x_{t_{k+1}}'
=
\Phi_\theta(x_{t_k})
+
d_k
\]

is dangerous even if

\[
\|d_k\| \ll 1.
\]

The problem is not merely the magnitude of \(d_k\), but whether the updated state remains in a region that the distilled student has actually learned to process.

This is especially severe for aggressively distilled models because:

- the native trajectory contains only a few states;
- the student is often trained on a narrow timestep schedule;
- state-distribution mismatch can compound immediately because there are only 1–4 denoising steps;
- arbitrary high-dimensional perturbations are likely to point outside the locally supported trajectory manifold.

Therefore, the refinement problem should be formulated as:

\[
\boxed{
\text{Improve the sampling trajectory without leaving the model-supported local geometry.}
}
\]

The target is not latent optimization or reward-guided search.

The target is a **new sampling rule**.

---

# 2. Key Principle

The central principle is:

\[
\boxed{
\text{Do not add arbitrary directions in ambient latent space.}
}
\]

Instead, refinement should satisfy two constraints:

1. the correction should lie in a **model-supported local subspace**;
2. the correction should preserve a local notion of **reliability / sensitivity / consistency**.

This leads to a general form

\[
d_k^\star
\in
\underbrace{\mathcal U_k}_{\text{model-supported subspace}}
\cap
\underbrace{T_{x_k}\mathcal S_k}_{\text{reliability-preserving tangent space}}.
\]

---

# 3. What Can Be Borrowed from Fisher-Preserving Guidance?

The paper **Fisher-Preserving Guidance: Training-Free Manifold Constraints for Safe Diffusion Control** introduces a useful geometric idea.

Their setting is diffusion policy / robotics rather than image generation, but the transferable part is the following:

## 3.1 Define a scalar reliability field

They define a Fisher-style sensitivity scalar over the current diffusion state,

\[
I_t(a_t;C)
=
\frac{1-\bar\alpha_t}{\bar\alpha_t}
\left\|
\nabla_C \epsilon_\theta(C,a_t,t)
\right\|_F^2.
\]

For fixed condition \(C\), this becomes a scalar field over \(a_t\).

A level set

\[
\mathcal S_{\kappa,t}
=
\left\{
a_t:
I_t(a_t;C)=\kappa
\right\}
\]

is interpreted as a local sensitivity-preserving surface.

Its normal direction is

\[
g_t
=
\nabla_{a_t} I_t(a_t;C).
\]

---

## 3.2 Project the proposed update into the tangent space

Given an arbitrary guidance vector \(u_t\), FPG removes the component parallel to the reliability-field normal:

\[
\Delta_t
=
u_t
-
\frac{u_t^\top g_t}
{\|g_t\|^2}
g_t.
\]

Therefore,

\[
g_t^\top \Delta_t = 0.
\]

By Taylor expansion,

\[
I_t(a_t+\Delta_t)
=
I_t(a_t)
+
O(\|\Delta_t\|^2).
\]

Thus the reliability statistic is preserved to first order.

---

# 4. Main Transferable Insight

The most useful idea is not specifically "Fisher information."

The important pattern is:

\[
\boxed{
\text{scalar reliability field}
\rightarrow
\text{isosurface}
\rightarrow
\text{tangent-space projection}
}
\]

For few-step image diffusion, we can replace their robotics-specific sensitivity measure by a statistic better aligned with the student model's native sampling geometry.

---

# 5. Why We Should Not Copy Their FDS Directly

In the robotics paper, the condition \(C\) represents observations such as images or navigation context.

Their sensitivity is approximately

\[
I_t
\propto
\left\|
\frac{\partial \hat a_0}
{\partial C}
\right\|_F^2.
\]

This measures how strongly the predicted action changes when the observation representation is perturbed.

For text-conditioned image diffusion, however,

\[
\left\|
\frac{\partial \hat x_0}
{\partial c}
\right\|_F^2
\]

does not necessarily characterize whether the current diffusion state is on or off the student's native state manifold.

Therefore, the geometry should be kept, but the reliability field should be redesigned.

---

# 6. Candidate Reliability Fields for Few-Step Distilled Diffusion

## 6.1 Candidate A: State Sensitivity

Define

\[
S_t(x_t)
=
\left\|
\frac{\partial \hat x_0(x_t,t,c)}
{\partial x_t}
\right\|_F^2.
\]

Interpretation:

- low/moderate sensitivity: the denoiser behaves stably under local state perturbations;
- abnormal sensitivity: small state changes produce large changes in predicted clean image;
- unusually high sensitivity may indicate that the sampler has moved into an unsupported region.

The local normal is

\[
g_t
=
\nabla_{x_t}S_t(x_t).
\]

A refinement direction \(d_t\) becomes

\[
d_t^\perp
=
d_t
-
\frac{
d_t^\top g_t
}{
\|g_t\|^2
}
g_t.
\]

This preserves state sensitivity to first order.

### Advantages

- direct analogue of FPG;
- mathematically clean;
- directly measures local denoiser stability.

### Drawbacks

- estimating the Jacobian norm can be expensive;
- differentiating the sensitivity field introduces second-order derivatives;
- low sensitivity does not automatically imply semantic correctness.

---

# 7. Candidate B: Denoising Consistency Field

This is likely more suitable for few-step distilled models.

Let the model predict

\[
\hat x_0^{(k)}
=
\hat x_0(x_{t_k},t_k,c).
\]

After one native step,

\[
x_{t_{k+1}}
=
\Phi_\theta(x_{t_k}),
\]

predict again:

\[
\hat x_0^{(k+1)}
=
\hat x_0(x_{t_{k+1}},t_{k+1},c).
\]

Define a local consistency score:

\[
S_k(x_{t_k})
=
\left\|
\hat x_0^{(k)}
-
\hat x_0^{(k+1)}
\right\|^2.
\]

A well-behaved trajectory should ideally satisfy approximate consistency:

\[
\hat x_0^{(k)}
\approx
\hat x_0^{(k+1)}.
\]

Large disagreement indicates that the trajectory is entering a region where the student dynamics are internally inconsistent.

The normal is

\[
g_k
=
\nabla_{x_{t_k}} S_k.
\]

A proposed sampling correction \(d_k\) is projected to

\[
d_k^\star
=
d_k
-
\frac{
d_k^\top g_k
}{
\|g_k\|^2
}
g_k.
\]

Then

\[
S_k(x_{t_k}+d_k^\star)
=
S_k(x_{t_k})
+
O(\|d_k^\star\|^2).
\]

This produces a **consistency-preserving correction**.

---

# 8. Candidate C: Native-Student Sensitivity Preservation

An even more conservative formulation is to avoid claiming that lower sensitivity means more on-manifold.

Instead, treat the native student trajectory as the reference distribution.

At native state \(x_{t_k}\), compute a local statistic

\[
S_k^{\text{native}}
=
S_k(x_{t_k}).
\]

The refinement should preserve that regime:

\[
S_k(x_{t_k}+\Delta_k)
\approx
S_k(x_{t_k}).
\]

First-order constraint:

\[
\nabla S_k(x_{t_k})^\top \Delta_k
=
0.
\]

This framing is weaker but easier to defend:

> the refinement is not assumed to recover the true data manifold; it only avoids leaving the local sensitivity regime naturally induced by the pretrained student sampler.

---

# 9. Second Important Idea from FPG: Low-Rank Projection

FPG observes that the model output head often induces a low-rank structure.

Writing approximately

\[
\epsilon_\theta
\approx
W h_\theta,
\]

the dominant variations lie in the span induced by \(W\).

Instead of operating in the full high-dimensional output space, the update can be projected into a lower-dimensional representation.

The general transferable idea is:

\[
\boxed{
\text{Do not refine in the full ambient latent space.}
}
\]

For image diffusion, define a local basis

\[
U_k
\in
\mathbb R^{D\times r},
\qquad
r \ll D.
\]

The correction is restricted to

\[
d_k
=
U_k\alpha_k.
\]

Possible constructions for \(U_k\):

- final prediction-head span;
- PCA/SVD of hidden features;
- local trajectory velocity span;
- previous velocity directions;
- attention-output subspace;
- low-rank covariance of model-supported perturbations;
- random orthogonal projection followed by model filtering;
- top directions from Jacobian-vector products;
- effective-rank-based adaptive subspace.

---

# 10. Combined Geometric Constraint

Let

\[
P_{U_k}
=
U_kU_k^\top.
\]

Given a proposed refinement direction \(r_k\), first restrict it to the supported subspace:

\[
\tilde r_k
=
P_{U_k}r_k.
\]

Let

\[
g_k
=
\nabla S_k(x_k).
\]

Project the reliability normal itself into the same subspace:

\[
\tilde g_k
=
P_{U_k}g_k.
\]

Then remove the sensitivity-changing component:

\[
r_k^\star
=
\tilde r_k
-
\frac{
\tilde r_k^\top \tilde g_k
}{
\|\tilde g_k\|^2+\epsilon
}
\tilde g_k.
\]

Thus,

\[
\boxed{
r_k^\star
\in
\operatorname{span}(U_k)
\cap
T_{x_k}\mathcal S_k
}
\]

where

\[
\mathcal S_k
=
\{x:S_k(x)=S_k(x_k)\}.
\]

---

# 11. Proposed Sampler

Let the base few-step sampler be

\[
x_{k+1}
=
\Phi_\theta(x_k).
\]

Assume we can construct a numerical refinement proposal

\[
r_k.
\]

The refined sampler becomes

\[
\boxed{
x_{k+1}
=
\Phi_\theta(x_k)
+
\lambda_k r_k^\star
}
\]

with

\[
r_k^\star
=
P_{\mathcal M_k}r_k,
\]

where

\[
P_{\mathcal M_k}
\approx
P_{U_k}
P_{g_k^\perp}.
\]

Conceptually:

\[
\boxed{
\text{refinement proposal}
\rightarrow
\text{model-supported projection}
\rightarrow
\text{consistency-preserving projection}
\rightarrow
\text{sampling update}
}
\]

---

# 12. Important Distinction: Refinement Is Not Guidance

The project should avoid framing \(r_k\) as external guidance.

Instead, \(r_k\) should be interpreted as a **numerical sampling correction**.

Examples:

- local curvature correction;
- solver truncation-error correction;
- previous-step velocity extrapolation;
- difference between first-order and second-order integration;
- clean-prediction inconsistency correction;
- trajectory self-consistency correction.

This makes the method a genuine **sampler**, rather than inference-time latent optimization.

---

# 13. Candidate Numerical Refinement Signals

## 13.1 Velocity Difference

For a flow model,

\[
v_k
=
v_\theta(x_k,t_k,c).
\]

A simple refinement signal is

\[
r_k
=
v_k-v_{k-1}.
\]

This approximates local change in the vector field.

Naively,

\[
v_k^{\text{ref}}
=
v_k+\lambda r_k.
\]

Our constrained version is

\[
v_k^{\text{ref}}
=
v_k+\lambda r_k^\star.
\]

Then

\[
x_{k+1}
=
x_k+h_kv_k^{\text{ref}}.
\]

---

# 14. Curvature-Based Refinement

Approximate trajectory curvature through finite differences:

\[
r_k
=
\frac{
v_k-v_{k-1}
}{
t_k-t_{k-1}
}.
\]

Then project

\[
r_k^\star
=
P_{\mathcal M_k}r_k.
\]

The sampler becomes

\[
x_{k+1}
=
x_k
+
h_kv_k
+
\beta_k h_k^2 r_k^\star.
\]

This resembles a second-order correction, but the correction is constrained to a locally reliable subspace.

---

# 15. Heun-Like Self-Correction

Predict:

\[
x_{k+1}^{E}
=
x_k+h_kv_k.
\]

Evaluate:

\[
v_{k+1}
=
v_\theta(x_{k+1}^{E},t_{k+1}).
\]

Standard Heun correction:

\[
r_k
=
\frac12(v_{k+1}-v_k).
\]

Our version:

\[
r_k^\star
=
P_{\mathcal M_k}r_k.
\]

Update:

\[
x_{k+1}
=
x_k
+
h_k
\left(
v_k+r_k^\star
\right).
\]

This can be described as a **manifold-preserving second-order sampler**.

---

# 16. Native-Timestep Constraint

A major issue with distilled models is that arbitrary intermediate timesteps can themselves be out-of-distribution.

If a student is distilled on

\[
t_4 \rightarrow t_3 \rightarrow t_2 \rightarrow t_1 \rightarrow t_0,
\]

querying at

\[
\frac{t_k+t_{k+1}}{2}
\]

may already be invalid.

Therefore, the safest formulation should initially enforce:

\[
\boxed{
\text{Only evaluate the model at native distilled timesteps.}
}
\]

This favors refinement signals based on:

- current and previous native velocities;
- current and previous clean predictions;
- native-step consistency;
- history-based trajectory geometry.

Avoid midpoint evaluations in the first version.

---

# 17. Trajectory Subspace

Instead of arbitrary low-rank features, a particularly natural subspace is constructed from model-generated trajectory directions.

Let

\[
V_k
=
[
v_k,
v_{k-1},
\ldots,
v_{k-r}
].
\]

Compute an orthonormal basis

\[
U_k
=
\operatorname{orth}(V_k).
\]

Then

\[
d_k
=
U_k\alpha_k.
\]

This guarantees that refinement remains in the local span of directions actually generated by the model.

A minimal version uses

\[
U_k
=
\operatorname{span}\{v_k,v_{k-1}\}.
\]

This is especially attractive for 4-step models because the subspace is extremely cheap.

---

# 18. Effective Rank

If the local feature covariance is

\[
C_k
=
\frac1N
\sum_i
(f_i-\bar f)(f_i-\bar f)^\top,
\]

with eigenvalues

\[
\lambda_1,\dots,\lambda_D,
\]

define normalized eigenvalues

\[
p_i
=
\frac{\lambda_i}{\sum_j\lambda_j}.
\]

The effective rank is

\[
r_{\text{eff}}
=
\exp
\left(
-\sum_i p_i\log p_i
\right).
\]

This can be used to adapt the subspace dimension:

\[
r_k
=
\lceil r_{\text{eff}}(C_k)\rceil.
\]

Potential hypothesis:

> distilled models may exhibit a strongly concentrated local trajectory spectrum, allowing refinement in a much smaller subspace than the full latent dimension.

---

# 19. Proposed Core Method

A clean initial method is:

## Consistency-Preserving Subspace Sampling

At native step \(k\):

### Step 1 — Native model prediction

\[
v_k
=
v_\theta(x_k,t_k,c).
\]

### Step 2 — Construct numerical refinement

For example,

\[
r_k
=
v_k-v_{k-1}.
\]

### Step 3 — Construct local trajectory subspace

\[
U_k
=
\operatorname{orth}
(
[v_k,v_{k-1},\ldots]
).
\]

### Step 4 — Define consistency field

\[
S_k
=
\left\|
\hat x_0^{(k)}
-
\hat x_0^{(k-1)}
\right\|^2
\]

or another native-step consistency statistic.

### Step 5 — Compute local normal

\[
g_k
=
\nabla_{x_k}S_k.
\]

### Step 6 — Restrict refinement

\[
\tilde r_k
=
U_kU_k^\top r_k.
\]

### Step 7 — Remove sensitivity-changing component

\[
r_k^\star
=
\tilde r_k
-
\frac{
\tilde r_k^\top U_kU_k^\top g_k
}{
\|U_kU_k^\top g_k\|^2+\epsilon
}
U_kU_k^\top g_k.
\]

### Step 8 — Refined sampler

\[
x_{k+1}
=
x_k
+
h_k
\left(
v_k
+
\lambda_k r_k^\star
\right).
\]

---

# 20. Pseudocode

```python
x = initial_noise()
v_prev = None

for k in native_timesteps:

    v = model_velocity(x, t[k], cond)

    if v_prev is None:
        x = native_step(x, v, k)
        v_prev = v
        continue

    # Numerical correction proposal
    r = v - v_prev

    # Local trajectory basis
    U = orthogonalize([v, v_prev])

    # Model-supported projection
    r_sub = U @ (U.T @ r)

    # Native consistency statistic
    S = consistency_score(x, t[k], cond)

    # Sensitivity normal
    g = grad(S, x)

    # Restrict normal to the same subspace
    g_sub = U @ (U.T @ g)

    # Tangent projection
    r_safe = r_sub - (
        dot(r_sub, g_sub)
        / (norm(g_sub)**2 + eps)
    ) * g_sub

    # Refined sampling direction
    v_ref = v + lambda_k * r_safe

    x = x + h[k] * v_ref

    v_prev = v
```

---

# 21. Low-Cost Approximation

Full

\[
g_k
=
\nabla_x S_k
\]

may require expensive higher-order derivatives.

A practical first version should avoid full second-order computation.

Possible approximations:

## A. Jacobian-vector products

Estimate only

\[
g^\top r
\]

instead of the full \(g\).

Since the projection only needs

\[
\frac{r^\top g}{\|g\|^2},
\]

directional derivatives or Hutchinson estimators may be sufficient.

---

## B. Low-rank hidden-space sensitivity

Let

\[
h_k
=
h_\theta(x_k,t_k,c)
\]

be a late hidden representation.

Define consistency in hidden space:

\[
S_k
=
\|h_k-h_{k-1}\|^2.
\]

This can be significantly cheaper than operating directly on the image latent Jacobian.

---

## C. OPS-inspired prediction-head approximation

If

\[
v_\theta
\approx
W h_\theta,
\]

perform the projection in hidden coordinates.

Let

\[
r_h
=
W^\top r.
\]

Let \(g_h\) be a hidden-space proxy for the consistency normal.

Use

\[
M_h
=
W^\top W.
\]

Then remove the parallel component under the pullback metric:

\[
r_h^\parallel
=
\frac{
g_h^\top M_h r_h
}{
g_h^\top M_h g_h
}
g_h.
\]

Then

\[
r_h^\perp
=
r_h-r_h^\parallel.
\]

Map back:

\[
r^\star
=
Wr_h^\perp.
\]

---

# 22. Novelty Positioning

The distinction from Fisher-Preserving Guidance should be explicit.

## FPG

Goal:

\[
\text{safe injection of external task guidance}
\]

into diffusion-policy sampling.

Their correction comes from an external task loss:

\[
u_t
=
\nabla L.
\]

Their reliability field measures observation-conditioned sensitivity.

---

## Proposed Method

Goal:

\[
\boxed{
\text{safe injection of numerical sampling refinement}
}
\]

into aggressively distilled image-generation trajectories.

The correction comes from the sampler itself:

\[
r_k
=
\text{trajectory / velocity / consistency correction}.
\]

The reliability field characterizes the student model's own native trajectory.

Therefore:

\[
\boxed{
\text{FPG preserves guidance reliability;}
\quad
\text{ours preserves sampling-trajectory reliability.}
}
\]

---

# 23. Stronger Research Hypothesis

A possible central hypothesis is:

> Few-step distilled samplers suffer not only from discretization error, but also from a narrow valid state distribution. Standard higher-order or extrapolative corrections may improve local numerical accuracy while simultaneously moving states outside the student's native trajectory distribution. Refinement should therefore be constrained to model-supported tangent directions.

This yields:

\[
\boxed{
\text{numerical accuracy}
+
\text{distributional validity}
}
\]

rather than numerical accuracy alone.

---

# 24. Suggested Experimental Setup

Start with a 4-step distilled image model.

Prefer a model where:

- native timesteps are clearly defined;
- the sampler exposes velocity or \(x_0\) prediction;
- inference fits on one A100;
- deterministic seed comparisons are easy.

Test:

1. native sampler;
2. naive velocity extrapolation;
3. naive Heun-like correction;
4. subspace-only refinement;
5. consistency-only projection;
6. subspace + consistency projection;
7. full proposed sampler.

---

# 25. Main Ablations

## A. Ambient vs Subspace Correction

Compare:

\[
r
\]

against

\[
P_Ur.
\]

Question:

> Does restricting correction to model-supported directions reduce artifacts?

---

## B. Projection Constraint

Compare:

\[
P_Ur
\]

with

\[
P_{g^\perp}P_Ur.
\]

Question:

> Does reliability-preserving projection improve stability beyond low-rank restriction?

---

## C. Subspace Type

Compare:

- trajectory velocity span;
- random orthogonal subspace;
- PCA hidden-state basis;
- final-head span;
- attention-feature span.

---

## D. Reliability Field

Compare:

- state Jacobian sensitivity;
- \(x_0\)-consistency;
- velocity consistency;
- hidden-feature consistency;
- native-step prediction disagreement.

---

## E. Rank

Test

\[
r
\in
\{1,2,4,8,16,32\}.
\]

Also test adaptive effective rank.

---

## F. Refinement Strength

Test

\[
\lambda
\in
\{0.05,0.1,0.2,0.5,1.0\}.
\]

Measure where quality improves before off-manifold artifacts appear.

---

# 26. Evaluation

Important metrics should separate:

## Image Quality

- FID
- ImageReward
- HPSv2
- PickScore
- aesthetic score

## Prompt Alignment

- CLIPScore
- text-image similarity
- VLM-based alignment evaluation

## Diversity

- LPIPS across seeds;
- DINO feature distance;
- pairwise CLIP image distance;
- feature covariance spectrum;
- effective rank of generated features.

## Sampling Stability

Measure:

\[
\|\hat x_0^{(k)}-\hat x_0^{(k+1)}\|
\]

along the trajectory.

Also measure:

\[
\|v_k-v_{k-1}\|.
\]

## Off-Manifold Proxy

Possible proxies:

- denoising sensitivity;
- consistency error;
- reconstruction instability;
- latent norm drift;
- hidden-feature distance from native trajectories;
- Jacobian spectral statistics.

---

# 27. Key Diagnostic Experiment

A particularly important experiment is:

1. collect native states \(x_k\);
2. add corrections of equal norm but different orientation:
   - random ambient direction;
   - trajectory-subspace direction;
   - sensitivity-tangent direction;
3. measure next-step error / consistency.

For fixed

\[
\|d\|=\epsilon,
\]

compare

\[
d_{\text{random}},
\quad
d_{\text{subspace}},
\quad
d_{\text{tangent}}.
\]

Hypothesis:

\[
E_{\text{consistency}}
(d_{\text{tangent}})
<
E_{\text{consistency}}
(d_{\text{subspace}})
<
E_{\text{consistency}}
(d_{\text{random}}).
\]

If this holds strongly, it directly supports the geometric premise of the method.

---

# 28. Failure Modes

## Reliability field may not equal manifold distance

A constant sensitivity level set is not guaranteed to coincide with the data manifold.

Therefore avoid claiming:

\[
\text{isosurface}
=
\text{true manifold}.
\]

Prefer:

> local model-supported sensitivity regime.

---

## Correction may be over-constrained

If

\[
r_k
\parallel
g_k,
\]

then

\[
r_k^\star\approx 0.
\]

The method may suppress useful refinement.

Potential fix:

\[
r_k^\star
=
r_k
-
\gamma
\frac{r_k^\top g_k}
{\|g_k\|^2}
g_k,
\qquad
0<\gamma<1.
\]

---

## Low-rank subspace may discard useful directions

Too small \(r\) can reduce flexibility.

Use adaptive effective rank or cumulative explained variance.

---

## Reliability gradients may be expensive

Need JVP/VJP approximations, hidden-space proxies, or OPS-style factorization.

---

## One-step models are harder

For a true one-step generator there is almost no trajectory history.

Possible alternatives:

- hidden-layer subspaces;
- prediction-head subspaces;
- multiple internal block features;
- Jacobian-derived local tangent approximation.

The 4-step case is the natural first target.

---

# 29. Recommended First Version

Do **not** start with the full Fisher-style machinery.

Start with:

\[
\boxed{
\text{4-step model}
+
\text{native timesteps only}
+
\text{trajectory velocity subspace}
+
\text{\(x_0\)-consistency field}
}
\]

Specifically:

\[
r_k=v_k-v_{k-1},
\]

\[
U_k=\operatorname{orth}([v_k,v_{k-1}]),
\]

\[
S_k
=
\|\hat x_0^{(k)}-\hat x_0^{(k-1)}\|^2,
\]

\[
r_k^\star
=
P_{U_k}r_k
-
\operatorname{proj}_{P_{U_k}g_k}(P_{U_k}r_k),
\]

and

\[
x_{k+1}
=
x_k+h_k(v_k+\lambda r_k^\star).
\]

This is the cleanest minimal prototype.

---

# 30. Possible Names

## Method-Oriented

- **Consistency-Preserving Sampling (CPS)**
- **Manifold-Preserving Sampling Refinement (MPSR)**
- **Trajectory-Preserving Refinement (TPR)**
- **Subspace-Constrained Sampling Refinement (SCSR)**
- **Native-Manifold Sampling Refinement (NMSR)**

## More Paper-Like

- **Stay on Track: Training-Free Trajectory-Preserving Sampling for Few-Step Diffusion Models**
- **Refine Without Drifting: Manifold-Preserving Sampling for Distilled Diffusion Models**
- **Tangent Sampling: Training-Free Refinement of Few-Step Diffusion Trajectories**
- **On-Track Sampling: Geometry-Constrained Refinement for Few-Step Generative Models**
- **Trajectory-Tangent Sampling for Distilled Diffusion Models**

Current favorite:

\[
\boxed{
\textbf{Trajectory-Tangent Sampling for Few-Step Diffusion Models}
}
\]

---

# 31. Core Story for a Paper

The paper story could be:

1. Few-step distilled models support only a narrow family of native states.
2. Existing refinement / higher-order correction can introduce off-distribution states.
3. Correction norm alone does not prevent this.
4. Native trajectories exhibit a low-dimensional local structure.
5. Define a student-derived reliability field.
6. Restrict numerical corrections to:
   - the local model-supported subspace;
   - the tangent space of a reliability isosurface.
7. Obtain a training-free sampler that adds test-time compute without arbitrary latent drift.

The core equation is:

\[
\boxed{
x_{k+1}
=
\Phi_\theta(x_k)
+
\lambda
\left[
P_{U_k}r_k
-
\operatorname{proj}_{P_{U_k}g_k}
(P_{U_k}r_k)
\right].
}
\]

where

\[
r_k
=
\text{numerical sampling refinement},
\]

\[
U_k
=
\text{model-supported local subspace},
\]

and

\[
g_k
=
\nabla S_k(x_k)
\]

is the normal to a local reliability surface.

---

# 32. Immediate Implementation Plan

### Phase 1 — Verify the premise

Implement native 4-step sampling and save at every step:

- \(x_k\);
- \(v_k\);
- \(\hat x_0^{(k)}\);
- hidden features if accessible.

Test equal-norm perturbations:

- random direction;
- velocity span;
- orthogonal-to-velocity;
- PCA feature directions.

Measure downstream degradation.

---

### Phase 2 — Simple subspace sampler

Implement

\[
r_k=v_k-v_{k-1}
\]

and

\[
r_k^{sub}=P_Ur_k.
\]

Check whether subspace restriction already improves naive extrapolation.

---

### Phase 3 — Consistency-preserving projection

Add

\[
S_k
=
\|\hat x_0^{(k)}-\hat x_0^{(k-1)}\|^2.
\]

Project the correction against the local normal.

---

### Phase 4 — Low-rank / efficient sensitivity

Replace expensive gradients by:

- JVP;
- Hutchinson estimation;
- hidden-space consistency;
- OPS-inspired projection.

---

### Phase 5 — Generalize

Test:

- 2-step model;
- 1-step model;
- multiple model families;
- CFG variation;
- different prompt complexity;
- diversity-sensitive settings.

---

# 33. Bottom Line

The most valuable idea from Fisher-Preserving Guidance is not to directly reuse their robotics-specific Fisher score.

The useful abstraction is:

\[
\boxed{
\text{refinement direction}
\rightarrow
\text{project onto a model-supported subspace}
\rightarrow
\text{remove the component that changes a reliability statistic}
}
\]

For few-step image diffusion, this naturally becomes:

\[
\boxed{
\textbf{training-free, geometry-constrained sampling refinement}
}
\]

rather than guidance or latent optimization.

The research question is therefore:

> **Can we spend additional inference compute to improve the numerical trajectory of a few-step distilled diffusion model while explicitly preserving the local state geometry that the student naturally supports?**

That is the central direction worth testing.
