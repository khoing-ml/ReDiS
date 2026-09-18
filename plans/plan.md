# ReDiS: Training-Free Diversity Restoration in Distilled Diffusion Models

## Implementation and falsification plan

**Working name:** ReDiS — Residual Diversity in Subspaces  
**Status:** proposed method; not yet empirically validated  
**Primary prototype:** `black-forest-labs/FLUX.1-schnell`  
**Primary question:** Does step distillation suppress seed-conditioned variation in intermediate hidden states, and can inference-time transformation of the surviving seed residuals restore image diversity without unacceptable fidelity loss?

---

## 1. Scope and claims

ReDiS is a **training-free** inference intervention. It does not update model weights. For a fixed prompt, it processes a group of different seeds jointly, measures their hidden-state residual geometry, and modifies those residuals at selected transformer layers and sampling timesteps.

The first implementation must separate three questions:

1. **Diagnosis:** Do hidden-state residuals become more concentrated with network depth or sampling time in the distilled model?
2. **Minimal intervention:** Is ordinary seed-residual amplification already sufficient?
3. **Geometry-aware intervention:** Does random-subspace isotropization provide a better diversity–fidelity trade-off than simpler amplification?

Do not claim that the model has “mode collapse” from low effective rank alone. Low hidden-state effective rank is evidence of anisotropic seed variation, not proof of reduced semantic coverage. Output diversity and feature-space coverage must be measured separately.

### Training-free does not mean free

The method preserves the checkpoint and the model's sampling-step count, but it adds:

- group inference with $B$ seeds for one prompt;
- activation storage at intervention layers;
- a random channel projection;
- a $B \times B$ eigendecomposition;
- cross-seed coupling for Gram isotropization.

Report the unchanged generator NFE separately from wall-clock latency, peak VRAM, batch size, and extra feature-transform compute.

---

## 2. Initial experimental setup

Start with:

```python
model_id = "black-forest-labs/FLUX.1-schnell"
num_inference_steps = 4
guidance_scale = 0.0
resolution = 512  # move to 768 only after the pipeline is stable
```

Use a small prompt suite of about 50 prompts, balanced across:

- single object;
- multiple objects and counting;
- pose and action;
- camera viewpoint;
- global scene composition;
- style;
- color and lighting.

Maintain a frozen manifest:

```text
prompt_id, prompt, seed, model_revision, scheduler,
num_steps, guidance_scale, height, width, dtype
```

All comparisons must use identical prompts, seeds, model revision, scheduler, resolution, and decoding settings.

### Important group constraint

Every ReDiS batch must contain **one prompt with multiple seeds**. Do not center or whiten a batch containing unrelated prompts. The output for one seed can depend on the other seeds in its group, so always record:

- group size $B$;
- group membership;
- grouping policy;
- random projection seed.

Permutation of group members should not change the corresponding outputs, up to numerical error. Changing group membership may change outputs and must be evaluated explicitly.

---

## 3. Hidden-state interface

Let a hidden activation at sampling timestep $t$ and transformer layer $\ell$ be

$$
H^{t,\ell} \in \mathbb{R}^{B \times N \times D},
$$

where $B$ is the number of seeds for one prompt, $N$ is the token or spatial sequence length, and $D$ is the channel dimension.

Use a block wrapper rather than a capture-only forward hook because later phases must modify the activation while preserving the block's original output structure.

```python
class ModifiedBlock(torch.nn.Module):
    def __init__(self, block, layer_id, controller):
        super().__init__()
        self.block = block
        self.layer_id = layer_id
        self.controller = controller

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        return self.controller.process_block_output(
            out=out,
            layer_id=self.layer_id,
        )
```

The adapter that extracts and reinserts the hidden tensor must handle the exact Diffusers block return type. Keep this model-specific logic in `models/flux_wrapper.py`; keep the mathematical transforms model-agnostic.

### Identity verification

Before implementing any method, verify:

```text
original pipeline output == wrapped pipeline output
```

under fixed seeds and deterministic settings. Compare both latent tensors and decoded images. The maximum absolute latent error should be zero when possible, or within the expected floating-point tolerance.

---

## 4. Phase I — Diagnose hidden-state diversity contraction

For one prompt and $B$ seeds, define the seed mean

$$
\bar H = \frac{1}{B}\sum_{i=1}^{B} H_i
$$

and residuals

$$
R_i = H_i - \bar H.
$$

Flatten each seed residual:

$$
R_f \in \mathbb{R}^{B \times M}, \qquad M = ND.
$$

Do not construct an $M \times M$ covariance matrix. Compute the small seed-space Gram matrix:

$$
G_R = \frac{1}{M} R_f R_f^\top \in \mathbb{R}^{B \times B}.
$$

Because the residuals are centered across seeds,

$$
\operatorname{rank}(G_R) \leq B-1.
$$

Let the nonnegative eigenvalues be $\lambda_1,\ldots,\lambda_q$, where $q \leq B-1$. Define

$$
p_i = \frac{\lambda_i}{\sum_j \lambda_j}.
$$

Measure:

$$
r_{\mathrm{eff}} = \exp\left(-\sum_i p_i \log(p_i + \varepsilon)\right),
$$

$$
r_{\mathrm{stable}} = \frac{\sum_i \lambda_i}{\max_i \lambda_i},
$$

$$
A_{\mathrm{top1}} = \frac{\lambda_1}{\sum_i \lambda_i},
$$

and the normalized effective rank

$$
\tilde r_{\mathrm{eff}} = \frac{r_{\mathrm{eff}}}{B-1}.
$$

The normalized form is required when comparing experiments with different group sizes.

```python
def seed_spectrum(hidden, eps=1e-12):
    # hidden: [B, N, D]
    residual = hidden.float() - hidden.float().mean(dim=0, keepdim=True)
    rf = residual.flatten(1)
    gram = (rf @ rf.T) / rf.shape[1]
    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0)

    tol = max(float(eigvals.max()) * 1e-6, eps)
    active = eigvals[eigvals > tol]
    if active.numel() == 0:
        return {
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "top1_ratio": 0.0,
        }

    p = active / active.sum().clamp_min(eps)
    effective_rank = torch.exp(-(p * torch.log(p + eps)).sum())
    stable_rank = active.sum() / active.max().clamp_min(eps)
    top1_ratio = active.max() / active.sum().clamp_min(eps)
    return {
        "effective_rank": effective_rank.item(),
        "stable_rank": stable_rank.item(),
        "top1_ratio": top1_ratio.item(),
    }
```

### Diagnostic sweep

Use $B=32$ seeds for diagnosis. Capture approximately four depths:

$$
\ell \in \{0.2L, 0.4L, 0.6L, 0.8L\},
$$

at each native sampling timestep.

Plot:

- effective rank versus layer;
- normalized effective rank versus layer;
- top-1 energy ratio versus layer;
- eigenvalue spectra at selected layers;
- the same measurements versus sampling timestep.

If a compatible non-distilled or less-distilled reference is available, run it with matched prompts and seeds. The main evidence is not merely low rank in the student, but a reproducible contraction relative to a reference or a progressive contraction through the student's own computation.

### Gate 1

Continue to geometry-aware restoration only if at least one layer/timestep region shows stable anisotropy or contraction across prompts. Otherwise, the isotropization story is unsupported; proceed only with generic perturbation baselines or change the hypothesis.

---

## 5. Phase II — Intervention hierarchy

Implement the methods in increasing order of complexity. Every complex method must beat the simpler method on a diversity–fidelity Pareto curve.

### 5.1 Identity

$$
H' = H.
$$

This verifies the intervention path and provides the vanilla distilled baseline.

### 5.2 Gaussian hidden noise baseline

$$
H' = H + \sigma \epsilon, \qquad \epsilon \sim \mathcal{N}(0,I).
$$

Scale $\epsilon$ to match a chosen fraction of the residual RMS. This baseline tests whether any gain is merely caused by adding hidden-state noise.

### 5.3 Seed-residual amplification

$$
H' = H + \gamma R
$$

or equivalently

$$
H_i' = \bar H + (1+\gamma)(H_i-\bar H).
$$

```python
def residual_amplify(hidden, gamma):
    mean = hidden.mean(dim=0, keepdim=True)
    residual = hidden - mean
    return hidden + gamma * residual
```

Initial sweep:

```text
gamma: 0.05, 0.10, 0.20, 0.40, 0.80
```

If this method already achieves the best Pareto trade-off, the project should pivot toward seed-residual geometry amplification rather than defend a more complex whitening formulation.

### 5.4 Random orthogonal channel projection

Sample

$$
A \in \mathbb{R}^{D \times k}, \qquad A_{ij} \sim \mathcal{N}(0,1),
$$

and compute a reduced QR factorization

$$
A = QR, \qquad Q^\top Q = I_k.
$$

Project residuals into the random channel subspace:

$$
Y = RQ \in \mathbb{R}^{B \times N \times k}.
$$

```python
def make_random_basis(d, k, *, device, dtype, generator):
    a = torch.randn(d, k, device=device, dtype=torch.float32,
                    generator=generator)
    q, _ = torch.linalg.qr(a, mode="reduced")
    return q.to(dtype=dtype)
```

Cache $Q$ by `(model_revision, layer, D, k, projection_seed)`. Do not resample $Q$ for every prompt, batch, or timestep in the main experiment. Repeat final results across at least three projection seeds to test robustness.

#### Projection-only ablation

Projection alone is not expected to restore diversity, but it is a useful information-loss control:

$$
H' = \bar H + RQQ^\top.
$$

It discards the residual component outside the random subspace. Treat it as an ablation, not the main method.

### 5.5 Projected residual amplification

Amplify only the projected residual:

$$
H' = H + \gamma RQQ^\top.
$$

```python
def projected_amplify(hidden, q, gamma):
    mean = hidden.mean(dim=0, keepdim=True)
    residual = hidden - mean
    y = residual @ q
    delta = y @ q.T
    return hidden + gamma * delta
```

The decisive comparison is

$$
H + \gamma R
\quad \text{versus} \quad
H + \gamma RQQ^\top.
$$

The projected version is useful only if it preserves fidelity better at matched output diversity or improves diversity at matched fidelity.

---

## 6. Phase III — Random-subspace isotropization by Gram whitening

This is the main ReDiS hypothesis.

Given

$$
Y = RQ \in \mathbb{R}^{B \times N \times k},
$$

flatten the token and projected-channel dimensions:

$$
Y_f \in \mathbb{R}^{B \times M_k}, \qquad M_k = Nk.
$$

Recenter numerically and compute the seed-space Gram matrix:

$$
G_Y = \frac{1}{M_k}Y_fY_f^\top.
$$

Let

$$
G_Y = U\Lambda U^\top.
$$

Since $Y_f$ is centered over seeds, one eigenvalue is theoretically zero. Never invert this centering null direction. Define an active set

$$
\mathcal{I} = \{i : \lambda_i > \tau\},
$$

with a relative threshold such as

$$
\tau = \max(10^{-6}\lambda_{\max}, 10^{-12}).
$$

Use the geometric mean of active eigenvalues as a dimensionally consistent reference:

$$
\lambda_{\mathrm{ref}}
=
\exp\left(
\frac{1}{|\mathcal{I}|}
\sum_{i\in\mathcal{I}} \log(\lambda_i+\epsilon)
\right).
$$

Define partial whitening scales

$$
s_i =
\begin{cases}
\left(\dfrac{\lambda_{\mathrm{ref}}}{\lambda_i+\epsilon}\right)^{\beta/2}, & i\in\mathcal{I},\\
0, & i\notin\mathcal{I},
\end{cases}
$$

where $\beta \in [0,1]$. Then

$$
A_\beta = U\operatorname{diag}(s)U^\top,
$$

$$
Y_{\mathrm{iso}} = A_\beta Y_f.
$$

At $\beta=0$, active directions are unchanged. At $\beta=1$, active seed-space eigenvalues are approximately equalized before later RMS matching and interpolation.

### Energy matching

Pure whitening changes energy. First use a global projected-residual energy match:

$$
Y_{\mathrm{iso}}
\leftarrow
Y_{\mathrm{iso}}
\frac{\lVert Y_f\rVert_F}
{\lVert Y_{\mathrm{iso}}\rVert_F+\epsilon}.
$$

Global matching preserves the relative whitening geometry. Per-seed matching is also worth ablating, but it changes the achieved spectrum.

Reshape $Y_{\mathrm{iso}}$ to $[B,N,k]$, compute the correction, and lift it to the original channel space:

$$
\Delta Y = Y_{\mathrm{iso}} - Y,
$$

$$
\Delta H = \Delta YQ^\top,
$$

$$
H' = H + \gamma \Delta H.
$$

### Reference implementation

```python
def gram_isotropize(
    hidden,
    q,
    *,
    beta,
    gamma,
    eps=1e-8,
    rel_tol=1e-6,
):
    """hidden [B,N,D], q [D,k]. Compute EVD in float32."""
    input_dtype = hidden.dtype
    h = hidden.float()
    qf = q.float()

    mean = h.mean(dim=0, keepdim=True)
    residual = h - mean
    y = residual @ qf                         # [B,N,k]
    yf = y.flatten(1)
    yf = yf - yf.mean(dim=0, keepdim=True)    # numerical recentering

    gram = (yf @ yf.T) / yf.shape[1]          # [B,B]
    eigvals, u = torch.linalg.eigh(gram)
    eigvals = eigvals.clamp_min(0)

    max_eval = eigvals.max()
    tol = torch.maximum(
        max_eval * rel_tol,
        torch.tensor(1e-12, device=eigvals.device),
    )
    active = eigvals > tol
    if active.sum() < 2 or beta == 0 or gamma == 0:
        return hidden

    active_vals = eigvals[active]
    log_ref = torch.log(active_vals + eps).mean()
    lambda_ref = torch.exp(log_ref)

    scales = torch.zeros_like(eigvals)
    scales[active] = (
        lambda_ref / (eigvals[active] + eps)
    ).pow(beta / 2)

    a_beta = (u * scales.unsqueeze(0)) @ u.T
    yf_iso = a_beta @ yf

    old_energy = yf.norm()
    new_energy = yf_iso.norm().clamp_min(eps)
    yf_iso = yf_iso * (old_energy / new_energy)

    y_iso = yf_iso.reshape_as(y)
    delta_h = (y_iso - y) @ qf.T
    h_new = h + gamma * delta_h
    return h_new.to(dtype=input_dtype)
```

### Batch-size limitation

The active seed-space rank is at most $B-1$.

- $B=2$: only one active direction; isotropization is mathematically unable to flatten a spectrum.
- $B=4$: at most three active directions; suitable only for a smoke test.
- $B=8$: minimum practical screening size.
- $B=16$ or $32$: preferred for stable analysis and final evaluation, subject to memory.

This is a batch-conditioned generation method. Do not present it as an independent single-image sampler unless a later version replaces batch statistics with frozen offline statistics.

---

## 7. RMS matching and activation safeguards

After an intervention, optionally match each sample's full-activation RMS:

$$
\operatorname{RMS}(X_i)
=
\sqrt{\frac{1}{ND}\sum_{n,d}X_{i,n,d}^2},
$$

$$
H_i' \leftarrow H_i'
\frac{\operatorname{RMS}(H_i)}
{\operatorname{RMS}(H_i')+\epsilon}.
$$

```python
def rms(x):
    dims = tuple(range(1, x.ndim))
    return x.square().mean(dim=dims, keepdim=True).sqrt()

def rms_match(modified, reference, eps=1e-8):
    return modified * (rms(reference) / (rms(modified) + eps))
```

Ablate:

```text
none
global projected-residual energy matching only
per-sample full-activation RMS matching only
both
```

Per-sample RMS matching may perturb the exact whitened spectrum. Therefore recompute the post-intervention effective rank rather than assuming that the intended transform was achieved.

Add hard safety checks:

- abort or fall back to identity on NaN/Inf;
- cap the relative correction norm $\lVert\Delta H\rVert_F / \lVert H\rVert_F$;
- log pre/post RMS, correction norm, and effective rank;
- run eigendecomposition in float32 even when the model uses bf16/fp16.

---

## 8. Layer and timestep sweep

There are two intervention axes:

$$
(\text{sampling timestep}, \text{transformer depth}).
$$

Do not fully sweep both axes at once.

### Stage A — Fix timestep, sweep depth

Modify only the first sampling timestep and test:

$$
\ell \in \{0.2L,0.4L,0.6L,0.8L\}.
$$

### Stage B — Fix best depth, sweep timestep

Test:

```text
step 0 only
step 1 only
steps 0 and 1
all native sampling steps
```

### Stage C — Small hyperparameter grid

For screening:

```text
prompts: 20
seeds per prompt: 8
layer: 20%, 40%, 60%, 80%
k: 64, 256
beta: 0.25, 0.50
gamma: 0.10, 0.30, 0.50
projection seeds: 1 during screening, 3 for finalists
```

This yields 48 main isotropization configurations before projection-seed replication. Keep only the Pareto-relevant configurations for $B=16$ or $32$ confirmation.

Do not implement SRHT or Hadamard projection in version 1. Dense Gaussian QR is the falsification baseline. Faster structured projection is an optimization only after the method works.

---

## 9. Metrics and statistical protocol

### 9.1 Representation metrics

At every intervention site, log:

- effective rank $r_{\mathrm{eff}}$;
- normalized effective rank $\tilde r_{\mathrm{eff}}$;
- stable rank;
- top-1 energy ratio;
- full active eigenvalue spectrum;
- residual RMS;
- relative correction norm;
- pre/post cosine similarity of flattened activations.

The key mechanistic plot is

$$
\Delta \tilde r_{\mathrm{eff}}
\quad \text{versus} \quad
\Delta \text{image diversity}.
$$

Correlation is supporting evidence, not proof of causality.

### 9.2 Prompt-local image diversity

For each prompt, compute all seed-pair distances and average within prompt before averaging across prompts:

- LPIPS distance for perceptual/local variation;
- DreamSim distance for semantic/perceptual variation;
- DINO feature distance for structural/semantic variation;
- $1-$CLIP image-image cosine similarity as an additional view.

Do not pool all images across prompts before computing diversity; prompt differences would dominate the measurement.

### 9.3 Fidelity and prompt alignment

Use a fixed, versioned set of evaluators:

- CLIP text-image similarity;
- HPSv2;
- ImageReward;
- an aesthetic or artifact score if needed;
- human inspection of a fixed contact sheet.

For larger evaluation, add FID/quality precision and feature-space recall. FID alone cannot establish same-prompt diversity or coverage.

### 9.4 Statistical reporting

- Use matched prompts and seeds across methods.
- Aggregate seed-pair metrics within each prompt.
- Report mean, median, and 95% paired bootstrap confidence intervals over prompts.
- Plot diversity versus each fidelity metric.
- Mark configurations that improve diversity but fail a predeclared fidelity floor.
- Repeat finalists with at least three random projection seeds.

---

## 10. Required baselines and ablations

Run at least:

1. Vanilla distilled model with IID seeds.
2. Gaussian hidden noise with matched correction RMS.
3. Raw seed-residual amplification.
4. Random projection only.
5. Random projected amplification.
6. Random-subspace Gram isotropization.
7. Gram isotropization without RMS matching.
8. Gram isotropization with RMS matching.
9. STRIDE, if a compatible implementation is available.
10. Feature Self-Guidance, if a compatible implementation is available.

Useful reference conditions, when accessible:

- a less-distilled/base model under matched prompts and seeds;
- a base-first-step hybrid as an upper/reference condition, clearly reporting its extra model and compute requirements.

For fair perturbation comparisons, match either the hidden correction RMS or the resulting fidelity degradation. Do not compare arbitrary perturbation strengths.

---

## 11. Configuration design

```yaml
model:
  id: black-forest-labs/FLUX.1-schnell
  revision: null
  dtype: bfloat16
  height: 512
  width: 512
  num_inference_steps: 4
  guidance_scale: 0.0

group:
  seeds_per_prompt: 8
  require_same_prompt: true

intervention:
  method: gram_isotropization
  timestep_ids: [0]
  layer_fractions: [0.4]
  gamma: 0.3
  rms_match: true
  max_relative_correction_norm: 0.25

projection:
  k: 256
  seed: 0
  cache_basis: true

whitening:
  beta: 0.5
  eps: 1.0e-8
  relative_eigenvalue_threshold: 1.0e-6
  energy_match: global

logging:
  save_images: true
  save_activations: false
  save_spectra: true
```

Every output directory should contain the resolved config, environment/package versions, Git commit, prompt manifest, projection seed, and metric versions.

---

## 12. Repository structure

```text
redis/
├── README.md
├── pyproject.toml
├── configs/
│   ├── flux_schnell_baseline.yaml
│   ├── flux_schnell_diagnosis.yaml
│   └── flux_schnell_redis.yaml
├── redis/
│   ├── models/
│   │   └── flux_wrapper.py
│   ├── hooks/
│   │   ├── capture.py
│   │   ├── controller.py
│   │   └── output_adapter.py
│   ├── methods/
│   │   ├── identity.py
│   │   ├── gaussian_noise.py
│   │   ├── residual_amplification.py
│   │   ├── random_projection.py
│   │   ├── projected_amplification.py
│   │   ├── gram_isotropization.py
│   │   └── rms_matching.py
│   ├── analysis/
│   │   ├── spectrum.py
│   │   ├── effective_rank.py
│   │   └── layer_timestep_analysis.py
│   ├── metrics/
│   │   ├── diversity.py
│   │   ├── fidelity.py
│   │   └── aggregation.py
│   └── utils/
│       ├── determinism.py
│       ├── manifests.py
│       └── logging.py
├── scripts/
│   ├── generate_baseline.py
│   ├── diagnose_hidden_states.py
│   ├── run_residual_amplification.py
│   ├── run_projected_amplification.py
│   ├── run_gram_isotropization.py
│   ├── evaluate_images.py
│   └── plot_pareto.py
├── tests/
│   ├── test_wrapper_identity.py
│   ├── test_projection.py
│   ├── test_spectrum.py
│   ├── test_isotropization.py
│   └── test_group_equivariance.py
└── outputs/
    ├── manifests/
    ├── images/
    ├── metrics/
    ├── spectra/
    └── figures/
```

---

## 13. Minimum tests before image sweeps

### Projection tests

Verify

$$
Q^\top Q \approx I_k.
$$

With $k=D$ and a full orthogonal $Q$,

$$
RQQ^\top \approx R,
$$

so projected amplification should match raw residual amplification.

### Isotropization tests

- `beta = 0` returns identity.
- `gamma = 0` returns identity.
- output contains no NaN/Inf.
- centered correction has near-zero seed mean before per-sample RMS matching.
- active eigenvalues become flatter as $\beta$ increases on synthetic anisotropic data.
- global energy matching preserves projected residual Frobenius norm.
- seed permutations produce correspondingly permuted outputs.
- $B=2$ triggers a warning or identity fallback for isotropization.

### Integration tests

- wrapper identity reproduces the original latent and image;
- fixed seeds and fixed $Q$ are deterministic;
- changing only the projection seed changes the intervention but not the vanilla baseline;
- activation capture is disabled after the selected layer to avoid memory leaks;
- a complete run writes its resolved configuration and manifest.

---

## 14. Implementation order

Implement in this exact order:

1. **Baseline generation:** fixed prompt/seed manifest and reproducible FLUX.1-schnell outputs.
2. **Block wrapper:** patch one block and prove identity behavior.
3. **Capture path:** save selected hidden states for $B=32$ seeds.
4. **Diagnosis:** plot spectra, effective rank, stable rank, and top-1 ratio across layers and timesteps.
5. **Raw residual amplification:** implement $H'=H+\gamma(H-\bar H)$.
6. **Random basis:** implement and cache dense Gaussian QR bases.
7. **Projection controls:** projection-only and projected amplification.
8. **Gram isotropization:** active-eigenspace partial whitening, global energy matching, and lifted correction.
9. **RMS safeguards:** add per-sample activation RMS matching and correction-norm caps.
10. **Unit/integration tests:** validate identities, determinism, permutation equivariance, and spectrum flattening.
11. **Depth sweep:** first sampling timestep only.
12. **Timestep sweep:** best one or two depth locations only.
13. **Metric pipeline:** prompt-local diversity, fidelity, confidence intervals, and Pareto plots.
14. **External baselines:** STRIDE and Feature Self-Guidance when compatible.
15. **Scale-up:** $B=16$ or $32$, more prompts, multiple projection seeds, and coverage metrics.
16. **Optimization only after success:** consider SRHT/Hadamard projection or frozen offline statistics.

---

## 15. Decision gates and falsification criteria

### Hypothesis H1 — representation contraction exists

Evidence required:

- effective rank decreases or top-1 energy increases through specific layers/timesteps; and/or
- the distilled model is more anisotropic than a matched reference;
- the effect is stable across prompts.

**Falsify or weaken H1** if the spectrum is not systematically more concentrated.

### Hypothesis H2 — surviving seed residuals control useful diversity

Evidence required:

- raw or projected residual amplification increases prompt-local semantic/structural diversity;
- the gain is not explained solely by artifacts or fidelity loss.

**Falsify H2** if matched-RMS Gaussian noise performs equally well or all interventions only damage quality.

### Hypothesis H3 — isotropization is better than amplification

Evidence required:

- Gram isotropization lies on a better diversity–fidelity Pareto frontier than raw and projected amplification;
- the result repeats across prompts, seeds, group compositions, and projection seeds;
- post-intervention effective-rank gain correlates with output-diversity gain.

**Falsify or pivot from H3** if plain residual amplification matches isotropization, or if whitening gains disappear after fair correction-norm/fidelity matching.

### Practical success criterion

A configuration is promising only if it:

1. improves at least two complementary prompt-local diversity metrics;
2. remains above a predeclared fidelity floor;
3. is robust across projection seeds;
4. does not rely on a small subset of prompts;
5. has acceptable latency and VRAM overhead;
6. remains beneficial when group membership changes.

---

## 16. First deliverables

The first complete experimental report should contain:

1. **Effective rank versus transformer layer and sampling timestep.**
2. **Representative eigenvalue spectra before intervention.**
3. **Diversity and fidelity versus $\gamma$ for raw residual amplification.**
4. **Matched comparison of raw versus projected amplification.**
5. **Post-whitening spectrum as $\beta$ changes.**
6. **Diversity–fidelity Pareto plot** for vanilla, Gaussian noise, residual amplification, projected amplification, and Gram isotropization.
7. **A fixed contact sheet** using identical prompts and seeds.
8. **Runtime table** with NFE, batch size, latency, and peak VRAM.

The central result is not “whitening increases effective rank.” That is true by construction. The meaningful result would be:

> A measured hidden-state contraction exists in distilled generation, and selectively flattening the surviving seed-residual geometry restores prompt-local image diversity more efficiently than simpler matched-strength perturbations.

Until the diagnostic and Pareto experiments support that statement, ReDiS should be presented as a testable hypothesis rather than a validated diversity-restoration method.
