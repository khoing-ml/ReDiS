# ReDiS: consistency-non-increasing sampling

Training-free, geometry-constrained sampling refinement for few-step FLUX.2
Klein. The implementation follows
`plans/training_free_manifold_preserving_sampling.md` and keeps the earlier
hidden-state intervention harness available for comparison.

At native step `k`, the sampler forms a numerical proposal, builds a local
trajectory span, computes the actual state-space normal with a VJP, and removes
only a proposal component that would increase `x0` inconsistency:

```text
r = proposal(v_k, v_{k-1}, x0_k, x0_{k-1})
r_sub = projection of r into span(v_k, v_{k-1}, ...)
e = x0_k - stopgrad(x0_previous)
g = grad_x (0.5 * mean(e^2)) = J_x0^T e
r_safe = projection of r_sub onto the half-space g^T r <= 0
v_refined = v_k + lambda * r_safe
```

All model calls remain on the checkpoint's native distilled timesteps. Model
weights and Diffusers source code are not modified.

For Google Colab Pro+, follow [COLAB.md](COLAB.md). The Colab installer reuses
the runtime's preinstalled CUDA-enabled PyTorch and auto-selects an execution
profile for the allocated GPU.

## Hardware note

The default smoke config is intentionally small (`256x256`, four steps). The
official model card reports about 13 GB VRAM for bf16 inference. On the current
6 GB GPU, use the 4-bit config and disk/CPU offload. Loading can still require
substantial host RAM and the first run downloads tens of gigabytes.

## Run one check at a time

```bash
bash bash/00_check_environment.sh
bash bash/01_unit_tests.sh
bash bash/02_check_model_access.sh
bash bash/03_generate_baseline.sh
bash bash/04_test_wrapper_identity.sh
bash bash/05_capture_hidden.sh
bash bash/06_test_gaussian_noise.sh
bash bash/07_test_residual_amplification.sh
bash bash/08_test_projected_amplification.sh
bash bash/09_test_gram_isotropization.sh
bash bash/10_evaluate_feature_diversity.sh outputs/METHOD/RUN_TIMESTAMP
bash bash/11_run_screening_site.sh A .runtime/colab_screening.yaml
bash bash/12_setup_output_metrics.sh
bash bash/13_evaluate_screening.sh outputs/screening_site_A/RUN_TIMESTAMP
bash bash/14_run_manifold_sampler.sh
bash bash/15_run_sampler_ablation.sh
bash bash/16_evaluate_sampler_ablation.sh outputs/sampler_ablation/RUN_TIMESTAMP
bash bash/17_evaluate_sampler_rewards.sh outputs/sampler_ablation/RUN_TIMESTAMP
```

`00` bootstraps `.venv` with Python 3.12. Set `REDIS_INSTALL_QUANT=0` to omit
bitsandbytes. Model scripts default to `configs/flux2_klein_4b_low_vram.yaml`.
The capture script uses its two-seed capture config by default.
Intervention scripts use a four-seed smoke config and modify dual-stream layer
0 at denoising step 0. Each run writes images, resolved config, environment,
runtime, correction norm, and pre/post seed spectra under `outputs/`.
Override the config with the first positional argument:

```bash
bash bash/03_generate_baseline.sh configs/flux2_klein_4b_smoke.yaml
```

## Consistency-constrained sampler

The smoke config runs the full subspace-plus-half-space method with an actual
VJP consistency normal. The previous prediction is a fixed, stop-gradient
reference:

```bash
bash bash/14_run_manifold_sampler.sh
```

Useful command-line overrides are:

```bash
# Exact native baseline; this path forwards the original velocity unchanged.
bash bash/14_run_manifold_sampler.sh \
  configs/flux2_klein_4b_sampler_smoke.yaml --mode native

# Equality-tangent ablation using the same actual VJP normal.
bash bash/14_run_manifold_sampler.sh \
  configs/flux2_klein_4b_sampler_smoke.yaml \
  --mode tangent --normal-estimator vjp

# Match correction magnitude across orientation ablations.
bash bash/14_run_manifold_sampler.sh \
  configs/flux2_klein_4b_sampler_smoke.yaml \
  --target-correction-norm 0.05
```

Supported proposals are `velocity_difference`, `curvature`, `x0_difference`,
`random_ambient`, and `random_velocity_orthogonal`. Supported modes are:

- `native`: unchanged Euler sampling;
- `naive`: add the ambient proposal;
- `subspace`: restrict it to recent native velocities;
- `tangent`: equality projection preserving consistency to first order;
- `non_increasing`: retain descent directions and remove only components for
  which `g^T r > 0`. This is the default method.

`normal_estimator: vjp` differentiates `x0` consistency with respect to the
current latent in the existing transformer forward pass. It computes a normal
in state space; the previous `x0` prediction is detached. The legacy
`residual_proxy` remains only as an explicit degeneracy control and must not be
interpreted as a geometric normal. VJP mode requires a single conditional pass
(`guidance_scale <= 1`) and materially more memory than inference-only mode.

Each sampler run writes images, the resolved config, environment information,
per-step norms and consistency diagnostics, and (when
`capture_trajectory: true`) `trajectory.pt` containing `x_k`, native velocity,
`x0`, proposal, velocity correction, and applied state correction.

### Finite correction check and trust region

The scheduler applies a velocity correction through the actual state
displacement

```text
delta_x = (sigma_next - sigma) * delta_v
```

Because `sigma_next - sigma` is negative for the Diffusers FlowMatch schedule,
the state-space directional derivative can have the opposite sign from
`g^T delta_v`. Both the half-space constraint and finite check therefore use
`delta_x`.

With `finite_consistency_check: true`, every active step performs an additional
forward evaluation at `x_k + delta_x`, at the same native timestep and against
the same detached previous `x0`. It logs:

- `consistency_before`, `consistency_after`, and `consistency_ratio`;
- `finite_directional_curvature`;
- `state_correction_norm` and its norm relative to the native state update.

To measure the unconstrained velocity extrapolation before enabling any
half-space projection, run:

```bash
bash bash/14_run_manifold_sampler.sh \
  configs/flux2_klein_4b_sampler_smoke.yaml \
  --mode naive --normal-estimator vjp --strength 0.2
```

This still computes the VJP and finite metrics, but applies the proposal
without a consistency constraint.

Enable trust-region backtracking with:

```bash
bash bash/14_run_manifold_sampler.sh \
  configs/flux2_klein_4b_sampler_smoke.yaml \
  --mode non_increasing --normal-estimator vjp --trust-region
```

Failed finite trials are scaled by `trust_region_shrink_factor` until they
satisfy `S_after <= S_before + tolerance`; a trial still failing after
`trust_region_max_shrinks` is rejected. This adds one forward per trial but no
additional backward pass. A per-step strength can be supplied with, for
example, `--strength-schedule 0 0.2 0.15 0.05`.

The broader orientation ablation in `configs/flux2_klein_4b_sampler_smoke.yaml`
includes both velocity- and `x0`-difference proposals. Since
`v_k-v_{k-1}` already lies in
`span(v_k, v_{k-1})`, its subspace projection is mathematically an identity;
the `x0` and seeded random controls are needed to test whether orientation,
rather than correction norm alone, drives stability. Its non-native
conditions are matched to correction norm `0.05`. The evaluation command
writes per-condition feature metrics plus a combined JSON/CSV table with
trajectory consistency and deltas against native sampling.

Per-step diagnostics distinguish the two projections:

- `subspace_retention = ||P_U r|| / ||r||`;
- `constraint_retention = ||r_safe|| / ||P_U r||`;
- `pre_projection_normal_cosine` and `post_projection_normal_cosine`;
- the directional derivatives `g^T r` before and after the constraint.

### Fixed-prompt CCSR baseline

`bash/15_run_sampler_ablation.sh` now defaults to the first validation grid in
`configs/flux2_klein_4b_ccsr_debug.yaml`: 20 fixed Pick-a-Pic held-out prompts,
four seeds, and exactly four conditions:

- native sampling;
- naive velocity refinement;
- first-order consistency projection;
- projection plus finite trust-region backtracking.

The pinned prompt files and source metadata live under `prompts/`. Rebuild or
verify them without silently overwriting changes with:

```bash
python scripts/prepare_benchmark_prompts.py
```

Each ablation run writes per-prompt images and diagnostics, plus aggregate
correction, consistency, first-order failure, shrink, and rejection statistics.
After generation, score all matched prompt/seed samples with PickScore and
compute paired deltas against native sampling:

```bash
bash bash/17_evaluate_sampler_rewards.sh outputs/sampler_ablation/RUN_TIMESTAMP
```

The evaluator writes `reward_scores.json` and `reward_summary.json`, including
paired bootstrap intervals and the correlation between finite consistency drift
and quality change. `RewardEvaluator` is a metric registry so HPSv2, CLIPScore,
and aesthetic backends can be added without changing the generation pipeline.

The decisive screening is fixed to three sites (A/B/C), eight prompts, eight
seeds, and per-seed correction norms 0.005/0.010/0.020. Run one site per
process. Its evaluator reports output diversity (DINO, DreamSim, LPIPS) against
fidelity (CLIP-T, HPSv2); hidden effective rank remains a mechanism diagnostic.

Useful environment variables:

- `HF_TOKEN`: optional Hub token (the 4B checkpoint is public).
- `HF_HOME`: Hugging Face cache location.
- `REDIS_OUTPUT_ROOT`: output root; defaults to `outputs`.
- `REDIS_INSTALL_QUANT=0`: install without bitsandbytes.

For `memory_mode: auto`, ReDiS maps the profile to Diffusers' supported
`device_map: balanced` strategy. An explicit `model.device_map` may be supplied
when another strategy is required by a particular Diffusers build.
