# ReDiS: trajectory-tangent sampling

Training-free, geometry-constrained sampling refinement for few-step FLUX.2
Klein. The implementation follows
`plans/training_free_manifold_preserving_sampling.md` and keeps the earlier
hidden-state intervention harness available for comparison.

At native step `k`, the sampler forms a numerical proposal, builds a local
trajectory span, and removes the component normal to an `x0`-consistency level
set:

```text
r = proposal(v_k, v_{k-1}, x0_k, x0_{k-1})
r_sub = projection of r into span(v_k, v_{k-1}, ...)
r_safe = r_sub - projection of r_sub onto the consistency normal
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

## Manifold-preserving sampler

The smoke config runs the full subspace-plus-tangent method with the low-cost
stop-gradient consistency normal:

```bash
bash bash/14_run_manifold_sampler.sh
```

Useful command-line overrides are:

```bash
# Exact native baseline; this path forwards the original velocity unchanged.
bash bash/14_run_manifold_sampler.sh \
  configs/flux2_klein_4b_sampler_smoke.yaml --mode native

# Full consistency gradient through the transformer (high VRAM cost).
bash bash/14_run_manifold_sampler.sh \
  configs/flux2_klein_4b_sampler_smoke.yaml \
  --mode tangent --normal-estimator exact

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
- `tangent`: additionally preserve the configured reliability field to first
  order.

`normal_estimator: proxy` treats the model prediction as locally constant and
is the practical default for quantized/offloaded inference.
`normal_estimator: exact` differentiates `x0` consistency with respect to the
current latent in one transformer pass; use full-precision/full-CUDA hardware
when possible. Exact mode currently requires a single conditional pass
(`guidance_scale <= 1`).

Each sampler run writes images, the resolved config, environment information,
per-step norms and consistency diagnostics, and (when
`capture_trajectory: true`) `trajectory.pt` containing `x_k`, native velocity,
`x0`, proposal, and applied correction.

The default ablation deliberately includes both velocity- and `x0`-difference
proposals. Since `v_k-v_{k-1}` already lies in
`span(v_k, v_{k-1})`, its subspace projection is mathematically an identity;
the `x0` and seeded random controls are needed to test whether orientation,
rather than correction norm alone, drives stability. Non-native default
conditions are matched to correction norm `0.05`. The evaluation command
writes per-condition feature metrics plus a combined JSON/CSV table with
trajectory consistency and deltas against native sampling.

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
