# ReDiS

Initial FLUX.2 Klein 4B harness for the experiments in `plans/plan.md`.

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

The decisive screening is fixed to three sites (A/B/C), eight prompts, eight
seeds, and per-seed correction norms 0.005/0.010/0.020. Run one site per
process. Its evaluator reports output diversity (DINO, DreamSim, LPIPS) against
fidelity (CLIP-T, HPSv2); hidden effective rank remains a mechanism diagnostic.

Useful environment variables:

- `HF_TOKEN`: optional Hub token (the 4B checkpoint is public).
- `HF_HOME`: Hugging Face cache location.
- `REDIS_OUTPUT_ROOT`: output root; defaults to `outputs`.
- `REDIS_INSTALL_QUANT=0`: install without bitsandbytes.
