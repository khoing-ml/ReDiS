# Running ReDiS on Colab Pro+

Select a GPU runtime first: **Runtime → Change runtime type → GPU**. GPU models
are allocated dynamically by Colab, so the setup script detects the actual GPU
instead of assuming that Pro+ always receives an A100.

## Notebook cells

```python
!git clone https://github.com/YOUR_USER/YOUR_REPO.git
%cd YOUR_REPO
!bash bash/00_setup_colab.sh
```

The setup installs into Colab's disposable Python runtime and records that
interpreter under `.runtime/python_path`. It deliberately does not replace
PyTorch or the CUDA runtime. This also avoids relying on `ensurepip`, which is
not functional in every Colab runtime image.

Some Colab images contain an optional `torchao` build that is incompatible
with the pinned Diffusers commit. ReDiS does not use TorchAO; setup probes its
API and removes it only when incompatible. The required bf16 and bitsandbytes
paths are unaffected.

The project supports Python 3.10 through 3.13 so it works with both current
and recent Colab runtime images.

Run checks and the row-normalized diagnostic one at a time:

```python
!bash bash/02_check_model_access.sh .runtime/colab_baseline.yaml
!bash bash/03_generate_baseline.sh .runtime/colab_baseline.yaml
!bash bash/04_test_wrapper_identity.sh .runtime/colab_baseline.yaml
!bash bash/05_capture_hidden.sh .runtime/colab_capture.yaml
!bash bash/06_test_gaussian_noise.sh .runtime/colab_intervention.yaml
!bash bash/07_test_residual_amplification.sh .runtime/colab_intervention.yaml
!bash bash/08_test_projected_amplification.sh .runtime/colab_intervention.yaml
!bash bash/09_test_gram_isotropization.sh .runtime/colab_intervention.yaml
```

## Trajectory-tangent sampler

The setup also writes `.runtime/colab_sampler.yaml`. Start with the proxy
normal, which works with quantization and CPU/disk offload:

```python
!bash bash/14_run_manifold_sampler.sh .runtime/colab_sampler.yaml
!bash bash/15_run_sampler_ablation.sh .runtime/colab_sampler.yaml
```

The ablation uses matched seeds and compares native, ambient, subspace-only,
and full tangent corrections. Evaluate every condition directory with:

```python
!bash bash/16_evaluate_sampler_ablation.sh outputs/sampler_ablation/RUN_TIMESTAMP
```

Only try `--normal-estimator exact` on a high-memory, non-quantized profile.
It enables gradients through the 4B transformer and is intentionally not the
automatic Colab default.

Each captured view now logs raw and per-seed row-normalized effective rank,
view RMS, and view energy relative to the full residual. The low-frequency
diagnostic and intervention use the same 2x2 average-and-lift operator.

## Decisive screening

Do not run another layer/rank/beta sweep. Run one fixed site at a time. Start
with site A; B and C are follow-ups:

```python
!bash bash/11_run_screening_site.sh A
# later, only if needed:
!bash bash/11_run_screening_site.sh B
!bash bash/11_run_screening_site.sh C
```

The generated `.runtime/colab_screening.yaml` fixes eight prompts, eight seeds,
`beta=0.5`, projection rank 256, projection seed 0, and target per-seed
correction norms 0.005/0.010/0.020. Every invocation patches exactly one of:

- A: `single_transformer_blocks.19`, timestep 0
- B: `single_transformer_blocks.4`, timestep 2
- C: `single_transformer_blocks.14`, timestep 3

It compares identity, Gaussian noise, residual amplification, full-hidden
isotropization, low-frequency ReDiS, and token-pooled ReDiS. `rms_match` is off
in this screening so it cannot introduce an extra full-hidden scaling term;
the final correction itself is norm-matched independently for every seed.

After generation, install the heavier evaluation-only packages and evaluate
the returned run directory:

```python
!bash bash/12_setup_output_metrics.sh
!bash bash/13_evaluate_screening.sh outputs/screening_site_A/RUN_TIMESTAMP
```

This writes per-prompt DINO, DreamSim, LPIPS, CLIP-T, and HPSv2 scores,
condition-level deltas against identity, a CSV summary, and
`dino_diversity_vs_fidelity.png`. HPSv2 is installed without its benchmark-only
pytest/protobuf pins, so it does not downgrade the working experiment stack.

For an older single-prompt run, evaluate DINO image diversity, CLIP image diversity, and CLIP prompt alignment
for any output directory containing at least two seed images:

```python
!bash bash/10_evaluate_feature_diversity.sh outputs/gram_isotropization/RUN_TIMESTAMP
```

The first model command downloads a large checkpoint. To persist the Hugging
Face cache across sessions, mount Drive before setup and set `HF_HOME`:

```python
from google.colab import drive
drive.mount("/content/drive")
%env HF_HOME=/content/drive/MyDrive/huggingface
```

Using Drive avoids re-downloading, but local Colab storage is usually faster.
Persist `outputs/` separately before the runtime terminates.

## Automatic hardware profiles

- At least 20 GiB VRAM: bf16/fp16, full CUDA, 512 px.
- 13–20 GiB VRAM with at least 24 GiB host RAM: component CPU offload,
  512 px.
- Below 13 GiB VRAM: bitsandbytes 4-bit with automatic CPU/disk offload,
  256 px.
- A 13–20 GiB GPU with less than 24 GiB host RAM also falls back to 4-bit.
- GPUs without bf16 support automatically use fp16.

Inspect `.runtime/colab_hardware.json` and the generated YAML files before a
long run. For the final diagnosis, increase the seed group from the smoke-test
sizes to B=8 or greater only after monitoring peak VRAM.
