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

Run checks and experiments one at a time:

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
