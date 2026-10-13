# Installing and running the RTRM answer-storedness probe

Two supported setups. Both use a Python `venv` — no conda, no system-wide
packages.

- **A: Jetson Orin (8 GB), JetPack 6** — GPU probe, ~1 minute for 60 pairs.
- **B: Windows 11, no GPU** — CPU probe, slower but works; precomputed
  results let you skip the compute entirely.

All 60 validation results are precomputed in `results/validation_60.json`,
so you can reproduce every table in the README without training anything
or running the probe at all.

---

## A. Jetson Orin 8 GB, JetPack 6

Tested on: JetPack 6.2 (L4T R36.4.x), Python 3.10, PyTorch 2.8.0 (NVIDIA
Jetson aarch64 build, CUDA-enabled).

**1. Check your JetPack version** (the PyTorch wheel must match it):

```bash
cat /etc/nv_tegra_release
# R36 (release), REVISION: 4.x  ->  JetPack 6.x
python3 --version
# want 3.10.x (3.11 works too, but match the wheel below to your Python)
```

**2. Create the venv** (in your home directory, or wherever you keep
project environments):

```bash
python3 -m venv ~/rtrm-venv
source ~/rtrm-venv/bin/activate
python -m pip install --upgrade pip
```

**3. Install PyTorch — NVIDIA Jetson wheel (this is the important step).**

Plain `pip install torch` gives you a CPU-only aarch64 wheel: it installs
fine but `torch.cuda.is_available()` returns `False` and the probe runs
~20x slower. For the GPU build, download the wheel NVIDIA built for your
JetPack from their redist index:

- Browse `https://developer.download.nvidia.com/compute/redist/jp/`
  and open the folder for your JetPack major version (`v62/` for
  JetPack 6.2, `v61/` for 6.1, `v60/` for 6.0), then `pytorch/`.
- Download the wheel matching your Python (`cp310` for Python 3.10)
  and architecture (`linux_aarch64`), e.g.:

```bash
wget <paste-the-wheel-URL-for-your-JetPack-and-Python-here>
pip install <downloaded-wheel-file>.whl
```

(Alternative index, same wheels: `https://pypi.jetson-ai-lab.dev/`
— pick the `jp6` path matching your CUDA version.)

Verify the GPU build:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# want: 2.x.x True
```

If it prints `False`, you have the CPU wheel — remove it
(`pip uninstall torch`) and redo step 3 with the NVIDIA wheel.

**4. Install the remaining packages** (plain PyPI is fine for these):

```bash
pip install transformers
```

**5. Run the probe.** Clone this repo, `cd` into this folder, activate
the venv, and probe any HuggingFace causal-LM id or local model path:

```bash
source ~/rtrm-venv/bin/activate
python rtrm_probe.py --model gpt2 --pairs demo_pairs.json --out my_results.json
python rtrm_probe.py --model gpt2 --pairs demo_pairs.json --layers 5,7,9,11
```

The first run downloads GPT-2 (~500 MB) from HuggingFace into
`~/.cache/huggingface` — one-time cost, needs internet. GPT-2 small fits
comfortably in the Orin Nano's 8 GB.

---

## B. Windows 11, no GPU

**1. Install Python 3.10 or newer** from python.org. During setup, tick
"Add python.exe to PATH".

**2. Create the venv** (in PowerShell):

```powershell
py -m venv $HOME\rtrm-venv
$HOME\rtrm-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell refuses to run the activate script, run once as
Administrator: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

**3. Install the CPU packages** (Windows PyTorch wheels are CPU-only by
default — no special index needed):

```powershell
pip install torch transformers
```

Verify:

```powershell
python -c "import torch; print(torch.__version__)"
```

**4. Run the probe** (Command Prompt or PowerShell, venv activated):

```powershell
python rtrm_probe.py --model gpt2 --pairs demo_pairs.json --out my_results.json
```

On CPU, allow up to ~20 minutes for 60 pairs (each pair is one forward
pass per requested layer). For a quick smoke test, make a 3-pair file
first. Or skip the compute: the next section reproduces the published
validation with zero torch required.

---

## Reproducing the published validation (no compute)

`results/validation_60.json` holds the full 60-pair probe output from the
README's validation table (mix_e2 model, layers 5/7/9/11, correct
first-piece rank, verdict, per-layer tops, and the behavioral outcome per
pair). It was generated with this exact `rtrm_probe.py`.

Print the summary table with plain Python — no torch, no model, no GPU:

```bash
# Jetson (venv not even needed — uses system python3):
python3 show_validation.py
# Windows:
python show_validation.py
```

Expected output:

```
group                          n   median rank   behavioral
expedition, reverse-trained   25         1.0      25/25 correct
expedition, reverse-untrained 25       675        0/25 correct
discovery, reverse-trained     5         1.0      5/5 correct
discovery, reverse-untrained   5       587        0/5 correct
```

(Trained max rank 9 < untrained min rank 13 — zero overlap.)

To re-run the probe yourself instead of trusting the file, build the
pairs from `results/validation_60.json` (each record's `prompt` and
`answer` fields) and run `rtrm_probe.py` against your own model with
`--layers 5,7,9,11`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is `False` (Jetson) | CPU-only torch from PyPI | Uninstall, redo step A.3 with the NVIDIA Jetson wheel |
| `pip install` fails on the wheel filename | Python/wheel tag mismatch | `python3 --version` must match the wheel's `cp310`/`cp311` tag |
| HuggingFace download stalls | No internet / proxy | The probe needs one-time access to huggingface.co for `gpt2` |
| Out of memory on the 8 GB Nano | Another process holds GPU memory | Close other apps; the probe alone uses ~1 GB |
| `Activate.ps1` blocked (Windows) | Execution policy | See step B.2 one-time policy command |
