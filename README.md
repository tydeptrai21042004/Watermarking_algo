# DIAL-QIM — One final proposal + five published-method baselines + ablations

**One proposal:** DIAL-QIM (*dual-moment integer algebraic lattice QIM*). Its only production implementation is `src/dial_qim.py`. It performs exact two-pixel integer embedding of two raw bits per pair, with a packed 8-bit residue kernel and an exact 16-bit fallback. No historical G-CGIM, SCOPE, contrast-carrier, CIRMM proposal, or alternative candidate is shipped.

**Exactly five baselines** (paper-equation reproductions with explicitly declared image adapters):

1. `Scalar-QIM` — Chen and Wornell (2001), https://doi.org/10.1109/18.923725
2. `STDM-QIM` — Chen and Wornell (2001), spread-transform dither modulation, same DOI
3. `SCS` — Eggers et al. (2003), https://doi.org/10.1109/TSP.2003.809366
4. `ISS` — Malvar and Florencio (2003), https://doi.org/10.1109/TSP.2003.809385
5. `SS` — conventional spread-spectrum baseline evaluated as the control in Malvar and Florencio (2003), same DOI

The `STDM-QIM`, `ISS`, and `SS` baselines are openly declared **8×8 luminance-DCT image adapters** of the published equations. They are **not** authors' official implementation files or exact reproductions of the papers' original experimental pipelines. `Scalar-QIM` and `SCS` use a separately declared 8×8-block sampling adapter. Different adapters may have different runtime costs. Baselines do **not** receive test-attack-specific settings.

## Quick start (Ubuntu)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest -q tests
python3 scripts/run_comparison.py --quick
python3 scripts/run_ablations.py --hosts airplane --attacks focused --retune-for-feasibility
python3 scripts/run_generalization.py --quick
```

## Full experiments

```bash
# Exactly 1 proposal + 5 baselines, 8 held-out hosts, 60 attacks per host.
python3 scripts/run_comparison.py

# Same proposed algorithm with controlled component removals; 8 hosts, 60 attacks.
python3 scripts/run_ablations.py --retune-for-feasibility

# No retuning: strictly identical h for the component ablations.
python3 scripts/run_ablations.py --output results/ablations_fixed_h

# Generalization across pixel formats / image and payload sizes.
python3 scripts/run_generalization.py

# Warm CPU benchmark against an optimized compiled Scalar-QIM kernel.
python3 scripts/benchmark_cpu.py
```

Results are generated in `results/` after execution; the archive **does not ship historical results or precomputed score tables**. Infeasible ablations and baseline quality/clean-BER failures are recorded; they are not silently dropped or called successes. Experiments use the original deterministic 5 families × 4 attack types × 3 strengths protocol and a separate clean condition.

## Ablation controls (not new proposed methods)

The production algorithm remains DIAL-QIM. `src/ablation_controls.py` only supports controlled experiments; it is **not** a collection of competing new methods.

| ID | Controlled removal/change | Scientific question |
|---|---|---|
| `full` | No removal | Main result |
| `arithmetic_projection` | Replace packed lookup by exact integer division | How much time does the packed finite-residue implementation save? **Images must be pixel-identical** |
| `single_branch` | Remove minimum-distortion choice between parity branches | What does the exact two-branch optimum contribute to PSNR, BER and runtime? |
| `static_phase` | Hold phase constant over every carrier | Does keyed phase cycling matter? |
| `sequential_carriers` | Use direct sequential disjoint carriers instead of affine keyed positions | What is the effect of the nonrepeating keyed carrier schedule? |
| `fixed_h8` | Fix `h=8` rather than the proposal's clean-host choice | Contribution of automatic parameter selection |

Ablations hold the original host, payload bits, attack randomness and quantizer spacing constant by default. Thus some variants may be **infeasible at 55 dB**, which is meaningful. Use `--retune-for-feasibility` to lower only the clean-host spacing for such variants; the CSV records `retuned_from` and the actual `h`. Do not interpret robustness changes between mismatched PSNR as intrinsically superior modulation.

## Directory map

```text
src/dial_qim.py                 # one production proposal
src/attack_certificate.py       # exact bounded-additive-integer attack certificate
src/ablation_controls.py        # evaluation-only component removals
src/attack_suite60.py           # fixed 60-attack protocol
src/cirmm_attacks.py            # source attack operators only; NOT a proposal method
src/ss_malvar2003.py            # SS classical baseline adapter
src/qr64_certified/...          # common DCT modules and classical STDM/ISS adapters
baselines/registry.py           # EXACTLY 5 published-method baselines
scripts/run_comparison.py       # 1 + 5, dev/test separated
scripts/run_ablations.py        # controlled ablations
scripts/run_generalization.py   # image/payload generalization
scripts/benchmark_cpu.py        # copied-output and kernel speed
tests/                         # exact proofs and smoke tests
data/host/                      # included original 512x512 BMP hosts
data/watermark/                 # sample payload images
```

## Guarantees and limitations

- Successful embedding uses a disjoint two-pixel carrier, exact bounded-integer projection, and metadata-assisted **aligned blind decoding**. The original image is not needed for extraction.
- Actual integer SSE is checked against the requested PSNR floor. Some payload/PSNR combinations are infeasible and **raise an error**; no universal 55 dB guarantee is possible for all inputs.
- No ECC, repetition, attack classifier, attack-conditioned embedding, denoising, DCT/FFT in the proposal, or geometric search. Baseline DCT transforms remain part of their respective classical published principles.
- Additive-integer attack certificates are exact only for the stated aligned bounded integer perturbation model—not JPEG, rotation, cropping or adversarial knowledge of a secure key.
- The affine carrier selection is invertible but **not a cryptographically secure permutation**.
- Speed advantage and robustness depend on the benchmark. No claim of "fastest watermarking ever" or first-in-literature lattice/QIM invention is made.

This package retains the upstream code's licensing notice in `LICENSE_BASE_CODE`.
