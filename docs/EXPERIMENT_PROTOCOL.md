# Experiment protocol and attribution

## Comparison

- One and only one proposal: DIAL-QIM, implemented in `src/dial_qim.py`.
- Five paper-derived controls: Scalar-QIM, STDM-QIM, SCS, ISS, SS.
- Training/development hosts: `house`, `pepper`, `sailboat`, `safari`.
- Evaluation hosts: `airplane`, `athens`, `baboon`, `butfly`, `Girl`, `manhatan`, `milkdrop`, `tiffany`.
- Fixed payload: 4,096 independent random raw bits per host; no ECC, repetition or CRC.
- Attack set: 60, from five families, four attack operators each, three severity levels. A clean baseline is evaluated separately.
- Baseline strength grid is declared in code, tuned only on the development hosts using clean PSNR and clean BER. Evaluation images are not used to pick parameters, and attacked BER is never used in tuning.
- Each experiment reports achieved PSNR, clean BER, runtime and attacked BER. Any failure to meet 55 dB or achieve zero clean BER is explicitly flagged as a constraint violation, not hidden.
- Speed should be reported independently for full copy-output and in-place modification. Report exact CPU, Python, libraries, warm-up protocol, payload and image dimensions in any paper.

## Ablation controls

The six rows are full DIAL-QIM and five *removals of individual components*. They are not six new proposed schemes. The `arithmetic_projection` control tests an alternative **evaluation of the same mathematical projector**; it must produce identical marked pixels and BER. The `single_branch` control removes the globally nearest branch and usually needs a lower spacing to meet the fidelity constraint. `static_phase` and `sequential_carriers` isolate phase spreading and location choice. `fixed_h8` isolates parameter selection.

The default fixed-`h` comparison is strongest for causal component attribution, but infeasibility is a valid outcome. `--retune-for-feasibility` permits clean-only retuning and reports the changed step; it is not an equal-step test anymore.

## Bibliographic provenance

- B. Chen and G. W. Wornell, *Quantization Index Modulation: A Class of Provably Good Methods for Digital Watermarking and Information Embedding*, IEEE Transactions on Information Theory 47 (2001). DOI: 10.1109/18.923725.
- J. J. Eggers, R. Bäuml, R. Tzschoppe and B. Girod, *Scalar Costa Scheme for Information Embedding*, IEEE Transactions on Signal Processing 51 (2003). DOI: 10.1109/TSP.2003.809366.
- H. S. Malvar and D. A. F. Florêncio, *Improved Spread Spectrum: A New Modulation Technique for Robust Watermarking*, IEEE Transactions on Signal Processing 51 (2003). DOI: 10.1109/TSP.2003.809385. SS refers to the conventional comparison equation, ISS to the improved scheme.

**Caution:** Paper-equation adapters are not official reference implementations. The provided host images are local benchmark assets; their presence does not by itself establish a redistribution license for external publication.
