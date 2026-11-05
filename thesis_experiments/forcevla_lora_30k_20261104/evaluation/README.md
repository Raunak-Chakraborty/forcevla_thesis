# ForceVLA LoRA — 30,000-update validation

Source baseline: cdf4677244513a52665a16f5319f662379ebe9bc
Experiment: LORA_5H_20261104

Dataset: Doosan Siemens/Cranfield assembly demonstrations.
Train: 255 episodes. Validation: 34 episodes, indices 255–288.
Test split: NOT evaluated.

Protocol: one midpoint per validation episode, 50-step
prediction horizon, normalized seven-dimensional action MAE.

The 2,000-step pilot MAE was 0.800993.
The 30,000-step model MAE was 0.591615.
Relative improvement: 26.14%.

Best observed validation checkpoint: 30,000.
Backup: 20,000 (MAE 0.604934).

This is an open-loop evaluation, not a physical
insertion-success measurement. Recorded-force benefits
were small and require further investigation.

Full Orbax checkpoints remain on the workstation SSD.
This Git archive intentionally excludes model weights,
optimizer tensors, and video data.

The evaluator source retains paths to the original lab
scripts; this folder also archives those scripts for
scientific provenance.
