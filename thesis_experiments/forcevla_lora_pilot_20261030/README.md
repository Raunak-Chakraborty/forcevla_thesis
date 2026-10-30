# ForceVLA Doosan LoRA pilot — October 30, 2026

## Training
- Architecture: ForceVLA with LoRA and force/LIMoE modules
- Completed updates: 2,000
- GPU configuration: two RTX A5000, data parallel
- Global batch size: 2
- CPU data-loader workers: 4
- Training episodes: 255
- Validation episodes: 34
- State: 25D, wrench indices 19:25
- Internal actions: 32D, semantic robot actions: first 7D
- Final checkpoint: step 1999 (stored locally, not in Git)

## Evaluation
Expanded validation: 34 episodes, 102 observations,
204 paired seed trials.

Normalized seven-channel action MAE:
- Recorded wrench: 0.843244
- Normalized-zero wrench: 0.838670
- Cross-episode shuffled wrench: 0.847909

Recorded-minus-zero MAE: +0.004574
Episode-bootstrap 95% interval: [-0.004702, +0.013494]

No reliable improvement from recorded force established.

## Limitations
- No explicit contact phase labels in exported metadata
- Cable/motion-induced wrench disturbances possible
- The zero-wrench ablation uses normalized zero,
  equivalent to training-set mean wrench
- Current training loss includes padded action channels
- Offline action MAE does not measure insertion success
- No separately trained force-disabled baseline
- Exact data-sampler resume not implemented

## Pending
- SSD headroom review
- Contact-aware evaluation
- Matched force-free training control
- Long-run checkpoint retention and schedule approval

No longer fine-tuning run has been authorized.
