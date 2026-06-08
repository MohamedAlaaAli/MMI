# EMAP

Empirical Multimodally-Additive Projection (Hessel & Lee, EMNLP 2020). Given a trained two-modality model `f(a, b)`, EMAP returns the closest additive function `f̂(a, b) = g(a) + h(b)`. If `f̂`'s metrics match `f`'s, the model isn't using cross-modal interactions.

## Files

- `projection.py` -> pure math, model-agnostic.
- `evaluator.py` -> model/data-specific. **Edit this file to adapt EMAP.**
- `run.py` -> using CLI.

## Usage

```bash
python -m emap.run --config configs/unet_petct.json --ckpt <ckpt.pth> --n 500 --k 3
```

## Current assumptions (in `evaluator.py`)


| Assumption                                                                             | Edit if your setup differs                                                                       |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `model(x)` with `x` shape `[B, 2, H, W]`, channels = `[PET, CT]`                       | For two-input model, change the `cat` + forward call inside `run()` to `model(a, b)`.            |
| Output `[B, 1, H, W]` binary logits, thresholded with `sigmoid > 0.5`                  | For multi-class: change the `1` in the buffer shapes to `C` and replace threshold with `argmax`. |
| Checkpoint dict has `"model_state_dict"`                                               | Change the key in `_init_model`.                                                                 |
| Dataset = `PETCTSliceDataset`, returns `(image[2,H,W], mask[1,H,W])`, augmentation off | Swap dataset class. **Augmentation must be disabled** — `f(a_i, b_j)` must be deterministic.     |


## Cost

`N²` forward passes per repeat. `N=500` → ~10–30 min/repeat on one GPU. Increase `--k` if EMAP dice varies across repeats.