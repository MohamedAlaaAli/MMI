"""CLI for running EMAP on a trained checkpoint.

    python -m emap.run --config configs/unet_petct.json \
                       --ckpt ckpts/Unet_best_model.pth \
                       --n 500 --k 3
"""
import argparse

from .evaluator import EMAPEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Training JSON config")
    parser.add_argument("--ckpt", required=True, help=".pth checkpoint with model_state_dict")
    parser.add_argument("--n", type=int, default=500, help="Slices per subsample")
    parser.add_argument("--k", type=int, default=1, help="Number of independent subsamples")
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    evaluator = EMAPEvaluator(
        config_path=args.config,
        ckpt_path=args.ckpt,
        n_samples=args.n,
        n_repeats=args.k,
        seed=args.seed,
    )
    out = evaluator.run()
    avg = out["avg"]

    print()
    print(f"=== EMAP results (N={out['n_samples']}, k={out['n_repeats']}) ===")
    print(f"{'':<12}{'dice':>10}{'precision':>12}{'recall':>10}{'hd95':>10}")
    for label, m in [("Full", avg["full"]), ("EMAP", avg["emap"])]:
        print(f"{label:<12}{m['dice']:>10.4f}{m['precision']:>12.4f}{m['recall']:>10.4f}{m['hd95']:>10.4f}")


if __name__ == "__main__":
    main()
