from __future__ import annotations

import argparse
from pathlib import Path

from bcosgnn.data.zinc_dichloro import BuildConfig, build_zinc_dichloro_dataset, save_dataset_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-scaffolds", type=int, default=2_000, help="Number of scaffolds (total graphs = 3*num_scaffolds)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-total-atoms", type=int, default=30)
    parser.add_argument("--max-total-atoms", type=int, default=50)
    parser.add_argument("--max-degree", type=int, default=6)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--use-rings-zinc-pickles", action="store_true", default=True)
    parser.add_argument("--rings-zinc-raw-dir", type=str, default="RINGS_v2/data/ZINC/raw")
    parser.add_argument("--no-fallback-random-alkanes", action="store_true", help="Disable random scaffold fallback")
    parser.add_argument("--out", type=str, default="data/zinc_dichloro/processed/splits.pt")
    args = parser.parse_args()

    cfg = BuildConfig(
        num_scaffolds=args.num_scaffolds,
        seed=args.seed,
        min_total_atoms=args.min_total_atoms,
        max_total_atoms=args.max_total_atoms,
        max_degree=args.max_degree,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        use_rings_zinc_pickles=args.use_rings_zinc_pickles,
        rings_zinc_raw_dir=args.rings_zinc_raw_dir,
        fallback_random_alkanes=not args.no_fallback_random_alkanes,
    )

    splits = build_zinc_dichloro_dataset(cfg)
    out_path = Path(args.out)
    save_dataset_splits(splits, out_path)
    print(f"Saved: {out_path}")
    for k, v in splits.items():
        print(k, len(v))


if __name__ == "__main__":
    main()
