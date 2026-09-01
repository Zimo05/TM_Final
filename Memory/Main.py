"""Compatibility entry point for Hawkes cold-start construction.

The implementation lives in :mod:`Train.ConstructTree`; keeping this thin
module preserves the historical ``python Main.py`` command without maintaining
a second copy of the trainer.
"""

from Train.ConstructTree import ConstructMemoryTree, parse_args


def main() -> None:
    args = parse_args()
    constructor = ConstructMemoryTree(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    constructor.cold_start_hawkes(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
