"""Command line: python main.py – all settings come from the one config*.yaml in the project folder."""

import sys

from tqdm import tqdm

from pore_detection import RunAborted, config_path, load_config, run


def main() -> int:
    try:
        config = load_config()
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2
    print(f"config: {config_path().name}")
    bar = None

    def progress(index: int, total: int, stage: str, name: str):
        nonlocal bar
        if bar is None:
            bar = tqdm(total=total, unit="image")
        bar.n = index
        bar.set_description(f"{name} {stage}" if name else stage)
        bar.refresh()

    try:
        result = run(config, progress=progress)
    except RunAborted as error:
        print(f"run aborted: {error}", file=sys.stderr)
        return 2
    finally:
        if bar is not None:
            bar.close()

    total = result.succeeded + result.failed
    print(f"{result.succeeded} of {total} images succeeded, {result.failed} failed "
          f"({result.duration_s:.1f} s) -> {result.results_dir}")
    for error in result.errors:
        print(f"  {error.image}: {error.stage} / {error.method}: {error.message}")
    return 1 if result.failed else 0


if __name__ == "__main__":
    sys.exit(main())
