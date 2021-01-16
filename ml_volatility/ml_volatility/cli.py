import argparse
import sys
from pathlib import Path
from ml_volatility.algos.extractor import Extractor


# Default paths for intermediate output files, meta paths
DATA_PATH: Path = Path(__file__).parent / "data"

DATA_PATH_RESULTS: Path = DATA_PATH


DATA_PATH.mkdir() if not DATA_PATH.exists() else None

MAIN_CMD_NAME = "volatility"


def get_prog_name() -> str:
    return MAIN_CMD_NAME + " " + sys.argv[1]


def run(data_path) -> None:

    save_link = DATA_PATH_RESULTS

    extractor_obj = Extractor(data_path, save_link)
    extractor_obj.run()


def run_cli() -> None:
    _ = argparse.ArgumentParser(
        prog=get_prog_name(),
        formatter_class=argparse.RawTextHelpFormatter,
        description="Volatility prediction.",
    )

    run(DATA_PATH)


def cli_strategy() -> None:
    argument_parser = argparse.ArgumentParser(
        prog=MAIN_CMD_NAME,
        formatter_class=argparse.RawTextHelpFormatter,
        description="Command line entry point for all the modules.",
    )
    argument_parser.add_argument(
        "strategy",
        help="Choose one of possible strategies. To see help for each of the strategies "
        'run with "-h" argument e.g.\n'
        f"{MAIN_CMD_NAME} run -h",
        choices=("run",),
    )
    args = argument_parser.parse_args(args=sys.argv[1:2])
    strategy_name_cli_func_map: dict = {
        "run": run_cli,
    }

    cli_func = strategy_name_cli_func_map[args.strategy]
    cli_func()


if __name__ == "__main__":
    run(DATA_PATH)
