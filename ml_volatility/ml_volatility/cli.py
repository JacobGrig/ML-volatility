import numpy as np
import argparse
import sys
from pathlib import Path
from ml_volatility.algos.extractor import Extractor


# Default paths for intermediate output files, meta paths
DATA_PATH: Path = Path(__file__).parent / "data"

DATA_PATH_DATASETS: Path = DATA_PATH / "full"
DATA_PATH_TESTDATA: Path = DATA_PATH / "test"
DATA_PATH_RESULTS_PROD: Path = DATA_PATH_DATASETS / "results"
DATA_PATH_RESULTS_TEST: Path = DATA_PATH_TESTDATA / "results"


DATA_PATH.mkdir() if not DATA_PATH.exists() else None
DATA_PATH_DATASETS.mkdir() if not DATA_PATH_DATASETS.exists() else None
DATA_PATH_TESTDATA.mkdir() if not DATA_PATH_TESTDATA.exists() else None
DATA_PATH_RESULTS_PROD.mkdir() if not DATA_PATH_RESULTS_PROD.exists() else None
DATA_PATH_RESULTS_TEST.mkdir() if not DATA_PATH_RESULTS_TEST.exists() else None

MAIN_CMD_NAME = "volatility"


def get_prog_name() -> str:
    return MAIN_CMD_NAME + " " + sys.argv[1]


def run(data_path_vec, test=True) -> None:

    doc_type_vec = []

    for (i_path, path) in enumerate(data_path_vec):
        doc_type_vec.append(Path(path).suffix)

    save_link = DATA_PATH_RESULTS_TEST if test else DATA_PATH_RESULTS_PROD

    extractor_obj = Extractor(
        np.array(doc_type_vec), data_path_vec, save_link
    )
    extractor_obj.run()


def run_testperf_cli() -> None:
    _ = argparse.ArgumentParser(
        prog=get_prog_name(),
        formatter_class=argparse.RawTextHelpFormatter,
        description="Running tests for volatility prediction.",
    )

    test_data_vec = np.array(
        [
            np.str(file)
            for file in (DATA_PATH_TESTDATA / "csv").glob("**/*")
            if file.is_file()
        ]
    )

    run(test_data_vec, test=True)


def run_calcprod_cli() -> None:
    _ = argparse.ArgumentParser(
        prog=get_prog_name(),
        formatter_class=argparse.RawTextHelpFormatter,
        description="Starts the volatility in production mode.",
    )

    prod_data_vec = np.array(
        [
            np.str(file)
            for file in (DATA_PATH_DATASETS / "csv").glob("**/*")
            if file.is_file()
        ]
    )

    run(prod_data_vec, test=False)


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
        f"{MAIN_CMD_NAME} calcprod -h"
        f"{MAIN_CMD_NAME} testperf -h",
        choices=("calcprod", "testperf"),
    )
    args = argument_parser.parse_args(args=sys.argv[1:2])
    strategy_name_cli_func_map: dict = {
        "testperf": run_testperf_cli,
        "calcprod": run_calcprod_cli,
    }

    cli_func = strategy_name_cli_func_map[args.strategy]
    cli_func()


if __name__ == "__main__":
    data_vec = np.array(
        [
            np.str(file)
            for file in (DATA_PATH_DATASETS / "csv").glob("**/*")
            if file.is_file()
        ]
    )
    run(data_vec, test=True)
