import subprocess
import argparse
import os
import sys
import logging

from pathlib import Path
from typing import List

LOG_LEVEL = "INFO"
MAX_LINE_LENGTH = 88
LOGGING_FORMAT = (
    "\n"
    + "-" * MAX_LINE_LENGTH
    + "\n%(asctime)s - %(name)s - %(levelname)s: %(message)s\n"
    + "-" * MAX_LINE_LENGTH
    + "\n"
)
logging.basicConfig(format=LOGGING_FORMAT, level=LOG_LEVEL)
IS_WINDOWS: bool = os.name == "nt"


def exec_cmd(command) -> None:
    if not IS_WINDOWS:
        command = " ".join(command)

    try:
        cmd_exec_str = f'executing command "{command}"'
        logging.info(cmd_exec_str + "...")
        exec_code = subprocess.call(command, shell=True)
        if exec_code == 0:
            logging.info(cmd_exec_str + ": SUCCESS")
        else:
            logging.fatal(f"command failed, exiting with code {exec_code}")
            sys.exit(exec_code)
    except KeyboardInterrupt:
        logging.fatal(f"Execution has been interrupted")
        sys.exit()
    except Exception as e:
        logging.fatal(f"Command has failed with the following exception: {e}")
        raise e


CURRENT_CONDA_ENV_PATH = os.environ["CONDA_PREFIX"]
CURRENT_CONDA_ENV_NAME = os.environ["CONDA_DEFAULT_ENV"]


CURRENT_PATH = Path(__file__).absolute().parent

MAIN_CMD_NAME = __file__


def get_prog_name() -> str:
    return MAIN_CMD_NAME + " " + sys.argv[1]


def update_conda_env_from_relfile(
    conda_env_path: str, req_relpath: Path, debug: bool = False
) -> None:

    conda_command_envupdate = [
        "conda",
        "env",
        "update",
        "--debug" if debug else "",
        "-p",
        conda_env_path,
        "-f",
    ] + [str(CURRENT_PATH / req_relpath)]

    exec_cmd(conda_command_envupdate)


def pip_install_modules_by_relpath(module_relpath_list: List[Path]) -> None:
    pip_install_command = [
        "pip",
        "install",
        "--trusted-host",
        "pypi.org",
        "--trusted-host",
        "pypi.python.org",
        "--trusted-host",
        "files.pythonhosted.org",
        "-e",
    ]

    for module_path in module_relpath_list:
        exec_cmd(pip_install_command + [str(CURRENT_PATH / module_path)])


def update(conda_env_path, mode, debug=False):

    mode2cmd_dict = {
        "all": (True, True),
        "products": (False, True),
        "packages": (True, False),
    }
    is_cmd_run_list = mode2cmd_dict[mode]

    if is_cmd_run_list[0]:
        update_conda_env_from_relfile(
            conda_env_path, Path("python_requirements.yml"), debug
        )

    if is_cmd_run_list[1]:
        pip_install_modules_by_relpath([Path(CURRENT_PATH)])


def update_cli():

    # noinspection PyTypeChecker
    argument_parser_obj = argparse.ArgumentParser(
        prog=MAIN_CMD_NAME,
        formatter_class=argparse.RawTextHelpFormatter,
        description="Volatility Python Environment Installer",
    )

    argument_parser_obj.add_argument(
        "mode",
        help="What to update" 'run with "-h" argument e.g.\n',
        choices=("products", "all", "packages"),
    )

    argument_parser_obj.add_argument(
        "-d", "--debug", action="store_true", help="update with debug logs"
    )

    arg_vec = argument_parser_obj.parse_args(args=sys.argv[2:])

    update(CURRENT_CONDA_ENV_PATH, arg_vec.mode, arg_vec.debug)


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser(
        prog=MAIN_CMD_NAME,
        formatter_class=argparse.RawTextHelpFormatter,
        description="Volatility Python Environment Installer",
    )

    argument_parser.add_argument(
        "command",
        help="Choose a command. To see help for each command "
        'run with "-h" argument e.g.\n'
        f"{MAIN_CMD_NAME} update -h"
        f"{MAIN_CMD_NAME} install -h",
        choices=("update",),
    )

    args = argument_parser.parse_args(args=sys.argv[1:2])
    command_name_cli_func_map: dict = {"update": update_cli}

    cli_func = command_name_cli_func_map[args.command]
    cli_func()
