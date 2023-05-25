"""
command line interface for experiments

Reads in a yaml file with experiment parameters and runs the experiment.
"""

import argparse
import sys
from collections import OrderedDict
from copy import deepcopy
from itertools import product
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "fl-sim"))

import yaml
from fl_sim.data_processing import (  # noqa: F401
    FedCIFAR100,
    FedEMNIST,
    FedMNIST,
    FedShakespeare,
    FedProxFEMNIST,
    FedProxMNIST,
)  # noqa: F401

from dataset import FedRotatedMNIST, FedRotatedCIFAR10  # noqa: F401
from algorithm import LCFLServer, LCFLServerConfig, LCFLClientConfig  # noqa: F401
from ifca import IFCAServer, IFCAServerConfig, IFCAClientConfig  # noqa: F401


def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="LCFL Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config_file_path",
        nargs=1,
        type=str,
        help="Config file (.yml or .yaml file) path",
    )

    args = vars(parser.parse_args())
    config_file_path = Path(args["config_file_path"]).resolve()

    assert config_file_path.exists(), f"Config file {config_file_path} not found"
    assert config_file_path.suffix in ["yml", "yaml"], (
        f"Config file {config_file_path} must be a .yml or .yaml file, "
        f"but got {config_file_path.suffix}."
    )

    file_content = config_file_path.read_text()

    configs = yaml.safe_load(file_content)
    # further process configs to a list of configs
    # by replacing values of the pattern ${{ matrix.key }} with the value of key
    # specified by configs["strategy"]["matrix"][key]
    matrix = OrderedDict(configs["strategy"]["matrix"])
    repalcements = list(product(*matrix.values()))
    keys = list(matrix.keys())
    configs = []
    for r in repalcements:
        new_file_content = deepcopy(file_content)
        for k, v in zip(keys, r):
            # replace all patterns of the form ${{ matrix.k }} in file_content with v
            new_file_content = new_file_content.replace(
                f"${{{{ matrix.{k} }}}}", str(v)
            )
        new_config = yaml.safe_load(new_file_content)
        new_config.pop("strategy")
        configs.append(new_config)

    return configs


def single_run(config: dict):
    pass


def main():
    configs = parse_args()
    for config in configs:
        single_run(config)
