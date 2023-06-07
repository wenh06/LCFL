"""
command line interface for experiments

Reads in a yaml file with experiment parameters and runs the experiment.
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent / "fl-sim"))

from fl_sim.data_processing import (
    FedCIFAR100,
    FedEMNIST,
    FedMNIST,
    FedShakespeare,
    FedProxSent140,
    FedProxFEMNIST,
    FedProxMNIST,
)
from fl_sim.algorithms import fedopt, fedprox
from fl_sim.cli import parse_config_file
from torch_ecg.cfg import CFG

from algorithm import LCFLServer, LCFLServerConfig, LCFLClientConfig
from algorithm_prox import (
    LCFLServer as LCFLProxServer,
    LCFLServerConfig as LCFLProxServerConfig,
    LCFLClientConfig as LCFLProxClientConfig,
)
from ifca import IFCAServer, IFCAServerConfig, IFCAClientConfig
from dataset import FedRotatedMNIST, FedRotatedCIFAR10


# create log directory if it does not exist
(Path(__file__).parent / ".logs").mkdir(exist_ok=True, parents=True)


def parse_args() -> List[CFG]:
    parser = argparse.ArgumentParser(
        description="LCFL Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "config_file_path",
        type=str,
        help="Config file (.yml or .yaml file) path",
    )

    args = vars(parser.parse_args())
    config_file_path = Path(args["config_file_path"]).resolve()

    assert config_file_path.exists(), f"Config file {config_file_path} not found"
    assert config_file_path.suffix in [".yml", ".yaml"], (
        f"Config file {config_file_path} must be a .yml or .yaml file, "
        f"but got {config_file_path.suffix}."
    )

    return parse_config_file(config_file_path)


def single_run(config: CFG):
    # run a single experiment
    config = CFG(config)
    config_bak = deepcopy(config)

    algorithm_pool = CFG(
        {
            "LCFL": {
                "server_config": LCFLServerConfig,
                "client_config": LCFLClientConfig,
                "server": LCFLServer,
            },
            "LCFL_Prox": {
                "server_config": LCFLProxServerConfig,
                "client_config": LCFLProxClientConfig,
                "server": LCFLProxServer,
            },
            "IFCA": {
                "server_config": IFCAServerConfig,
                "client_config": IFCAClientConfig,
                "server": IFCAServer,
            },
            "FedAvg": {
                "server_config": fedopt.FedAvgServerConfig,
                "client_config": fedopt.FedAvgClientConfig,
                "server": fedopt.FedAvgServer,
            },
            "FedProx": {
                "server_config": fedprox.FedProxServerConfig,
                "client_config": fedprox.FedProxClientConfig,
                "server": fedprox.FedProxServer,
            },
        }
    )

    dataset_pool = {
        c.__name__: c
        for c in [
            FedCIFAR100,
            FedEMNIST,
            FedMNIST,
            FedShakespeare,
            FedProxSent140,
            FedProxFEMNIST,
            FedProxMNIST,
            FedRotatedMNIST,
            FedRotatedCIFAR10,
        ]
    }

    # set random seed
    seed = config.pop("seed", None)  # global seed
    if config.dataset.get("seed", None) is None:
        config.dataset.seed = seed
    if config.algorithm.server.get("seed", None) is None:
        config.algorithm.server.seed = seed
    assert config.dataset.seed is not None and config.algorithm.server.seed is not None

    # dataset and model selection
    ds_cls = dataset_pool[config.dataset.pop("name")]
    ds = ds_cls(**(config.dataset))
    model = ds.candidate_models[config.model.pop("name")]

    # fill default values
    if (
        "batch_size" not in config.algorithm.client
        or config.algorithm.client.batch_size is None
    ):
        config.algorithm.client.batch_size = ds.DEFAULT_BATCH_SIZE
    if (
        "num_clients" not in config.algorithm.server
        or config.algorithm.server.num_clients is None
    ):
        config.algorithm.server.num_clients = ds.DEFAULT_TRAIN_CLIENTS_NUM

    # server and client configs
    server_config_cls = algorithm_pool[config.algorithm.name]["server_config"]
    client_config_cls = algorithm_pool[config.algorithm.name]["client_config"]
    server_config = server_config_cls(**(config.algorithm.server))
    client_config = client_config_cls(**(config.algorithm.client))

    # setup the experiment
    server_cls = algorithm_pool[config.algorithm.name]["server"]
    s = server_cls(
        model,
        ds,
        server_config,
        client_config,
        lazy=False,
    )

    s._logger_manager.log_message(f"Experiment config:\n{config_bak}")

    # s._setup_clients()

    # execute the experiment
    s.train_federated()

    # destroy the experiment
    del s, ds, model


def main():
    configs = parse_args()
    for config in configs:
        single_run(config)


if __name__ == "__main__":
    # typical usage:
    # nohup python -u cli.py {{config.yml}} > .logs/cli.log 2>&1 & echo $! > .logs/cli.pid
    # replace {{config.yml}} with the path to your config file,
    # e.g. ./configs/lcfl-rot-mnist.yml

    # full examples:
    # nohup python -u cli.py configs/lcfl-rot-mnist.yml > .logs/cli-mnist.log 2>&1 & echo $! > .logs/cli-mnist.pid
    # nohup python -u cli.py configs/lcfl-rot-cifar10.yml > .logs/cli-cifar10.log 2>&1 & echo $! > .logs/cli-cifar10.pid
    # nohup python -u cli.py configs/single-run-test.yml > .logs/cli-test.log 2>&1 & echo $! > .logs/cli-test.pid
    # nohup python -u cli.py configs/single-run-test-1.yml > .logs/cli-test-1.log 2>&1 & echo $! > .logs/cli-test-1.pid

    main()
