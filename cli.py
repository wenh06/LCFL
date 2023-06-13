"""
command line interface for experiments

Reads in a yaml file with experiment parameters and runs the experiment.
"""

try:
    from fl_sim.cli import main as fl_sim_main
except ModuleNotFoundError:

    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent / "fl-sim"))
    from fl_sim.cli import main as fl_sim_main

# import the following modules to register them with the registry
import algorithm  # noqa: F401
import algorithm_prox  # noqa: F401
import dataset  # noqa: F401


# create log directory if it does not exist
(Path(__file__).parent / ".logs").mkdir(exist_ok=True, parents=True)


def main():
    fl_sim_main()


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
