"""
Miscellaneous functions,
for collecting experiment results, visualizing them, etc.
"""

import re
from pathlib import Path
from typing import Union, Sequence, Optional, Tuple, List

import matplotlib.pyplot as plt

try:
    from fl_sim.nodes import Node
    from fl_sim.utils.const import LOG_DIR
except ModuleNotFoundError:
    # not installed,
    # import from the submodule instead
    import sys

    sys.path.append(str(Path(__file__).parent / "fl-sim"))

    from fl_sim.nodes import Node
    from fl_sim.utils.const import LOG_DIR


__all__ = [
    "find_log_files",
    "get_config_from_log",
    "plot_curve",
]


def find_log_files(
    directory: Union[str, Path] = LOG_DIR, filters: str = "", show: bool = False
) -> Union[List[Path], None]:
    """Find log files in the given directory, recursively.

    Parameters
    ----------
    directory : Union[str, pathlib.Path], default fl_sim.utils.const.LOG_DIR
        The directory to search for log files.
    filters : str, default ""
        Filters for the log files.
        Only files fitting the pattern of `filters` will be returned.
    show : bool, default False
        Whether to print the found log files.
        If True, the found log files will be printed and **NOT** returned.

    Returns
    -------
    List[pathlib.Path]
        The list of log files.

    """
    log_files = [
        item
        for item in Path(directory).rglob("*.json")
        if item.is_file() and re.search(filters, item.name)
    ]
    if show:
        for idx, fn in enumerate(log_files):
            print(idx, "---", fn.stem)
    else:
        return log_files


def get_config_from_log(file) -> dict:
    file = Path(file)
    if not file.exists():
        print("File not found")
        return {}
    if file.suffix == ".json":
        file = file.with_suffix(".txt")
    if not file.exists():
        print("Corresponding text log file not found")
        return {}
    contents = file.read_text().splitlines()
    flag = False
    for idx, line in enumerate(contents):
        if "FLSim - INFO - Experiment config:" in line:
            flag = True
            break
    if flag:
        return eval(contents[idx + 1])
    else:
        print("Config not found")
        return {}


def plot_curve(
    files: Union[str, Path, Sequence[Union[str, Path]]],
    part: str = "val",
    metric: str = "acc",
    fig_ax: Optional[Tuple[plt.Figure, plt.Axes]] = None,
    labels: Union[str, Sequence[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    curves = []
    stems = []
    if isinstance(files, (str, Path)):
        files = [files]
    for file in files:
        curves.append(
            Node.aggregate_results_from_json_log(
                file,
                part=part,
                metric=metric,
            )
        )
        stems.append(Path(file).stem)
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig, ax = fig_ax
    plot_config = dict(marker="*")
    if labels is None:
        labels = stems
    for idx, curve in enumerate(curves):
        plot_config["label"] = labels[idx]
        ax.plot(curve, **plot_config)
    ax.legend(loc="best", fontsize=18)
    ax.set_xlabel("Global Iter.", fontsize=14)
    ax.set_ylabel(f"{part} {metric}", fontsize=14)
    return fig, ax
