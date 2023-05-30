"""
Miscellaneous functions,
for collecting experiment results, visualizing them, etc.
"""

from pathlib import Path
from typing import Union, Sequence, Optional, Tuple

import matplotlib.pyplot as plt

try:
    from fl_sim.nodes import Node
except ModuleNotFoundError:
    # not installed,
    # import from the submodule instead
    import sys

    sys.path.append(str(Path(__file__).parent / "fl-sim"))

    from fl_sim.nodes import Node


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
                part="val",
                metric="acc",
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
