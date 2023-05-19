"""
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

import numpy as np  # noqa: F401
import torch  # noqa: F401
import torch.utils.data as data  # noqa: F401
import torchvision.transforms as transforms  # noqa: F401

try:
    from fl_sim.utils.misc import CACHED_DATA_DIR  # noqa: F401
    from fl_sim.models import nn as mnn  # noqa: F401
    from fl_sim.models.utils import top_n_accuracy
    from fl_sim.data_processing import FedVisionDataset
except ModuleNotFoundError:
    # not installed,
    # import from the submodule instead
    import sys

    sys.path.append(str(Path(__file__).parent / "fl-sim"))

    from fl_sim.utils.misc import CACHED_DATA_DIR  # noqa: F401
    from fl_sim.models import nn as mnn  # noqa: F401
    from fl_sim.models.utils import top_n_accuracy
    from fl_sim.data_processing import FedVisionDataset


_mnist_label_mapping = {i: str(i) for i in range(10)}

_ciFar10_label_mapping = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


class FedRotatedMNIST(FedVisionDataset):
    """
    To write

    """

    __name__ = "FedRotatedMNIST"

    def _preload(self, datadir: Optional[Union[str, Path]] = None) -> None:
        raise NotImplementedError

    def get_dataloader(
        self,
        train_bs: Optional[int] = None,
        test_bs: Optional[int] = None,
        client_idx: Optional[int] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        return [
            "n_class",
        ] + super().extra_repr_keys()

    def get_class(self, label: torch.Tensor) -> str:
        return _mnist_label_mapping[label.item()]

    def get_classes(self, labels: torch.Tensor) -> List[str]:
        return [_mnist_label_mapping[label.item()] for label in labels]

    def evaluate(self, probs: torch.Tensor, truths: torch.Tensor) -> Dict[str, float]:
        return {
            "acc": top_n_accuracy(probs, truths, 1),
            "top3_acc": top_n_accuracy(probs, truths, 3),
            "top5_acc": top_n_accuracy(probs, truths, 5),
            "loss": self.criterion(probs, truths).item(),
            "num_samples": probs.shape[0],
        }

    @property
    def url(self) -> str:
        raise NotImplementedError

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """
        a set of candidate models
        """
        raise NotImplementedError

    @property
    def doi(self) -> List[str]:
        raise NotImplementedError

    def view_image(self, client_idx: int, image_idx: int) -> None:
        raise NotImplementedError


class FedRotatedCIFAR10(FedVisionDataset):
    """
    To write

    """

    __name__ = "FedRotatedCIFAR10"

    def _preload(self, datadir: Optional[Union[str, Path]] = None) -> None:
        raise NotImplementedError

    def get_dataloader(
        self,
        train_bs: Optional[int] = None,
        test_bs: Optional[int] = None,
        client_idx: Optional[int] = None,
    ) -> Tuple[data.DataLoader, data.DataLoader]:
        raise NotImplementedError

    def extra_repr_keys(self) -> List[str]:
        return [
            "n_class",
        ] + super().extra_repr_keys()

    def get_class(self, label: torch.Tensor) -> str:
        return _ciFar10_label_mapping[label.item()]

    def get_classes(self, labels: torch.Tensor) -> List[str]:
        return [_ciFar10_label_mapping[label.item()] for label in labels]

    def evaluate(self, probs: torch.Tensor, truths: torch.Tensor) -> Dict[str, float]:
        return {
            "acc": top_n_accuracy(probs, truths, 1),
            "top3_acc": top_n_accuracy(probs, truths, 3),
            "top5_acc": top_n_accuracy(probs, truths, 5),
            "loss": self.criterion(probs, truths).item(),
            "num_samples": probs.shape[0],
        }

    @property
    def url(self) -> str:
        raise NotImplementedError

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """
        a set of candidate models
        """
        raise NotImplementedError

    @property
    def doi(self) -> List[str]:
        raise NotImplementedError

    def view_image(self, client_idx: int, image_idx: int) -> None:
        raise NotImplementedError
