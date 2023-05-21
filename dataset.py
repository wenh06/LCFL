"""
Datasets used in the numerical experiments in
"An Efficient Framework for Clustered Federated Learning"

Reference
---------
1. https://github.com/jichan3751/ifca
2. https://arxiv.org/abs/2006.04088

"""

import pickle
import posixpath
import gzip
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

import numpy as np
import requests
import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate

try:
    from fl_sim.utils.const import (
        CACHED_DATA_DIR,
        MNIST_LABEL_MAP,
        CIFAR10_LABEL_MAP,
    )
    from fl_sim.utils._download_data import http_get
    from fl_sim.models import nn as mnn
    from fl_sim.models.utils import top_n_accuracy
    from fl_sim.data_processing import FedVisionDataset
except ModuleNotFoundError:
    # not installed,
    # import from the submodule instead
    import sys

    sys.path.append(str(Path(__file__).parent / "fl-sim"))

    from fl_sim.utils.const import (
        CACHED_DATA_DIR,
        MNIST_LABEL_MAP,
        CIFAR10_LABEL_MAP,
    )
    from fl_sim.utils._download_data import http_get
    from fl_sim.models import nn as mnn
    from fl_sim.models.utils import top_n_accuracy
    from fl_sim.data_processing import FedVisionDataset


__all__ = [
    "FedRotatedMNIST",
    "FedRotatedCIFAR10",
]


class FedRotatedMNIST(FedVisionDataset):
    """MNIST augmented with rotations.

    The rotations are fixed and are multiples of 360 / num_clusters.

    The original MNIST dataset contains 60,000 training images and 10,000 test images.
    Images are 28x28 grayscale images in 10 classes (0-9 handwritten digits).

    Parameters
    ----------
    datadir : str or Path, optional
        Path to store the dataset. If not specified, the default path is used.
    num_clusters : int, default 4
        Number of clusters to partition the dataset into.
    num_clients : int, default 4800
        Number of clients to simulate.

    References
    ----------
    .. [1] https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html#MNIST
    .. [2] "An Efficient Framework for Clustered Federated Learning"

    """

    __name__ = "FedRotatedMNIST"

    def __init__(
        self,
        datadir: Optional[Union[Path, str]] = None,
        num_clusters: int = 4,
        num_clients: int = 4800,
    ) -> None:
        self.num_clusters = num_clusters
        self.num_clients = num_clients
        assert self.num_clients % self.num_clusters == 0
        super().__init__(datadir)

    def _preload(self, datadir: Optional[Union[str, Path]] = None) -> None:
        default_datadir = CACHED_DATA_DIR / "fed-rotated-mnist"
        self.datadir = Path(datadir or default_datadir).expanduser().resolve()

        # download if needed
        self.download_if_needed()

        self.DEFAULT_BATCH_SIZE = 20
        self.DEFAULT_TRAIN_CLIENTS_NUM = self.num_clients
        self.DEFAULT_TEST_CLIENTS_NUM = self.num_clients
        self.DEFAULT_TRAIN_FILE = {
            "images": self.url["train-images"],
            "labels": self.url["train-labels"],
        }
        self.DEFAULT_TEST_FILE = {
            "images": self.url["test-images"],
            "labels": self.url["test-labels"],
        }
        self._IMGAE = "image"
        self._LABEL = "label"

        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss()

        # set transforms for creating dataset
        self.transform = transforms.Compose(
            [
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # load data
        self._train_data_dict = {}
        self._test_data_dict = {}
        for key, fn in self.url.items():
            with gzip.open(self.datadir / fn, "rb") as f:
                part, name = key.split("-")
                if name == "images":
                    data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(
                        -1, 28, 28
                    )
                    name = self._IMGAE
                else:  # name == "labels"
                    data = np.frombuffer(f.read(), np.uint8, offset=8)
                    name = self._LABEL
                if part == "train":
                    self._train_data_dict[name] = data
                else:  # part == "test"
                    self._test_data_dict[name] = data

        original_num_images = {
            "train": len(self._train_data_dict[self._LABEL]),
            "test": len(self._test_data_dict[self._LABEL]),
        }

        # set n_class
        self._n_class = len(
            np.unique(
                np.concatenate(
                    [
                        self._train_data_dict[self._LABEL],
                        self._test_data_dict[self._LABEL],
                    ]
                )
            )
        )

        # distribute data to clients
        self.indices = {}
        self.indices["train"] = distribute_images(
            original_num_images["train"],
            self.num_clients // self.num_clusters,
            random=True,
        )
        self.indices["test"] = distribute_images(
            original_num_images["test"],
            self.num_clients // self.num_clusters,
            random=False,
        )

        # perform rotation, and distribute data to clients
        print("Performing rotation...")
        angles = np.arange(0, 360, 360 / self.num_clusters)[1:]
        raw_images = {
            "train": torch.from_numpy(self._train_data_dict[self._IMGAE].copy()),
            "test": torch.from_numpy(self._test_data_dict[self._IMGAE].copy()),
        }
        raw_labels = {
            "train": self._train_data_dict[self._LABEL].copy(),
            "test": self._test_data_dict[self._LABEL].copy(),
        }
        for idx, angle in enumerate(angles):
            transform = FixedDegreeRotation(angle)
            self._train_data_dict[self._IMGAE] = np.concatenate(
                [
                    self._train_data_dict[self._IMGAE],
                    transform(raw_images["train"]).numpy(),
                ]
            )
            self._train_data_dict[self._LABEL] = np.concatenate(
                [
                    self._train_data_dict[self._LABEL],
                    raw_labels["train"].copy(),
                ]
            )
            self._test_data_dict[self._IMGAE] = np.concatenate(
                [
                    self._test_data_dict[self._IMGAE],
                    transform(raw_images["test"]).numpy(),
                ]
            )
            self._test_data_dict[self._LABEL] = np.concatenate(
                [
                    self._test_data_dict[self._LABEL],
                    raw_labels["test"].copy(),
                ]
            )
            self.indices["train"].extend(
                distribute_images(
                    np.arange(original_num_images["train"])
                    + (idx + 1) * original_num_images["train"],
                    self.num_clients // self.num_clusters,
                    random=True,
                )
            )
            self.indices["test"].extend(
                distribute_images(
                    np.arange(original_num_images["test"])
                    + (idx + 1) * original_num_images["test"],
                    self.num_clients // self.num_clusters,
                    random=False,
                )
            )
        del raw_images, raw_labels

    def get_dataloader(
        self,
        train_bs: Optional[int] = None,
        test_bs: Optional[int] = None,
        client_idx: Optional[int] = None,
    ) -> Tuple[torchdata.DataLoader, torchdata.DataLoader]:
        """Get dataloader for training and testing.

        Parameters
        ----------
        train_bs : int, default None
            Batch size for training.
        test_bs : int, default None
            Batch size for testing.
        client_idx : int, default None
            Index of client.
            If None, return dataloader for all clients.

        Returns
        -------
        tuple of torch.utils.data.DataLoader
            Dataloader for training and testing.

        """
        if client_idx is None:
            train_slice = slice(None)
            test_slice = slice(None)
        else:
            train_slice = self.indices["train"][client_idx]
            test_slice = self.indices["test"][client_idx]

        train_ds = torchdata.TensorDataset(
            self.transform(
                torch.from_numpy(
                    self._train_data_dict[self._IMGAE][train_slice]
                ).float()
            ).unsqueeze(1),
            torch.from_numpy(self._train_data_dict[self._LABEL][train_slice]).long(),
        )
        train_dl = torchdata.DataLoader(
            dataset=train_ds,
            batch_size=train_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        test_ds = torchdata.TensorDataset(
            self.transform(
                torch.from_numpy(self._test_data_dict[self._IMGAE][test_slice]).float()
            ).unsqueeze(1),
            torch.from_numpy(self._test_data_dict[self._LABEL][test_slice]).long(),
        )
        test_dl = torchdata.DataLoader(
            dataset=test_ds,
            batch_size=test_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )

        return train_dl, test_dl

    def extra_repr_keys(self) -> List[str]:
        return [
            "n_class",
        ] + super().extra_repr_keys()

    def evaluate(self, probs: torch.Tensor, truths: torch.Tensor) -> Dict[str, float]:
        return {
            "acc": top_n_accuracy(probs, truths, 1),
            "top3_acc": top_n_accuracy(probs, truths, 3),
            "top5_acc": top_n_accuracy(probs, truths, 5),
            "loss": self.criterion(probs, truths).item(),
            "num_samples": probs.shape[0],
        }

    @property
    def mirror(self) -> Dict[str, str]:
        return {
            "lecun": "http://yann.lecun.com/exdb/mnist/",
            "aws": "https://ossci-datasets.s3.amazonaws.com/mnist/",
        }

    @property
    def url(self) -> Dict[str, str]:
        return {
            "train-images": "train-images-idx3-ubyte.gz",
            "train-labels": "train-labels-idx1-ubyte.gz",
            "test-images": "t10k-images-idx3-ubyte.gz",
            "test-labels": "t10k-labels-idx1-ubyte.gz",
        }

    def download_if_needed(self) -> None:
        # check if mirror "lecun" is available
        if requests.get(self.mirror["lecun"]).status_code == 200:
            base_url = self.mirror["lecun"]
        else:
            base_url = self.mirror["aws"]
        for key, fn in self.url.items():
            url = posixpath.join(base_url, fn)
            local_fn = self.datadir / fn
            if local_fn.exists():
                print(f"{key} exists, skip downloading")
                continue
            http_get(url, self.datadir, extract=False)

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """
        a set of candidate models
        """
        return {
            "cnn_mnist": mnn.CNNMnist(num_classes=self.n_class),
            "cnn_femmist_tiny": mnn.CNNFEMnist_Tiny(num_classes=self.n_class),
            "cnn_femmist": mnn.CNNFEMnist(num_classes=self.n_class),
            # "resnet10": mnn.ResNet10(num_classes=self.n_class),
            "mlp": mnn.MLP(dim_in=28 * 28, dim_out=self.n_class, ndim=2),
        }

    @property
    def doi(self) -> List[str]:
        # TODO: add doi of MNIST and IFCA
        return None

    @property
    def label_map(self) -> dict:
        return MNIST_LABEL_MAP

    def view_image(self, client_idx: int, image_idx: int) -> None:
        import matplotlib.pyplot as plt

        if client_idx >= self.num_clients:
            raise ValueError(
                f"client_idx must be less than {self.num_clients}, got {client_idx}"
            )

        total_num_images = len(self.indices["train"][client_idx]) + len(
            self.indices["test"][client_idx]
        )
        if image_idx >= total_num_images:
            raise ValueError(
                f"image_idx must be less than {total_num_images}, got {image_idx}"
            )
        if image_idx < len(self.indices["train"][client_idx]):
            image = (
                255
                - self._train_data_dict[self._IMGAE][
                    self.indices["train"][client_idx][image_idx]
                ]
            )
            label = self._train_data_dict[self._LABEL][
                self.indices["train"][client_idx][image_idx]
            ]
            image_idx = self.indices["train"][client_idx][image_idx]
            angle = (
                image_idx
                // (len(self._train_data_dict[self._IMGAE]) // self.num_clusters)
                * (360 // self.num_clusters)
            )
        else:
            image_idx -= len(self.indices["train"][client_idx])
            image = (
                255
                - self._test_data_dict[self._IMGAE][
                    self.indices["test"][client_idx][image_idx]
                ]
            )
            label = self._test_data_dict[self._LABEL][
                self.indices["test"][client_idx][image_idx]
            ]
            image_idx = self.indices["test"][client_idx][image_idx]
            angle = (
                image_idx
                // (len(self._test_data_dict[self._IMGAE]) // self.num_clusters)
                * (360 // self.num_clusters)
            )
        plt.imshow(image, cmap="gray")
        plt.title(
            f"image_idx: {image_idx}, label: {label} ({self.label_map[int(label)]}), "
            f"angle: {angle}"
        )
        plt.show()


class FedRotatedCIFAR10(FedVisionDataset):
    """CIFAR10 dataset with rotation augmentation.

    The rotations are fixed and are multiples of 360 / num_clusters.

    The original CIFAR10 dataset contains 50k training images and 10k test images.
    Images are 32x32 RGB images in 10 classes.

    Parameters
    ----------
    datadir : str or Path, optional
        Path to store the dataset. If not specified, the default path is used.
    num_clusters : int, default 2
        Number of clusters to partition the dataset into.
    num_clients : int, default 200
        Number of clients to simulate.

    References
    ----------
    .. [1] https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10
    .. [2] "An Efficient Framework for Clustered Federated Learning"

    """

    __name__ = "FedRotatedCIFAR10"

    def __init__(
        self,
        datadir: Optional[Union[Path, str]] = None,
        num_clusters: int = 2,
        num_clients: int = 200,
    ) -> None:
        self.num_clusters = num_clusters
        self.num_clients = num_clients
        assert self.num_clients % self.num_clusters == 0
        super().__init__(datadir=datadir)

    def _preload(self, datadir: Optional[Union[str, Path]] = None) -> None:
        default_datadir = CACHED_DATA_DIR / "fed-rotated-cifar10"
        self.datadir = Path(datadir or default_datadir).expanduser().resolve()

        # download data
        self.download_if_needed()

        self.DEFAULT_BATCH_SIZE = 20
        self.DEFAULT_TRAIN_CLIENTS_NUM = self.num_clients
        self.DEFAULT_TEST_CLIENTS_NUM = self.num_clients

        self.DEFAULT_TRAIN_FILE = [f"data_batch_{i}" for i in range(1, 6)]
        self.DEFAULT_TEST_FILE = ["test_batch"]
        self._IMGAE = "image"
        self._LABEL = "label"

        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss()

        # set transforms for creating dataset
        self.transform = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010),
                ),
            ]
        )

        # load data
        self._train_data_dict = {
            self._IMGAE: np.empty((0, 3, 32, 32), dtype=np.uint8),
            self._LABEL: np.empty((0,), dtype=np.int64),
        }
        self._test_data_dict = {
            self._IMGAE: np.empty((0, 3, 32, 32), dtype=np.uint8),
            self._LABEL: np.empty((0,), dtype=np.int64),
        }

        for file in self.DEFAULT_TRAIN_FILE:
            data = pickle.loads((self.datadir / file).read_bytes(), encoding="bytes")
            self._train_data_dict[self._IMGAE] = np.concatenate(
                [
                    self._train_data_dict[self._IMGAE],
                    data[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8),
                ]
            )
            self._train_data_dict[self._LABEL] = np.concatenate(
                [
                    self._train_data_dict[self._LABEL],
                    np.array(data[b"labels"]).astype(np.int64),
                ]
            )
        data = pickle.loads(
            (self.datadir / self.DEFAULT_TEST_FILE[0]).read_bytes(),
            encoding="bytes",
        )
        self._test_data_dict[self._IMGAE] = (
            data[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8)
        )
        self._test_data_dict[self._LABEL] = np.array(data[b"labels"]).astype(np.int64)

        original_num_images = {
            "train": len(self._train_data_dict[self._LABEL]),
            "test": len(self._test_data_dict[self._LABEL]),
        }

        # set n_class
        self._n_class = len(
            np.unique(
                np.concatenate(
                    [
                        self._train_data_dict[self._LABEL],
                        self._test_data_dict[self._LABEL],
                    ]
                )
            )
        )

        # distribute data to clients
        self.indices = {}
        self.indices["train"] = distribute_images(
            original_num_images["train"],
            self.num_clients // self.num_clusters,
            random=True,
        )
        self.indices["test"] = distribute_images(
            original_num_images["test"],
            self.num_clients // self.num_clusters,
            random=False,
        )

        # perform rotation, and distribute data to clients
        print("Performing rotation...")
        angles = np.arange(0, 360, 360 / self.num_clusters)[1:]
        raw_images = {
            "train": torch.from_numpy(self._train_data_dict[self._IMGAE].copy()),
            "test": torch.from_numpy(self._test_data_dict[self._IMGAE].copy()),
        }
        raw_labels = {
            "train": self._train_data_dict[self._LABEL].copy(),
            "test": self._test_data_dict[self._LABEL].copy(),
        }
        for idx, angle in enumerate(angles):
            transform = FixedDegreeRotation(angle)
            self._train_data_dict[self._IMGAE] = np.concatenate(
                [
                    self._train_data_dict[self._IMGAE],
                    transform(raw_images["train"]).numpy(),
                ]
            )
            self._train_data_dict[self._LABEL] = np.concatenate(
                [
                    self._train_data_dict[self._LABEL],
                    raw_labels["train"].copy(),
                ]
            )
            self._test_data_dict[self._IMGAE] = np.concatenate(
                [
                    self._test_data_dict[self._IMGAE],
                    transform(raw_images["test"]).numpy(),
                ]
            )
            self._test_data_dict[self._LABEL] = np.concatenate(
                [
                    self._test_data_dict[self._LABEL],
                    raw_labels["test"].copy(),
                ]
            )
            self.indices["train"].extend(
                distribute_images(
                    np.arange(original_num_images["train"])
                    + (idx + 1) * original_num_images["train"],
                    self.num_clients // self.num_clusters,
                    random=True,
                )
            )
            self.indices["test"].extend(
                distribute_images(
                    np.arange(original_num_images["test"])
                    + (idx + 1) * original_num_images["test"],
                    self.num_clients // self.num_clusters,
                    random=False,
                )
            )
        del raw_images, raw_labels

    def get_dataloader(
        self,
        train_bs: Optional[int] = None,
        test_bs: Optional[int] = None,
        client_idx: Optional[int] = None,
    ) -> Tuple[torchdata.DataLoader, torchdata.DataLoader]:
        if client_idx is None:
            train_slice = slice(None)
            test_slice = slice(None)
        else:
            train_slice = self.indices["train"][client_idx]
            test_slice = self.indices["test"][client_idx]

        train_ds = torchdata.TensorDataset(
            self.transform(
                torch.from_numpy(
                    self._train_data_dict[self._IMGAE][train_slice]
                ).float()
            ).unsqueeze(1),
            torch.from_numpy(self._train_data_dict[self._LABEL][train_slice]).long(),
        )
        train_dl = torchdata.DataLoader(
            dataset=train_ds,
            batch_size=train_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=True,
            drop_last=False,
        )

        test_ds = torchdata.TensorDataset(
            self.transform(
                torch.from_numpy(self._test_data_dict[self._IMGAE][test_slice]).float()
            ).unsqueeze(1),
            torch.from_numpy(self._test_data_dict[self._LABEL][test_slice]).long(),
        )
        test_dl = torchdata.DataLoader(
            dataset=test_ds,
            batch_size=test_bs or self.DEFAULT_BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )

        return train_dl, test_dl

    def extra_repr_keys(self) -> List[str]:
        return [
            "n_class",
        ] + super().extra_repr_keys()

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
        return "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    @property
    def candidate_models(self) -> Dict[str, torch.nn.Module]:
        """
        a set of candidate models
        """
        return {
            "cnn_cifar": mnn.CNNCifar(num_classes=self.n_class),
            "resnet10": mnn.ResNet10(num_classes=self.n_class),
        }

    @property
    def doi(self) -> List[str]:
        # TODO: add doi of CIFAR10 and IFCA
        return None

    @property
    def label_map(self) -> dict:
        return CIFAR10_LABEL_MAP

    def view_image(self, client_idx: int, image_idx: int) -> None:
        import matplotlib.pyplot as plt

        if client_idx >= self.num_clients:
            raise ValueError(
                f"client_idx must be less than {self.num_clients}, got {client_idx}"
            )

        total_num_images = len(self.indices["train"][client_idx]) + len(
            self.indices["test"][client_idx]
        )
        if image_idx >= total_num_images:
            raise ValueError(
                f"image_idx must be less than {total_num_images}, got {image_idx}"
            )
        if image_idx < len(self.indices["train"][client_idx]):
            image = self._train_data_dict[self._IMGAE][
                self.indices["train"][client_idx][image_idx]
            ]
            label = self._train_data_dict[self._LABEL][
                self.indices["train"][client_idx][image_idx]
            ]
            image_idx = self.indices["train"][client_idx][image_idx]
            angle = (
                image_idx
                // (len(self._train_data_dict[self._IMGAE]) // self.num_clusters)
                * (360 // self.num_clusters)
            )
        else:
            image_idx -= len(self.indices["train"][client_idx])
            image = self._test_data_dict[self._IMGAE][
                self.indices["test"][client_idx][image_idx]
            ]
            label = self._test_data_dict[self._LABEL][
                self.indices["test"][client_idx][image_idx]
            ]
            image_idx = self.indices["test"][client_idx][image_idx]
            angle = (
                image_idx
                // (len(self._test_data_dict[self._IMGAE]) // self.num_clusters)
                * (360 // self.num_clusters)
            )
        # image: channel first to channel last
        image = image.transpose(1, 2, 0)
        plt.imshow(image)
        plt.title(
            f"image_idx: {image_idx}, label: {label} ({self.label_map[int(label)]}), "
            f"angle: {angle}"
        )
        plt.show()


class FixedDegreeRotation(torch.nn.Module):
    """Fixed Degree Rotation Transformation"""

    __name__ = "FixedDegreeRotation"

    def __init__(self, degree: float = 0.0) -> None:
        super().__init__()
        self.degree = degree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rotate(x, self.degree)


def distribute_images(
    total: Union[int, np.ndarray], num_clients: int, random: bool = True
) -> List[np.ndarray]:
    """Distribute images to clients.

    Parameters
    ----------
    total : int or np.ndarray
        Total number of images,
        or an array of indices of images.
    num_clients : int
        Number of clients.
    random : bool, default True
        Whether to distribute images randomly.

    Returns
    -------
    list of np.ndarray
        A list of arrays of indices of images.

    """
    if isinstance(total, int):
        indices = np.arange(total)
    else:
        indices = total.copy()
    if random:
        np.random.shuffle(indices)
    return np.array_split(indices, num_clients)
