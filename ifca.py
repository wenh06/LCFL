"""
Based on the paper, serving as the baseline for LCFL.

`An Efficient Framework for Clustered Federated Learning. <https://arxiv.org/abs/2102.04803>`_

Codebase URL: https://github.com/jichan3751/ifca

"""

import warnings
from copy import deepcopy
from typing import List, Dict, Any

import torch
from torch_ecg.utils.misc import add_docstring

try:
    from fl_sim.nodes import ClientMessage
    from fl_sim.algorithms.fedprox import (
        FedProxClient,
        FedProxServer,
        FedProxClientConfig,
        FedProxServerConfig,
    )
except ModuleNotFoundError:
    # not installed,
    # import from the submodule instead
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent / "fl-sim"))

    from fl_sim.nodes import ClientMessage
    from fl_sim.algorithms.fedprox import (
        FedProxClient,
        FedProxServer,
        FedProxClientConfig,
        FedProxServerConfig,
    )


class IFCAServerConfig(FedProxServerConfig):
    """Server config for the IFCA algorithm.

    Parameters
    ----------
    num_clusters : int
        The number of clusters.
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    **kwargs : dict, optional
        Additional keyword arguments:

        - ``txt_logger`` : bool, default True
            Whether to use txt logger.
        - ``csv_logger`` : bool, default True
            Whether to use csv logger.
        - ``eval_every`` : int, default 1
            The number of iterations to evaluate the model.

    """

    __name__ = "IFCAServerConfig"

    def __init__(
        self,
        num_clusters: int,
        num_iters: int,
        num_clients: int,
        **kwargs: Any,
    ) -> None:
        if kwargs.pop("clients_sample_ratio", None) is not None:
            warnings.warn(
                "`clients_sample_ratio` is not used in IFCA, and always set to 1",
                RuntimeWarning,
            )
        if kwargs.pop("vr", None) is not None:
            warnings.warn(
                "`vr` is not used in IFCA, and always set to False", RuntimeWarning
            )
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio=1,
            vr=False,
            **kwargs,
        )
        self.algorithm = "IFCA"
        self.num_clusters = num_clusters


class IFCAClientConfig(FedProxClientConfig):
    """Client config for the IFCA algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float, default 1e-2
        The learning rate.

    """

    __name__ = "IFCAClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            vr=False,
        )
        self.algorithm = "IFCA"


@add_docstring(FedProxServer.__doc__.replace("FedProx", "IFCA"))
class IFCAServer(FedProxServer):

    __name__ = "IFCAServer"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        check compatibility of server and client configs,
        and set cluster centers
        """
        super()._post_init()
        assert self.config.num_clusters > 0
        self._cluster_centers = {
            cluster_id: {
                "center_model_params": [
                    p.detach().clone() for p in self.model.parameters()
                ],
                "client_ids": [],  # not used currently
            }
            for cluster_id in range(self.config.num_clusters)
        }

    @property
    def client_cls(self) -> type:
        return IFCAClient

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": IFCAServerConfig,
            "client": IFCAClientConfig,
        }

    @property
    def required_config_fields(self) -> List[str]:
        return ["num_clusters"]

    def communicate(self, target: "IFCAClient") -> None:
        """Send cluster centers to client"""
        target._received_messages = {
            "cluster_centers": {
                cluster_id: deepcopy(cluster["center_model_params"])
                for cluster_id, cluster in self._cluster_centers.items()
            }
        }

    @torch.no_grad()
    def update(self) -> None:
        """Update cluster centers"""
        # check the size of each cluster from the received messages
        cluster_sizes = {
            cluster_id: 0 for cluster_id in range(self.config.num_clusters)
        }
        for m in self._received_messages:
            cluster_sizes[m["cluster_id"]] += 1
        # update the cluster centers via averaging
        for cluster_id, cluster in self._cluster_centers.items():
            cluster["client_ids"] = []
            # check if the cluster is empty
            # leave it unchanged if it is empty
            if cluster_sizes[cluster_id] == 0:
                continue
            # initialize the cluster center
            for p in cluster["center_model_params"]:
                p.data.fill_(0)
        # sum up the parameters from each client
        for m in self._received_messages:
            cluster_id = m["cluster_id"]
            cluster = self._cluster_centers[cluster_id]
            for p, p_client in zip(cluster["center_model_params"], m["parameters"]):
                p.data.add_(p_client.data, alpha=1 / cluster_sizes[cluster_id])
            cluster["client_ids"].append(m["client_id"])


@add_docstring(FedProxClient.__doc__.replace("FedProx", "IFCA"))
class IFCAClient(FedProxClient):

    __name__ = "IFCAClient"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes for maintaining itermidiate states
        """
        super()._post_init()
        self.cluster_id = -1

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "IFCAServer") -> None:
        message = {
            "client_id": self.client_id,
            "cluster_id": self.cluster_id,
            "parameters": self.get_detached_model_parameters(),
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        target._received_messages.append(ClientMessage(**message))

    def update(self) -> None:
        """Perform clustering and local training."""
        losses = {}
        # cache current model parameters and metrics
        local_model_weights = self.model.state_dict()
        prev_metrics = self._metrics.copy()
        with torch.no_grad():
            for cluster_id, cluster in self._received_messages[
                "cluster_centers"
            ].items():
                # load cluster center into self.model
                for p, p_center in zip(self.model.parameters(), cluster):
                    p.data.copy_(p_center.data)
                # evaluate the loss
                losses[cluster_id] = self.evaluate(part="train")["loss"]
        # restore model parameters and metrics
        self.model.load_state_dict(local_model_weights)
        self._metrics = prev_metrics.copy()
        del local_model_weights, prev_metrics
        # select the cluster with the minimum loss
        self.cluster_id = min(losses, key=losses.get)

        # set the cluster center as the center of the proximal term
        self._cached_parameters = [
            p.detach().clone().to(self.device)
            for p in self._received_messages["cluster_centers"][self.cluster_id]
        ]
        self.solve_inner()  # alias of self.train()
