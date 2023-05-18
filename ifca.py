"""
Based on the paper, serving as the baseline for LCFL.

`An Efficient Framework for Clustered Federated Learning. <https://arxiv.org/abs/2102.04803>`_

Codebase URL: https://github.com/jichan3751/ifca
"""

from copy import deepcopy
from typing import List, Optional

import numpy as np  # noqa: F401
import torch
from tqdm.auto import tqdm  # noqa: F401

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
    from fl_sim.algorithms.fedprox import (
        FedProxClient,
        FedProxServer,
        FedProxClientConfig,
        FedProxServerConfig,
    )


class IFCAServerConfig(FedProxServerConfig):

    __name__ = "IFCAServerConfig"

    def __init__(
        self,
        num_clusters: int,
        num_iters: int,
        num_clients: int,
    ) -> None:
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio=1,
            vr=False,
        )
        self.algorithm = "IFCA"
        self.num_clusters = num_clusters


class IFCAClientConfig(FedProxClientConfig):

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
            i: {"center_model": deepcopy(self.model), "client_ids": []}
            for i in range(self.config.num_clusters)
        }

    @property
    def client_cls(self) -> "IFCAClient":
        return IFCAClient

    @property
    def required_config_fields(self) -> List[str]:
        return ["num_clusters"]

    def communicate(self, target: "IFCAClient") -> None:
        """Send cluster centers to client"""
        raise NotImplementedError

    @torch.no_grad()
    def update(self) -> None:
        """Update cluster centers"""
        raise NotImplementedError

    def train_federated(self, extra_configs: Optional[dict] = None) -> None:
        """Federated (distributed) training, conducted on the clients and the server.

        Parameters
        ----------
        extra_configs : dict, optional
            The extra configs for federated training.

        Returns
        -------
        None

        TODO
        ----
        Run clients training in parallel.

        """
        raise NotImplementedError


class IFCAClient(FedProxClient):

    __name__ = "IFCAClient"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        and set attributes for maintaining itermidiate states
        """
        super()._post_init()
        self._cluster_id = -1

    @property
    def required_config_fields(self) -> List[str]:
        return []

    def communicate(self, target: "IFCAServer") -> None:
        message = {
            "client_id": self.client_id,
            "cluster_id": self._cluster_id,
            "parameters": self.get_detached_model_parameters(),
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        if self.config.vr:
            # currently, config.vr is always False
            message["gradients"] = [
                p.grad.detach().clone() for p in self.model.parameters()
            ]
        target._received_messages.append(ClientMessage(**message))
