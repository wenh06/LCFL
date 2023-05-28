"""
"""

from copy import deepcopy
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from torch_ecg.utils.misc import add_docstring
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids
from tqdm.auto import tqdm

try:
    from fl_sim.nodes import ClientMessage
    from fl_sim.algorithms.fedopt import (
        FedAvgClient as BaseClient,
        FedAvgServer as BaseServer,
        FedAvgClientConfig as BaseClientConfig,
        FedAvgServerConfig as BaseServerConfig,
    )

    _base_algorithm = "FedAvg"
except ModuleNotFoundError:
    # not installed,
    # import from the submodule instead
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent / "fl-sim"))

    from fl_sim.nodes import ClientMessage
    from fl_sim.algorithms.fedopt import (
        FedAvgClient as BaseClient,
        FedAvgServer as BaseServer,
        FedAvgClientConfig as BaseClientConfig,
        FedAvgServerConfig as BaseServerConfig,
    )

    _base_algorithm = "FedAvg"


__all__ = [
    "LCFLServer",
    "LCFLClient",
    "LCFLServerConfig",
    "LCFLClientConfig",
]


class LCFLServerConfig(BaseServerConfig):
    """Server config for the LCFL algorithm.

    Parameters
    ----------
    num_clusters : int
        The number of clusters.
    num_iters : int
        The number of (outer) iterations.
    num_clients : int
        The number of clients.
    clients_sample_ratio : float, default 1
        The ratio of clients to participate in each round.
    cluster_method : str, default "kmedoids"
        The clustering method to use based on the distance matrix.
        Currently only support "kmedoids" and "dbscan".
    num_warmup_iters : int, default 10
        The number of warmup iterations.
    local_warmup : bool, default False
        Whether to use local warmup or federated warmup.
    warmup_clients_sample_ratio : float, optional
        The ratio of clients to participate in each warmup round.
        If not specified, will use the same ratio as ``clients_sample_ratio``.
    **kwargs : dict, optional
        Additional keyword arguments:

        - ``txt_logger`` : bool, default True
            Whether to use txt logger.
        - ``csv_logger`` : bool, default False
            Whether to use csv logger.
        - ``json_logger`` : bool, default True
            Whether to use json logger.
        - ``eval_every`` : int, default 1
            The number of iterations to evaluate the model.
        - ``seed`` : int, default 0
            The random seed.
        - ``verbose`` : int, default 1
            The verbosity level.
        - ``gpu_proportion`` : float, default 0.2
            The proportion of clients to use GPU.
            Used to similate the system heterogeneity of the clients.
            Not used in the current version, reserved for future use.

    """

    __name__ = "LCFLServerConfig"

    def __init__(
        self,
        num_clusters: int,
        num_iters: int,
        num_clients: int,
        clients_sample_ratio: float = 1,
        cluster_method: str = "kmedoids",
        num_warmup_iters: int = 10,
        local_warmup: bool = False,
        warmup_clients_sample_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            num_iters,
            num_clients,
            clients_sample_ratio=clients_sample_ratio,
            **kwargs,
        )
        self.algorithm = "LCFL"
        self.num_clusters = num_clusters
        self.cluster_method = cluster_method
        self.num_warmup_iters = num_warmup_iters
        self.local_warmup = local_warmup
        self.warmup_clients_sample_ratio = (
            warmup_clients_sample_ratio or clients_sample_ratio
        )


class LCFLClientConfig(BaseClientConfig):
    """Client config for the LCFL algorithm.

    Parameters
    ----------
    batch_size : int
        The batch size.
    num_epochs : int
        The number of epochs.
    lr : float, default 1e-2
        The learning rate.
    **kwargs : dict, optional
        Additional keyword arguments:

        - ``scheduler`` : dict, optional
            The scheduler config.
            None for no scheduler, using constant learning rate.
        - ``verbose`` : int, default 1
            The verbosity level.
        - ``latency`` : float, default 0.0
            The latency of the client.
            Not used in the current version, reserved for future use.

    """

    __name__ = "LCFLClientConfig"

    def __init__(
        self,
        batch_size: int,
        num_epochs: int,
        lr: float = 1e-2,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_epochs=num_epochs,
            lr=lr,
            **kwargs,
        )
        self.algorithm = "LCFL"


@add_docstring(BaseServer.__doc__.replace(_base_algorithm, "LCFL"))
class LCFLServer(BaseServer):

    __name__ = "LCFLServer"

    def _post_init(self) -> None:
        """
        check if all required field in the config are set,
        check compatibility of server and client configs,
        and set cluster centers
        """
        super()._post_init()
        assert self.config.num_clusters > 0
        if self.config.cluster_method.lower() == "kmedoids":
            self._cluster_method = KMedoids(
                n_clusters=self.config.num_clusters, metric="precomputed"
            )
        elif self.config.cluster_method.lower() == "dbscan":
            self._cluster_method = DBSCAN(metric="precomputed")
        else:
            raise ValueError(
                "Currenly only support 'dbscan' and 'kmedoids' as cluster method, "
                f"got {self.config.cluster_method}"
            )
        # self._cluster_centers = {
        #     i: {"center_model": deepcopy(self.model), "client_ids": []}
        #     for i in range(self.config.num_clusters)
        # }
        self._cluster_centers = None

    @property
    def client_cls(self) -> type:
        return LCFLClient

    @property
    def config_cls(self) -> Dict[str, type]:
        return {
            "server": LCFLServerConfig,
            "client": LCFLClientConfig,
        }

    @property
    def required_config_fields(self) -> List[str]:
        return ["num_clusters", "num_warmup_iters", "local_warmup"]

    def communicate(self, target: "LCFLClient") -> None:
        """Send cluster centers to client"""
        if self._cluster_centers is None:
            # the warm up stage
            if self.config.local_warmup:
                # local warmup, do nothing on the server side
                return
            super().communicate(target)
        else:
            # NOTE: for simplicity,
            # the clustering is done on the server side,
            # which indeed should be done on the clients

            # send cluster centers to client
            target._received_messages = {
                "parameters": [
                    p.detach().clone()
                    for p in self._cluster_centers[target._cluster_id][
                        "center_model"
                    ].parameters()
                ]
            }

    @torch.no_grad()
    def update(self) -> None:
        """Update cluster centers"""
        if self._cluster_centers is None:
            # the warm up stage
            if self.config.local_warmup:
                # local warmup, do nothing on the server side
                return
            super().update()
        else:
            # federated training on each cluster
            for cluster_id in self._cluster_centers:
                cluster_center = self._cluster_centers[cluster_id]
                cluster_size = len(cluster_center["client_ids"])
                alpha = 1 / (cluster_size * self.config.clients_sample_ratio)
                # average the cluster center model using received parameters
                for idx, p in enumerate(cluster_center["center_model"].parameters()):
                    # add received delta_parameters to the cluster center model
                    for m in self._received_messages:
                        if m["cluster_id"] != cluster_id:
                            continue
                        p.add_(m["delta_parameters"][idx].to(self.device), alpha=alpha)

    def train_federated(self, extra_configs: Optional[dict] = None) -> None:
        """Federated (distributed) training,
        conducted on the clients and the server.

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
        if self._complete_experiment:
            # reset before training if a previous experiment is completed
            self._reset()

        self._logger_manager.log_message("Training federated...")

        # warm up stage
        self._warmup()

        # cluster clients
        self._perform_clustering()

        # perform federated training on each cluster
        self._train_cluster_federated()

        self._logger_manager.log_message("Federated training finished...")
        self._logger_manager.flush()
        # self._logger_manager.reset()

        self._complete_experiment = True

    def _warmup(self) -> None:
        """Warm up stage."""
        self._logger_manager.log_message("Warming up...")

        self.n_iter = 0
        with tqdm(
            range(self.config.num_warmup_iters),
            total=self.config.num_warmup_iters,
            desc=f"{self.config.algorithm} Warmup",
            unit="iter",
            mininterval=1.0,
        ) as outer_pbar:
            for self.n_iter in outer_pbar:
                # selected_clients = list(range(self.config.num_clients))
                selected_clients = self._sample_clients(
                    clients_sample_ratio=self.config.warmup_clients_sample_ratio
                )
                with tqdm(
                    total=len(selected_clients),
                    desc=f"Warm Up Iter {self.n_iter+1}/{self.config.num_warmup_iters}",
                    unit="client",
                    mininterval=1.0,
                    disable=self.config.verbose < 1,
                ) as pbar:
                    for client_id in selected_clients:
                        client = self._clients[client_id]
                        if not self.config.local_warmup:
                            self._communicate(client)
                        if (self.n_iter > 0) and (
                            (self.n_iter + 1) % self.config.eval_every == 0
                        ):
                            for part in self.dataset.data_parts:
                                metrics = client.evaluate(part)
                                metrics["cluster_id"] = -1
                                # print(f"metrics: {metrics}")
                                self._logger_manager.log_metrics(
                                    client_id,
                                    metrics,
                                    step=self.n_iter,
                                    epoch=self.n_iter,
                                    part=part,
                                )
                        client._update()
                        if not self.config.local_warmup:
                            client._communicate(self)
                        pbar.update(1)
                    if (
                        (self.n_iter > 0)
                        and ((self.n_iter + 1) % self.config.eval_every == 0)
                        and (not self.config.local_warmup)
                    ):
                        self.aggregate_client_metrics()
                    if not self.config.local_warmup:
                        self._update()

    def _perform_clustering(self) -> None:
        """Perform clustering.

        For simplicity, we omit the transimission of client models
        and directly use the client models for clustering
        the transmission of distance vectors is also simplified
        dist stored in server.
        """
        self._logger_manager.log_message("Perform clustering...")

        dist = {
            client_id: np.zeros((self.config.num_clients,))
            for client_id in range(self.config.num_clients)
        }
        with tqdm(
            range(self.config.num_clients),
            total=self.config.num_clients,
            desc="Compute distance vectors",
            unit="client",
            mininterval=1.0,
        ) as pbar:
            for client_id in pbar:
                client = self._clients[client_id]
                half_dist_vec = np.zeros(len(self._clients))
                for another_client_id in range(self.config.num_clients):
                    # server broadcast model parameters of another_client_id
                    # to client_id
                    if client_id == another_client_id:
                        continue
                    another_client = self._clients[another_client_id]
                    client_data, client_label = client.get_all_data()
                    half_dist_vec[another_client_id] = (
                        client.criterion(
                            client.model(client_data.to(client.model.device)),
                            client_label.to(client.model.device),
                        ).cpu()
                        - another_client.criterion(
                            another_client.model(
                                client_data.to(another_client.model.device)
                            ),
                            client_label.to(another_client.model.device),
                        ).cpu()
                    ).abs()
                # transmit half_dist_vec to server
                dist[client_id] = half_dist_vec

        dist_mat = np.zeros((self.config.num_clients, self.config.num_clients))
        for i in range(self.config.num_clients - 1):
            for j in range(i + 1, self.config.num_clients):
                dist_mat[i][j] = dist[i][j] + dist[j][i]
        # the diagonal of dist_mat is 0, so we can simply add dist_mat and its transpose
        dist_mat = dist_mat + dist_mat.T
        if isinstance(self._cluster_method, DBSCAN):
            self._cluster_method.eps = np.percentile(
                dist_mat, 100 / (self.config.num_clusters - 1)
            )
        cluster_ids = self._cluster_method.fit_predict(dist_mat)

        # form cluster centers
        self._cluster_centers = {
            i: {
                "center_model": deepcopy(self.model),
                "client_ids": np.where(cluster_ids == i)[0],
            }
            for i in np.unique(cluster_ids)
        }
        # assign cluster id to clients
        for client_id in range(self.config.num_clients):
            self._clients[client_id]._cluster_id = cluster_ids[client_id]

    def _train_cluster_federated(self) -> None:
        """Perform federated training on each cluster."""
        self._logger_manager.log_message(
            "Perform federated training on each cluster..."
        )

        total_iters = self.config.num_warmup_iters + self.config.num_iters
        with tqdm(
            range(self.config.num_warmup_iters, total_iters),
            total=self.config.num_iters,
            desc=f"{self.config.algorithm} Clustered Federated Training",
            unit="iter",
            mininterval=1.0,
        ) as outer_pbar:
            for self.n_iter in outer_pbar:
                for cluster_id in self._cluster_centers:
                    # selected_clients = self._cluster_centers[cluster_id]["client_ids"]
                    selected_clients = self._sample_clients(
                        subset=self._cluster_centers[cluster_id]["client_ids"],
                        clients_sample_ratio=self.config.clients_sample_ratio,
                    )
                    with tqdm(
                        total=len(selected_clients),
                        desc=f"Iter {self.n_iter+1}/{total_iters} | Cluster {cluster_id}",
                        unit="client",
                        mininterval=1.0,
                        disable=self.config.verbose < 1,
                    ) as pbar:
                        for client_id in selected_clients:
                            client = self._clients[client_id]
                            self._communicate(client)
                            if (self.n_iter + 1) % self.config.eval_every == 0:
                                for part in self.dataset.data_parts:
                                    metrics = client.evaluate(part)
                                    metrics["cluster_id"] = cluster_id
                                    # print(f"metrics: {metrics}")
                                    self._logger_manager.log_metrics(
                                        client_id,
                                        metrics,
                                        step=self.n_iter,
                                        epoch=self.n_iter,
                                        part=part,
                                    )
                            client._update()
                            client._communicate(self)
                            pbar.update(1)
                        if (self.n_iter + 1) % self.config.eval_every == 0:
                            self.aggregate_client_metrics()
                        self._update()


@add_docstring(BaseClient.__doc__.replace(_base_algorithm, "LCFL"))
class LCFLClient(BaseClient):

    __name__ = "LCFLClient"

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

    def communicate(self, target: "LCFLServer") -> None:
        delta_parameters = self.get_detached_model_parameters()
        for dp, rp in zip(delta_parameters, self._cached_parameters):
            dp.data.add_(rp.data, alpha=-1)
        message = {
            "client_id": self.client_id,
            "cluster_id": self._cluster_id,
            "delta_parameters": delta_parameters,
            "train_samples": len(self.train_loader.dataset),
            "metrics": self._metrics,
        }
        target._received_messages.append(ClientMessage(**message))
