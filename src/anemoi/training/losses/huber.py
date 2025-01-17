# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging

import torch

from anemoi.training.losses.weightedloss import BaseWeightedLoss

LOGGER = logging.getLogger(__name__)


class WeightedHuberLoss(BaseWeightedLoss):
    """Node-weighted Huber loss."""

    name = "whuber"

    def __init__(
        self,
        node_weights: torch.Tensor,
        delta: float = 1.0,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Node- and feature weighted Huber Loss.

        See `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`_ for more information.

        Parameters
        ----------
        node_weights : torch.Tensor of shape (N, )
            Weight of each node in the loss function
        delta : float, optional
            Threshold for Huber loss, by default 1.0
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss, by default False
        """
        super().__init__(
            node_weights=node_weights,
            ignore_nans=ignore_nans,
            **kwargs,
        )
        self.delta = delta

    def huber(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the Huber loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Huber loss
        """
        diff = torch.abs(pred - target)
        return torch.where(diff < self.delta, 0.5 * torch.square(diff), self.delta * (diff - 0.5 * self.delta))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        feature_indices: torch.Tensor | None = None,
        feature_scale: bool = True,
    ) -> torch.Tensor:
        """Calculates the lat-weighted Huber loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        feature_indices:
            feature indices (relative to full model output) of the features passed in pred and target
        feature_scale:
            If True, scale the loss by the feature_weights

        Returns
        -------
        torch.Tensor
            Weighted Huber loss
        """
        out = self.huber(pred, target)

        if feature_scale:
            out = self.scale_by_variable_scaling(out, feature_indices)
        return self.scale_by_node_weights(out, squash)
