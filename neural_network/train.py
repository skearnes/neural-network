"""
Train.

Copyright 2014 Stanford University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import division

import numpy as np

from .dataset import DatasetIterator


class SGDTrainer(object):
    """
    SGD trainer.

    Parameters
    ----------
    cost : Cost
        Training cost.
    network : Network, optional
        Network to train.
    batch_size : int, optional (default 100)
        Batch size.
    stratified : bool, optional (default False)
        Use stratified partitioning to construct batches.
    shuffle : bool, optional (default False)
        Shuffle order of examples for each epoch.
    random_state : int or RandomState, optional
        Random state.
    learning_rate : float, optional (default 1)
        Learning rate.
    kwargs : dict, optional
        Keyword arguments for DatasetIterator.
    """
    def __init__(self, cost, network=None, learning_rate=1, **kwargs):
        self.cost = cost
        self.network = network
        self.learning_rate = learning_rate
        self.dataset_iterator = DatasetIterator(**kwargs)

    def set_network(self, network):
        """
        Set network.

        Parameters
        ----------
        network : Network
            Network to train.
        """
        self.network = network

    def fit(self, X, y, n_epochs=100):
        """
        Train a network.

        Parameters
        ----------
        X : array_like
            Training examples.
        y : array_like, optional
            Training labels.
        n_epochs : int
            Number of training epochs.
        """
        self.dataset_iterator.set_dataset(X, y)
        for i in xrange(n_epochs):
            self.epoch()

    def epoch(self):
        """
        Perform a single training epoch.
        """
        for X, y in self.dataset_iterator:
            self.fit_batch(X, y)

    def fit_batch(self, X, y):
        """
        Perform a single training step.

        Parameters
        ----------
        X : array_like
            Training examples.
        y : array_like
            Training labels.
        """
        # get activations and gradients
        activations, gradients = self.network.get_activations_and_gradients(
            np.asmatrix(X).T)

        # get errors
        output_error = np.multiply(self.cost.gradient(y, activations[-1]),
                                   gradients[-1])
        errors = self.network.backpropagate_errors(output_error, gradients)

        # update weights and biases
        # updates are cumulative over all examples in the batch
        for i, layer in enumerate(self.network.layers[:-1]):
            layer.update_weights(
                self.learning_rate * errors[i+1] * activations[i].T)
            layer.update_biases(
                self.learning_rate * np.sum(errors[i+1], axis=1))


class Cost(object):
    """
    Training cost.
    """
    def cost(self, y_true, y_pred):
        """
        Compute cost.

        Parameters
        ----------
        y_true : array_like
            True labels.
        y_pref : array_like
            Predicted labels.
        """
        raise NotImplementedError

    def gradient(self, y_true, y_pred):
        """
        Compute cost gradient.

        Parameters
        ----------
        y_true : array_like
            True labels.
        y_pref : array_like
            Predicted labels.
        """
        raise NotImplementedError


class SquaredError(Cost):
    """
    Squared error.
    """
    def cost(self, y_true, y_pred):
        """
        Compute cost.

        Parameters
        ----------
        y_true : array_like
            True labels.
        y_pref : array_like
            Predicted labels.
        """
        return np.mean(np.square(y_true - y_pred)) / 2

    def gradient(self, y_true, y_pred):
        """
        Compute cost gradient.

        Parameters
        ----------
        y_true : array_like
            True labels.
        y_pref : array_like
            Predicted labels.
        """
        return (y_pred - y_true) / len(y_true)
