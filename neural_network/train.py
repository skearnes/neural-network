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


class Trainer(object):
    """
    Trainer.

    Parameters
    ----------
    network : Network
        Network to train.
    dataset : Dataset
        Training dataset.
    cost : Cost
        Training cost.
    learning_rate : float
        Learning rate.
    """
    def __init__(self, network, dataset, cost, learning_rate=0.1):
        self.network = network
        self.dataset = dataset
        self.cost = cost

    def epoch(self):
        """
        Perform a single training epoch.
        """
        for X, y in self.dataset:
            self.train(X, y)

    def train(self, X, y):
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
        activations, gradients = self.network.get_activations_and_gradients(X)

        # get errors
        errors = self.network.backpropagate_errors(
            self.cost.gradient(y, activations[-1]), gradients)

        # update weights and biases


class Cost(object):
    """
    Training cost.
    """
    def cost(y_true, y_pred):
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

    def gradient(y_true, y_pred):
        """
        Compute cost gradient.

        Parameters
        ----------
        y_true : array_like
            True labels.
        y_pref : array_like
            Predicted labels.
        """
