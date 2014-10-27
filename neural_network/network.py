"""
Network.

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
import numpy as np


class Network(object):
    """
    Network containing one or more layers.

    Parameters
    ----------
    layers : list
        Layers.
    trainer : Trainer, optional
        Trainer.
    """
    def __init__(self, layers, trainer=None):
        self.layers = layers
        self.trainer = trainer

    def set_trainer(self, trainer):
        """
        Set trainer.

        Parameters
        ----------
        trainer : Trainer
            Trainer.
        """
        self.trainer = trainer
        self.trainer.set_network(self)

    def fit(self, X, y=None, n_epochs=500):
        """
        Train the network.

        Parameters
        ----------
        X : array_like
            Training examples.
        y : array_like, optional
            Training labels.
        n_epochs : int
            Number of training epochs.
        """
        self.trainer.train(self, n_epochs=n_epochs)

    def predict(self, X):
        """
        Predict labels for examples.

        Parameters
        ----------
        x : array_like
            Input values.
        """
        return self.forward(X)

    def forward(self, X):
        """
        Forward propagation.

        Parameters
        ----------
        x : array_like
            Input values.
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def get_activations_and_gradients(self, X):
        """
        Get activations and gradient for each layer.

        Parameters
        ----------
        x : array_like
            Input values.
        """
        activations, gradients = [], []
        for layer in self.layers:
            X, g = layer.get_activations_and_gradient(X)
            activations.append(X)
            gradients.append(g)
        return activations, gradients

    def backpropagate_errors(self, error, gradients):
        """
        Backpropagate errors.

        Parameters
        ----------
        error : array_like
            Error (cost gradient).
        gradients : list
            Gradients for each layer.
        """
        errors = []
        for layer, gradient in zip(self.layers[::-1], gradients[::-1]):
            error = np.multiply(layer.backward(error), gradient)
            errors.append(error)
        return errors
