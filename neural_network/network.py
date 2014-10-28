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
import scipy.stats

from .layer import InputLayer


class Network(object):
    """
    Network containing one or more layers.

    The network is responsible for all the plumbing between layers.

    Parameters
    ----------
    layers : list
        Layers.
    input_dim : int
        Input dimensionality.
    trainer : Trainer, optional
        Trainer.
    """
    #TODO don't require explicit weights (we don't know layers[0] dim)
    #TODO get input_dim from dataset?
    def __init__(self, layers, input_dim, trainer=None):
        input_layer = InputLayer(input_dim)
        self.layers = np.concatenate(([input_layer], layers))
        self.trainer = trainer
        self.setup()

    def setup(self):
        """
        Network setup.

        This method handles network plumbing, weight initialization, etc.
        """
        for i, layer in enumerate(self.layers[:-1]):
            if layer.weights is None:
                layer.weights = np.asmatrix(
                    scipy.stats.norm(layer.scale).rvs(
                        (self.layers[i+1].size, layer.size)))
            if layer.biases is None:
                layer.biases = np.asmatrix(
                    np.zeros((self.layers[i+1].size, 1)))

        # handle output layer
        if self.layers[-1].weights is None:
            self.layers[-1].weights = np.asmatrix(
                np.identity(self.layers[-1].size))
        if self.layers[-1].biases is None:
            self.layers[-1].biases = np.asmatrix(
                np.zeros((self.layers[-1].size, 1)))

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
        self.trainer.fit(X, y, n_epochs=n_epochs)

    def predict(self, X):
        """
        Predict labels for examples.

        Parameters
        ----------
        x : array_like
            Input values.
        """
        return self.forward(np.asmatrix(X).T)

    def forward(self, z):
        """
        Forward propagation.

        Parameters
        ----------
        z : array_like
            Transformed input.
        """
        a = None
        for layer in self.layers:
            a = layer.activate(z)
            z = layer.transform(a)
        return a

    def get_activations_and_gradients(self, z):
        """
        Get activations and gradient for each layer.

        Parameters
        ----------
        z : array_like
            Transformed input.
        """
        activations, gradients = [], []
        for i, layer in enumerate(self.layers):
            a, g = layer.get_activations_and_gradient(z)
            activations.append(a)
            gradients.append(g)
            z = layer.transform(a)
        return activations, gradients

    def backpropagate_errors(self, output_error, gradients):
        """
        Backpropagate errors.

        Parameters
        ----------
        output_error : array_like
            Output error (cost gradient).
        gradients : list
            Gradients for each layer.
        """
        errors = [output_error]
        error = output_error
        for i in range(len(self.layers)-1)[::-1]:
            error = np.multiply(self.layers[i].weights.T * error, gradients[i])
            errors.append(error)
        return errors[::-1]
