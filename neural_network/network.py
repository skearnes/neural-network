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
    input_weights : array_like
        Input weights.
    input_biases : array_like, optional
        Input biases.
    trainer : Trainer, optional
        Trainer.
    """
    #TODO don't require explicit weights (we don't know layers[0] dim)
    #TODO get input_dim from dataset?
    def __init__(self, layers, input_dim, input_weights, input_biases=None,
                 trainer=None):
        input_layer = InputLayer(input_dim, input_weights, biases=input_biases)
        self.layers = np.concatenate(([input_layer], layers))
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

    def forward(self, a):
        """
        Forward propagation.

        Parameters
        ----------
        a : array_like
            Input activations.
        """
        for i, layer in enumerate(self.layers[:-1]):
            z = layer.transform(a)
            a = self.layers[i+1].activate(z)
        return a

    def get_activations_and_gradients(self, a):
        """
        Get activations and gradient for each layer.

        Parameters
        ----------
        a : array_like
            Input activations.
        """
        activations, gradients = [], []
        for i, layer in enumerate(self.layers[:-1]):
            z = layer.transform(a)
            a, g = layer.get_activations_and_gradient(z)
            activations.append(a)
            gradients.append(g)
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
