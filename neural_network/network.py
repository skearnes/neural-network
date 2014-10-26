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
    """
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        """
        Forward propagation.

        Parameters
        ----------
        x : array_like
            Input values.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def get_activations_and_gradients(self, x):
        """
        Get activations and gradient for each layer.

        Parameters
        ----------
        x : array_like
            Input values.
        """
        activations, gradients = [], []
        for layer in self.layers:
            x, g = layer.get_activations_and_gradient(x)
            activations.append(x)
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
