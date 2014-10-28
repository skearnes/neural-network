"""
Layer.

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


class Layer(object):
    """
    Layer containing weights and biases for neurons of the same type.

    Parameters
    ----------
    neuron : Neuron
        Neuron associated with this layer.
    size : int
        Layer size.
    weights : array_like
        Weight matrix.
    biases : array_like, optional
        Biases. Defaults to 0 for each neuron.
    """
    def __init__(self, size, weights, biases=None):
        self.size = size
        self.weights = np.asmatrix(weights)
        if weights is not None and biases is None:
            biases = np.zeros((weights.shape[0], 1), dtype=float)
        self.biases = np.asmatrix(biases)

    def transform(self, a):
        """
        Transform input.

        Parameters
        ----------
        a : array_like
            Input activations, with examples as columns.
        """
        return self.weights * a + self.biases

    def activate(self, z):
        """
        Compute activation on transformed input.

        Parameters
        ----------
        z : float
            Transformed input.
        """
        raise NotImplementedError

    def gradient(self, z):
        """
        Compute gradient.

        Parameters
        ----------
        z : float
            Weighted and biased input value.
        """
        raise NotImplementedError

    def get_activations_and_gradient(self, z):
        """
        Compute activations and gradient.

        Parameters
        ----------
        z : float
            Weighted and biased input value.
        """
        return self.activate(z), self.gradient(z)

    def update_weights(self, update):
        """
        Update weights.

        Parameters
        ----------
        update : array_like
            Update for weights.
        """
        self.weights += update

    def update_biases(self, update):
        """
        Update biases.

        Parameters
        ----------
        update : array_like
            Update for biases.
        """
        self.biases += update


class InputLayer(Layer):
    """
    Input layer.

    Parameters
    ----------
    size : int
        Layer size.
    weights : array_like
        Weight matrix.
    biases : array_like, optional
        Biases. Defaults to 0 for each neuron.
    """
    def activate(self, z):
        """
        Compute activation.

        Parameters
        ----------
        z : array_like
            Transformed input.
        """
        return z

    def gradient(self, z):
        """
        Compute gradient.

        Parameters
        ----------
        z : array_like
            Transformed input.
        """
        return np.asmatrix(np.ones_like(z))


class SigmoidLayer(Layer):
    """
    Sigmoid layer.

    Parameters
    ----------
    size : int
        Layer size.
    weights : array_like
        Weight matrix.
    biases : array_like, optional
        Biases. Defaults to 0 for each neuron.
    """
    def activate(self, z):
        """
        Compute activation.

        Parameters
        ----------
        z : array_like
            Transformed input.
        """
        return 1 / (1 + np.exp(-z))

    def gradient(self, z):
        """
        Compute gradient.

        Parameters
        ----------
        z : array_like
            Transformed input.
        """
        a = self.activate(z)
        return np.multiply(a, 1 - a)

    def get_activations_and_gradient(self, z):
        """
        Compute activations and gradient.

        Parameters
        ----------
        z : array_like
            Transformed input.
        """
        a = self.activate(z)
        return a, np.multiply(a, 1 - a)
