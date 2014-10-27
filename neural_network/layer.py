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

from .neuron import SigmoidNeuron


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
        Input weight matrix.
    biases : array_like, optional
        Neuron biases. Defaults to 0 for each neuron.
    """
    def __init__(self, neuron, size, weights, biases=None):
        self.neuron = neuron
        self.size = size
        self.weights = weights
        if biases is None:
            biases = np.zeros(size, dtype=float)
        self.biases = biases

    def forward(self, x):
        """
        Forward propagation.

        Parameters
        ----------
        x : array_like
            Input values.
        """
        z = np.asmatrix(self.weights) * np.asmatrix(x) + self.biases
        return self.neuron(z)

    def backward(self, x):
        """
        Backward propagation.

        Parameters
        ----------
        x : array_like
            Input values.
        """
        z = np.asmatrix(self.weights).T * np.asmatrix(x)
        return self.neuron(z)

    def get_activations_and_gradient(self, x):
        """
        Get activations and gradient for this layer.

        Parameters
        ----------
        x : array_like
            Input values.
        """
        z = np.asmatrix(self.weights) * np.asmatrix(x) + self.biases
        return self.neuron.get_activations_and_gradient(z)

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
        Input weight matrix.
    biases : array_like, optional
        Neuron biases. Defaults to 0 for each neuron.
    """
    def __init__(self, size, weights, biases=None):
        super(InputLayer, self).__init__(
            neuron=None, size=size, weights=weights, biases=biases)


class SigmoidLayer(Layer):
    """
    Sigmoid layer.

    Parameters
    ----------
    size : int
        Layer size.
    weights : array_like
        Input weight matrix.
    biases : array_like, optional
        Neuron biases. Defaults to 0 for each neuron.
    """
    def __init__(self, size, weights, biases=None):
        super(SigmoidLayer, self).__init__(
            neuron=SigmoidNeuron(), size=size, weights=weights, biases=biases)
