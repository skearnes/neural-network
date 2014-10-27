"""
Neuron.

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


class Neuron(object):
    """
    Neuron.
    """
    def __call__(self, z):
        """
        Compute activation.

        Parameters
        ----------
        z : float
            Weighted and biased input value.
        """
        return self.activate(z)

    def activate(self, z):
        """
        Compute activation.

        Parameters
        ----------
        z : float
            Weighted and biased input value.
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


class SigmoidNeuron(Neuron):
    """
    Sigmoid neuron.
    """
    def activate(self, z):
        """
        Compute activation.

        Parameters
        ----------
        z : float
            Weighted and biased input value.
        """
        return 1 / (1 + np.exp(-z))

    def gradient(self, z):
        """
        Compute gradient.

        Parameters
        ----------
        z : float
            Weighted and biased input value.
        """
        a = self.activate(z)
        return np.multiply(a, 1 - a)

    def get_activations_and_gradient(self, z):
        """
        Compute activations and gradient.

        Parameters
        ----------
        z : float
            Weighted and biased input value.
        """
        a = self.activate(z)
        return a, np.multiply(a, 1 - a)
