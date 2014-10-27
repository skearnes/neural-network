"""
Dataset.

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

from sklearn.cross_validation import KFold, StratifiedKFold


class DatasetIterator(object):
    """
    Dataset iterator.

    Parameters
    ----------
    X : array_like, optional
        Examples.
    y : array_like, optional
        Labels.
    batch_size : int, optional (default 100)
        Batch size.
    stratified : bool, optional (default False)
        Use stratified partitioning to construct batches.
    shuffle : bool, optional (default False)
        Shuffle order of examples for each epoch.
    random_state : int or RandomState, optional
        Random state.
    """
    def __init__(self, X=None, y=None, batch_size=100, stratified=False,
                 shuffle=False, random_state=None):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.stratified = stratified
        self.shuffle = shuffle
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)
        self.random_state = random_state

    def set_dataset(self, X, y=None):
        """
        Set dataset.

        Parameters
        ----------
        X : array_like
            Examples.
        y : array_like, optional
            Labels.
        """
        self.X = X
        self.y = y

    def __iter__(self):
        """
        Iterate through the dataset.
        """
        n_batches = int(np.ceil(len(self.X) / self.batch_size))
        if self.stratified:
            cv = StratifiedKFold(
                self.y, n_folds=n_batches, shuffle=self.shuffle,
                random_state=self.random_state)
        else:
            cv = KFold(
                len(self.X), n_folds=n_batches, shuffle=self.shuffle,
                random_state=self.random_state)
        for _, batch in cv:
            yield self.X[batch], self.y[batch]
