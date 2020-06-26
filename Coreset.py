"""*****************************************************************************************
MIT License
Copyright (c) 2020 Ibrahim Jubran, Murad Tukan, Alaa Maalouf, Dan Feldman
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*****************************************************************************************"""


"""
This module is for generating a coreset given the sensitivities

Author: Murad Tukan
"""


import numpy as np
import time
import Utils
import copy
import PointSet


class Coreset(object):
    """
    ################## Coreset ####################
    Functions:
        - __init__ : instructor
        - computeCoreset
        - mergeCoreset
    """

    def __init__(self, prob_dep_vars=None, is_uniform=False):
        """

        :param prob_dep_vars: Problem dependant variables (not used)
        :param is_uniform: A boolean variable stating whether uniform or importance sampling, i.e., using sensitivity
        (Default value: False)
        """
        self.weights = []
        self.S = []
        self.probability = []
        self.is_uniform = is_uniform
        self.prob_dep_vars = prob_dep_vars

    def computeCoreset(self, P, sensitivity, sampleSize, weights=None, core_seed=1.0):
        """
        :param P: A list of point sets.
        :param sensitivity: A vector of n entries (number of point sets in P) which describes the sensitivity of
                            each point set.
        :param sampleSize: An integer describing the size of the coreset.
        :param weights: A weight vector of the data points (Default value: None)
        :return: A subset of P (the datapoints alongside their respected labels), weights for that subset and the
        time needed for generating the coreset.
        """

        startTime = time.time()
        if weights is None:
            weights = np.ones((len(P), 1)).flatten()

        # Compute the sum of sensitivities.
        t = np.sum(sensitivity)

        # The probability of a point prob(p_i) = s(p_i) / t
        self.probability = sensitivity.flatten() / t

        # The number of points is equivalent to the number of rows in P.
        n = len(P)

        # initialize new seed
        np.random.seed()

        # Multinomial distribution.
        indxs = np.random.choice(n, sampleSize, p=self.probability.flatten())
        """countcorset = 0; importantidxes = 218
        for idx in indxs:
            if idx  <= importantidxes: 
                countcorset+=1
        print ("choosen {} points from state".format(countcorset))"""
        # Compute the frequencies of each sampled item.
        hist = np.histogram(indxs, bins=range(n))[0].flatten()
        indxs = copy.deepcopy(np.nonzero(hist)[0])

        # Select the indices.
        # S = P[indxs]
        S = copy.deepcopy([P[i] for i in indxs])


        # Compute the weights of each point: w_i = (number of times i is sampled) / (sampleSize * prob(p_i))
        weights = np.asarray(np.multiply(weights[indxs], hist[indxs]), dtype=float).flatten()

        # Compute the weights of the coreset
        weights = np.multiply(weights, 1.0 / (self.probability[indxs]*sampleSize))
        timeTaken = time.time() - startTime

        for idx, p_set in enumerate(S):
            p_set[0].updateWeight(weights[idx])

        return S, timeTaken
