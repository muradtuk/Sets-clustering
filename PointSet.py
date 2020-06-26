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

import numpy as np
from sklearn.metrics import pairwise_distances as pwd
import Utils


class PointSet(object):
    def __init__(self, P, weight=None):
        self.P = P
        if weight is None:
            weight = 1.0

        self.weight = weight
        self.deprecated = []
        self.n, self.d = P.shape

    def getNumberOfPoints(self):
        return self.n

    def resetTail(self):
        self.deprecated = []

    def computeDistanceToQuery(self, center):
        # print('Center is : {}'.format(center.flatten()))
        dists = np.linalg.norm(self.P - center.flatten(), axis=1) ** Utils.Z
        dists[self.deprecated] = np.inf
        idx_of_smallest_val = np.argmin(dists).astype(np.int)

        if isinstance(idx_of_smallest_val, list):
            idx_of_smallest_val = idx_of_smallest_val[0]
        return np.min(dists * self.weight), int(idx_of_smallest_val)

    def computeDistanceToKQuery(self, k_centers):
        dists_and_idxs = np.apply_along_axis(lambda x: self.computeDistanceToQuery(np.expand_dims(x, 1)),
                                             axis=1, arr=k_centers)
        idx_min_dist = np.argmin(dists_and_idxs[:, 0]).astype(np.int)
        return dists_and_idxs[idx_min_dist, 0], int(dists_and_idxs[idx_min_dist, 1]), idx_min_dist

    def putIntoTail(self, idx):
        self.deprecated.append(idx)

    def computeDistanceBetweenPointSets(self, Q):
        optimal_dist, idx_from_Q, idx_from_P = Q.computeDistanceToKQuery(self.P)
        return optimal_dist, idx_from_Q, idx_from_P

    def updateWeight(self, weight):
        self.weight = weight
