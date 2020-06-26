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
from heapq import nsmallest
from Utils import fraction_lambda
from Utils import robust_median_sample_size
import Utils


class RobustMedian(object):
    def __init__(self, set_P, gamma, fig, ax):
        self.set_P = set_P
        self.fraction = fraction_lambda(len(set_P), gamma)
        self.fig = fig
        self.ax = ax
        self.gamma = gamma
        self.Z = (lambda N: np.ceil(N/Utils.K).astype(np.int))

    def updateSet(self, set_P, gamma=None):
        if gamma is not None:
            self.fraction = fraction_lambda(len(set_P), gamma)
        else:
            self.fraction = fraction_lambda(len(set_P), self.gamma)
        self.set_P = set_P

    def compute2Approx(self):
        optimal_center_from_input = []
        optimal_dist = np.inf
        optimal_cluster = []

        if Utils.MED_SUBSAMPLE:
            np.random.seed()
            indices = np.random.choice(range(len(self.set_P)), size=120) #robust_median_sample_size(1.0))) # was with np.unique
            set_Q = np.array(self.set_P)[indices.astype(int)].tolist()
        else:
            set_Q = self.set_P

        for (P, _) in set_Q:
            for idx_p, p in enumerate(P.P):
                if idx_p not in P.deprecated:
                    sum_dists_and_idxs = [Q.computeDistanceToQuery(p) + (idx, ) for (Q, idx) in set_Q]
                    smallest_dist_fraction = self.attainSmallestFraction(sum_dists_and_idxs,
                                                                         frac=fraction_lambda(len(sum_dists_and_idxs),
                                                                                              self.gamma))
                    sum_dist_fraction = np.sum(np.array([x[0] for x in smallest_dist_fraction]))
                    if sum_dist_fraction < optimal_dist:
                        optimal_dist = sum_dist_fraction
                        optimal_center_from_input = p
                        optimal_cluster = smallest_dist_fraction

        if Utils.MED_SUBSAMPLE:
            sum_dists_and_idxs = [Q.computeDistanceToQuery(optimal_center_from_input) + (idx,) for (Q, idx)
                                  in self.set_P]

            # smallest_dist_fraction = self.attainSmallestFraction(sum_dists_and_idxs, frac=self.fraction / 2)
            smallest_dist_fraction = self.attainSmallestFractionByCost(sum_dists_and_idxs)
            optimal_cluster = smallest_dist_fraction

        return optimal_center_from_input, optimal_cluster

    def attainSmallestFraction(self, dist_idx_tuple, frac=1.0):
        return nsmallest(int(np.ceil(frac)), dist_idx_tuple, key=lambda x: x[0])

    def attainSmallestFractionByCost(self, dist_idx_tuple):
        # sum_dist = np.sum([x[0] for x in dist_idx_tuple])
        # N = len(dist_idx_tuple)
        # return [x for x in dist_idx_tuple if x[0] <= (sum_dist / N)]
        N = len(dist_idx_tuple)
        dists_sorted = [x[0] for x in dist_idx_tuple]
        dists_sorted.sort()
        sum_dist = np.sum(dists_sorted[:int(np.ceil(N / (2.0 * Utils.K)))])
        if dists_sorted[self.Z(N)] < ((2.0 * Utils.K * sum_dist) / N):
            return [x for x in dist_idx_tuple if x[0] <= ((2.0 * Utils.K * sum_dist) / N)]
        else:
            return [x for x in dist_idx_tuple if x[0] <= dists_sorted[self.Z(N)]]
