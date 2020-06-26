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
import RecursiveRobustMedian
import PointSet
from Utils import GAMMA
import copy

RHO = 2.0



class KMeansForSetsSensitivityBounder(object):
    def __init__(self, set_P, k, fig=None, ax=None):
        self.set_P = set_P
        self.k = k
        self.gamma = 1.0 / (2.0 * self.k)  #1.0 / 2.0
        self.sensitivity = np.ones((len(set_P), )) * -np.inf

        self.fig = fig
        self.ax = ax
        self.upper_bound = 50 #8.0  #(2.0 * (3.0 * RHO ** 2.0) ** self.set_P[0][0].getNumberOfPoints())
        self.idxs_per_m = None

        self.recursive_robust_median = RecursiveRobustMedian.RecursiveRobustMedian(set_P, self.gamma, self.fig, self.ax)

    def checkStoppingCriterion(self):
        return True
        # return self.upper_bound  <= np.count_nonzero(
        #     np.isinf(self.sensitivity)) * ((0.5 * self.gamma) ** self.set_P[0][0].getNumberOfPoints())

    def partitionNMSetByM(self):
        self.idxs_per_m = {}

        for P in self.set_P:
            if P[0].n not in self.idxs_per_m.keys():
                self.idxs_per_m[P[0].n] = [P[1]]
            else:
                self.idxs_per_m[P[0].n] += [P[1]]

    def getAllIdxsWithMSetWithUndefinedSens(self, m):
        idxs = np.where(np.isinf(self.sensitivity))[0]
        return [idx for idx in idxs if idx in self.idxs_per_m[m]]


    def boundSensitivity(self):
        # Q = self.set_P
        self.partitionNMSetByM()
        for key in self.idxs_per_m.keys():
            Q = [P for P in self.set_P if P[0].n == key]
            previous_b = None
            self.recursive_robust_median.updateSet(Q)
            while self.checkStoppingCriterion():
                Q_i = self.recursive_robust_median.applyRecursiveRobustMedian()
                if len(Q_i) < self.upper_bound:
                    break
                idxs = [P[1] for P in Q_i]
                b = self.upper_bound / (len(Q_i))
                self.sensitivity[idxs] = b
                if previous_b is None:
                    previous_b = b
                else:
                    assert(previous_b <= b, 'Must be be a non-descending series...')
                    assert(b <= 1, 'Stopping criterion is not working properly!')
                    previous_b = b
                idxs = self.getAllIdxsWithMSetWithUndefinedSens(key)
                Q_updated = [P for P in Q if P[1] in idxs]
                for P in Q_updated:
                    P[0].resetTail()
                self.recursive_robust_median.updateSet(Q_updated)

            idxs = self.getAllIdxsWithMSetWithUndefinedSens(key)
            # self.sensitivity[np.where(np.isinf(self.sensitivity))[0]] = 1  # 4.0 * (3.0 * RHO ** 2.0) ** Q[0][0].getNumberOfPoints()
            self.sensitivity[idxs] = 1.0

        return self.sensitivity
