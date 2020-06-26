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
import PointSet
import RobustMedian
from Utils import fraction_lambda


class RecursiveRobustMedian(object):
    def __init__(self, set_P, gamma, fig, ax):
        self.set_P = set_P
        self.robust_median = RobustMedian.RobustMedian(set_P, gamma, fig, ax)
        self.m = set_P[0][0].getNumberOfPoints()
        self.fig = fig
        self.ax = ax

    def updateSet(self, set_P):
        self.set_P = set_P
        self.m = set_P[0][0].getNumberOfPoints()
        for (P, _) in set_P:
            P.resetTail()
        self.robust_median.updateSet(self.set_P)

    def applyRecursiveRobustMedian(self):
        for i in range(self.m):
            _, optimal_cluster = self.robust_median.compute2Approx()
            all_idxs_in_set_P = np.array([x[1] for x in self.set_P])
            closest_set = [val_tuple + (np.where(all_idxs_in_set_P == val_tuple[-1])[0][0], ) for val_tuple in
                           optimal_cluster]
            # closest_set = self.robust_median.attainSmallestFraction(optimal_cluster,
            #                                                         frac=fraction_lambda(len(optimal_cluster), 0.5))
            # closest_set = self.robust_median.attainSmallestFractionByCost(optimal_cluster)

            closest_set_idxs = [tuple_val[3] for tuple_val in closest_set]
            for tuple_val in closest_set:
                self.set_P[tuple_val[3]][0].putIntoTail(tuple_val[1])

            # for idx in range(len(self))
            self.set_P = [self.set_P[idx] for idx in closest_set_idxs]
            self.robust_median.updateSet(self.set_P)

        return self.set_P
