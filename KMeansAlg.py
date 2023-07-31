import numpy as np
import Utils
from multiprocessing.dummy import Pool as ThreadPool
import time
import copy


class KMeansAlg(object):
    def __init__(self, d, k, n_init=3, max_iter=5):
        self.d = d
        self.k = k
        self.n_init = n_init
        self.max_iter = max_iter

        if self.max_iter < 1 or self.max_iter % 1 != 0:
            raise ValueError('Number of maximum iteration must be a positive integer!')

        if self.n_init < 1 or self.n_init % 1 != 0:
            raise ValueError('Number of initializations must be a positive integer!')

    def KmeansPP(self, P):
        pass

    def minibatchKmeans(self, P):
        pass

    def generateKRandomPoints(self, i, set_P=None, is_uniform=False):
        np.random.seed()

        if set_P is not None:
            set_idxs = np.random.choice(len(set_P), size=self.k)
            from_P_idxs = [np.random.choice(set_P[set_idx][0].getNumberOfPoints(), size=1)[0] for set_idx in set_idxs]
            Q = np.array([set_P[idx_set][0].P[from_P_idxs[idx], :] for idx, idx_set in enumerate(set_idxs)])
        else:
            Q = np.random.rand(self.k, self.d)

        return Q

    def computeKmeansCost(self, set_P, Q):
        sum_dist = 0
        for P in set_P:
            dist_P, _, _ = P[0].computeDistanceToKQuery(Q)
            sum_dist += dist_P
        return sum_dist
    
    def updateKmeansParamas(self, n_init, max_iter):
        self.n_init = n_init
        self.max_iter = max_iter

    def runKmeans(self, set_P, Q):
        iter = 0
        opt_Q = None
        opt_dist = np.inf
        previous_dist = np.inf
        num_iters_converged = 0
        #print("new love")
        while iter != self.max_iter and num_iters_converged <= Utils.CHECK_TILL_CONVERGED:
            
            idxs = [[] for i in range(self.k)]
            for P in set_P:
                _, idx_in_P, idx_query = P[0].computeDistanceToKQuery(Q)
                idxs[idx_query].append((P[0], idx_in_P))

            for idx_query, idxs_cluster in enumerate(idxs):
                if idxs_cluster:
                    if not Utils.KMEANS_USE_SUB_SAMPLE:
                        cluster_points = np.array([x[0].P[x[1], :] for x in idxs_cluster])
                        weights = np.array([x[0].weight for x in idxs_cluster])
                        Q[idx_query, :] = np.average(cluster_points, weights=weights, axis=0)
                    else :
                        # For every cluster, a random sample of 1 point from the cluster is a 2 approximation with probability 0.5.
                        # Therefore, we sample many such points and test every one of them.
                        # Observation: the mean of the points that are the closest to that sampled point is always better than the point.
                        # -> We sample point from the input, compute all the closest points to it from every m-set, and take their mean as the new cluster center.
                        np.random.seed()
                        size_of_sample = np.min( [len(idxs_cluster), 50])
                        
                        indices = np.random.choice(range(len(idxs_cluster)), size=size_of_sample,replace=False) #robust_median_sample_size(1.0))) # was with np.unique
                        optimal_dist = np.inf
                        idxes = indices.astype(int)
                        set_Q = np.array([x[0] for x in np.array(idxs_cluster)[idxes]])
                        optimal_cluster_points = None
                        optimal_cluster_weights = None
                        for P in set_Q:
                            for p in P.P:
                                sum_dist = 0.0
                                cluster_weights = []
                                cluster_points = []
                                sum_weights = 0.0
                                weighted_mean = np.zeros(p.shape)
                                for Z in idxs_cluster:
                                    dist, idx_in_Z = Z[0].computeDistanceToQuery(p)
                                    sum_dist += dist
                                    sum_weights += Z[0].weight
                                    weighted_mean += Z[0].weight * Z[0].P[idx_in_Z, :] 
                                 
                                weighted_mean /= sum_weights
                                weighted_sum =  sum([Z[0].computeDistanceToQuery(weighted_mean)[0] for Z in idxs_cluster])
                                if weighted_sum < optimal_dist:
                                    optimal_center = copy.deepcopy(weighted_mean)
                                    optimal_dist = weighted_sum
                            
                        Q[idx_query, :] = optimal_center
                        
            sum_dist = self.computeKmeansCost(set_P, Q);#print (sum_dist)
            iter += 1
            #print(sum_dist)
            if opt_dist > sum_dist :
                opt_Q = copy.deepcopy(Q)
                opt_dist = sum_dist
            if previous_dist > sum_dist:
                previous_dist = sum_dist
                num_iters_converged = 0
            else:
                num_iters_converged += 1
            
        return opt_dist, opt_Q

    def computeKmeans(self, set_P, is_uniform=False):
        start_time = time.time()
        optimal_sum_dist = np.inf
        # final_Q = []
        set_Q = copy.deepcopy([self.generateKRandomPoints(i, set_P, is_uniform=is_uniform)
                 if Utils.FROM_INPUT else self.generateKRandomPoints(i, is_uniform=is_uniform)
                 for i in range(self.n_init)])

        for P in set_P:
            P[0].resetTail()

        if Utils.CONSIDER_HISTORY:
            if is_uniform:
                set_Q = set_Q + Utils.Unifrom_History
            else:
                set_Q = set_Q + Utils.Our_History

        for Q in set_Q:
            sum_dist, output_Q = self.runKmeans(set_P, copy.deepcopy(Q))
            if sum_dist < optimal_sum_dist:
                optimal_sum_dist = sum_dist
                final_Q = copy.deepcopy(output_Q)

        if Utils.CONSIDER_HISTORY:
            if is_uniform:
                Utils.Unifrom_History.append(final_Q)
            else:
                Utils.Our_History.append(final_Q)
        time_taken = time.time() - start_time
        print('Our k-means algorithm finished in {:.3f} seconds'.format(time_taken))
        # print('k-means centers: {}'.format(final_Q))
        return optimal_sum_dist, final_Q, time_taken
