import math
from time import time

import numpy as np
import scipy.stats as st
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans

# from statsmodels.distributions.empirical_distribution import ECDF

length = 2000
qseq = np.linspace(1e-6, 1 - (1e-6), length)
# Taken from Seaborns implementation
class ECDF:
    """Univariate empirical cumulative distribution estimator."""

    def __init__(self, x):
        """Inner function for ECDF of one variable."""
        self.x_max = max(x)
        self.x_min = min(x)
        sorter = x.argsort()
        x = x[sorter]
        weights = np.ones_like(x)
        weights = weights[sorter]
        y = weights.cumsum()

        self.x = np.r_[0, x]
        self.y = np.r_[0, y]
        self.y = self.y / self.y.max()
        self.inter = interp1d(self.x, self.y, fill_value=0.0)

    def __call__(self, x, weights=None):
        if x > self.x_max:
            return 1.0
        if x < self.x_min:
            return 0.0

        return self.inter(x)


class ECDFWrapper:
    def __init__(self, k, inverse):

        # Filter out non-sensical, this is some a bug earlier in the
        # pipeline
        k = np.asarray(k)
        filter = k < 10e7
        k = k[filter]
        filter = k > (1.0 / 1440)
        k = k[filter]

        self.data = np.asarray(k, dtype="float32")
        self.ecdf = ECDF(self.data)

        self.x_min = min(self.data)
        self.x_max = max(self.data)
        # Dont go to negative inf X

        if inverse:
            self.inverse = quantile(self, qseq)
            assert not np.isnan(self.inverse).any()
        else:
            self.inverse = None

    def _re_init(self, length=1000):
        qseq = np.linspace(1e-6, 1 - (1e-6), length)
        print(self.x_min, self.x_max)
        self.inverse = quantile(self, qseq)

    def __call__(self, x):
        return self.ecdf(x)

    def data(self):
        return self.data

    def sample(self, num=1, random_state=int(time())):
        np.random.seed(random_state)
        samples = []
        # TODO: Fix bug with lowest percentile
        for _ in range(0, num):
            p = np.random.randint(0, high=length - 1)
            samples.append(1 / self.inverse[p])
        return samples


def ecdf(keys, inverse=True):
    return ECDFWrapper(keys, inverse)


def emd(m, n):
    return st.wasserstein_distance(m, n)


def quantile(cdf, qs):
    return [_quantile_help(cdf, q) for q in qs]


def _quantile_help(cdf, q):
    try:
        l = math.log10(min(cdf.data))
        h = math.log10(max(cdf.data))
    except:
        for x in cdf.data:
            if x > 10e5:
                print(x)
            if x < 0:
                print(x)
        print(cdf.data.size)
        print(cdf.x_min, cdf.x_max)
        exit(0)
    qseq = np.logspace(l, h, 100000)
    # qseq = np.linspace(cdf.x_min, cdf.x_max, 1000000)
    lo = 0
    hi = len(qseq) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        v = qseq[mid]
        if cdf(v) < q:
            lo = mid + 1
        else:
            hi = mid

    lo = qseq[lo]

    return lo


def epmeans_cluster(clist, k=2):
    myn = len(clist)
    print(myn)
    qmat = []
    for i in range(0, myn):
        if np.isnan(clist[i].inverse).any():
            continue
        qmat.append(clist[i].inverse)
    print("Clustering")
    cluster = KMeans(n_clusters=k, init="k-means++")
    cluster.fit(qmat)

    means = cluster.cluster_centers_
    list2 = []
    list1 = []

    print("Done Clustering")
    for i in range(0, myn):
        list1.append(ecdf(qmat[i], inverse=False))

    for i in range(0, k):
        list2.append(ecdf(means[i]))

    distance_matrix = []
    for i, m in enumerate(list1):
        tmp = []
        for _, n in enumerate(list2):
            tmp.append(emd(m.data, n.data))
        distance_matrix.append(tmp)

    labels = np.argmin(distance_matrix, axis=1)

    # Get the total EMD for the clusters and their associated models attached
    # to it
    s = 0
    for i, l in enumerate(labels):
        s += distance_matrix[i][l]

    return (list2, labels, s)
