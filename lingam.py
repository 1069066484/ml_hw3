# -*- coding: utf-8 -*-

"""Estimate LiNGAM model. 
"""
# Author: Taku Yoshioka, Shohei Shimizu
# License: MIT

from copy import deepcopy
from munkres import Munkres
import numpy as np
from sklearn.decomposition import FastICA

def _nzdiaghungarian(w):
    """Permurate rows of w to minimize sum(diag(1 / w)). 
    """
    assert(0 < np.min(np.abs(w)))

    w_ = 1 / np.abs(w)
    m = Munkres()
    ixs = np.vstack(m.compute(deepcopy(w_)))

    # Sort by row indices
    ixs = ixs[np.argsort(ixs[:, 0]), :]

    # Return permutation indices
    # r-th row moves to `ixs[r]`. 
    return ixs[:, 1]

def _slttestperm(b_i):
    """Permute rows and cols of the given matrix. 
    """
    n = b_i.shape[0]
    remnodes = np.arange(n)
    b_rem = deepcopy(b_i)
    p = list()

    for i in range(n):
        # Find the row with all zeros
        #ixs = np.where(np.sum(np.abs(b_rem), axis=1) < 1e-12)[0]
        ixs = np.where(np.sum(np.abs(b_rem), axis=1) < 1e-0)[0]

        if len(ixs) == 0:
            # If empty, return None
            return None
        else:
            # If more than one, rbitrarily select the first
            ix = ixs[0]
            p.append(remnodes[ix])

            # Remove the node (and the row and column from b_rem)
            remnodes = np.hstack((remnodes[:ix], remnodes[(ix + 1):]))
            ixs = np.hstack((np.arange(ix), np.arange(ix + 1, len(b_rem))))
            b_rem = b_rem[ixs, :]
            b_rem = b_rem[:, ixs]

    return np.array(p)

def _sltprune(b):
    """Finds an permutation for approximate lower triangularization. 
    """
    n = b.shape[0]
    assert(b.shape == (n, n))

    # Sort the elements of b
    ixs = np.argsort(np.abs(b).ravel())

    for i in range(int(n * (n + 1) / 2) - 1, (n * n) - 1):
        b_i = deepcopy(b)

        # NOTE: `ravel()` returns a view of the given array
        b_i.ravel()[ixs[:i]] = 0

        ixs_perm = _slttestperm(b_i)

        if ixs_perm is not None:
            b_opt = deepcopy(b)
            b_opt = b_opt[ixs_perm, :]
            b_opt = b_opt[:, ixs_perm]
            return b_opt, ixs_perm

    raise ValueError("Failed to do lower triangularization.")

def estimate(xs, random_state=1234):
    """Estimate LiNGAM model. 

    Parameters
    ----------
    xs : numpy.ndarray, shape=(n_samples, n_features)
        Data matrix.
    seed : int
        The seed of random number generator used in the function. 

    Returns
    -------
    b_est : numpy.ndarray, shape=(n_features, n_features)
        Estimated coefficient matrix with LiNGAM. This can be transformed to 
        a strictly lower triangular matrix by permuting rows and columns, 
        implying that the directed graph represented by b_est is acyclic. 
        NOTE: Each row of `b` corresponds to each variable, i.e., X = BX. 
    """
    n_samples, n_features = xs.shape
    ica = FastICA(random_state=random_state, max_iter=2000,whiten=True).fit(xs)
    w = np.linalg.pinv(ica.mixing_)

    assert(w.shape == (n_features, n_features))

    # TODO: check statistical independence of icasig
    # icasig = ica.components_

    # Permute rows of w so that np.sum(1/np.diag(w_perm)) is minimized
    # Permutation order does not make sense in the following processing, 
    # because the permutation is canceled with independent components, whose 
    # order is arbitrary. 
    ixs_perm = _nzdiaghungarian(w)
    w_perm = np.zeros_like(w)
    w_perm[ixs_perm] = w

    # Divide each row of wp by the diagonal element
    w_perm = w_perm / np.diag(w_perm)[:, np.newaxis]

    # Estimate b
    b_est = np.eye(n_features) - w_perm

    # Permute the rows and columns of b_est
    b_csl, p_csl = _sltprune(b_est)

    # Set the upper triangular to zero
    b_csl = np.tril(b_csl, -1)

    # Permute b_csl back to the original variable
    b_est = b_csl # just rename here
    b_est[p_csl, :] = deepcopy(b_est)
    b_est[:, p_csl] = deepcopy(b_est)

    return b_est


if __name__ == '__main__':
    from Ldata_helper import *
    from Lglobal_defs import *
    np.set_printoptions(precision=2,suppress=True)
    data = read_dataset_2(Datasets2.diamonds)
    print(data.shape)
    dd = data.copy()
    dd[:,0] = data[:,-1]
    dd[:,-1] = data[:,0]
    data = dd
    b = estimate(data,random_state=1)
    print(b)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if abs(b[i][j]) > 0.5:
                print(j,i)

"""
[
[   0.      0.      0.      0.      0.      0.      0.      0.      0.      0.  ]
[  -0.61    0.      0.      0.      0.      0.      0.      0.      0.      0.  ]
 [   2.65    0.06    0.      0.      0.      0.      0.      0.      0.      0.  ]
 [  -2.44   -0.05    0.29    0.     -0.05    0.      0.      0.      0.      0.  ]
 [   0.52   -1.07   -0.      0.      0.      0.      0.      0.      0.      0.  ]
 [  -0.38   -0.05    0.06   -0.38   -0.57    0.      0.      0.      0.      0.  ]
 [8545.24  118.17  -88.37  156.02  -44.82   -1.69    0.   -569.63  -11.94      8.54]
 [   1.28   -0.01    0.02    0.02   -0.05   -0.02    0.      0.      0.      0.  ]
 [  -0.01   -0.01    0.      0.     -0.01   -0.      0.      0.98    0.      0.  ]
 [   0.02   -0.     -0.      0.      0.06   -0.      0.      0.53    0.07      0.  ]
 
 ]

 0 1
0 2
0 3
0 4
1 4
4 5
0 6
1 6
2 6
3 6
4 6
5 6
7 6
8 6
9 6
0 7
7 8
7 9


 the currency column only contain one value 'USD'. so can i drop it

"vpp_lic": Vpp Device Based Licensing Enabled
can anyone elaborate it. i don't understand the meaning

"ver" : Latest version code 
i may be wrong but this column is un-necessary during visualizing of data
if i am wrong, please where can i use it.

thats all for now




0.00    &0.00   &0.00   &0.00   &0.00   &0.00   &0.00   &0.00   &0.00   &0.00\\
-0.61   &0.00   &0.00   &0.00   &0.00   &0.00   &0.00   &0.00   &0.00   &0.00\\
2.65    &0.06   &0.00   &0.00   &0.00   &0.00   &0.00   &0.00   &0.00   &0.00\\
-2.44   &-0.05  &0.29   &0.00   &-0.05  &0.00   &0.00   &0.00   &0.00   &0.00\\
0.52    &-1.07  &-0.00  &0.00   &0.00   &0.00   &0.00   &0.00   &0.00   &0.00\\
-0.38   &-0.05  &0.06   &-0.38  &-0.57  &0.00   &0.00   &0.00   &0.00   &0.00\\
8545.24 & 118.17& -88.37& 156.02& -44.82&-1.69  & 0.00  &-569.63& -11.94& 8.54\\
1.28    &-0.01  &0.02   &0.02   &-0.05  &-0.02  &0.00   &0.00   &0.00   &0.\\
-0.01   &-0.01  & 0.00  &0.00   &-0.01  &-0.00  &0.00   &0.98   & 0.00  &0.00\\
0.02    &-0.00  & -0.00 & 0.00  & 0.06  & -0.00 & 0.00  &0.53   &0.07   & 0.00\\





[[    0.       0.       0.       0.       0.       0.       0.       0.       0.       0.  ]
 [   -0.09     0.       0.       0.       0.       0.       0.       0.      0.      -0.57]
 [   -0.04     0.06     0.       0.       0.       0.       0.       0.       0.       2.56]
 [    0.03    -0.06     0.31     0.       0.       0.       0.       0.       0.      -2.93]
 [   -0.05    -1.07    -0.       0.09     0.       0.       0.       0.        0.       0.5 ]
 [   -0.02    -0.08     0.06    -0.37    -0.59     0.       0.       0.        0.      -0.29]
 [   14.43   114.16  -101.25   147.98   -65.62    -5.51     0.   -1123.51       -24.38 10087.47]
 [   -0.04    -0.01     0.02     0.01    -0.05    -0.02     0.       0.        0.       1.24]
 [   -0.72    -0.01     0.       0.       0.04    -0.01     0.       1.43       0.       0.  ]
 [   -0.01     0.       0.       0.       0.       0.       0.       0.       0.       0.  ]]
"""