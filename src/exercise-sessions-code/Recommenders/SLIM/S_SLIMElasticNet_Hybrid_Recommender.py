import scipy as sp

from Recommenders.BaseCBFRecommender import BaseItemCBFRecommender
from Recommenders.Recommender_utils import check_matrix
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
import scipy.sparse as sps
import numpy as np


class S_SLIMElasticNet_Hybrid_Recommender(MultiThreadSLIM_SLIMElasticNetRecommender):
    """ ItemKNN_CFCBF_Hybrid_Recommender"""

    RECOMMENDER_NAME = "S_SLIMElasticNet_Hybrid_Recommender"

    def __init__(self, URM_train, ICM_train, verbose=True):
        super(S_SLIMElasticNet_Hybrid_Recommender, self).__init__(URM_train, verbose=verbose)

        # assert self.n_items == ICM_train.shape[0], "{}: URM_train has {} items but ICM_train has {}".format(
        #     self.RECOMMENDER_NAME, self.n_items, ICM_train.shape[0])

        self.ICM_train = check_matrix(ICM_train.copy(), 'csr', dtype=np.float32)
        self.ICM_train.eliminate_zeros()

    def fit(self, ICM_weight=1.0, **fit_args):
        self.ICM_train = self.ICM_train * ICM_weight
        self.URM_train = sps.vstack([self.URM_train, self.ICM_train.T], format='csr')

        super(S_SLIMElasticNet_Hybrid_Recommender, self).fit(**fit_args)
