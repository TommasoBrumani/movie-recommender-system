import os

import scipy.sparse as sp
from skopt.space import Integer, Categorical

from Evaluation.Evaluator import EvaluatorHoldout
from HyperparameterTuning.SearchAbstractClass import SearchInputRecommenderArgs
from HyperparameterTuning.SearchBayesianSkopt import SearchBayesianSkopt
from Recommenders.DataIO import DataIO
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.Cython.MatrixFactorization_Cython import MatrixFactorization_FunkSVD_Cython, \
    MatrixFactorization_AsySVD_Cython, MatrixFactorization_BPR_Cython
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM.SLIMElasticNetRecommender import MultiThreadSLIM_SLIMElasticNetRecommender
from src.Utils.ICM_preprocessing import combine
from src.Utils.dataset_splits import dataset_splits
from src.Utils.load_ICM import load_ICM
from src.Utils.load_URM import load_URM
from src.Utils.load_data import load_data
from src.Utils.write_submission import write_submission


def model_training():
    ######################### DATA PREPARATION ###########################################

    # generate the dataframes for URM and ICMs
    URM, ICM_channel, ICM_event, ICM_genre, ICM_subgenre = load_data("kaggle-data")

    # turn the ICMs into matrices
    # icm_channel_train = sp.csr_matrix((ICM_channel['data'], (ICM_channel['row'], ICM_channel['col'])))
    # icm_event_train = sp.csr_matrix((ICM_event['data'], (ICM_event['row'], ICM_event['col'])))
    # icm_genre_train = sp.csr_matrix((ICM_genre['data'], (ICM_genre['row'], ICM_genre['col'])))
    # icm_subgenre_train = sp.csr_matrix((ICM_subgenre['data'], (ICM_subgenre['row'], ICM_subgenre['col'])))

    # concatenate the ICMs
    # icm_mixed_train = sp.hstack([icm_channel_train, icm_genre_train, icm_subgenre_train])

    # concatenate ICM_subgenre to URM
    urm_train = load_URM("kaggle-data/data_train.csv")
    ICM_subgenre = load_ICM("kaggle-data/data_ICM_subgenre.csv")
    ICM_combine = combine(ICM_subgenre, urm_train)

    # turn the URM into a matrix without splitting
    # urm_train_validation, urm_train, urm_validation, urm_test = dataset_splits(URM, 0.0, 0.0)

    ################### EVALUATORS ############################

    # evaluator_test = EvaluatorHoldout(urm_test, [10])

    ############# STICK UR HOT AND JUICY RECOMMENDER HERE #####################################

    recommender = MultiThreadSLIM_SLIMElasticNetRecommender(urm_train)
    recommender.fit(topK=1139, l1_ratio= 6.276359878274636e-05, alpha=0.12289267654724283)

    ###################### SEE RESULTS ########################

    # result_df, _ = evaluator_test.evaluateRecommender(recommender)

    # print("TEST MAP: {}".format(result_df.loc[10]["MAP"]))

    ################### SUBMIT THAT SHIT ##############################

    write_submission(recommender=recommender, target_users_path="kaggle-data/data_target_users_test.csv",
                      out_path='out/{}_submission.csv'.format(recommender.RECOMMENDER_NAME))


model_training()