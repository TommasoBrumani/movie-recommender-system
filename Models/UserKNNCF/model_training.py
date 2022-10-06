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
from Recommenders.NonPersonalizedRecommender import TopPop
from src.Utils.dataset_splits import dataset_splits
from src.Utils.load_data import load_data
from src.Utils.write_submission import write_submission


def model_training():
    ######################### DATA PREPARATION ###########################################

    # generate the dataframes for URM and ICMs
    URM, ICM_channel, ICM_event, ICM_genre, ICM_subgenre = load_data("kaggle-data")

    # turn the ICMs into matrices
    icm_channel_train = sp.csr_matrix((ICM_channel['data'], (ICM_channel['row'], ICM_channel['col'])))
    # icm_event_train = sp.csr_matrix((ICM_event['data'], (ICM_event['row'], ICM_event['col'])))
    icm_genre_train = sp.csr_matrix((ICM_genre['data'], (ICM_genre['row'], ICM_genre['col'])))
    icm_subgenre_train = sp.csr_matrix((ICM_subgenre['data'], (ICM_subgenre['row'], ICM_subgenre['col'])))

    # concatenate the ICMs
    icm_mixed_train = sp.hstack([icm_channel_train, icm_genre_train, icm_subgenre_train])

    # turn the URM into a matrix without splitting
    urm_train_validation, urm_train, urm_validation, urm_test = dataset_splits(URM, 0.0, 0.0)

    ############# STICK UR HOT AND JUICY RECOMMENDER HERE #####################################à

    recommender = UserKNNCFRecommender(urm_train)

    recommender.fit(shrink=0, topK=497, similarity="tversky", normalize=True, tversky_alpha=2.0, tversky_beta=1.0492068509506025)

    ################### SUBMIT THAT SHIT ##############################à

    write_submission(recommender=recommender, target_users_path="kaggle-data/data_target_users_test.csv",
                     out_path='out/{}_submission.csv'.format(recommender.RECOMMENDER_NAME))
