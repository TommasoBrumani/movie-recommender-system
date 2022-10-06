import scipy.sparse as sp
from matplotlib import pyplot

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.Utils.dataset_splits import dataset_splits
from src.Utils.load_data import load_data


def hyperparameter_range_selection():
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

    # split the URM df in training, validation and testing and turn it into a matrix
    urm_train_validation, urm_train, urm_validation, urm_test = dataset_splits(URM,
                                                                               validation_percentage=0.10,
                                                                               testing_percentage=0.10)

    ####################################### EVALUATORS #################################################

    evaluator_validation = EvaluatorHoldout(urm_validation, [10])
    evaluator_test = EvaluatorHoldout(urm_test, [10])

    ########################### TRY DIFFERENT HYPERPARAMETERS ##########################################

    ################ TOPK ##################
    x_tick = [2, 5, 7, 10, 20, 30]
    MAP_per_k = []

    for topK in x_tick:
        recommender = ItemKNNCBFRecommender(urm_train, icm_mixed_train)
        recommender.fit(shrink=0.0, topK=topK)

        result_df, _ = evaluator_validation.evaluateRecommender(recommender)

        MAP_per_k.append(result_df.loc[10]["MAP"])

    pyplot.plot(x_tick, MAP_per_k)
    pyplot.ylabel('MAP')
    pyplot.xlabel('TopK')
    pyplot.show()

    max_MAP = max(MAP_per_k)
    max_index = MAP_per_k.index(max_MAP)
    max_topK = x_tick[max_index]

    print("Best topK: {}".format(max_topK))

    ################ SHRINK ####################
    x_tick = [0, 10, 25, 50, 75, 100]
    MAP_per_shrinkage = []

    for shrink in x_tick:
        recommender = ItemKNNCBFRecommender(urm_train, icm_mixed_train)
        recommender.fit(shrink=shrink, topK=max_topK)

        result_df, _ = evaluator_test.evaluateRecommender(recommender)

        MAP_per_shrinkage.append(result_df.loc[10]["MAP"])

    pyplot.plot(x_tick, MAP_per_shrinkage)
    pyplot.ylabel('MAP')
    pyplot.xlabel('Shrink')
    pyplot.show()

    max_MAP = max(MAP_per_shrinkage)
    max_index = MAP_per_shrinkage.index(max_MAP)
    max_shrink = x_tick[max_index]

    print("Best shrink: {}".format(max_shrink))

    ############## NORMALIZATION #######################
    x_tick = [True, False]
    MAP_per_norm = []

    for normalization in x_tick:
        recommender = ItemKNNCBFRecommender(urm_train, icm_mixed_train)
        recommender.fit(shrink=max_shrink, topK=max_topK, normalize=normalization)

        result_df, _ = evaluator_test.evaluateRecommender(recommender)

        MAP_per_norm.append(result_df.loc[10]["MAP"])

    pyplot.plot(x_tick, MAP_per_norm)
    pyplot.ylabel('MAP')
    pyplot.xlabel('Normalization')
    pyplot.show()

    max_MAP = max(MAP_per_norm)
    max_index = MAP_per_norm.index(max_MAP)
    max_norm = x_tick[max_index]

    print("Best normalization: {}".format(max_norm))

    ############## FEATURE WEIGHTING ##########################
    x_tick = ["BM25", "TF-IDF", "none"]
    MAP_per_weighting = []

    for weighting in x_tick:
        recommender = ItemKNNCBFRecommender(urm_train, icm_mixed_train)
        recommender.fit(shrink=max_shrink, topK=max_topK, normalize=max_norm, feature_weighting=weighting)

        result_df, _ = evaluator_test.evaluateRecommender(recommender)

        MAP_per_weighting.append(result_df.loc[10]["MAP"])

    pyplot.plot(x_tick, MAP_per_weighting)
    pyplot.ylabel('MAP')
    pyplot.xlabel('Feature weighting')
    pyplot.show()

    max_MAP = max(MAP_per_weighting)
    max_index = MAP_per_weighting.index(max_MAP)
    max_weighting = x_tick[max_index]

    print("Best weighting: {}".format(max_weighting))

    ###################################### FINAL RESULTS ###############################################àà

    recommender = ItemKNNCBFRecommender(urm_train, icm_mixed_train)
    recommender.fit(shrink=max_shrink, topK=max_topK, normalize=max_norm, feature_weighting=max_weighting)

    result_df, _ = evaluator_test.evaluateRecommender(recommender)

    print("FINAL MAP: {}".format(result_df.loc[10]["MAP"]))