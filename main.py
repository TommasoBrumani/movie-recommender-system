from hyperparameter_range_selection import hyperparameter_range_selection
from model_training import model_training
from hyperparameter_tuning import hyperparameter_tuning

if __name__ == '__main__':

    #################### COMMENT THE ONE YOU ARE NOT USING ################

    # train on selected hyperparameters chosen by hand to see distribution
    # hyperparameter_range_selection()

    # train the model on a variety of randomly generated hyperparameter combinations to find the best
    # hyperparameter_tuning()

    # train the model on the entire URM and print the output for submission
    model_training()

    #######################################################################
    # ################## PREVIOUS ATTEMPTS ##################################
    #######################################################################

    #######################################################################
    #   Hybrid item KNN BY TOM   #  Model: ItemKNN_CFCBF_Hybrid_Recommender
    #   {'topK': 209, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True, 'asymmetric_alpha': 0.0, 'feature_weighting': 'TF-IDF', 'ICM_weight': 0.01}  #   MAP: 0.2333424
    #   {'topK': 180, 'shrink': 113, 'similarity': 'tversky', 'normalize': True, 'tversky_alpha': 0.0, 'tversky_beta': 0.9749037991238488, 'ICM_weight': 37.618076717283955}} #   MAP: 0.2312498
    #######################################################################

    #######################################################################
    #   MultVAE BY ME   #  Model: MultVAERecommender_OptimizerMask
    #   {'epochs': 23, 'learning_rate': 0.0015961592417506301, 'l2_reg': 0.0006490776439334211, 'dropout': 0.5791277104923426, 'total_anneal_steps': 146901, 'anneal_cap': 0.3054246574668649, 'batch_size': 256, 'encoding_size': 309, 'next_layer_size_multiplier': 9, 'max_n_hidden_layers': 1, 'max_layer_size': 5000.0}        #   MAP: 0.2219075
    #######################################################################

    #######################################################################
    #   IALS BY ME   #  Model: IALSRecommender
    #   {'num_factors': 37, 'epochs': 16, 'confidence_scaling': 'linear', 'alpha': 1.680037230198647, 'epsilon': 0.04350824882819495, 'reg': 0.000569693545326997}      #   MAP: 0.2304352  #   Submission: 0.39873
    #######################################################################

    #######################################################################
    #   LightMFCF BY TOM   #  Model: LightFMCFRecommender
    #   {'epochs': 5, 'n_components': 182, 'loss': 'bpr', 'sgd_mode': 'adagrad', 'learning_rate': 3.453755506640821e-05, 'item_alpha': 2.5481055240434452e-05, 'user_alpha': 0.0006713436460513105}      #   MAP: 0.16889952151717974
    #######################################################################

    #######################################################################
    #   P3ABeta BY TOM   #  Model: RP3betaRecommender
    #   {'topK': 580, 'alpha': 0.859081662043375, 'beta': 0.42062133692252907, 'normalize_similarity': True}      #   MAP: 0.21440596625718747
    #######################################################################

    #######################################################################
    #   P3Alpha BY TOM   #  Model: P3alphaRecommender
    #   {'topK': 5, 'alpha': 0.647292462313285, 'normalize_similarity': False}      #   MAP: 0.19110185156094517
    #   {'topK': 6, 'alpha': 0.647292462313285, 'normalize_similarity': False}      #   MAP: 0.1948523271928806
    #######################################################################

    #######################################################################
    #   PureSVD Item BY TOM   #  Model: PureSVDItemRecommender
    #   {'num_factors': 16, 'topK': 642}  #   MAP: 0.2268649
    #######################################################################

    #######################################################################
    #   PureSVD BY TOM   #  Model: PureSVDRecommender
    #   {'num_factors': 17}  #   MAP: 0.2282129446358606
    #   {'num_factors': 21}  #    MAP: 0.2312214897199509   #   submission: 0.40101
    #######################################################################

    #######################################################################
    #   IALS BY TOM   #  Model: IALSRecommender
    #   epochs = 300,
    #             num_factors = 20,
    #             confidence_scaling = "linear",
    #             alpha = 1.0,
    #             epsilon = 1.0,
    #             reg = 1e-3,
    #             init_mean=0.0,
    #             init_std=0.1
    #   MAP: 0.22734136874943975
    #######################################################################

    #######################################################################
    #   FUNKSVD BPR BY TOM   #  Model: MatrixFactorization_BPR_Cython
    #   epochs=300, batch_size = 1000,
    #             num_factors=10, positive_threshold_BPR = None,
    #             learning_rate = 0.001,
    #             use_bias = True,
    #             use_embeddings = True,
    #             sgd_mode='sgd',
    #             negative_interactions_quota = 0.0,
    #             dropout_quota = None,
    #             init_mean = 0.0, init_std_dev = 0.1,
    #             user_reg = 0.0, item_reg = 0.0, bias_reg = 0.0, positive_reg = 0.0, negative_reg = 0.0,
    #             random_seed = None
    #   MAP: 0.0011389508519484308
    #######################################################################

    #######################################################################
    #   FUNKSVD BY TOM   #  Model: MatrixFactorization_FunkSVD_Cython
    #   epochs=300, batch_size = 1000,
    #             num_factors=10, positive_threshold_BPR = None,
    #             learning_rate = 0.001,
    #             use_bias = True,
    #             use_embeddings = True,
    #             sgd_mode='sgd',
    #             negative_interactions_quota = 0.0,
    #             dropout_quota = None,
    #             init_mean = 0.0, init_std_dev = 0.1,
    #             user_reg = 0.0, item_reg = 0.0, bias_reg = 0.0, positive_reg = 0.0, negative_reg = 0.0,
    #             random_seed = None
    #   MAP: ? (bassa)
    #######################################################################

    #######################################################################
    #   SLIM BPR BY TOM   #  Model: SLIM_BPR_Python
    #   topK = 100, epochs = 25, lambda_i = 0.0025, lambda_j = 0.00025, learning_rate = 0.05  #   MAP: 0.19998925973634635
    #   {'topK': 135, 'epochs': 150, 'symmetric': True, 'sgd_mode': 'sgd', 'lambda_i': 0.003549697095754257, 'lambda_j': 1.4859794378425793e-05, 'learning_rate': 0.009299268968379654}     #     MAP: 0.2244522    #   submission: 0.37067
    #######################################################################

    #######################################################################
    #   SLIM Elastic Net BY TOM   #  Model: SLIMElasticNetRecommender / MultiThreadSLIM_SLIMElasticNetRecommender
    #   l1_ratio=0.1, alpha = 1.0, positive_only=True, topK = 100  #   MAP: 0.18890385893134967
    #   {'topK': 1000, 'l1_ratio': 1.4146440846824416e-05, 'alpha': 0.41532324269493487}    #   50 iterations   #   MAP: 0.2437318  #   submission: 0.45522
    #   {'topK': 1051, 'l1_ratio': 4.288143601437429e-05, 'alpha': 0.12974800685199264}     #   80 iterations   #   MAP: ~0.248     #   submission: 0.47428
    #   {'topK': 1139, 'l1_ratio': 6.276359878274636e-05, 'alpha': 0.12289267654724283}     #   100 iterations  #   MAP: ~0.2490015 #   submission: 0.47484
    #######################################################################

    #######################################################################
    #   USER KNN CF BY MISTERY MAN   #  Model: UserKNNCFRecommender
    #   Hyperparameters: {'topK': 497, 'shrink': 0, 'similarity': 'tversky', 'normalize': True, 'tversky_alpha': 2.0, 'tversky_beta': 1.0492068509506025}
    #   CUTOFF: 10 - PRECISION: 0.3790485, PRECISION_RECALL_MIN_DEN: 0.3807380, RECALL: 0.0672539, MAP: 0.2340507, MAP_MIN_DEN: 0.2349674, MRR: 0.6445074, NDCG: 0.3971381, F1: 0.1142386, HIT_RATE: 0.9657650, ARHR_ALL_HITS: 1.2059595, NOVELTY: 0.0054557, AVERAGE_POPULARITY: 0.5984961, DIVERSITY_MEAN_INTER_LIST: 0.8611292, DIVERSITY_HERFINDAHL: 0.9861066, COVERAGE_ITEM: 0.1123540, COVERAGE_ITEM_CORRECT: 0.0481754, COVERAGE_USER: 0.9993407, COVERAGE_USER_CORRECT: 0.9651282, DIVERSITY_GINI: 0.0076710, SHANNON_ENTROPY: 7.1596668, RATIO_DIVERSITY_HERFINDAHL: 0.9864870, RATIO_DIVERSITY_GINI: 0.0309442, RATIO_SHANNON_ENTROPY: 0.5778496, RATIO_AVERAGE_POPULARITY: 2.9454224, RATIO_NOVELTY: 0.0256689,
    #   Submission: 0.40799
    #######################################################################

    #######################################################################
    #   ITEM KNN CF BY MISTERY MAN   #  Model: ItemKNNCFRecommender
    #   Hyperparameters: {'topK': 601, 'shrink': 89, 'similarity': 'tversky', 'normalize': True, 'tversky_alpha': 0.0, 'tversky_beta': 2.0}
    #   CUTOFF: 10 - PRECISION: 0.3608167, PRECISION_RECALL_MIN_DEN: 0.3623790, RECALL: 0.0619888, MAP: 0.2195377, MAP_MIN_DEN: 0.2203445, MRR: 0.6263426, NDCG: 0.3787411, F1: 0.1058009, HIT_RATE: 0.9579210, ARHR_ALL_HITS: 1.1523062, NOVELTY: 0.0053422, AVERAGE_POPULARITY: 0.6446964, DIVERSITY_MEAN_INTER_LIST: 0.8204139, DIVERSITY_HERFINDAHL: 0.9820354, COVERAGE_ITEM: 0.0264688, COVERAGE_ITEM_CORRECT: 0.0211529, COVERAGE_USER: 0.9993407, COVERAGE_USER_CORRECT: 0.9572894, DIVERSITY_GINI: 0.0037081, SHANNON_ENTROPY: 6.4357012, RATIO_DIVERSITY_HERFINDAHL: 0.9824142, RATIO_DIVERSITY_GINI: 0.0149582, RATIO_SHANNON_ENTROPY: 0.5194191, RATIO_AVERAGE_POPULARITY: 3.1727914, RATIO_NOVELTY: 0.0251350,
    #   Submission: 0.36528
    #######################################################################

    #######################################################################
    #   ITEM KNN CBF REC BY TOM   #     Model: ItemKNNCBFRecommender
    #   #### PURE ####
    #   ICM: channel, shrink: 0, topK: 50   #   MAP@10 - 0.0304475 #   submission -  #
    #   ICM: event, shrink: 0, topK: 50   #   MAP@10 - 0.0005795 #   submission -  #
    #   ICM: genre, shrink: 0, topK: 50   #   MAP@10 - 0.0035598 #   submission -  #
    #   ICM: subgenre, shrink: 0, topK: 50   #   MAP@10 - 0.0122477 #   submission -  #
    #   #### MIXED 2 #### (just those that improved all their components)
    #   ICM: subgenre + genre, shrink: 0, topK: 50   #   MAP@10 - 0.0124698 #   submission -  #
    #   ICM: subgenre + channel, shrink: 0, topK: 50  # MAP@10 - 0.0320515 #   submission -  #
    #   #### MIXED 3 #### (just those that improved all their mixed 2 components)
    #   ICM: subgenre + genre + channel, shrink: 0, topK: 50   #   MAP@10 - 0.0328533 #   submission -  #
    #   ICM: subgenre + genre + channel,  {'topK': 5, 'shrink': 648, 'similarity': 'cosine', 'normalize': True}   #   MAP@10 - 0.0448411 #   submission - 0.04947 #
    #######################################################################

    #######################################################################
    #   TOP POP REC BY TOM   #  Model: TopPop
    #   MAP@10 - 0.1661706 (split 0.01/0.20)    #   submission(0.79) - 0.13957  # submission(0.98) - 0.22996  #
    #######################################################################

    #######################################################################
    #   RANDOM REC BY TOM   #   Model: Random
    #   seed = 69   #   MAP@10 - 0.0014101 (split 0.01/0.20)   #
    #######################################################################




