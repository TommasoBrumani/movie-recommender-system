# Movie Recommender System
## Overview
The repository contains the `Python` files used to fine tune and combine already provided <b>recommender system models</b> with the purpose of producing <b>recommendations for movies</b>, based on a User Rating Matrix (URM) and Item Content Matrix (ICM).

### Authors
- <b>Andrea Alesani</b> (andrea.alesani@mail.polimi.it)
- <b>Tommaso Brumani</b> (tommaso.brumani@mail.polimi.it)

### License
The project was carried out as part of a challenge for the 2021/2022 '<b>Recommender Systems</b>' course at <b>Politecnico of Milano</b>. 

The base recommender models in `Python` and `Cython`, as well as the functions for hyperparameter tuning, were implemented by the course staff and provided to the students as is. 

## Project Specifications
The challenge required the usage of the provided URM and ICM matrices to produce user recommendations, which were evaluated based on their MAP score on 10 predictions.

In order to achieve this goal all of the provided algorithms for recommender systems and parameter tuning could be employed, as well as the students' own code and hybridization techniques.

A number of different models were trained, compared, tuned and combined, and ultimately a linear combination of `SLIM`, `MultVAE`, and `ItemKNN` tuned using `bayesian search` provided the best results.

## File System Structure
* `kaggle-data`: the matrices containing the user and item data
* `Models`: saved parameters for trained recommender models, allowing to load them without having to train again 
* `out`: output recommendations of various recommender models
* `results`: file containing details, parameters, and scores of the various recommender models trained
* `src`: code provided by the course staff containing base recommender models and parameter tuners