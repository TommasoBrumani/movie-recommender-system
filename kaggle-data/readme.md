# Data provided by the teacher

### Below you can find a description for each csv file inside this folder

**data_train.csv**: Contains the training set, describing implicit preferences expressed by the users.
- _row_ : identifier of the user
- _col_ : identifier of the item
- _data_ : "1" if the user interacted with the item.

**data_ICM_genre.csv**: Contains the genres of the items. TV shows (items) may have multiple genres. All genres are anonymized and described only by a numerical identifier.

- _row_ : identifier of the item
- _col_ : identifier of the genre
- _data_ : "1" if the item is described by the genre 

**data_ICM_subgenre.csv**: Contains the subgenres of the items. TV shows (items) may have multiple subgenres. All subgenres are anonymized and described only by a numerical identifier.

- row : identifier of the item
- col : identifier of the subgenre
- data : "1" if the item is described by the subgenre 

**data_ICM_channel.csv**: Contains the channels of an item. TV shows (items) are broadcasted on one or more channels. All TV channles are anonymized and described only by a numerical identifier. The file is composed of 3 columns:

- _row_ : identifier of the item
- _col_ : identifier of the TV channel
- _data_ : "1" if the item has been broadcasted on that TV channel

**data_ICM_event.csv**: Contains the episodes of an item. TV shows (items) might contain one or more episodes. All episodes are anonymized and described only by a numerical identifier. The file is composed of 3 columns:

- row : identifier of the item
- col : identifier of the episode
- data : "1" if the episode belongs to the item

**data_target_users_test.csv**: Contains the ids of the users that should appear in your submission file. The submission file should contain all and only these users.

**sample_submission.csv**: A sample submission file in the correct format: [user_id],[ordered list of recommended items]. Be careful with the spaces and be sure to recommend the correct number of items to every user. IMPORTANT: first line is mandatory and must be properly formatted.