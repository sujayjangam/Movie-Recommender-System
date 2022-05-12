# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Recommender Systems

### Summary:
In this project, I used the Movielens(100k) dataset to create a recommender engine that predicts user ratings for a movie in an attempt to recommend the most relevent title to them.
Metrics used are Root Mean Squared Error, `rmse` and the maximum time taken for a prediction output of **3 seconds**.


### Contents:
- [Problem Statement](#Problem-Statement)
- [Datasets](#Datasets)
- [Proposed Model](#Proposed-Model)
- [Summary of Analysis](#Summary-of-Analysis)
- [Recommendations](#Recommendations)
- [Future Works](#Future-Works)
- [Sources](#Sources)

---

### Problem Statement

**How can we implement our own recommender system to accurately predict the rating a user may give the movie?**

---

### Datasets


* [`movies.csv`](./datasets/movies.csv): Dataset containing MovieIds, Movie Titles, and the Titles of those movies
* [`ratings.csv`](./datasets/ratings.csv): Dataset containing UserIds, MovieIds, Ratings, and Timestamp
* [`links.csv`](./datasets/links.csv): Dataset containing MovieIds, imdbIds and imdbIds
* [`tags.csv`](./datasets/tags.csv): Dataset containing UserIds, MovieIds, and user generated Tags
* [`movies+avg_rating.csv`](./datasets/movies+avg_rating.csv): Dataset containing Movies data as well as mean rating for each movie and the number of times each movie was rated.
* [`merged_users+movies.csv`](./datasets/merged_users+movies.csv): Merged dataframe between ratings and movies
* [`merged_users+movies+tags.csv`](./datasets/merged_users+movies+tags.csv): Merged dataframe between ratings, movies, and tags
* [`model_scores_overall.csv`](./datasets/model_scores_overall.csv): Scores for the SurPRISE models


---
### Keywords
- Recommender, Movielens, Collaborative-Filtering Recommender Systems, Scikit-Surprise, SurPRISE library, TensorFlow-Recommenders, Singular Value Decomposition, GridsearchCV, Cross Validation, WordCloud

---

### Summary of Analysis

#### EDA:
1. As the dataset was very clean, I was able to jump right into EDA.
2. I found that the ratings were not distributed normally around 3, but rather there was a bias in the rating.
3. I also found that in general, the more times a movie was rated, the higher that movie's mean rating was.
4. I also investigated, using wordclouds, whether or not there was any difference made by user preference in the top words for genres, and movie titles.


#### SurPrise Library Models:
1. As I wanted to remain model agnostic, I tried not to pre-select any model, but instead I ran baseline models for all available models by using a custom function.
2. Afterwhich I selected the `SVD+`, `SVD` and `KNNBaseline` models to tune hyperparameters, based on which had the lowest `rmse` score.
3. After a considerable amount of tuning, I found that the `SVD+` model had the lowest `rmse` score, however its prediction time was **5.2** seconds, which is too long as per the threshold of **3 seconds** that we mentioned earlier. So our best model instead is the `SVD` with an `rmse` score of `0.8533`. The hyperparameters can be found in notebook 3.
4. I found that although the model performs well, it fails when it comes to making predictions for users who give out of the norm ratings. e.g. A user gives a rating of 0.5 to a movie with a mean rating of 4.

#### TensorFlow-Recommenders:
1. I began with building a simple model, following the guide on the [TensorFlow-Recommenders site](https://www.tensorflow.org/recommenders) without any regard for model performance.
2. Afterwhich in notebook 5, I created embeddings for all existing features in the dataset and added them into the model, testing for number of layers and for learning rate. Much more tuning is still required, which might appear in notebook 6 (will be created in future).
3. Our top `Retrieval` model was a model of depth 3, which had a `val top 100 accuracy` of **0.047%** which was 23 times better than our simple model.
4. Our top `Ranking` model was a model of depth 3, which has an `rmse` score of `0.8713`. This is pretty close to our `SVD` model considering we haven't yet tuned the `TFRS`.
5. An added benefit to the `TFRS` is that it is able to solve the cold start problem with creating unknown tokens in the embedding process. Of course, for accurate recommendations, more implicit features are required.


---

#### Conclusions and Recommendations:
**`SVD`**
#### Pros
* The `SVD` model is able to convert the sparse data into two low rank matrix approximation, which can remove noise from the data
* The model was also able to pick up on underlying characteristics of the interactions between users and items. 
* Because of the above first two points, the model can scale to large amounts of data fairly easily. With a larger amount of data, the model's recommendation may be able to be improved.
* The model is able to output predictions very quickly.

#### Cons
* The `SVD` model is only able to extract the feature vectors of `userId`s and `movieId`s. This means that we are likely to lose some meaningful signals, that might not be lost if we use contextual features. Some interesting research is being done in this area to combine the `SVD` model with contextual features, like `CFSVD` etc. [linked here](https://www.sciencedirect.com/science/article/abs/pii/S0045790621003311)
* Transformed data is hard to make sense of, e.g. even though the `SVD` model is able to identify latent underlying features, we don't know what these features are, and we will not be able to map these features either, for example, movie length.
* The SVD algorithm is not able to make predictions for new users that are not already in the training set. This would result in a cold start problem. Our solution to this, was to recommend the top_n movies, based on popularity and rating. This is a common solution to the cold start problem, however, it is not the only solution.
* The SVD algorithm is also not able to handle users that are outliers as we saw during error analysis.

**`TFRS`**
### Pros
* The `NN` can actually accept more than just the `userId`, `movieId` and `rating` features. I believe this is one of the biggest advantages of this model over the `SVD`.
* Since you can add in an additional embeddings for unknown tokens, the model above was actually able to give recommendations for a completely new user, that did not exist in the dataset. In order to evaluate the recommendations, we would need to create another simulation where we withhold that user data from the trainset.
* The ranking model was able to reach close to the `SVD` model `rmse` with a relatively low amount of effort considering that we haven't yet tuned the model. We have only touched on the depth of the model, and features in the model, no hyperparameter tuning.

#### Cons
* The `NN` model is not easy to train, and it is complicated to learn about the model as compared to the `SVD` model.
* Time taken to train the `NN` model can be hours or even days. In comparison, the `SVD` took just minutes to train on the 100k dataset.
* For deeper models, careful tuning of all the hyperparameters is required, including the number of dimensions for feature embedding. Without tuning, the model's performance will not be great.



#### Future Works
**`SVD`**
1. Exploration of other models, outside of the `SurPRISE` library like the `CFSVD` as mentioned earlier.
2. Further tuning of the learning rate for bias terms etc, instead of use the `lr_all` parameter.


**`TFRS`**
1. Further research can be done on how to obtain other user features like `age`, `occupation`, `geography` etc. With those kinds of features, especially those that are not user history dependent, we can solve the cold start problem and give the user some level of personalization in the recommendations presented to them.
2. Tuning the model. This goes without saying as we have mentioned preivously that most likely we need to explore tuning the model used here with the `Keras Tuner` library.
3. `Google` itself uses a `NN` recommender system for YouTube, and they have decided to implement a extreme multiclass classification model for the candidate retrieval part. I think further exploration into that is also possible, and some learnings from there can be implemented in your own `NN`. 

---

### Sources
All Sources are linked within the notebooks themselves.
