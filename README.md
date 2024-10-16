# Spotify Genre Classifying
This repository contains code I wrote for a project in STAT 154 (Modern Statistical Prediction and Machine Learning) at UC Berkeley, focused on attempting to classify musical genres.


### The Data

`spotify_songs.csv` contains data on 30,000 songs from Spotify, obtained via the `spotifyr` package in R. This dataset was accessed on [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs). 

For every track, the first 9 features contain information about the song, artist, popularity, and the album. The remaining features include the genre, subgenre, and 12 features that measure aspects about the song itself, including key, tempo, "energy", and duration. Some of the quantitative features are based on "perception" of a song as opposed to more direct measurements, like the song duration or tempo.

`process_data.py` reads the spotify dataset, excludes some features (most metadata, like track titles, are not relevant for classification) and separates the label of interest for classification--playlist genre--and the remaining features. All features are scaled, and a random split is made to create a training set `genre_train.csv` and a test set `genre_test.csv`.

### Data Analysis

Before training a classifier, I wanted to look at separability between genres in the training dataset. In `genre_projection.py`, I perform two rank-2 projections of the training data and plot a color-coded map of the projections. The first projection is a standard PCA projection, projecting onto the first two principal components. The second projection uses t-distributed Stochastic Neighbor Embeddings (or t-SNE).

### Training and Classifying

`genre_classifying.py` contains the code that trains and tests genre classifier models, using more "traditional" methods. The methods include a Decision tree, Random forests, and AdaBoost. 

`neural_net_classifier.py`, as the name suggests, trains and tests a neural network genre classifier. Written with PyTorch, This simple neural network has three hidden layers and uses the rectified linear unit (ReLU) for nonlinear activation. The network is trained over 100 epochs with a batch size of 10, working to minimize cross-entropy loss.

### Results

All models trained in this project display their test error. The values are as shown:

|Model        |Test Error    |
|-------------|--------------|
