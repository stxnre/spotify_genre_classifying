# Spotify Genre Classifying
This repository contains code I wrote for a project in STAT 154 (Modern Statistical Prediction and Machine Learning) at UC Berkeley, focused on attempting to classify musical genres.


### The Data

`spotify_songs.csv` contains data on 30,000 songs from Spotify, obtained via the `spotifyr` package in R. This dataset was accessed on [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs). 

For every track, the first 9 features contain information about the song, artist, popularity, and the album. The remaining features include the genre, subgenre, and 12 features that measure aspects about the song itself, including key, tempo, "energy", and duration. Some of the quantitative features are based on "perception" of a song as opposed to more direct measurements, like the song duration or tempo.

In ``
