import numpy as numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Data
data = pd.read_csv("data/spotify_songs.csv")
data.drop(['track_id','track_name','track_artist','track_album_id','track_album_name',
           'track_album_release_date','playlist_name','playlist_subgenre','key','mode','playlist_id'],axis=1,inplace=True)
# data = data[data['playlist_genre'].isin(['rap','edm'])]
y = data['playlist_genre'].to_numpy()
X = data.drop(['playlist_genre'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10,random_state=30)

# Scale Features
scaler = StandardScaler()
Xscl_train = scaler.fit_transform(X_train)
Xscl_test = scaler.transform(X_test)

# Construct DataFrames
train = pd.DataFrame(Xscl_train,columns="scaled_" + X_train.columns)
train['label'] = y_train
test = pd.DataFrame(Xscl_test,columns="scaled_" + X_train.columns)
test['label'] = y_test

# Save as CSVs
train.to_csv("data/genre_train.csv",index=False)
test.to_csv("data/genre_test.csv",index=False)
