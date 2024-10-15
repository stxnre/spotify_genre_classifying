import numpy as numpy
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import time

train = pd.read_csv('hw6/genre_train.csv')
test = pd.read_csv('hw6/genre_test.csv')

X_train = train.drop('label',axis=1)
y_train = train['label']
X_test = test.drop('label',axis=1)
y_test = test['label']

reg_tree = DecisionTreeClassifier(criterion='gini',ccp_alpha=0.0001,random_state=42)
start=time.time()
reg_tree.fit(X_train,y_train)
stop=time.time()
print(f"Training time: {stop - start}s",reg_tree.score(X_test,y_test))

rf = RandomForestClassifier(n_estimators=50,criterion='gini',random_state=42,ccp_alpha=0)
start=time.time()
rf.fit(X_train,y_train)
stop=time.time()
print(f"Training time: {stop - start}s",rf.score(X_test,y_test))

boost = AdaBoostClassifier(n_estimators=200,learning_rate=1,random_state=42)
start=time.time()
boost.fit(X_train,y_train)
rf.fit(X_train,y_train)
stop=time.time()
print(f"Training time: {stop - start}s",boost.score(X_test,y_test))