# can start the model where we left off
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib       # saving and loading models

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(x.values, y)

joblib.dump(model, 'music_recommender.joblib')        # for dumping the model into a folder
# model = joblib.load('music_recommender.joblib')     # for loading the model from the folder
# predictions = model.predict([ [21, 1]])
# predictions
