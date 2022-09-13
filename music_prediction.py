import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split  # splitting data into testing and training data sets
from sklearn.metrics import accuracy_score            # findind accuracy  

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])                # dividing into datasets
y = music_data['genre']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# New Instance (creating a model)
model = DecisionTreeClassifier()
model.fit(x_train.values, y_train)                    # Training the model
predictions = model.predict(x_test.values)            # Make predictions

score = accuracy_score(y_test, predictions)           # compare score to output value
score
