from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn import tree

music_data = pd.read_csv('music.csv')
x = music_data.drop(columns=['genre'])
y = music_data['genre']

model = DecisionTreeClassifier()
model.fit(x.values, y)

tree.export_graphviz(model, out_file = 'music_recommender.dot',
                    feature_names = ['age', 'gender'],
                    class_names = sorted(y.unique()),
                    label = 'all',
                    rounded = True,      # rounded edges
                    filled = True)       # tables filled with colors
