import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('drug200.csv')
x = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

enc = LabelEncoder()
y = enc.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
# column encoding
cholesterol_map = {'NORMAL':0, 'HIGH':1}
BP_map = {'LOW':0, 'NORMAL': 1,  'HIGH':2}
sex_map = {'F':0, 'M':1}
x_train['BP'] = x_train['BP'].map(BP_map)
x_train['Sex'] = x_train['Sex'].map(sex_map)
x_train['Cholesterol'] = x_train['Cholesterol'].map(cholesterol_map)
clf = DecisionTreeClassifier(
    criterion='gini',
    random_state=42,
)
clf.fit(x_train, y_train)

import json

def extract_tree_data(clf, feature_names=None, class_names=None):
    t = clf.tree_
    return {
        "n_nodes":        t.node_count,
        "feature_names":  list(feature_names) if feature_names is not None else None,
        "class_names":    list(class_names)   if class_names is not None  else None,
        "feature":        t.feature.tolist(),
        "threshold":      t.threshold.tolist(),
        "impurity":       t.impurity.tolist(),
        "n_node_samples": t.n_node_samples.tolist(),
        "value":          t.value.tolist(),
        "children_left":  t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
    }

tree_data = extract_tree_data(clf, feature_names=x_train.columns, class_names=enc.classes_)
with open('tree.json', 'w') as f:
    json.dump(tree_data, f, indent=2)