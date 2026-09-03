from flask import Flask, render_template, request, jsonify
import numpy as np
import json
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


app = Flask(__name__)


def extract_tree_data(clf, feature_names=None, class_names=None):
    """
    Accepts a fitted sklearn DecisionTreeClassifier (clf.fit(X_train, y_train))
    and returns a JSON-serialisable dict consumed by the frontend renderer.
    """
    t = clf.tree_

    fn = list(feature_names) if feature_names is not None else [
        f"feature_{i}" for i in range(clf.n_features_in_)
    ]
    cn = list(class_names) if class_names is not None else [
        str(c) for c in clf.classes_
    ]

    return {
        "n_nodes":        int(t.node_count),
        "feature_names":  fn,
        "class_names":    cn,
        "max_depth":      int(clf.get_depth()),
        "n_leaves":       int(clf.get_n_leaves()),
        "feature":        t.feature.tolist(),
        "threshold":      [round(float(v), 6) for v in t.threshold],
        "impurity":       [round(float(v), 6) for v in t.impurity],
        "n_node_samples": t.n_node_samples.tolist(),
        "value":          t.value.tolist(),
        "children_left":  t.children_left.tolist(),
        "children_right": t.children_right.tolist(),
    }


@app.route("/", methods=["GET"])
def homepage():
    """Interactive Gini index explanation."""
    return render_template("index.html")


@app.route("/tree", methods=["POST"])
def tree():
    """
    Accepts JSON body produced by extract_tree_data() or the helper below,
    then renders the interactive tree visualizer page.

    Expected body (application/json):
    {
        "tree_data": { ...output of extract_tree_data()... }
    }
    """
    body = request.get_json(force=True, silent=True)
    if not body or "tree_data" not in body:
        return jsonify({"error": "Send JSON with key 'tree_data'"}), 400

    tree_data = body["tree_data"]
    return render_template("tree.html", tree_data=json.dumps(tree_data))


@app.route("/demo", methods=["GET"])
def demo():
    """
    Convenience endpoint: loads the drug200 demo tree without needing
    a real clf object — useful for testing the visualizer directly.
    """
    tree_data = extract_tree_data(clf, feature_names=x_train.columns, class_names=enc.classes_)
    return render_template("decision_tree_visualizer.html", tree_data=json.dumps(tree_data))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
