# Decision Tree Gini Visualizer

An interactive Flask web application for learning and visualizing how a Decision Tree classifier uses the **Gini Index** to split data.

This project trains a `DecisionTreeClassifier` on the `drug200.csv` dataset and provides a browser-based visual explanation of the tree structure, node impurity, class distributions, and Gini calculations.

## Overview

The project is designed as an educational tool for understanding:

- Decision Tree classification
- Gini impurity
- Class probability distributions
- Tree node splitting
- Leaf prediction behavior
- Interactive tree exploration

The app includes two main views:

1. **Gini Index Explanation Page**
   - Explains how Gini impurity is calculated.
   - Includes interactive class distribution controls.
   - Demonstrates impurity through sampling experiments.

2. **Decision Tree Visualizer**
   - Displays a trained decision tree as an interactive SVG-based tree.
   - Shows each node’s:
     - Gini value
     - Sample count
     - Class distribution
     - Split condition
     - Predicted class for leaf nodes
   - Includes zoom controls, detail panels, and formula breakdowns.


## Features

- Train a Decision Tree classifier using `scikit-learn`
- Use Gini impurity as the splitting criterion
- Encode categorical variables for model training
- Extract decision tree internals into JSON
- Render an interactive tree visualization in the browser
- Explore node-level Gini calculations
- View class proportions and impurity formulas
- Load custom tree JSON data through the frontend
- Includes a demo endpoint using the included drug dataset

## Dataset

The project uses the `drug200.csv` dataset.

The dataset contains patient-related features:

| Column | Description |
|---|---|
| `Age` | Patient age |
| `Sex` | Patient sex |
| `BP` | Blood pressure level |
| `Cholesterol` | Cholesterol level |
| `Na_to_K` | Sodium-to-potassium ratio |
| `Drug` | Target drug class |

The target variable is `Drug`, which is encoded before training.

## Technologies Used

- Python 3.12
- Flask
- Pandas
- NumPy
- scikit-learn
- HTML
- CSS
- JavaScript
- SVG

## Installation

Clone the repository:

Create and activate your Python environment.

Install the required dependencies:
```conda install flask numpy pandas scikit-learn```
## Running the Application

Start the Flask app:
``` python app.py```


## How It Works

The application performs the following steps:

1. Loads the `drug200.csv` dataset.
2. Separates features and target labels.
3. Encodes the target drug classes.
4. Converts categorical input columns into numeric values.
5. Splits the dataset into training and testing data.
6. Trains a `DecisionTreeClassifier` using the Gini criterion.
7. Extracts internal tree information from the trained model.
8. Sends the extracted tree data to the frontend.
9. Renders the tree interactively in the browser.

## Gini Index Formula

The Gini Index measures how impure a node is.
```Gini = 1 - Σ Pi²```

Where:

- `Pi` is the probability of class `i` in the node.
- A Gini value of `0` means the node is pure.
- Higher Gini values mean the node contains a more mixed class distribution.

## Notes

- The model is trained when `app.py` starts.
- The current implementation encodes categorical columns manually.
- The visualizer uses tree data extracted from `sklearn.tree.DecisionTreeClassifier`.
- `tree.json` can be used as a saved representation of a trained tree.

## Possible Improvements

- Add model accuracy reporting
- Add prediction form for new patient data
- Save trained models using `pickle` or `joblib`
- Add automatic preprocessing pipelines
- Add support for entropy-based trees
- Add support for regression trees
- Add unit tests
- Add deployment instructions

## License

This project is open-source and available for learning and experimentation.

You may add a license such as MIT if you plan to share or publish it publicly.

## Author

Created by `Chinmay Bhandare`.