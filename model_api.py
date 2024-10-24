from flask import Flask, jsonify, request
import os
import json
import joblib
import pandas as pd
import numpy as np
import ssl
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

app = Flask(__name__)

# Directory to store the artefacts
ARTEFACTS_DIR = "artefacts"
BEST_MODEL_PATH = os.path.join(ARTEFACTS_DIR, 'best_model.pkl')
ARTEFACTS_FILE = os.path.join(ARTEFACTS_DIR, 'tuning_artefacts.json')
DATASET_PATH = "boston_dataset.txt"
os.makedirs(ARTEFACTS_DIR, exist_ok=True)


@app.route('/')
def home():
    return "Boston House Predict Using Random Forest"


def load_boston_dataset():
    """
    Function is used to load boston dataset from given url from scikit-learn
    Note that load_boston from scikit-learn is disable from version 1.2
    This is an alternative method to load and extract the dataset
    return: data (13 features) with MEDV as the target (label)
    """

    if os.path.exists(DATASET_PATH):
        raw_df = pd.read_csv(DATASET_PATH, sep="\s+", skiprows=22, header=None)
        data = np.hstack(
            [raw_df.values[::2, :], raw_df.values[1::2, :2]])  # 13 FEATURES
        target = raw_df.values[1::2, 2]  # MEDV
    else:
        raise FileExistsError("Err: Dataset not found")

    return data, target


def build_rf_model():
    """
    Function is used to build or train Random Forest model.
    Also perform hyperparamter tuning and evaluate using MSE across validation set
    return: grid search results (artefacts)
    """
    X, y = load_boston_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=23)

    # Initate parameter grid for GridSearch
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Build Random Forest model
    rf_model = RandomForestRegressor()

    # Grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=rf_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Collecting artefacts and metrics for each trial
    artefacts = []
    for i, params in enumerate(grid_search.cv_results_['params']):
        test_score = -grid_search.cv_results_['mean_test_score'][i]
        artefacts.append({
            'trial': i + 1,
            'hyperparameters': params,
            'test_score': test_score
        })

    # Save artefacts to file
    with open(ARTEFACTS_FILE, 'w') as f:
        json.dump(artefacts, f, indent=4)

    # Save the best model
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, BEST_MODEL_PATH)

    return artefacts, best_model


# if the model has already exists
if os.path.exists(BEST_MODEL_PATH):
    best_model = joblib.load(BEST_MODEL_PATH)
else:
    # If the model doesn't exist, perform tuning and select the best model
    artefacts, model = build_rf_model()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict house price using Random Forest. 
    param: features ({}) - 13 features
    return: serve the best model with best predict (json)
    """

    try:
        data = request.json
        features = data['features']

        # invalid input error handling
        if None in features or len(features) != 13:
            return jsonify({'error': '13 features are required'}), 400

        features = [float(f) for f in features]

        # Reshape features to 2D array for model prediction
        features = np.array(features).reshape(1, -1)

        # predict
        prediction = best_model.predict(features)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/artefacts', methods=['GET'])
def get_artefacts():
    """
    Retrieve artefacts, whereby each trial of hyperparameter tuning is stored along with their evalution MSE score
    return: artefacts (json)
    """

    try:
        with open(ARTEFACTS_FILE, 'r') as f:
            artefacts = json.load(f)
        return jsonify({'trials': artefacts})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
