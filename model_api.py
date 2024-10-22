from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/predict', method=['POST'])
def predict():
    """
    Predict house price using Random Forest. Store each trial of hyperparameter tuning as artefact along with their evaluation metrics
    param: n_estimators
    param: max_depth
    param: min_samples_split
    param: min_samples_leaf
    return: serve the best model with best predict
    """

    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
