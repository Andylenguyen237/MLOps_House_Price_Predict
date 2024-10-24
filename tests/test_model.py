import pytest
import joblib
import numpy as np

model = joblib.load('artefacts/best_model.pkl')


def test_rf_model():

    input = np.array([[0.00632, 18.0, 2.31, 0.0, 0.538, 6.575,
                     65.2, 4.09, 1.0, 296.0, 15.3, 396.9, 4.98]])
    prediction = model.predict(input)
    assert prediction is not None
    assert len(prediction) == 1
