"""Test Depth Prediction Model

To run these tests,

1. download the file linked here: https://aquabyte.atlassian.net/browse/RS-143
2. Edit ``LOCAL_PATH_TO_PARQUET`` to be the path to the downloaded file
3. use the following command: `python -m pytest test_predict_depth.py`"""

import json
import os

import pytest
import numpy as np

from predict_depth import train_model, get_data, DepthModel, train_all_models_and_serialize, load_model

LOCAL_PATH_TO_PARQUET = '/Users/siddharthsachdeva/Downloads/weekly-fd.parquet'

def test_load_data():
    df = get_data(LOCAL_PATH_TO_PARQUET)
    assert df.shape == (2165, 19)

def test_train_model():
    df = get_data(LOCAL_PATH_TO_PARQUET)
    model, report = train_model(df)
    assert isinstance(model, DepthModel)

def test_predict():
    df = get_data(LOCAL_PATH_TO_PARQUET)
    model, report = train_model(df)
    test_inp = np.array([2.5, 1.7])
    preds = model.predict(test_inp)
    assert isinstance(preds, type(test_inp))
    assert preds.shape == (2, )

def test_serialize_deserialize(tmp_path):
    df = get_data(LOCAL_PATH_TO_PARQUET)
    model, report = train_model(df)
    test_inp = [2.5, 1.7]
    preds_before_serializing = model.predict(test_inp)
    params_path = tmp_path / "linear_model_params.json"
    with params_path.open('w') as out_params:
        json.dump(model.get_weights(), out_params)
    with params_path.open('r') as in_params:
        deserialized_model = DepthModel.init_from_weights(json.load(in_params))
        np.testing.assert_almost_equal(deserialized_model.predict(test_inp), preds_before_serializing)

def test_train_all_models_and_predict(tmp_path):
    savepath = tmp_path / 'save.json'
    test_inp = [2.5, 1.7]
    pen_ids = train_all_models_and_serialize(LOCAL_PATH_TO_PARQUET, str(savepath))
    for pen_id in pen_ids:
        model = load_model(str(savepath))
        assert model.predict(test_inp).shape == (2, )
