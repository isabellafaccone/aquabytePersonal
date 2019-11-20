from typing import Tuple, Dict, Union, List
import json

import pyarrow.parquet as pq
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
import numpy as np

PEN_ID_COLUMN = 'pen_id'
CROP_METADATA_COLUMN = 'left_crop_metadata'
MODEL_TYPE = LinearRegression
HYPERPARAMS = {LinearRegression: {'normalize': True}}
MIN_DATAPOINTS = 10

class DepthModel:
    def __init__(self, model: LinearRegression) -> None:
        self.model = model

    def predict(self, crop_areas: Union[list, np.array]) -> np.array:
        crop_areas = np.array([crop_areas]).T
        return self.model.predict(crop_areas)

    def get_weights(self) -> Dict[str, float]:
        assert len(self.model.coef_.shape) == 1
        assert self.model.coef_.shape[0] == 1
        assert isinstance(self.model.intercept_, np.float64) == 1
        return {'coef': float(self.model.coef_[0]),
                'intercept': float(self.model.intercept_)}

    @classmethod
    def init_from_weights(cls, weights: Dict[str, float]) -> 'DepthModel':
        model = MODEL_TYPE(**HYPERPARAMS[MODEL_TYPE])
        model.coef_ = np.array([weights['coef']])
        model.intercept_ = np.array([weights['intercept']])
        return cls(model)


def get_data(local_path_to_parquet: str) -> pd.DataFrame:
    # Load the data in
    pdf = pd.read_parquet(local_path_to_parquet)

    # Clean the data
    pdf = pdf[pdf[CROP_METADATA_COLUMN].apply(type) == str]
    expanded = pdf[CROP_METADATA_COLUMN].apply(json.loads).apply(pd.Series)
    pdf = pd.concat([pdf, expanded], axis=1)
    pdf = pdf[pdf.distance_from_camera.notnull() & pdf.crop_area.notnull()]
    return pdf

def get_performance_report(
        y_true: np.array,
        y_pred: np.array,
        metrics=[explained_variance_score, mean_absolute_error, mean_squared_error]
    ) -> dict:
    report = {'evaluation': {m.__name__: m(y_true, y_pred) for m in metrics}}
    report['preds'] = y_pred
    report['y_true'] = y_true
    report['residuals'] = list(y_true - y_pred)
    return report

def train_model(pdf: pd.DataFrame) -> Tuple[DepthModel, dict]:
    x, y = pdf[['crop_area']], pdf['distance_from_camera']
    model = MODEL_TYPE(**HYPERPARAMS[MODEL_TYPE])
    model.fit(x, y)
    pred = model.predict(x)
    report = get_performance_report(y, pred)
    model = DepthModel(model)
    return model, report

def train_all_models_and_serialize(local_path_to_parquet: str, savepath: str) -> List[int]:
    """Main function for training models from parquet file

    Parameters
    ----------
    local_path_to_parquet: Local path to parquet file with crop data
    savepath: Local path where you want model params,results to be saved

    Returns
    -------
    The pen ids we trained models for
    """

    pen_specific_models = dict()
    pdf = get_data(local_path_to_parquet)
    for pen_id in pdf[PEN_ID_COLUMN].unique():
        pdf = pdf[pdf[PEN_ID_COLUMN] == pen_id]
        try:
            assert pdf.shape[0] >= MIN_DATAPOINTS, f'Need at least {MIN_DATAPOINTS} datapoints for 2 params'
            model, report = train_model(pdf, pen_id)
            pen_specific_models[pen_id] = {'weights': model.get_weights(),
                                           'performance_report': report}
        except Exception as e:
            print(e)
            pass
    with open(savepath, 'w') as out:
        json.dump(pen_specific_models, out)
    return list(pen_specific_models.keys())

def load_model(savepath: str, pen_id: int) -> DepthModel:
    with open(savepath, 'r') as in_f:
        model_weight = json.load(in_f)['weights']
        return DepthModel.init_from_weights(model_weight)
