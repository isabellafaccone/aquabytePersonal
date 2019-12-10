from aquabyte.data_access_utils import RDSAccessUtils
from get_calibration_report import CalibrationReport, DEPTH_REPORT_SAVENAME, IMAGE_DEPTH_DIR
import json
import os
import pytest
import numpy as np

SAVE_DIR = 'save_dir'

@pytest.fixture
def keypoint_table():
    pen_id = 64
    date_range = ('11-29-2019', '12-06-2019')
    credentials = json.load(open(os.environ["PROD_SQL_CREDENTIALS"]))
    s3 = RDSAccessUtils(credentials)
    query = f"""SELECT * FROM keypoint_annotations WHERE (pen_id = {pen_id}) AND (captured_at BETWEEN '{date_range[0]}' AND '{date_range[1]})')"""
    keypoint_table = s3.extract_from_database(query)
    return keypoint_table

def test_get_distance(keypoint_table, tmpdir):
    save_dir = tmpdir.mkdir(SAVE_DIR)
    report = CalibrationReport(keypoint_table, SAVE_DIR)
    report.clean_input()
    report.get_distance()
    assert report.keypoint_table['distance'].isnull().sum() == 0

@pytest.fixture
def calibration_report(keypoint_table, tmpdir):
    save_dir = tmpdir.mkdir(SAVE_DIR)
    report = CalibrationReport(keypoint_table, save_dir)
    report.clean_input()
    report.get_distance()
    return report

def test_build_depth_report(calibration_report):
    save_dir = calibration_report.save_dir
    print(save_dir)
    results = calibration_report.build_depth_report()
    assert os.path.exists(os.path.join(save_dir, DEPTH_REPORT_SAVENAME))
    np.testing.assert_almost_equal(results['mean'], 1.0859345539996017)
    np.testing.assert_almost_equal(results['median'], 1.0894455899882012)
    np.testing.assert_almost_equal(results['std'], 0.20423878258196673)

def test_plot_example_images(calibration_report):
    calibration_report.plot_example_images(depths=[0.4, 0.6, 0.8], dpi=10)
    image_dir = os.path.join(calibration_report.save_dir, IMAGE_DEPTH_DIR)
    assert os.path.exists(image_dir)
    assert len(os.listdir(image_dir)) == 2
