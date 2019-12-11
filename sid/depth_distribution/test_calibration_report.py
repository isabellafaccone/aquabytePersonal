from get_calibration_report import CalibrationReport, DEPTH_REPORT_SAVENAME, IMAGE_DEPTH_DIR
from get_depths import KeypointDepths, TemplateMatchingDepths, DEPTH_COL
import json
import os
import pytest
import numpy as np

SAVE_DIR = 'save_dir'

def build_keypoint_depth_table():
    depths = KeypointDepths.from_pen_id_and_date_range_using_db(
        pen_id=64, date_range=('11-29-2019', '12-06-2019'))
    depths.get_depth_table()
    return depths.depth_table

def build_parquet_depth_table():
    depths = TemplateMatchingDepths.from_week_date_and_pen_id(
        pen_id=64, week_date='2019-11-25')
    depths.get_depth_table()
    return depths.depth_table

@pytest.mark.parametrize("depth_table", [
    #build_keypoint_depth_table()
    build_parquet_depth_table()
])
class TestCalibrationReport:

    def test_get_distance(self, depth_table, tmpdir):
        save_dir = tmpdir.mkdir(SAVE_DIR)
        calibration_report = CalibrationReport(depth_table, save_dir)
        assert calibration_report.depth_table[DEPTH_COL].isnull().sum() == 0
    
    
    def test_build_depth_report(self, depth_table, tmpdir):
        save_dir = tmpdir.mkdir(SAVE_DIR)
        calibration_report = CalibrationReport(depth_table, save_dir)
        results = calibration_report.build_depth_report()
        assert os.path.exists(os.path.join(save_dir, DEPTH_REPORT_SAVENAME))
        np.testing.assert_almost_equal(results['mean'], 1.0859345539996017, decimal=2)
        np.testing.assert_almost_equal(results['median'], 1.0894455899882012, decimal=2)
        np.testing.assert_almost_equal(results['std'], 0.20423878258196673, decimal=2)
    
    def test_plot_example_images(self, depth_table, tmpdir):
        save_dir = tmpdir.mkdir(SAVE_DIR)
        calibration_report = CalibrationReport(depth_table, save_dir)
        calibration_report.plot_example_images(depths=[0.4, 0.6, 0.8], dpi=10)
        image_dir = os.path.join(calibration_report.save_dir, IMAGE_DEPTH_DIR)
        assert os.path.exists(image_dir)
        assert len(os.listdir(image_dir)) == 2
