from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base


class SqlClient:

    def __init__(credentials):
        url = "postgresql://{}:{}@{}:{}/{}".format(credentials["user"],
                                                   credentials["password"],
                                                   credentials["host"],
                                                   credentials["port"],
                                                   credentials["database"])
        engine = create_engine((url)
        session_class = sessionmaker(bind=engine)
        self.session = session_class()

        self.Base = automap_base()
        self.Base.prepare(engine, reflect=True)

    def get_calibration_parameters(enclosure_id):
        """ get camera parameters from table """
        Calibration = self.Base.classes.calibrations
        calibration = self.session.query(Calibration) \
                                  .filter(Calibration.enclosure_id == enclosure_id) \
                                  .order_by(Calibration.utc_timestamp.desc()) \
                                  .first()
        key = calibration.stereo_camera_rectification_params_s3_key
        bucket = calibration.s3_bucket
        return bucket, key

    def populate_data_collection(meta_bucket, meta_key, metadata):
        """ populate gtsf data collection RDS table """
        GtsfDataCollection = self.Base.classes.gtsf_data_collections
        _, _, pen_number, date, fish_identifier = meta_key.split("/")

        new_gtsf_data_collection = GtsfDataCollection(
            enclosure_id = metadata["enclosure_id"],
            pen_number = pen_number,
            gtsf_fish_identifier = fish_identifier,
            date = date,
            num_stereo_frame_pairs = len(rectified_image_files) / 2,
            ground_truth_metadata = json.dumps(metadata),
            raw_frames_s3_folder = raw_folder,
            rectified_frames_s3_folder = rectified_folder,
            s3_bucket = meta_bucket
        )
        self.session.add(new_gtsf_data_collection)
        self.session.commit()

    def populate_stereo_frame_pairs():
        """ for each data collection fish, create stereo frame pairs """
        continue



