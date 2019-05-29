import json
import os

import skimage.io as io
from sqlalchemy import create_engine, MetaData, Table, select, and_, func


def lice_main(base_folder, s3_client):
    """ query from postgres table"""

    sql_credentials = json.load(open(os.environ["SQL_CREDENTIALS"]))
    sql_engine = create_engine(
        "postgresql://{}:{}@{}:{}/{}".format(sql_credentials["user"], sql_credentials["password"],
                                             sql_credentials["host"], sql_credentials["port"],
                                             sql_credentials["database"]))

    metadata = MetaData()
    # step 1 - download crops + json
    # get the two tables we care about
    fish_crops = Table('lati_fish_detections', metadata, autoload=True, autoload_with=sql_engine)
    lice_crops = Table('lati_fish_detections_lice_annotations_reconciled', metadata, autoload=True,
                       autoload_with=sql_engine)

    # inner join on fish crop id
    # TODO @Thomas debug this
    query = select([fish_crops.c.image_key, lice_crops.c.lice_bbox_list]) \
        .select_from(lice_crops.join(fish_crops, lice_crops.c.lati_fish_detections_id == fish_crops.c.id)) \
        .where(and_(fish_crops.c.site_id == 23,
                    lice_crops.c.lice_bbox_list != None,
                    # func.json_array_length(lice_crops.c.lice_bbox_list) > 0,
                    lice_crops.c.created_by == "gunnar@aquabyte.ai"))

    json_files = []
    counter = 0
    with sql_engine.connect() as conn:
        for row in conn.execute(query):
	    if len(row) == 0:
	    	continue
            # [image_key, lice_json]
            results = {}
            key = row[0]
            _, farm, penid, date, image_name = key.split('/')
            results["key"] = key
            results["farm"] = farm
            results["penid"] = penid
            results["date"] = date
            results["image_name"] = image_name
            results["detections"] = row[1]
            results["processed"] = False
            destination = os.path.join(base_folder, "crops", farm, date, penid)

            results["image_path"] = os.path.join(destination, image_name)
            if not os.path.isdir(destination):
                os.makedirs(destination)
            with open(os.path.join(destination, image_name.replace("jpg", "json")), "w") as f:
                json.dump(results, f)
            if not os.path.isfile(os.path.join(destination, image_name)):
                s3_client.download_file("aquabyte-crops", key, os.path.join(destination, image_name))
                counter += 1
            json_files.append(os.path.join(destination, image_name.replace("jpg", "json")))
    print("{} new files have downloaded".format(counter))

    # step 2 - create training and validation sets
    for jf in json_files:
        with open(jf, "r") as f:
            annotations = json.load(f)
        if annotations["processed"]:
            continue
        image = io.imread(annotations["image_path"])
        farm = annotations["farm"]
        date = annotations["date"]
        penid = annotations["penid"]
        image_name = annotations["image_name"]
        for (i, annotation) in enumerate(annotations['detections']):
            category = annotation['category']
            position = annotation['position']
            x1, height, y1, width = position["left"], position["height"], position["top"], position["width"]
            destination = os.path.join(base_folder, "lice_only", farm, date, penid, category)
            if not os.path.isdir(destination):
                os.makedirs(destination)
            lice_name = image_name + ".lice_{}.jpg".format(i)
            io.imsave(os.path.join(destination, lice_name), image[y1:y1+height, x1:x1+width, :])
        # tag as processed
        annotations["processed"] = True
        with open(jf, "w") as f:
            json.dump(annotations, f)



