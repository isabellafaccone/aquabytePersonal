from sqlalchemy import create_engine
import json, os
import pandas as pd
import requests
import shutil
import torch
from multiprocessing import Pool
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from loader_new import TRANSFORMS
import cv2
from sklearn.metrics import precision_score, recall_score, roc_auc_score

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

from research.utils.data_access_utils import RDSAccessUtils
from config import SKIP_CLASSIFIER_TEST_IMAGE_DIRECTORY, SKIP_CLASSIFIER_MODEL_DIRECTORY
from model_new import ImageClassifier

rds_access_utils = RDSAccessUtils(json.load(open(os.environ['DATA_WAREHOUSE_SQL_CREDENTIALS'])))

ACCEPT_LABEL, SKIP_LABEL = 'ACCEPT', 'SKIP'

id2state = {
    4: 'SKIPPED_ANN',
    6: 'SKIPPED_QA',
    7: 'VERIFIED'
}

def get_label(state):
    if state == 'VERIFIED':
        return 1
    elif state == 'SKIPPED_ANN':
        return 0
    elif state == 'SKIPPED_QA':
        return 0
    else:
        return None

def get_url(row):
    if isinstance(row['left_crop_url'], str):
        return row['left_crop_url']
    elif isinstance(row['right_crop_url'], str):
        return row['right_crop_url']
    else:
        assert False

def image_to_array(file_path):
    # Read an image with OpenCV
    image = cv2.imread(file_path)

    # By default OpenCV uses BGR color space for color images,
    # so we need to convert the image to RGB color space.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = TRANSFORMS['pad'](image=image)['image']
    return image

def download_image(_row, exclude_images=[]):
    _, row = _row

    url, local_path = row['url'], row['local_path']
    if local_path not in exclude_images:
        response = requests.get(url, stream=True)
        with open(local_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
    return local_path

num_processes = 20
device = 1

def evaluate(name, start_date):
    query = """SELECT pen_id, annotation_state_id, base_key, url_key, left_crop_url, right_crop_url, 
        left_crop_metadata, right_crop_metadata, annotation FROM prod.crop_annotation WHERE (service_id=1) 
        AND (annotation_state_id IN (4, 6, 7)) AND captured_at > '%s'""" % (start_date, )

    production_data = rds_access_utils.extract_from_database(query)

    production_data['site_id'] = production_data['base_key'].str.split('/').apply(lambda ps: ps[1])
    production_data['state'] = production_data['annotation_state_id'].apply(lambda id: id2state[id] if id in id2state else None)

    production_data = production_data[production_data['state'].notnull()]

    qa_accepts = production_data[production_data['state'] == 'VERIFIED']
    pen_counts = qa_accepts.site_id.value_counts()

    all_pens = list(production_data.site_id.unique())
    naccepts_per_pen = 200
    sampled_accepts = pd.DataFrame([], columns=qa_accepts.columns)

    for s in all_pens:
        this_pen_accepts = qa_accepts[qa_accepts['site_id'] == s]
        this_pen_count = 0 if s not in pen_counts else pen_counts[s]
        this_pen_sample = this_pen_accepts.sample(min(naccepts_per_pen, len(this_pen_accepts)))
        sampled_accepts = pd.concat([sampled_accepts, this_pen_sample])

    pen_counts = sampled_accepts['site_id'].value_counts()

    cogito_skips = production_data[production_data['state'] == 'SKIPPED_ANN']

    all_pens = list(production_data.site_id.unique())
    nskips_per_pen = int(round((len(qa_accepts)*2)/len(all_pens), 0))
    sampled_skips = pd.DataFrame([], columns=cogito_skips.columns)

    for p in all_pens:
        this_pen_skips = cogito_skips[cogito_skips['site_id'] == p]
        this_pen_count = 0 if p not in pen_counts else pen_counts[p]
        this_pen_sample = this_pen_skips.sample(min(this_pen_count, len(this_pen_skips)))
        sampled_skips = pd.concat([sampled_skips, this_pen_sample])

    eval_data = pd.concat([sampled_accepts, sampled_skips])
    eval_data['url'] = eval_data.apply(get_url, axis=1)

    production_eval_img_dir = os.path.join(SKIP_CLASSIFIER_TEST_IMAGE_DIRECTORY, name)
    os.makedirs(production_eval_img_dir, exist_ok = True)

    def get_local_path(url):
        name = '_PATHSEP_'.join(url.split('/')[3:])
        return os.path.join(production_eval_img_dir, name)

    eval_data['local_path'] = eval_data.url.apply(get_local_path)

    already_downloaded = os.listdir(production_eval_img_dir)
    already_downloaded = [os.path.join(production_eval_img_dir, url) for url in already_downloaded]

    production_data_images = eval_data[~eval_data.url.duplicated()]
    need_to_download = production_data_images[~production_data_images['local_path'].isin(already_downloaded)]

    pool = Pool(num_processes)

    list(tqdm(pool.imap(download_image, need_to_download.iterrows()), total=len(need_to_download)))

    downloaded_production_data = production_data_images
    downloaded_production_data['label'] = downloaded_production_data['state'].apply(get_label)
    
    # Load the model

    path = os.path.join(SKIP_CLASSIFIER_MODEL_DIRECTORY, name, 'model.pt')
    new_model = ImageClassifier(['ACCEPT', 'SKIP'], savename=None)
    new_model.load_state_dict(torch.load(path))
    new_model.to(device)
    new_model.cuda()
    new_model.eval()

    def path2newmodelpredictions(file_path):
        image = image_to_array(file_path)
        cuda_inputs = torch.unsqueeze(image.to(device), dim=0)
        return new_model(cuda_inputs)

    downloaded_production_data['production_predicted_accept_prob'] = \
        downloaded_production_data.apply(lambda row:
                                         row['left_crop_metadata']['quality_score'] if row['left_crop_metadata']
                                         else row['right_crop_metadata']['quality_score'], axis=1)

    tqdm.pandas()
    
    downloaded_production_data['new_model_predicted_accept_prob'] = downloaded_production_data[
        'local_path'].progress_apply(
        path2newmodelpredictions)

    accept_label_idx = 0

    output = downloaded_production_data['new_model_predicted_accept_prob'].values
    labels = downloaded_production_data['label'].values

    outputs = torch.cat(list(output.values))
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    preds = preds == accept_label_idx
    pred_probs = outputs[:, accept_label_idx]
    pred_probs = pred_probs.detach().cpu().numpy()

    precision = precision_score(labels, preds)

    recall = recall_score(labels, preds)

    auc = roc_auc_score(labels, pred_probs)

    return {
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

if __name__ == '__main__':
    metrics = evaluate('2021-03-16', '2021-03-09')

    print(metrics)