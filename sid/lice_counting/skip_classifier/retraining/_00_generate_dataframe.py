# Use research-exploration/sid/lice_counting/skip_classifier/Generate%20Stratified%20Skip%20Classifier%20Dataset.ipynb
# Copy the code from there to here to create your script

import os
import json
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta

from research.utils.data_access_utils import RDSAccessUtils

from config import SKIP_CLASSIFIER_DATASET_DIRECTORY

rds_access_utils = RDSAccessUtils(json.load(open(os.environ['DATA_WAREHOUSE_SQL_CREDENTIALS'])))

def get_dataframe(retraining_name, pen_ids, start_date, end_date):
    pen_id_string = ', '.join([str(pen_id) for pen_id in pen_ids])

    query = '''
        SELECT service_id, pen_id, annotation_state_id, left_crop_url, left_crop_metadata, right_crop_url,
        right_crop_metadata, camera_metadata, captured_at, skip_reasons, annotation FROM prod.crop_annotation
        WHERE pen_id IN (%s)
        AND captured_at >= '%s'
        AND captured_at <= '%s'
        AND annotation_state_id IN (3, 4, 7) AND (service_id=1)
    ''' % (pen_id_string, start_date, end_date)

    print(query)

    # Get the annotations

    annotations = rds_access_utils.extract_from_database(query)

    annotations = annotations[(annotations['left_crop_url'].notnull()) | (annotations['right_crop_url'].notnull())]

    annotations['time_bucket'] = annotations['captured_at'].apply(lambda c: c.hour // 2)
    annotations['is_qa_accept'] = annotations['annotation_state_id'] == 7

    accepts = annotations[annotations['is_qa_accept']]

    # Get the accepts

    accepts['visibleBodySections'] = accepts['annotation'].apply(
        lambda ann: ann['visibleBodySections'] if 'visibleBodySections' in ann else None)
    accepts['isPartial'] = accepts['annotation'].apply(
        lambda ann: ann['isPartial'] if 'isPartial' in ann else None)

    accepts = accepts[~(accepts['isPartial'] & accepts['visibleBodySections'].isnull())]

    # Get the bodyparts

    bodyparts = list(accepts['visibleBodySections'].explode().unique())

    for part in bodyparts:
        if part is not None:
            accepts['HAS_' + part] = accepts['visibleBodySections'].apply(
                lambda sections: part in sections if sections is not None else True)
            print(part, accepts['HAS_' + part].value_counts(normalize=True).loc[True])

    SAMPLES_PER_PEN = 5000
    SAMPLE_SIZE = SAMPLES_PER_PEN // len(annotations['time_bucket'].unique())
    print('Sample size', SAMPLE_SIZE)

    def sample_from_pen(pen_rows, sample_strat='random', sample_size=SAMPLE_SIZE):
        if sample_strat == 'random':
            return pen_rows.sample(min(len(pen_rows), int(sample_size)))
        elif sample_strat == 'recent':
            sorted_rows = pen_rows.sort_values(['is_qa_accept', 'captured_at'], ascending=False)
            sorted_rows.drop_duplicates(subset='left_crop_url', inplace=True)
            sorted_rows.drop_duplicates(subset='right_crop_url', inplace=True)
            return sorted_rows.head(sample_size)
        else:
            assert False

    accepts = annotations[annotations['annotation_state_id'].isin([7])]
    accepts = accepts.groupby(['pen_id', 'time_bucket'], group_keys=False).apply(sample_from_pen)

    print('Accepts', len(accepts))

    bucket_sample_sizes = accepts[['pen_id', 'time_bucket']].apply(lambda row: tuple(row), axis=1).value_counts()

    def sample_equal_skips_to_accepts(pen_rows):
        sample_size = pen_rows.sample_size.unique()
        assert len(sample_size) == 1
        sample_size = sample_size[0]
        return sample_from_pen(pen_rows, sample_size=sample_size)

    from tqdm import tqdm
    tqdm.pandas()

    # Get Cogito skips

    cogito_skips = annotations[annotations['annotation_state_id'] == 4]
    cogito_skips.drop_duplicates('left_crop_url', inplace=True)
    cogito_skips.drop_duplicates('right_crop_url', inplace=True)

    def get_sample_size(row):
        try:
            return bucket_sample_sizes[(row['pen_id'], row['time_bucket'])]
        except:
            return 0

    cogito_skips['sample_size'] = cogito_skips.progress_apply(get_sample_size , axis=1)
    chosen_cogito_skips = cogito_skips.groupby(['pen_id', 'time_bucket'], group_keys=False).apply(sample_equal_skips_to_accepts)

    print('Cogito skips', len(chosen_cogito_skips))

    # If we still need skips, get them from the leftover skips

    still_need = accepts.pen_id.value_counts() - chosen_cogito_skips.pen_id.value_counts()
    print(still_need.sum() + len(chosen_cogito_skips))

    leftover_skips = cogito_skips[~cogito_skips['left_crop_url'].isin(chosen_cogito_skips['left_crop_url'])]
    leftover_skips = leftover_skips[~cogito_skips['right_crop_url'].isin(chosen_cogito_skips['right_crop_url'])]

    leftover_skips['sample_size'] = leftover_skips['pen_id'].progress_apply(
        lambda p: still_need[p])
    chosen_cogito_skips2 = leftover_skips.groupby(['pen_id'], group_keys=False).apply(sample_equal_skips_to_accepts)
    chosen_cogito_skips = pd.concat([chosen_cogito_skips, chosen_cogito_skips2])

    print('Cogito skips', len(chosen_cogito_skips))

    still_need = accepts.time_bucket.value_counts() - chosen_cogito_skips.time_bucket.value_counts()
    extras = still_need[still_need<0].sum() * -1
    still_need = still_need[still_need>0]
    still_need = ((still_need / still_need.sum()) * extras).apply(int)

    leftover_skips = cogito_skips[~cogito_skips['left_crop_url'].isin(chosen_cogito_skips['left_crop_url'])]
    leftover_skips = leftover_skips[~cogito_skips['right_crop_url'].isin(chosen_cogito_skips['right_crop_url'])]

    leftover_skips['sample_size'] = leftover_skips['time_bucket'].progress_apply(
        lambda p: 0 if p not in still_need else still_need[p])
    chosen_cogito_skips2 = leftover_skips.groupby(['time_bucket'], group_keys=False).apply(sample_equal_skips_to_accepts)
    chosen_cogito_skips = pd.concat([chosen_cogito_skips, chosen_cogito_skips2])

    print('Cogito skips', len(chosen_cogito_skips))

    # Start building the skip dataset

    skip_dataset = pd.concat([chosen_cogito_skips, accepts.sample(len(chosen_cogito_skips))])

    def get_label(state_id):
        if state_id == 4:
            return 'SKIP'
        elif state_id in [3,7]:
            return 'ACCEPT'
        else:
            assert False

    def get_url(row):
        if row['left_crop_url'] is None:
            return row['right_crop_url']
        if row['right_crop_url'] is None:
            return row['left_crop_url']
        else:
            if row['left_crop_metadata']['quality_score'] > row['left_crop_metadata']['quality_score']:
                return row['left_crop_url']
            else:
                return row['right_crop_url']

    skip_dataset['label'] = skip_dataset['annotation_state_id'].apply(get_label)
    skip_dataset['url'] = skip_dataset.apply(get_url, axis=1)

    skip_dataset = skip_dataset[~skip_dataset.url.duplicated()]

    assert len(skip_dataset) == len(skip_dataset['url'].unique())

    # Write the skip dataset to a file
    dataset_file_name = os.path.join(SKIP_CLASSIFIER_DATASET_DIRECTORY, retraining_name + '.csv')

    skip_dataset.to_csv(dataset_file_name)

    print('Number of skips', len(skip_dataset))

    print('Wrote file', dataset_file_name)

    metadata = {
        'num_rows': len(skip_dataset),
        'pen_ids': pen_ids,
        'start_date': start_date,
        'end_date': end_date
    }

    return dataset_file_name, metadata


if __name__ == '__main__':
    three_months_ago = date.today() + relativedelta(months=-3)
    start_date = three_months_ago.strftime('%Y-%m-%d')
    today = (date.today()).strftime('%Y-%m-%d')

    pen_ids = [56, 60, 37, 85, 86, 66, 83, 84, 95, 100, 61, 1, 4, 126, 128, 129, 133, 122, 123,
               137, 114, 119, 116, 131, 132, 5, 145, 171, 173, 138, 149, 159, 210, 211, 67, 193, 140, 142, 216]

    name = '%s_%s' % (start_date, today)

    skip_dataset, out_file = get_dataframe(name, pen_ids, start_date, today)




