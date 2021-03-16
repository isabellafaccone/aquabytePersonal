from datetime import date
from dateutil.relativedelta import relativedelta

from _00_generate_dataframe import get_dataframe
from _10_download_data import download_images_to_local_dir
from _20_train_skip_classifier import run

from research_api.skip_classifier import add_model, add_train_dataset

if __name__ == '__main__':
#     three_months_ago = date.today() + relativedelta(months=-3)
#     start_date = three_months_ago.strftime('%Y-%m-%d')
#     today = (date.today()).strftime('%Y-%m-%d')

#     pen_ids = [56, 60, 37, 85, 86, 66, 83, 84, 95, 100, 61, 1, 4, 126, 128, 129, 133, 122, 123,
#                137, 114, 119, 116, 131, 132, 5, 145, 171, 173, 138, 149, 159, 210, 211, 67, 193, 140, 142, 216]

#     # Write the skip dataset to a file
#     retraining_name = '%s' % (today, )

#     # Get the dataframe
#     dataset_file_name, metadata = get_dataframe(retraining_name, pen_ids, start_date, today)

#     # Download the images from the dataframe
#     dataset_file_name, metadata = download_images_to_local_dir(retraining_name, metadata)

#     # Add to the train dataset
#     add_train_dataset(retraining_name, dataset_file_name, metadata)

    retraining_name = '2021-03-16'

    # Train the new model
    model_file_name = run(retraining_name, retraining_name, 'pad', 'full_fish', 64, 0.8, 0, None)

    # Add the new model
    add_model(retraining_name, model_file_name, True)
