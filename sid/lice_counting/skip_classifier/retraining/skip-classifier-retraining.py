from datetime import date
from dateutil.relativedelta import relativedelta

from _00_generate_training_dataframe import get_dataframe
from _10_download_training_data import download_images_to_local_dir
from _20_train_skip_classifier import run
from _30_evaluate_skip_classifier import evaluate

from research_api.skip_classifier import add_model, add_train_dataset, add_retraining, get_train_dataset

if __name__ == '__main__':
    print('Running skip classifier retraining')

    three_months_ago = date.today() + relativedelta(months = -3)
    seven_days_ago = date.today() + relativedelta(days = -7)

    today = date.today().strftime('%Y-%m-%d')
    start_date = three_months_ago.strftime('%Y-%m-%d')
    end_date = seven_days_ago.strftime('%Y-%m-%d')

    print('today', today)
    print('start_date', start_date)
    print('end_date', end_date)

    pen_ids = [56, 60, 37, 85, 86, 66, 83, 84, 95, 100, 61, 1, 4, 126, 128, 129, 133, 122, 123,
               137, 114, 119, 116, 131, 132, 5, 145, 171, 173, 138, 149, 159, 210, 211, 67, 193, 140, 142, 216]

    # Write the skip dataset to a file
    retraining_name = '%s' % (today, )

    print('Retraining name', retraining_name)

    trainDataset = get_train_dataset(pen_ids, start_date, end_date)

    if trainDataset:
        trainDatasetId = trainDataset.id
    else:
        # Get the dataframe
        dataset_file_name, metadata = get_dataframe(retraining_name, pen_ids, start_date, end_date)

        print('dataset_file_name', metadata)

        # Download the images from the dataframe
        dataset_file_name, metadata = download_images_to_local_dir(retraining_name, metadata)

        print('dataset_file_name', metadata)

        # Add to the train dataset
        trainDatasetId = add_train_dataset(retraining_name, dataset_file_name, metadata)

    print('trainDatasetId', trainDatasetId)

    # Train the new model
    model_file_name, metadata = run(retraining_name, retraining_name, 'pad', 'full_fish', 64, 0.8, 0, None)

    print('model_file_name', model_file_name)
    print('metadata', metadata)

    # Add the new model
    modelId = add_model(retraining_name, model_file_name, True)

    print('modelId', modelId)

    retrainingId = add_retraining(trainDatasetId, modelId, retraining_name, metadata)

    print('retrainingId', retrainingId)

    print('Completed skip classifier retraining')

    metrics = evaluate(retraining_name, end_date)

    print('Metrics', metrics)