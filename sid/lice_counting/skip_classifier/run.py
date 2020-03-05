from data import download_images_to_local_dir
from train import run

if __name__ == '__main__':
    fname = 'qa_accept_cogito_skips_03-04-2020'
    print('\n\n\n\nGENERAL 10k {fname}')
    run(fname, fname, 'pad', 64, 0.8, 0, None)
    useful_labels = [
            'BLURRY',
            'BAD_CROP',
            'BAD_ORIENTATION',
            'OBSTRUCTION',
            'TOO_DARK'
    ]
    for lab in useful_labels:
        fname = f'qa_accept_{lab}_skips_03-04-2020'
        print('\n\n\n\n{lab} specific {fname}')
        download_images_to_local_dir(fname=fname)
        run(fname, fname, 'pad', 64, 0.8, 0, None)

    fname = 'qa_accept_cogito_skips_03-04-2020_100k'
    print('\n\n\n\n 100k {fname}')
    download_images_to_local_dir(fname)
    run(fname, fname, 'pad', 64, 0.8, 0, None)
