import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from data import download_images_to_local_dir
from train import run

if __name__ == '__main__':
    fname = 'qa_accept_cogito_skips_03-04-2020_100k'
    print('\n\n\n\n 100k {fname}')
    run(fname, fname, 'pad', 64, 0.8, 0, None)
