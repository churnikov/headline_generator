import argparse
import csv
import gzip
import json
import os

from tqdm import tqdm
import requests

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

def download_dataset(size: str='20', save_path='data/dataset') -> None:
    """
    :param size: size of dataset to download. Accepts following values:
                 - "all"  download all files
                 - "full" full archive
                 - "1k"   first 1k news
                 - "20"   first 20 news
    """

    links = {
        'full': 'https://github.com/RossiyaSegodnya/ria_news_dataset/raw/master/ria.json.gz',
        '1k'  : 'https://github.com/RossiyaSegodnya/ria_news_dataset/raw/master/ria_1k.json',
        '20'  : 'https://github.com/RossiyaSegodnya/ria_news_dataset/raw/master/ria_20.json'
    }

    if size == 'all':
        for sz, link in links.items():
            download_dataset(sz, save_path)
    else:
        link = links[size]
        resp = requests.get(link, stream=True)

        full_save_path = os.path.join(CURRENT_PATH, save_path)
        file_name = link.split('/')[-1]

        if not os.path.isdir(full_save_path):
            os.makedirs(full_save_path)

        with open(os.path.join(CURRENT_PATH, save_path, file_name), 'wb') as f:
            for data in tqdm(resp.iter_content(), desc=f'Downloading file {file_name}'):
                f.write(data)


def json_iterator(file_name=os.path.join(CURRENT_PATH, 'ria.json.gz')):
    if file_name.endswith('.gz'):
        with gzip.open(file_name) as f:
            for l in f:
                yield json.loads(l)
    else:
        with open(file_name, 'r') as f:
            for l in f:
                yield json.loads(l)


def jsons2csvs(file_name: str):
    out_fn = file_name.split('.')[0] + '.csv'
    with open(out_fn, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'title'])
        for jsn in tqdm(json_iterator(file_name)):
            writer.writerow([jsn['text'], jsn['title']])


def __jsons2csvs(dataset_size, save_path):
    if dataset_size == 'all':
        __jsons2csvs('20', save_path)
        __jsons2csvs('1k', save_path)
        __jsons2csvs('full', save_path)
    else:
        file_name = os.path.join(save_path, (f'ria_{dataset_size}.json' if dataset_size != 'full' else 'ria.json.gz'))
        jsons2csvs(file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_size', default='20')
    parser.add_argument('--save_path', default='data/dataset')
    parser.add_argument('--make_csvs', default=True)

    args = parser.parse_args()

    download_dataset(args.dataset_size, args.save_path)
    if args.make_csvs:
        __jsons2csvs(args.dataset_size, args.save_path)