import argparse
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_size', default='20')
    parser.add_argument('--save_path', default='data/dataset')

    args = parser.parse_args()

    download_dataset(args.dataset_size, args.save_path)
