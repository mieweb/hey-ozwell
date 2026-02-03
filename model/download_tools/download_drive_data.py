import pandas as pd
import os
import simplejson as json
from pathlib import Path
import shutil
import re
from zipfile import ZipFile
import gdown
import argparse    
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Downloading data from google drive')
    parser.add_argument('--data-dir', type=str, default='../data',
                       help='path to directory where unzipped data should end up')
    parser.add_argument('--delete-after-extract', action='store_true',
                       help='deletes zip file after extraction')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    # download data zipfile from google drive
    if not (data_dir.parent / 'data.zip').exists():
        gdown.download('https://drive.google.com/uc?id=1FKdgd-hNHBWzxcUDbetY4FcTPBItRmzA', str(data_dir.parent / 'data.zip'))
    # unzip data
    zip_obj = ZipFile(data_dir.parent / 'data.zip')
    zip_obj.extractall(data_dir.parent)

    if args.delete_after_extract:
        os.remove(data_dir.parent / 'data.zip')

    for phrase in ['hey-ozwell', "ozwell-i'm-done", 'go-ozwell', 'ozwell-go']:
        # restructure directory
        (data_dir / phrase / 'test' / 'positive').mkdir(parents=True, exist_ok=True)
        (data_dir / phrase / 'test' / 'negative').mkdir(parents=True, exist_ok=True)
        (data_dir / phrase / 'train' / 'negative').mkdir(parents=True, exist_ok=True)
        positive_data_path = (data_dir / phrase / 'train' / 'positive')
        if not positive_data_path.exists():
            shutil.move((data_dir / phrase / 'positive'), positive_data_path)
        else:
            for file in os.listdir(positive_data_path):
                shutil.move((data_dir / phrase / 'positive' / file), positive_data_path)
            shutil.rmtree((data_dir / phrase / 'positive' / file))

        # Create manifest
        df = pd.DataFrame(os.listdir(str(positive_data_path)), columns=['file'])
        df.insert(1, 'label', 1)
        df.insert(2, 'text', phrase.replace('-', ' '))
        df['voice'] = df['file'].map(lambda x: re.match(r'\d+_(.*)_aug_\d+\.wav', x)[1].replace('_', ' ') if '_aug_' in x else re.match(r'\d+_(.*)\.wav', x)[1].replace('_', ' '))
        manifest = {'train': {'positive_samples': df.to_dict(orient='records'), 'negative_samples': []}, 'test': {'positive_samples': [], 'negative_samples': []}}
        
        # Save manifest
        with open(data_dir / phrase / 'training_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=4, ignore_nan=True)


if __name__=='__main__':
    main()