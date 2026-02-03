import pandas as pd
from elevenlabs import ElevenLabs
from dotenv import load_dotenv
import os
import string
import simplejson as json
from zip import ZipInfo
import time
import shutil
import re
import multiprocessing
import argparse
from zipfile import ZipFile
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))

remove_punctuation = str.maketrans('', '', string.punctuation)
clean_text = lambda text: text.lower().translate(remove_punctuation)

def load_phrases(wake_phrases, csv_path: str, positive: bool, apply_cleaning: bool):
    phrase_type = 'positive' if positive else 'negative'
    for group_name, group in pd.read_csv(csv_path).groupby('wake_phrase'):
        key = group_name.replace(' ', '-')
        if key not in wake_phrases.keys():
            wake_phrases[key] = {'positive': [], 'negative': []}
        phrases = group[f'{phrase_type}_phrase']
        if apply_cleaning:
            phrases = phrases.map(clean_text)
        wake_phrases[key][phrase_type] = list(set(wake_phrases[key][phrase_type] + phrases.to_list()))

def get_all_history(all_phrases: list, output_path: str, filter: bool = True):
    response = client.history.list(model_id='eleven_multilingual_v2')
    records = response.history
    while (response.has_more):
        response = client.history.list(model_id='eleven_multilingual_v2', date_before_unix=response.scanned_until+1) # add one to make sure we don't skip over items with same timestamp
        records.extend(response.history)
    df = pd.DataFrame.from_records(list(map(lambda x: x.__dict__, records))).drop_duplicates('history_item_id')
    if filter:
        df = df[df['text'].map(clean_text).isin(all_phrases)]
    df.to_json(output_path, orient='records', indent=4)

def save_audio(history_item_ids: list, zip_path: str):
    audio_files = client.history.download(history_item_ids=history_item_ids, output_format='wav')
    with open(zip_path, "wb") as f:
        for chunk in audio_files:
            if chunk:
                f.write(chunk)
    with open(zip_path, 'rb') as f:
        data = f.read()

    central_dir_offset, central_dir_size, total_entries = ZipInfo.find_central_directory(data)
    if central_dir_offset: 
        metadata = pd.DataFrame(ZipInfo.parse_central_directory(data, central_dir_offset, total_entries))
        metadata['zip_path'] = zip_path
        metadata['history_item_id'] = history_item_ids
        return metadata
    return None


def move_from_zip(zip_obj, file_path, export_path):
    with zip_obj.open(file_path) as f:
        data = f.read()
    with open(export_path, 'wb') as f:
        f.write(data)
    

def main():
    parser = argparse.ArgumentParser(description='Downloading data from Eleven labs')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='number of files to download at a time')
    parser.add_argument('--negative-phrase-csv', type=str, default='../negative_phrases.csv',
                       help='path to negative phrase csv to load')
    parser.add_argument('--data-dir', type=str, default='../data',
                       help='path to directory where unzipped data should end up')
    parser.add_argument('--split-factor', type=float, default=.8,
                       help='fraction of data to be placed in train split for each phrase/sample_type (the rest will be put in test directory)')
    parser.add_argument('--delete-after-extract', action='store_true',
                       help='deletes all data stored in temp dir after file extraction')

    args = parser.parse_args()

    batch_size = args.batch_size
    temp_dir = './eleven_lab_download'
    os.makedirs(temp_dir, exist_ok=True)

    wake_phrases = {
        'hey-ozwell': {'positive': ['hey ozwell', 'hey oswell', 'hay ozwell'], 'negative': ['hey oswald', 'hey amal', 'hey paul', 'nay ozwell', 'he is well']},
        "ozwell-i'm-done": {'positive': ['ozwell im done', 'ozwell i am done', 'oswell im done', 'ozwell done'], 'negative': ["oswald im done", "ozwell im not done", 'ozwell is fun', "oh swell im done"]},
        'go-ozwell': {'positive': ['go ozwell', 'go oswell'], 'negative': ['go oswald', 'no ozwell', 'so ozwell', 'show ozwell', 'go amal', 'go call', 'gauze well', 'go with the flow']},
        'ozwell-go': {'positive': ['ozwell go', 'oswell go'], 'negative': ['oswald go', 'ozwell no', 'ozwell show', 'ozwell dont', 'ozwell knows', 'is it slow', 'was mellow', "oh we'll know", "cause we'll know"]}
    }
 
    history_path = os.path.join(temp_dir, 'eleven_labs_history.json')
    if not os.path.exists(history_path):
        for k,v in wake_phrases.items():
            v['positive'] = list(map(clean_text, v['positive']))
            v['negative'] = list(map(clean_text, v['negative']))
        load_phrases(wake_phrases, args.negative_phrase_csv, positive=False, apply_cleaning=True)
        all_phrases =  sum([v['positive'] + v['negative'] for v in wake_phrases.values()], [])
        get_all_history(all_phrases, history_path,  True)

    df = pd.read_json(history_path)
    history_item_ids = df['history_item_id'].to_list()

    if not os.path.exists(os.path.join(temp_dir, f'metadata.json')):
        batches = [history_item_ids[i:min(len(history_item_ids), i + batch_size)] for i in range(0, len(history_item_ids), batch_size)]
        
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        pool = multiprocessing.Pool()
        pool = multiprocessing.Pool(processes=len(batches))
        result_async = [pool.apply_async(save_audio, args = (batch, os.path.join(temp_dir, f'audio_{i}.zip'))) for i, batch in enumerate(batches)]
        s = time.time()
        metadata = [r.get() for r in result_async]
        metadata_path = os.path.join(temp_dir, f'metadata.json')
        metadata = pd.concat(metadata)
        metadata.to_json(metadata_path, indent=4, index=False, orient='records')
        e = time.time()
        logger.info(f'Finished in {e-s} seconds')
    else:
        metadata = pd.read_json(os.path.join(temp_dir, f'metadata.json'), orient='records')

    
    df['phrase'] = None
    df['sample_type'] = None
    for k,v in wake_phrases.items():
        cleaned_text = df['text'].map(clean_text)
        df.loc[cleaned_text.isin(v['positive'] + v['negative']), 'phrase'] = k
        df.loc[cleaned_text.isin(v['positive']), 'sample_type'] = 'positive'
        df.loc[cleaned_text.isin(v['negative']), 'sample_type'] = 'negative'
    
    df = df.merge(metadata, on='history_item_id')
    df.rename(columns={'filename': 'src_dir_path', 'voice_name': 'voice'}, inplace=True)
    
    zip_objects = {file: ZipFile(file) for file in df['zip_path'].unique()}
    manifests = {}
    data_dir = args.data_dir
    for phrase in wake_phrases.keys():
        manifest_path = os.path.join(data_dir, phrase, 'training_manifest.json')
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifests[phrase] = json.load(f)
        else:
            manifests[phrase] = {'train': {'positive_samples': [], 'negative_samples': []}, 'test': {'positive_samples': [], 'negative_samples': []}}

    for (phrase, sample_type), group in df.groupby(['phrase', 'sample_type']):
        train = group.sample(frac=args.split_factor, replace=False).get(['text', 'voice', 'zip_path', 'src_dir_path']).copy()
        path = os.path.join(data_dir, phrase, 'train', sample_type)
        os.makedirs(path, exist_ok=True)
        if len(manifests[phrase]['train'][f'{sample_type}_samples']) > 0:
            start_idx = pd.DataFrame(manifests[phrase]['train'][f'{sample_type}_samples'])['file'].map(lambda x: int(x.split('/')[-1].split('_')[0])).max() + 1
        else:
            start_idx = 0
        train.insert(0, 'file', [f'{(start_idx + i):04d}_{re.sub(r' +', '_', re.sub(r' - .*', '', voice)).strip('_')}.wav' for i, voice in enumerate(train['voice'].to_list())])
        train.insert(1, 'label', int(sample_type == 'positive'))
        train.apply(lambda row: move_from_zip(zip_objects[row['zip_path']], row['src_dir_path'], os.path.join(path, row['file'])), axis=1)
        train.drop(columns=['zip_path', 'src_dir_path'], inplace=True)
        manifests[phrase]['train'][f'{sample_type}_samples'].extend(train.to_dict(orient='records'))
        logger.info(phrase, sample_type, 'train', len(train))
        
        test = group.drop(index=train.index).get(['text', 'voice', 'zip_path', 'src_dir_path']).copy()
        path = os.path.join(data_dir, phrase, 'test', sample_type)
        os.makedirs(path, exist_ok=True)
        if len(manifests[phrase]['test'][f'{sample_type}_samples']) > 0:
            start_idx = pd.DataFrame(manifests[phrase]['test'][f'{sample_type}_samples'])['file'].map(lambda x: int(x.split('/')[-1].split('_')[0])).max() + 1
        else:
            start_idx = 0
        test.insert(0, 'file', [f'{(start_idx + i):04d}_{re.sub(r' +', '_', re.sub(r' - .*', '', voice)).strip('_')}.wav' for i, voice in enumerate(test['voice'].to_list())])
        test.insert(1, 'label', int(sample_type == 'positive'))
        test.apply(lambda row: move_from_zip(zip_objects[row['zip_path']], row['src_dir_path'], os.path.join(path, row['file'])), axis=1)
        test.drop(columns=['zip_path', 'src_dir_path'], inplace=True)
        manifests[phrase]['test'][f'{sample_type}_samples'].extend(test.to_dict(orient='records'))
        logger.info(phrase, sample_type, 'test', len(test))

    for k, v in manifests.items():
        with open(os.path.join(data_dir, k, 'training_manifest.json'), 'w') as f:
            json.dump(v, f, indent=4, ignore_nan=True)
    
    if args.delete_after_extract:
        shutil.rmtree(temp_dir)
    
if __name__ == '__main__':
    main()
   
