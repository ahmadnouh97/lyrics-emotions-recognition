import sys
import os
import pandas as pd
from tqdm import tqdm
import yaml

sys.path.append(os.path.abspath('.'))
from Code.config import Config
from Code.preprocessing import process_text

with open(Config.PARAMS_PATH) as f:
    params = yaml.load(f.read())['prepare']


def prepare_data():
    raw_data = pd.read_csv(Config.RAW_DATA_PATH, index_col=False)
    # songID,SongTitle,Singer,Lyrics,Label,SongDialect,loudness
    data = raw_data[raw_data['Label'].apply(lambda x: x in {'Happy', 'Sad'})].reset_index(drop=True)
    data = data[data['Lyrics'].apply(lambda x: x and len(str(x)) > 0)].reset_index(drop=True)
    processed_lyrics = []
    with tqdm(total=len(data)) as p_bar:
        for lyrics in list(data['Lyrics']):
            processed_lyrics.append(
                process_text(str(lyrics),
                             remove_stopwords=bool(params['remove_stopwords']),
                             remove_shadda=bool(params['remove_shadda']),
                             remove_tashkeel=bool(params['remove_tashkeel']),
                             remove_tatweel=bool(params['remove_tatweel']),
                             remove_punc=bool(params['remove_punc']),
                             remove_repeats=bool(params['remove_repeats']),
                             normalize=bool(params['normalize']),
                             remove_nonarabic=bool(params['remove_nonarabic']),
                             stemming=bool(params['stemming']))
            )
            p_bar.update(1)

    data['Lyrics'] = processed_lyrics
    data = data[data['Lyrics'].apply(lambda x: len(x.split()) > 1)].reset_index(drop=True)
    data.to_csv(Config.PROCESSED_DATA_PATH, index=False)


prepare_data()
