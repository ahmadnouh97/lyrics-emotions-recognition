import sys
import os
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath('.'))
from Code.config import Config
from Code.preprocessing import process_text


def prepare_data():
    raw_data = pd.read_csv(Config.RAW_DATA_PATH, index_col=False)
    raw_data.columns = ['ID', 'Tweet', 'Class']
    data = raw_data[raw_data['Class'].apply(lambda x: x in {'happiness', 'sadness'})].reset_index(drop=True)
    data = data[data['Tweet'].apply(lambda x: x and len(str(x)) > 0)].reset_index(drop=True)
    segmented_tweets = []
    with tqdm(total=len(data)) as p_bar:
        for tweet in list(data['Tweet']):
            segmented_tweets.append(
                process_text(str(tweet))
            )
            p_bar.update(1)

    data['Tweet'] = segmented_tweets
    data = data[data['Tweet'].apply(lambda x: len(x.split()) > 1)].reset_index(drop=True)
    data.to_csv(Config.PROCESSED_DATA_PATH, index=False)


prepare_data()