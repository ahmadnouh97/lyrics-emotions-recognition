import sys
import os
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath('.'))
from Code.config import Config
from Code.preprocessing import process_text

data = pd.read_csv(os.path.join(Config.RAW_DATA_DIR, 'habibi.csv'), index_col=False)
data = data.sort_values(['songID', 'LyricsOrder'], ascending=True).groupby(['songID']).agg({
    'Singer': 'first',
    'SongTitle': 'first',
    'SongWriter': 'first',
    'Composer': 'first',
    # 'LyricsOrder': 'first',
    'Lyrics': '\n'.join,
    'SingerNationality': 'first',
    'SongDialect': 'first',
}).reset_index()

# print(data.head())
print(f'number of songs = {len(data)}')

raw_lyrics = list(data['Lyrics'])
processed_lyrics = []

print('processing lyrics..')
with tqdm(total=len(raw_lyrics)) as p_bar:
    for lyrics in raw_lyrics:
        processed_lyrics.append(
            process_text(lyrics)
        )
        p_bar.update(1)
data['ProcessedLyrics'] = processed_lyrics
data.to_csv(os.path.join(Config.PROCESSED_DATA_DIR, 'habibi_processed.csv'), index=False)
