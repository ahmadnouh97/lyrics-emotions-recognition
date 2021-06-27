import sys
import os
import numpy as np
import pandas as pd
import codecs
import json

sys.path.append(os.path.abspath('.'))
from Code.config import Config

data = pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, 'habibi_labeled.csv'), index_col=False, encoding="utf-8")
data = data[['songID', 'Singer', 'Lyrics', 'Label', 'SongDialect']]

parts = np.array_split(data, 5)

for i, part in enumerate(parts):
    part.to_json(os.path.join(Config.PROCESSED_DATA_DIR, f'habibi_labeled_part_0{i+1}.json'),
                 orient='records')

    with open(os.path.join(Config.PROCESSED_DATA_DIR, f'habibi_labeled_part_0{i+1}.json'), 'r', encoding='utf-8') as file:
        part_data = json.load(file)

    with codecs.open(os.path.join(Config.PROCESSED_DATA_DIR, f'habibi_labeled_part_0{i+1}.json'), 'w', encoding='utf-8') as file:
        json.dump(part_data, file, ensure_ascii=False)
