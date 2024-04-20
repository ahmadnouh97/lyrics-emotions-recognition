import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.abspath('.'))
from Code.config import Config
from Code.text_vectorization import standardize_ar_text, split_ar_text

data = pd.read_csv(os.path.join(Config.PROCESSED_DATA_DIR, 'habibi_processed.csv'), index_col=False)

model = tf.keras.models.load_model(Config.MODEL_PATH)

processed_lyrics = list(data['ProcessedLyrics'])

predictions = model.predict(processed_lyrics)
predictions = predictions.reshape((-1,))
labels = np.where(predictions >= 0.5, 'happy', 'sad')

data['Label'] = labels
# data = data[['SongDialect', 'Singer', 'Lyrics', 'Label']]

data.to_csv(os.path.join(Config.PROCESSED_DATA_DIR, 'habibi_labeled.csv'), index=False)
