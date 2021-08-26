import numpy as np
import pandas as pd
import pyarabic.araby as ar
# import tensorflow as tf
import json
import pickle as pkl
# setup gpu
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("Num GPUs Available: ", physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import os
import sys
import yaml
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.append(os.path.abspath('.'))
from Code.config import Config
from Code.preprocessing import stop_words, process_text
# from Code.text_vectorization import standardize_ar_text, split_ar_text
# from Code.models import build_cnn_model
params = yaml.safe_load(open(Config.PARAMS_PATH))['train']

# os.makedirs(Config.TENSORBOARD_DIR, exist_ok=True)
os.makedirs(Config.METRICS_DIR, exist_ok=True)


def prepare_data():
    return pd.read_csv(Config.PROCESSED_DATA_PATH, index_col=False)


def plot_training_history(_history):
    # Plot training & validation accuracy values
    plt.plot(_history.history['accuracy'])
    plt.plot(_history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.savefig(str(Config.PLOT_TRAINING_CURVE_ACC_FILE))
    plt.show()
    plt.clf()

    # Plot training & validation loss values
    plt.plot(_history.history['loss'])
    plt.plot(_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(str(Config.PLOT_TRAINING_CURVE_LOSS_FILE))
    plt.show()
    plt.clf()


data = prepare_data()
data = data.sample(frac=1).reset_index(drop=True)

print(f'class_count:\n{data["Label"].value_counts()}')

data['Label'] = data['Label'].replace({'Happy': 1.0, 'Sad': 0.0})

lyrics = np.array(list(data['Lyrics'])).astype(str)
labels = np.array(list(data['Label']))

# max_len = max([len(item) for item in x])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# print(f'x = {len(x)}')

tfidf = TfidfVectorizer(
    sublinear_tf=True,
    min_df=5,
    norm='l2',
    encoding='utf-8',
    ngram_range=(1, 2),
    tokenizer=ar.tokenize,
    stop_words=stop_words)
x = lyrics
# x = tfidf.fit_transform(lyrics).toarray()
y = labels


classifier = Pipeline(steps=[
    ('tfidf', tfidf),
    ('cls', LinearSVC(C=0.1))
])

classifier.fit(x, y)

with open(Config.MODEL_PATH, 'wb') as file:
    pkl.dump(classifier, file)

