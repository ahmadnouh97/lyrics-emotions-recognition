import os
import sys
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, \
    roc_auc_score

sys.path.append(os.path.abspath('.'))
from Code.config import Config
from Code.text_vectorization import standardize_ar_text, split_ar_text
from Code.preprocessing import process_text

with open(Config.PARAMS_PATH) as f:
    params = yaml.load(f.read())['prepare']


def prepare_data():
    return pd.read_json(Config.EVAL_DATA_PATH, orient='records')


data = prepare_data()

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
y_test_num = np.array(list(data['Label'].replace({'Happy': 1.0, 'Sad': 0.0})))

x_test = np.array(list(data['Lyrics']))
y_test = np.array(list(data['Label']))

model = tf.keras.models.load_model(Config.MODEL_PATH)

predictions = model.predict(x_test)

y_pred = ['Happy' if val >= 0.5 else 'Sad' for val in predictions.reshape(-1)]

predictions_dict = {
    'actual': y_test,
    'predicted': y_pred
}

class_report = classification_report(y_test, y_pred, output_dict=True)

print(class_report)

accuracy = {
    'accuracy': accuracy_score(y_test, y_pred)
}

happy_f1_metrics = {
    'f1-score': class_report['Happy']['f1-score']
}

happy_precision_metrics = {
    'precision': class_report['Happy']['precision']
}

happy_recall_metrics = {
    'recall': class_report['Happy']['recall']
}

sad_f1_metrics = {
    'f1-score': class_report['Sad']['f1-score']
}

sad_precision_metrics = {
    'precision': class_report['Sad']['precision']
}

sad_recall_metrics = {
    'recall': class_report['Sad']['recall']
}

total_f1_metrics = {
    'f1-score': f1_score(y_test, y_pred, average='macro')
}

total_precision_metrics = {
    'precision': precision_score(y_test, y_pred, average='macro')
}

total_recall_metrics = {
    'recall': recall_score(y_test, y_pred, average='macro')
}

print(y_test_num)

print(predictions.reshape(-1))

roc_auc_metrics = {
    'roc_auc_score': roc_auc_score(y_test_num, predictions.reshape(-1), average='macro')
}

os.makedirs(Config.METRICS_DIR, exist_ok=True)

with open(os.path.join(Config.METRICS_DIR, 'f1_score.json'), 'w') as file:
    json.dump(total_f1_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'precision.json'), 'w') as file:
    json.dump(total_precision_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'recall.json'), 'w') as file:
    json.dump(total_recall_metrics, file)

# ---------------------------------------------------------------------------

with open(os.path.join(Config.METRICS_DIR, 'happy_f1_score.json'), 'w') as file:
    json.dump(happy_f1_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'happy_precision.json'), 'w') as file:
    json.dump(happy_precision_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'happy_recall.json'), 'w') as file:
    json.dump(happy_recall_metrics, file)

# ---------------------------------------------------------------------------
with open(os.path.join(Config.METRICS_DIR, 'sad_f1_score.json'), 'w') as file:
    json.dump(sad_f1_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'sad_precision.json'), 'w') as file:
    json.dump(sad_precision_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'sad_recall.json'), 'w') as file:
    json.dump(sad_recall_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'roc_auc.json'), 'w') as file:
    json.dump(roc_auc_metrics, file)

with open(os.path.join(Config.METRICS_DIR, 'accuracy.json'), 'w') as file:
    json.dump(accuracy, file)


# ---------------------------------------------------------------------------
def save_to_csv(obj, file_path):
    key = list(obj.keys())[0]
    df = pd.DataFrame(obj, index=list(range(len(obj.get(key)))))
    df.to_csv(file_path, index=False)


save_to_csv(predictions_dict, os.path.join(Config.PLOT_DIR, 'classes.csv'))
