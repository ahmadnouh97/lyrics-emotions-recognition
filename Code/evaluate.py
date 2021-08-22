import os
import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import json
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

sys.path.append(os.path.abspath('.'))
from Code.config import Config
from Code.text_vectorization import standardize_ar_text, split_ar_text


def prepare_data():
    return pd.read_json(Config.EVAL_DATA_PATH, orient='records')


data = prepare_data()

x_test = np.array(list(data['lyrics']))
y_test = np.array(list(data['actual']))

model = tf.keras.models.load_model(Config.MODEL_PATH)

predictions = model.predict(x_test)

y_pred = [row.argmax() for row in predictions]

class_report = classification_report(y_test, y_pred, output_dict=True)

accuracy = {
    'f1-accuracy': accuracy_score(y_test, y_pred)
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
    'f1-score': precision_score(y_test, y_pred, average='macro')
}

total_recall_metrics = {
    'f1-score': recall_score(y_test, y_pred, average='macro')
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

# ---------------------------------------------------------------------------
