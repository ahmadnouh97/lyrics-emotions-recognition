import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import json

# setup gpu
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
import os
import sys
import yaml
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('.'))
from Code.config import Config
from Code.text_vectorization import standardize_ar_text, split_ar_text
from Code.models import build_cnn_model
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

x = np.array(list(data['Lyrics'])).astype(str)
y = np.array(list(data['Label']))

max_len = max([len(item) for item in x])

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(f'x = {len(x)}')

text2vec_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=int(params['vocab_size']),
    standardize=standardize_ar_text,
    split=split_ar_text,
    output_sequence_length=int(params['input_length'])
)

text2vec_layer.adapt(x)

with tf.device('/GPU:0'):
    model = build_cnn_model(
        text2vec_layer,
        vocab_size=int(params['vocab_size']),
        embedding_dim=int(params['embedding_dim']),
        cnn_units=list(params['cnn_units']),
        cnn_kernels=list(params['cnn_kernels']),
        dense_units=list(params['dense_units']),
        hidden_activation=str(params['hidden_activation']),
        regularization_factor=float(params['regularization_factor']),
        dropout_factor=float(params['dropout_factor'])
    )
    print('model created')
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(params['learning_rate'])),
        metrics=[
            'accuracy'
        ]
    )
    early_stopping_cp = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                         patience=5, restore_best_weights=True)

    history = model.fit(
        x,
        y,
        validation_split=float(params['validation_split']),
        epochs=int(params['epochs']),
        batch_size=int(params['batch_size']),
        callbacks=[
            early_stopping_cp
        ]
    )

    model.save(Config.MODEL_PATH)
    plot_training_history(history)

