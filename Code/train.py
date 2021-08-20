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

params = yaml.safe_load(open(Config.PARAMS_PATH))['train']

# os.makedirs(Config.TENSORBOARD_DIR, exist_ok=True)
os.makedirs(Config.METRICS_PATH, exist_ok=True)


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

print(f'class_count:\n{data["Class"].value_counts()}')

data['Class'] = data['Class'].replace({'happiness': 1.0, 'sadness': 0.0})

X = np.array(list(data['Tweet'])).astype(str)
y = np.array(list(data['Class']))

max_len = max([len(item) for item in X])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

val_len = int(len(X_train) * float(params["validation_split"]))

print(f'X_train_len = {len(X_train) - val_len}')
print(f'X_val_len = {val_len}')
print(f'X_test_len = {len(X_test)}')

text2vec_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=int(params['vocab_size']),
    standardize=standardize_ar_text,
    split=split_ar_text,
    output_sequence_length=int(params['input_length'])
)

text2vec_layer.adapt(X_train)

vocab_size = len(text2vec_layer.get_vocabulary())


def create_model():
    _model = tf.keras.Sequential()

    _model.add(tf.keras.Input(shape=(1,), dtype=tf.string))

    _model.add(text2vec_layer)

    _model.add(
        tf.keras.layers.Embedding(
            input_dim=int(params['vocab_size']) + 1,
            output_dim=int(params['embedding_dim']),
            # Use masking to handle the variable sequence lengths
            mask_zero=True
            # input_length=int(params['input_length'])
        )
    )

    _model.add(
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(params['lstm_01_units']),
                                                           # return_sequences=True,
                                                           # activation=params['hidden_activation'],
                                                           # recurrent_regularizer=tf.keras.regularizers.L2(
                                                           #     float(params['regularization_factor'])),
                                                           # kernel_regularizer=tf.keras.regularizers.L2(
                                                           #     float(params['regularization_factor']))
                                                           )
                                      )
    )

    # _model.add(
    #     tf.keras.layers.Bidirectional(
    #         tf.keras.layers.LSTM(int(params['lstm_02_units']),
    #                              activation=params['hidden_activation'],
    #                              recurrent_regularizer=tf.keras.regularizers.L2(float(params['regularization_factor'])),
    #                              kernel_regularizer=tf.keras.regularizers.L2(float(params['regularization_factor']))
    #                              )
    #     )
    # )
    # _model.add(
    #     tf.keras.layers.Dropout(float(params['dropout_factor']))
    # )

    _model.add(
        tf.keras.layers.Dense(int(params['dense_01_units']),
                              activation=params['hidden_activation'],
                              # kernel_regularizer=tf.keras.regularizers.L2(float(params['regularization_factor']))
                              )
    )
    # _model.add(
    #     tf.keras.layers.Dropout(float(params['dropout_factor']))
    # )
    # _model.add(
    #     tf.keras.layers.Dense(int(params['dense_02_units']),
    #                           activation=params['hidden_activation'],
    #                           kernel_regularizer=tf.keras.regularizers.L2(float(params['regularization_factor'])))
    # )
    # _model.add(
    #     tf.keras.layers.Dropout(float(params['dropout_factor']))
    # )
    _model.add(
        tf.keras.layers.Dense(1, activation='sigmoid')
    )
    return _model


with tf.device('/GPU:0'):
    model = create_model()
    print('model created')
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(params['learning_rate'])),
        metrics=[
            'accuracy'
            # tfa.metrics.F1Score(num_classes=1)
        ]
    )
    early_stopping_cp = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                         patience=5, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        validation_split=float(params['validation_split']),
        epochs=int(params['epochs']),
        batch_size=int(params['batch_size']),
        callbacks=[
            early_stopping_cp
        ]
    )

    model.save(Config.MODEL_PATH)
    plot_training_history(history)

    metrics = model.evaluate(x=X_test, y=y_test, return_dict=True)

    with open(os.path.join(Config.METRICS_PATH, 'metrics.json'), 'w') as file:
        json.dump(metrics, file)
