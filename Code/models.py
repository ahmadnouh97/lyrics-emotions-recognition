import tensorflow as tf


def build_bi_lstm_model(text2vec_layer,
                        vocab_size: int,
                        embedding_dim: int,
                        bi_lstm_units: list,
                        dense_units: list,
                        hidden_activation: str,
                        regularization_factor: float,
                        dropout_factor: float
                        ):
    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))

    model.add(text2vec_layer)

    model.add(
        tf.keras.layers.Embedding(
            input_dim=int(vocab_size) + 1,
            output_dim=int(embedding_dim),
            # Use masking to handle the variable sequence lengths
            mask_zero=True
            # input_length=int(params['input_length'])
        )
    )

    for i, units in enumerate(bi_lstm_units):
        if i == len(bi_lstm_units) - 1:
            model.add(
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(units),
                                                                   activation=hidden_activation,
                                                                   recurrent_regularizer=tf.keras.regularizers.L2(
                                                                       float(regularization_factor)),
                                                                   kernel_regularizer=tf.keras.regularizers.L2(
                                                                       float(regularization_factor))
                                                                   )
                                              )
            )
        else:
            model.add(
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(int(units),
                                                                   return_sequences=True,
                                                                   activation=hidden_activation,
                                                                   recurrent_regularizer=tf.keras.regularizers.L2(
                                                                       float(regularization_factor)),
                                                                   kernel_regularizer=tf.keras.regularizers.L2(
                                                                       float(regularization_factor))
                                                                   )
                                              )
            )

    if dropout_factor > 0:
        model.add(
            tf.keras.layers.Dropout(float(dropout_factor))
        )
    for units in dense_units:
        model.add(
            tf.keras.layers.Dense(int(units),
                                  activation=hidden_activation,
                                  kernel_regularizer=tf.keras.regularizers.L2(float(regularization_factor))
                                  )
        )
        if dropout_factor > 0:
            model.add(
                tf.keras.layers.Dropout(float(dropout_factor))
            )
    model.add(
        tf.keras.layers.Dense(1, activation='sigmoid')
    )

    return model


def build_cnn_model(text2vec_layer,
                    vocab_size: int,
                    embedding_dim: int,
                    cnn_units: list,
                    cnn_kernels: list,
                    dense_units: list,
                    hidden_activation: str,
                    regularization_factor: float,
                    dropout_factor: float
                    ):
    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))

    model.add(text2vec_layer)

    model.add(
        tf.keras.layers.Embedding(
            input_dim=int(vocab_size) + 1,
            output_dim=int(embedding_dim),
            # Use masking to handle the variable sequence lengths
            mask_zero=True
            # input_length=int(params['input_length'])
        )
    )

    for i, (units, kernels) in enumerate(zip(cnn_units, cnn_kernels)):
        if i == len(cnn_units) - 1:
            model.add(tf.keras.layers.Conv1D(units, kernels, activation=hidden_activation))
            model.add(tf.keras.layers.GlobalMaxPooling1D())

    for units in dense_units:
        model.add(
            tf.keras.layers.Dense(int(units),
                                  activation=hidden_activation,
                                  kernel_regularizer=tf.keras.regularizers.L2(float(regularization_factor))
                                  )
        )
        if dropout_factor > 0:
            model.add(
                tf.keras.layers.Dropout(float(dropout_factor))
            )
    model.add(
        tf.keras.layers.Dense(1, activation='sigmoid')
    )

    return model
