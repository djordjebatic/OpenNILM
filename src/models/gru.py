"""https://dl.acm.org/doi/10.1145/3200947.3201011"""

import keras
import numpy as np
import tensorflow as tf
import keras.backend as K


class GRU_NILM(keras.Model):
    def __init__(self, input_window_length, **kwargs):
        super().__init__(**kwargs)

        self.network = get_nework(input_window_length)


    def compile(self, optimizer, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)

    #@tf.function
    def train_step(self, data):

        x, y = data

        with tf.GradientTape() as tape:
            tape.watch(x)
            # forward pass
            y_pred = self(x, training=True)
            # loss function is configured in 'compile()'
            prediction_loss = self.compiled_loss(y, y_pred)
            total_loss = prediction_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred)

        output = {m.name: m.result() for m in self.metrics}

        return output

    @tf.function
    def test_step(self,  data):

        x, y = data

        y_pred = self(x, training=False)

        self.compiled_loss(y, y_pred)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, *args, **kwargs):
        return self.network(inputs)


def mse_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    loss = mse(y_true, y_pred)
    return loss


def mae_loss(y_true, y_pred):
    mae = tf.keras.losses.MeanAbsoluteError()
    loss = mae(y_true, y_pred)
    return loss


def get_nework(input_window_length):

    input_layer = tf.keras.layers.Input(shape=(input_window_length,), dtype='float32')
    reshape_layer  = tf.keras.layers.Reshape((input_window_length, 1))(input_layer)
    conv_layer_1 = tf.keras.layers.Conv1D(8, 4, activation='relu', padding="same", strides=1)(reshape_layer)
    bidi_layer_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, activation='tanh', return_sequences=True), merge_mode='concat')(conv_layer_1)
    drop_layer_1 = tf.keras.layers.Dropout(0.5)(bidi_layer_1)
    bidi_layer_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, activation='tanh', return_sequences=False), merge_mode='concat')(drop_layer_1)
    drop_layer_2 = tf.keras.layers.Dropout(0.5)(bidi_layer_2)
    dense_layer_1 = tf.keras.layers.Dense(64, activation='relu')(drop_layer_2)
    drop_layer_3 = tf.keras.layers.Dropout(0.5)(dense_layer_1)
    output_layer = tf.keras.layers.Dense(1, activation='linear')(drop_layer_3)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model
