"""https://arxiv.org/pdf/1612.09106"""

import keras
import numpy as np
import tensorflow as tf


class CNN_NILM(keras.Model):
    def __init__(self, input_window_length, **kwargs):
        super().__init__(**kwargs)

        self.network = get_network(input_window_length)

        self.freq_limit = 150
        self.limit_softness = 0.2

    def compile(self, optimizer, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)

    @tf.function
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


def get_network(input_window_length):

    input_layer = tf.keras.layers.Input(shape=(input_window_length,), dtype='float32')
    reshape_layer = tf.keras.layers.Reshape((input_window_length, 1))(input_layer)
    conv_layer_1 = tf.keras.layers.Conv1D(30, 10, activation="relu", strides=1)(reshape_layer)
    conv_layer_2 = tf.keras.layers.Conv1D(30, 8, activation="relu", strides=1)(conv_layer_1)
    conv_layer_3 = tf.keras.layers.Conv1D(40, 6, activation="relu", strides=1)(conv_layer_2)
    conv_layer_4 = tf.keras.layers.Conv1D(50, 5, activation="relu", strides=1)(conv_layer_3)
    drop_layer_1 = tf.keras.layers.Dropout(0.2)(conv_layer_4)
    conv_layer_5 = tf.keras.layers.Conv1D(50, 5, activation="relu", strides=1)(drop_layer_1)
    drop_layer_2 = tf.keras.layers.Dropout(0.2)(conv_layer_5)
    flatten_layer = tf.keras.layers.Flatten()(drop_layer_2)
    dense_layer_1 = tf.keras.layers.Dense(1024, activation='relu')(flatten_layer)
    drop_layer_3 = tf.keras.layers.Dropout(0.2)(dense_layer_1)
    dense_layer_2 = tf.keras.layers.Dense(1, activation='linear')(drop_layer_3)

    model = tf.keras.Model(inputs=input_layer, outputs=dense_layer_2)

    model.summary()

    return model