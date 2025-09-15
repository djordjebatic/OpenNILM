"""https://arxiv.org/pdf/1902.08736"""

import tensorflow as tf
import numpy as np
import keras
from keras.layers import ReLU
from keras.regularizers import l2
import tensorflow_addons as tfa


class TCN_NILM(keras.Model):
    def __init__(self, input_window_length, depth, nb_filters, res_l2, stacks, dropout, **kwargs):
        super().__init__(**kwargs)

        self.n_blocks = 4
        self.kernel_sizes = [3]
        self.dilation_rate = [2, 4, 8, 16]
        self.filters = [32]

        self.network = get_network(input_window_length, depth, nb_filters, res_l2, stacks, dropout)


    def compile(self, optimizer, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)

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


def get_network(input_window_length=1440, depth=9, nb_filters=[512, 256, 256, 128, 128, 256, 256, 256, 512],
                res_l2=0., stacks=1, dropout=0.1):

    if len(nb_filters) == 1:
        nb_filters = np.ones(depth, dtype='int') * nb_filters[0]

    inpt = tf.keras.Input(shape=(input_window_length, 1))

    # Initial Feature mixing layer
    out = tf.keras.layers.Conv1D(nb_filters[0], 1, padding='same', use_bias='True', kernel_regularizer=l2(res_l2))(inpt)

    skip_connections = [out]

    # Create main wavenet structure
    for j in range(stacks):
        for i in range(depth):
            # "Signal" output
            signal_out = tf.keras.layers.Conv1D(nb_filters[i], 2, dilation_rate=2 ** i, padding='causal',
                                                use_bias='True', kernel_regularizer=l2(res_l2))(out)

            signal_out = tf.keras.layers.Activation('relu')(signal_out)

            # "Gate" output
            gate_out = tf.keras.layers.Conv1D(nb_filters[i], 2, dilation_rate=2 ** i, padding='causal',
                                              use_bias='True', kernel_regularizer=l2(res_l2))(out)
            gate_out = tf.keras.layers.Activation('sigmoid')(gate_out)

            # Multiply signal by gate to get gated output
            gated = tf.keras.layers.Multiply()([signal_out, gate_out])

            out = gated

            # Droupout for regularization
            if dropout != 0:
                out = tf.keras.layers.Dropout(dropout)(out)

            skip_connections.append(out)

    out = tf.keras.layers.Concatenate()(skip_connections)

    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='linear'))(out)
    out = tf.keras.layers.LeakyReLU(alpha=0.1)(out)

    model = tf.keras.Model(inpt, out)

    #print(model.summary())

    return model


def TCN_block(filters, kernel_size, dilation_rate, dropout, input):


    residual = tf.keras.layers.Conv1D(1, 1)(input)
    print('Residual shape:')
    print(residual.shape)
    x1 = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate))(input)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Dropout(dropout)(x1)
    x1 = tfa.layers.WeightNormalization(
        tf.keras.layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate))(x1)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Dropout(dropout)(x1)
    x1 = tf.keras.layers.Add()([residual, x1])
    print('Block output shape:')
    return x1





