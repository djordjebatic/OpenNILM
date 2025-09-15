import os

import tensorflow as tf
from omegaconf import DictConfig
import numpy as np

from src.models.gru import mae_loss, mse_loss

from keras.callbacks import Callback


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.model_name = cfg.model.name
        self.cfg = cfg

    def train(self, model, data=None, optimizer_params=None):
        train_model(self.cfg, model, data, optimizer_params)


def train_model(cfg, model, data, optimizer_params):

    if optimizer_params is None:
        optimizer = tf.keras.optimizers.Adam(**cfg.training.optimizer)
    else:
        optimizer = tf.keras.optimizers.Adam(**optimizer_params)

    # initialize early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(**cfg.callbacks.early_stopping)
    # initialize model checkpoint callback
    filepath = os.path.join(os.getcwd(), 'checkpoint', 'model.h5')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                          **cfg.callbacks.model_checkpoint)
    
    # initialize tensorboard callback
    log_dir = os.path.join(os.getcwd(), 'tensorboard')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # compile model
    model.compile(optimizer=optimizer, **cfg.training.compile)

    data_train = data.train
    data_val = data.val

    history = model.fit(data_train,
                        epochs=cfg.training.epochs,
                        verbose=1,
                        callbacks=[early_stopping, model_checkpoint, tensorboard],
                        validation_data=data_val,
                        shuffle=False)


    losses = min(history.history['val_loss'])

    best_loss = round(np.min(losses), 3)
    best_epoch = np.argmin(losses)

    print(f'Best epoch: {best_epoch} | Best loss: {best_loss}')