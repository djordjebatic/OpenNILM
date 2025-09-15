import os

# remove tensorboard logger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorboard import program


def value_checks(cfg) -> None:
    if cfg.model.name == 'seq2subseq' or cfg.model.name == 'seq2seq':
        if cfg.model.init.input_window_length % 2 != 0:
            raise ValueError('Input width must be divisible with 2 for seq2subseq model')

    if cfg.model.name == 'cnn':
        if cfg.model.init.input_window_length % 2 != 1:
            raise ValueError('Input width must be and odd number for seq2point model')

    assert cfg.appliance_name in ['washingmachine', 'dishwasher', 'microwave', 'kettle']
    assert cfg.model.init.input_window_length in [599, 1024]


def create_experiment_directories(root_dir, directories) -> None:
    [os.makedirs(os.path.join(root_dir, directory)) for directory in directories]


def run_tensorboard() -> None:
    # automatically run tensorboard
    tb = program.TensorBoard()
    # path where tensorboard is saved
    log_dir = os.path.join(os.getcwd(), 'tensorboard')
    # configure tensorboard
    tb.configure(argv=[None, '--logdir', log_dir])
    print(f'Tensorboard listening on {tb.launch()}')


def set_seeds(seed: int = 6):
    #os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'

    os.environ['PYTHONHASHSEED'] = str(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    #tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    print("Seeds set!")