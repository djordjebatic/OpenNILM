from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import pandas as pd
import tensorflow as tf
from omegaconf import DictConfig
from hydra.utils import to_absolute_path
import os


class DatasetNILM:
    """
    A TensorFlow data loader for NILM experiments.

    This class reads processed training, validation, and test data from CSV files,
    and prepares them as `tf.data.Dataset` objects suitable for training and
    evaluating various NILM models (CNN, GRU, TCN, etc.). It handles the specific
    input/output windowing required by each model architecture.
    """
    def __init__(self, cfg: DictConfig, train: bool = True, model_name: str = None,
                 batch_size: int = None, input_window_length: int = None,
                 **kwargs) -> None:
        """
        Initializes the DatasetNILM loader.

        Args:
            cfg (DictConfig): The Hydra configuration object.
            train (bool, optional): If True, loads train, validation, and test sets.
                                    If False, loads only the test set. Defaults to True.
            model_name (str, optional): Overrides the model name from the config.
                                        Defaults to None.
            batch_size (int, optional): Overrides the batch size from the config.
                                        Defaults to None.
            input_window_length (int, optional): Overrides the input window length
                                                 from the config. Defaults to None.
        """
        self.cfg = cfg
        # Allow overriding config values for flexibility
        self.model_name = model_name or cfg.model.name
        self.batch_size = batch_size or cfg.model.batch_size
        self.input_window_length = input_window_length or cfg.model.init.input_window_length

        # Construct the path to the processed data directory
        data_dir = to_absolute_path(f'{cfg.data.processed_path}/{cfg.dataset.name}/{cfg.appliance.name}')

        if train:
            # Load all data splits for a training run
            train_data = np.array(pd.read_csv(os.path.join(data_dir, 'training_.csv')))
            val_data = np.array(pd.read_csv(os.path.join(data_dir, 'validation_.csv')))
            test_data = np.array(pd.read_csv(os.path.join(data_dir, 'test_.csv')))

            self.train_data = train_data[:, 0]    # Aggregate power
            self.train_labels = train_data[:, 1]  # Appliance power

            self.val_data = val_data[:, 0]
            self.val_labels = val_data[:, 1]

            self.test_data = test_data[:, 0]
            self.test_labels = test_data[:, 1]
        else:
            # Load only the test data for evaluation/inference
            test_data_path = os.path.join(data_dir, self.cfg.data.test_file)
            test_data = np.array(pd.read_csv(test_data_path))

            self.test_data = test_data[:, 0]
            self.test_labels = test_data[:, 1]
            self.test_window_size = len(test_data)

    def create_dataset(self, data: np.ndarray, labels: np.ndarray,
                       input_window_length: int, batch_size: int,
                       model_name: str, shuffle: bool = False) -> tf.data.Dataset:
        """
        Creates a `tf.data.Dataset` from numpy arrays based on the model type.

        This method handles the specific slicing and windowing required by different
        NILM architectures (seq2point, seq2seq, etc.).

        Args:
            data (np.ndarray): The input aggregate power data.
            labels (np.ndarray): The target appliance power data.
            input_window_length (int): The length of the input sequence window.
            batch_size (int): The batch size for the dataset.
            model_name (str): The name of the model ('cnn', 'gru', 'tcn'), which
                              determines the windowing strategy.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.

        Returns:
            tf.data.Dataset: The fully prepared and batched TensorFlow dataset.
        """
        if model_name in ['cnn', 'gru']:  # seq2point models
            if model_name == 'cnn':
                # Target is the center point of the window
                targets = labels[input_window_length // 2 :]
            else: # gru
                # Target is the last point of the window
                targets = labels[input_window_length - 1:]

            dataset = tf.keras.utils.timeseries_dataset_from_array(
                data=data,
                targets=targets,
                sequence_length=input_window_length,
                sequence_stride=1,
                shuffle=shuffle,
                seed=self.cfg.random_seed,
                batch_size=batch_size,
            )

        elif model_name == 'tcn':  # seq2seq models
            # Use non-overlapping sliding windows for inputs and targets
            # This creates sequences of `input_window_length`
            inputs = sliding_window_view(data, (input_window_length,))[::input_window_length, :]
            targets = sliding_window_view(labels, (input_window_length,))[::input_window_length, :]

            # Ensure inputs and targets have the same number of complete windows
            min_len = min(len(inputs), len(targets))
            inputs, targets = inputs[:min_len], targets[:min_len]

            # Add a channel dimension for Keras Conv1D layers
            inputs = np.expand_dims(inputs, axis=2).astype(np.float32)
            targets = np.expand_dims(targets, axis=2).astype(np.float32)

            dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))

            if shuffle:
                dataset = dataset.shuffle(buffer_size=len(data), seed=self.cfg.random_seed)

            dataset = dataset.batch(batch_size, drop_remainder=True)

        else:
            raise ValueError(f"Unsupported model_name for dataset creation: {model_name}")

        return dataset

    @property
    def train(self) -> tf.data.Dataset:
        """Returns the training dataset, shuffled and batched."""
        return self.create_dataset(self.train_data, self.train_labels,
                                   self.input_window_length, self.batch_size,
                                   self.model_name, shuffle=True)

    @property
    def val(self) -> tf.data.Dataset:
        """Returns the validation dataset, batched."""
        return self.create_dataset(self.val_data, self.val_labels,
                                   self.input_window_length, self.batch_size,
                                   self.model_name)

    @property
    def test(self) -> tf.data.Dataset:
        """Returns the test dataset, batched."""
        return self.create_dataset(self.test_data, self.test_labels,
                                   self.input_window_length, self.batch_size,
                                   self.model_name)