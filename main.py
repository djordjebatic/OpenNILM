import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.data_loader.data_loader import DatasetNILM
from src.trainer import Trainer
from src.tester import Tester
from src.utils import set_seeds
from src.utils import create_experiment_directories, run_tensorboard
import tensorflow as tf
from hydra.utils import get_original_cwd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix


log = logging.getLogger(__name__)


@hydra.main(config_path="cfg", config_name="config")
def run(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))
    # asserts
    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

    assert tf.test.is_gpu_available()
    assert tf.test.is_built_with_cuda()

    print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

    # set random seeds
    # set matplotlib styles
    # plt.style.use(cfg.matplotlib_styles)

    # create directories used for training
    create_experiment_directories(os.getcwd(), cfg.experiment_directories)

    # automatically run tensorboard
    # run_tensorboard()

    # create data
    data = DatasetNILM(cfg, train=True)
    # create model
    model = instantiate(cfg.model.init)
    #optimizer = tf.keras.optimizers.Adam(**cfg.training.optimizer)
    #model.compile(optimizer='adam', **cfg.training.compile)

    #lr_finder = LargeScaleLearningRateFinder(model, data)
    #lrs, losses = lr_finder.find()
    #lr_finder.plot_loss(lrs, losses)

    
    # create trainer
    trainer = Trainer(cfg)
    # train model
    trainer.train(model, data)
    # log results
    # evaluate model
    loss = model.evaluate(data.test)
    log.info(f'test results -> mse:{round(loss[0], 4)}')

    # create tester
    tester = Tester(cfg)

    # saved model path [{base_dir}/outputs/{date}/{time}/checkpoint/model.h5
    base_dir = os.getcwd()
    model_path = os.path.join(base_dir, 'checkpoint', 'model.h5')
    print(model_path)

    # load weights
    model.load_weights(model_path)

    # test
    tester.test(model, data)
   


def load_model(cfg):
    # get saved model info
    info = cfg.dataset.trained_models[cfg.model.name][cfg.appliance.name]
    # get base dir location
    base_dir = get_original_cwd()
    # create saved model path [{base_dir}/outputs/{date}/{time}/checkpoint/model.h5
    model_path = os.path.join(base_dir, 'outputs', info.date, info.time, 'checkpoint', 'model.h5')

    model = tf.keras.models.load_model(model_path)

    print(model.summary())

    return model


def compute_step_function(data, threshold):
    step_data = (data >= threshold) * 1
    return step_data


def acc_precision_recall_f1_score(status, status_pred):
    assert status.shape == status_pred.shape

    tn, fp, fn, tp = confusion_matrix(status, status_pred, labels=[0, 1]).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / np.max((tp + fp, 1e-9))
    recall = tp / np.max((tp + fn, 1e-9))
    f1_score = 2 * (precision * recall) / np.max((precision + recall, 1e-9))

    return f1_score, acc, precision, recall


if __name__ == '__main__':
    #tf.config.run_functions_eagerly(True)
    set_seeds(1)
    run()
