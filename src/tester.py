import os
import tensorflow as tf
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from src.data_loader.data_loader import DatasetNILM
import numpy as np
from sklearn.metrics import mean_absolute_error, confusion_matrix
import csv
import json


class Tester:
    def __init__(self, cfg: DictConfig, transfer_test=False) -> None:
        self.model_name = cfg.model.name
        self.cfg = cfg
        self.transfer_test = transfer_test

    def test(self, model, data, reduction_test_data=None, save_results=True):

        if reduction_test_data:
            predictions = np.squeeze(model.predict(reduction_test_data)).flatten()
            self.input_window_length = data.input_window_length
            self.model_name = data.model_name
        else:
            predictions = np.squeeze(model.predict(data.test)).flatten()
            self.input_window_length = self.cfg.model.init.input_window_length
        ground_truth = data.test_labels
        agg = data.test_data

        mae, f1, precision, recall, acc = self.model_test(self.cfg, predictions, ground_truth, agg)

        if save_results:
            #self.write_results_to_json(mae, f1, precision, recall, acc)
            self.write_results_to_csv(mae, f1, precision, recall, acc, self.cfg.appliance.name)

        return mae, f1

    def write_results_to_json(self, mae, f1, precision, recall, acc):

        working_dir = os.getcwd()
        original_dir = get_original_cwd()

        dataset = self.cfg.dataset.name
        appliance = self.cfg.appliance.name
        model = self.cfg.model.name
        
        json_dir = os.path.join(original_dir, 'results', dataset, 'results.json')

        with open(json_dir, 'r') as results_file:
            results = json.load(results_file)
            data = results[model][appliance]

        if f1 >= data['F1']:
            results[model][appliance]['Directory'] = working_dir
            results[model][appliance]['MAE'] = mae
            results[model][appliance]['F1'] = f1
            results[model][appliance]['Precision'] = precision
            results[model][appliance]['Recall'] = recall
            results[model][appliance]['Acc'] = acc

        with open(json_dir, 'w') as results_file:
            json.dump(results, results_file, indent=2)

    def write_results_to_csv(self, mae, f1, precision, recall, acc, app_name):

        save_path = os.path.join(os.getcwd(), 'metrics', f'{app_name}_results.csv')
        with open(save_path, 'w', newline='') as csv_file:
            header_key = ['Model', 'MAE', 'F1', 'Precision', 'Recall', 'Acc']
            writer = csv.DictWriter(csv_file, fieldnames=header_key)

            writer.writeheader()

            writer.writerow({'Model': self.model_name,
                             'MAE': round(mae, 3),
                             'F1': round(f1, 3),
                             'Precision': round(precision, 3),
                             'Recall': round(recall, 3),
                             'Acc': round(acc, 3)})

    def model_test(self, cfg, predictions, ground_truth, agg):
        # loss = model.evaluate(data.test)

        if self.model_name == 'cnn':
            ground_truth = ground_truth[int(self.input_window_length / 2) - 1:]
            ground_truth = ground_truth[:len(predictions)]

            agg = agg[int(self.input_window_length / 2) - 1:]
            agg = agg[:len(predictions)]

        if self.model_name == 'gru':
            agg = agg[(self.input_window_length - 1):]
            ground_truth = ground_truth[(self.input_window_length - 1):]

        if self.model_name == 'tcn':
            ground_truth = ground_truth[:len(predictions)]
            agg = agg[:len(predictions)]

        assert len(ground_truth) == len(predictions), print(f'GT: {len(ground_truth)} | PRED: {len(predictions)}')

        agg = agg * cfg.dataset.aggregate[cfg.appliance.name].std + cfg.dataset.aggregate[cfg.appliance.name].mean
        ground_truth = ground_truth * cfg.dataset.cutoff[cfg.appliance.name]
        predictions = predictions * cfg.dataset.cutoff[cfg.appliance.name]

        predictions[predictions < cfg.dataset.threshold[cfg.appliance.name]] = 0
        predictions[predictions > cfg.dataset.cutoff[cfg.appliance.name]] = cfg.dataset.cutoff[cfg.appliance.name]

        mae = mean_absolute_error(ground_truth, predictions)
        print(f'MAE: {round(mae, 4)}')

        gt_step = compute_step_function(ground_truth, cfg.dataset.threshold[cfg.appliance.name])
        pred_step = compute_step_function(predictions, cfg.dataset.threshold[cfg.appliance.name])

        f1, acc, precision, recall = acc_precision_recall_f1_score(gt_step, pred_step)

        print(f'F1: {f1} | ACC: {acc}')

        return mae, f1, precision, recall, acc

    def tcn_test(self, model, data, cfg):
        # loss = model.evaluate(data.test)

        predictions = np.squeeze(model.predict(data.test)).flatten()

        agg = data.test_data
        ground_truth = data.test_labels

        agg = agg[:int(len(agg) / self.input_window_length) * self.input_window_length]
        ground_truth = ground_truth[:int(len(ground_truth) / self.input_window_length) * self.input_window_length]

        assert len(ground_truth) == len(predictions), print(f'GT: {len(ground_truth)} | PRED: {len(predictions)}')

        agg = agg * cfg.dataset.aggregate[cfg.appliance.name].std + cfg.dataset.aggregate[cfg.appliance.name].mean
        ground_truth = ground_truth * cfg.dataset.cutoff[cfg.appliance.name]
        predictions = predictions * cfg.dataset.cutoff[cfg.appliance.name]

        predictions[predictions < cfg.dataset.threshold[cfg.appliance.name]] = 0
        predictions[predictions > cfg.dataset.cutoff[cfg.appliance.name]] = cfg.dataset.cutoff[cfg.appliance.name]

        mae = mean_absolute_error(ground_truth, predictions)
        print(f'MAE: {round(mae, 4)}')

        gt_step = compute_step_function(ground_truth, cfg.dataset.threshold[cfg.appliance.name])
        pred_step = compute_step_function(predictions, cfg.dataset.threshold[cfg.appliance.name])

        f1, acc, precision, recall = acc_precision_recall_f1_score(gt_step, pred_step)

        print(f'F1: {f1} | ACC: {acc}')

        f1, acc, precision, recall = acc_precision_recall_f1_score(gt_step, pred_step)

        print(f'F1: {f1} | ACC: {acc}')

        return mae, f1, precision, recall, acc


def compute_step_function(data, threshold):
    step_data = (data >= threshold) * 1
    # plt.plot(step_data)
    return step_data


def acc_precision_recall_f1_score(status, status_pred):
    assert status.shape == status_pred.shape

    tn, fp, fn, tp = confusion_matrix(status, status_pred, labels=[0, 1]).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / np.max((tp + fp, 1e-9))
    recall = tp / np.max((tp + fn, 1e-9))
    f1_score = 2 * (precision * recall) / np.max((precision + recall, 1e-9))

    return f1_score, acc, precision, recall


def seq2point_test(model, data, cfg):
    pass


def load_model(cfg):
    # get saved model info
    info = cfg.dataset.trained_models[cfg.appliance.name]
    # get base dir location
    base_dir = get_original_cwd()
    # create saved model path [{base_dir}/outputs/{date}/{time}/checkpoint/model.h5
    model_path = os.path.join(base_dir, 'outputs', info.date, info.time, 'checkpoint', 'model.h5')

    model = tf.keras.models.load_model(model_path)

    return model


def load_model_test(cfg):

    # get saved model info
    info = cfg.dataset.trained_models[cfg.appliance.name]
    # get base dir location
    base_dir = get_original_cwd()
    # create saved model path [{base_dir}/results/models/{dataset}/{appliance}/model.h5
    print(f'Date: {info.date} | Time: {info.time}')
    model_path = os.path.join(base_dir, 'results', 'models', cfg.dataset.name, cfg.appliance.name, 'model.h5')

    model = tf.keras.models.load_model(model_path)

    return model

def seq2subseq_test():
    pass


def accuracy(results):
    tn, fp, fn, tp = results
    return (tp + tn) / (tp + fn + fp + tn)


def sensitivity(results):
    tn, fp, fn, tp = results
    return tp / (tp + fn + 1e-7)


def specificity(results):
    tn, fp, fn, tp = results
    return tn / (tn + fp + 1e-7)


def precision(results):
    tn, fp, fn, tp = results
    return tp / (tp + fp + 1e-7)


def recall(results):
    tn, fp, fn, tp = results
    return tp / (tp + fn + 1e-7)


def f1_score(results):
    tn, fp, fn, tp = results
    return 2 * tp / (2 * tp + fp + fn + 1e-7)

def seq2seq_test():
    pass
