from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


LIBFM_VAL_FILENAME = 'val.libfm'
PREDICTION_FILENAME = 'libfm_pred.txt'


def calculate_metrics(dataset_dir):
    predictions_path = dataset_dir / PREDICTION_FILENAME
    input_path = dataset_dir / LIBFM_VAL_FILENAME
    with input_path.open() as input_file:
        labels = np.array([bool(int(line[0])) for line in input_file])
    with predictions_path.open() as predictions_file:
        probabilities = np.array([float(line) for line in predictions_file])
    roc_auc = roc_auc_score(labels, probabilities)
    predictions = probabilities > .5
    accuracy = accuracy_score(labels, predictions)
    print(f'Dataset: {dataset_dir.name} ROC AUC: {roc_auc:.3f} accuracy {accuracy:.3f}')


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('dataset_dir', type=Path)
    args = arg_parser.parse_args()

    calculate_metrics(args.dataset_dir)
