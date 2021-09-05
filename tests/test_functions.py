import pytest
from helpers_and_functions import utils, main_functions as mf
import numpy as np


def test_vector_normalization():
    assert max(mf.vec_nor(np.arange(1, 10), a=-1, b=1)) <= 1  # or min(mf.vec_nor(np.arange(1, 10), a=-1, b=1)) <= -1


def test_confusion_matrix():
    score_0, score_1 = 0.5, 0.5
    class_balance = {0: 50, 1: 50}
    cm = utils.cm_from_class_scores(score_0, score_1, class_balance)
    assert cm[0][0] == 25 and cm[1][1] == 25
    assert sum(sum(cm)) == 100

def test_evaluation_metrics():
    true_p, false_p, true_n, false_n = 0, 0, 1, 0
    conf_matrix = np.array([[true_p, false_n], [false_p, true_n]], dtype=int)
    eval_metrics = utils.evaluation_metrics(conf_matrix, out=None)
    assert eval_metrics[1] == 0.0 and eval_metrics[2] == 0.0

    true_p, false_p, true_n, false_n = 1, 0, 0, 0
    conf_matrix = np.array([[true_p, false_n], [false_p, true_n]], dtype=int)
    eval_metrics = utils.evaluation_metrics(conf_matrix, out=None)
    assert eval_metrics[1] != 0.0 and eval_metrics[2] != 0.0
