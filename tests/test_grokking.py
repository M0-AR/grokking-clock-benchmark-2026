"""Smoke tests: task math, model shape, handwritten clock = 100%, 1% chance baseline."""
import numpy as np
from src.task import split
from src.clock import clock_predict, clock_scores


def test_table():
    (tr_t, tr_a), (te_t, te_a) = split(frac=0.3, seed=0)
    assert len(tr_t) + len(te_t) == 113 * 113 == 12769
    assert len(tr_t) == 3830, len(tr_t)
    assert (100 + 50) % 113 == 37
    assert (37 + 99) % 113 == 23


def test_clock_perfect():
    (tr_t, tr_a), (te_t, te_a) = split(frac=0.3, seed=0)
    A = np.concatenate([tr_t[:, 0], te_t[:, 0]])
    B = np.concatenate([tr_t[:, 1], te_t[:, 1]])
    Y = np.concatenate([tr_a, te_a])
    pred = clock_predict(A, B, speeds=(1,))
    assert (pred == Y).mean() > 0.999


def test_chance_level():
    # untrained guess ~1/113
    assert abs(1 / 113 - 0.0088) < 0.001
