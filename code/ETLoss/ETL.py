import numpy as np
import math


def my_closure_xgb1(vector):
    def T_Loss(y, data):
        t = data.get_label()
        coff_T = vector['airTemperature'].apply(
            lambda x: 1 * 2 / (1 + 1 * math.exp(-x + 30 + 273.15)) if x > 30 + 273.15 else 1)
        grad = [i * j for i, j in zip((y - t), coff_T)]
        hess = coff_T
        return grad, hess

    return T_Loss


def my_closure_xgb2(vector):
    def Terror(y, data):
        t = data.get_label()
        coff_T = vector['airTemperature'].apply(
            lambda x: 1 * 2 / (1 + 1 * math.exp(-x + 30 + 273.15)) if x > 30 + 273.15 else 1)
        loss = [1 / 2 * i * j for i, j in zip((y - t) ** 2, coff_T)]
        return "T_loss", np.mean(loss)

    return Terror


def my_closure_lgb1(y, data):
    d = data.get_data()
    t = data.get_label()
    coff_T = d['airTemperature'].apply(
        lambda x: 1 * 2 / (1 + 1 * math.exp(-x + 30 + 273.15)) if x > 30 + 273.15 else 1)
    grad = [i * j for i, j in zip(2 * (y - t), coff_T)]
    hess = 2 * coff_T
    return grad, hess


def my_closure_lgb2(y, data):
    d = data.get_data()
    t = data.get_label()
    w = data.get_weight()
    coff_T = d['airTemperature'].apply(
        lambda x: 1 * 2 / (1 + 1 * math.exp(-x + 30 + 273.15)) if x > 30 + 273.15 else 1)
    loss = [i * j for i, j in zip((y - t) ** 2, coff_T)]
    return "T_loss", np.mean(loss), False
