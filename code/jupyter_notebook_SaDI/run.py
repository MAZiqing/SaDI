# coding=utf-8

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import sys

path = os.getcwd()
# print(path)
f_path, _ = os.path.split(path)
sys.path.insert(0, f_path)
# print(f_path)
import lightgbm as lgb
from datetime import datetime
from sklearn.linear_model import LinearRegression
# from dmatrix2np import dmatrix_to_numpy
# from sklearn.datasets import load_svmlight_file
# import math
# from dmatrix2np import dmatrix_to_numpy
# from sklearn.datasets import load_svmlight_file
# import xgboost as xgb
# import pdb
from src.featurizers import *
from src.feature_ensemblers import *
from src.evaluation import *
from ETLoss.ETL import *
from adtk.detector import PersistAD

## feature engineering
datefea = DateTimeFeaturizer(feature_col=['index_15min', 'year', 'day', 'day_of_year',
                                          'day_of_year_sin', 'day_of_year_cos',
                                          'month', 'day_of_month_sin',
                                          'day_of_month_cos',
                                          'day_of_week', 'is_weekend',
                                          'day_of_week_sin',
                                          'day_of_week_cos', 'hour',
                                          'is_special_workday',
                                          'is_special_holiday'])
difffea = DifferenceFeaturizer(offsets=192,
                               feature_col=['airTemperature', 'cloudCoverage',
                                            'dewTemperature', 'precipDepth1HR',
                                            'precipDepth6HR', 'seaLvlPressure', 'windDirection', 'windSpeed'])

difffea2 = DifferenceFeaturizer(offsets=192,
                                feature_col=['airTemperature'])

rfea1 = RollingStatsFeaturizer(offsets=192,
                               feature_col=['load'], wins=[1, 7],
                               quantiles=[0.25, 0.75])
rfea2 = RollingStatsFeaturizer(offsets=192,
                               wins=[7, 14],
                               quantiles=[0.25, 0.75],
                               feature_col=['load'],
                               is_interval=True)
feature_ensembler = FeatureEnsembler(featurizers=[datefea, difffea, rfea1, rfea2])
feature_ensembler2 = FeatureEnsembler(featurizers=[datefea, difffea2, rfea1])


def pred_one_component(feature_en, df, feature_engi=True, diff=False, model='lgb', xslice='all', max_depth=10):
    # choose to do feautre engineering or not
    if feature_engi:
        dffea = feature_en.transform(df)
    else:
        dffea = df
    dffea2 = dffea.copy()

    # choose to predict differencing or not
    if diff:
        Y_ori = dffea['load'].copy()
        dffea2 = dffea.copy()
        dffea2['load'] = (dffea2['load'] - dffea2['load'].shift(192))
    Y = dffea2.pop('load')
    X = dffea2

    # choose split mode: 'summer' or 'all'
    split1 = datetime(2017, 6, 1, 0, 15)
    split2 = datetime(2017, 7, 1, 0, 15)

    if xslice == 'summer':
        slice0 = slice(datetime(2016, 6, 1), datetime(2016, 9, 1))
        slice1 = slice(datetime(2017, 6, 1), datetime(2017, 7, 1))
        X_train = pd.concat([X[slice0], X[slice1]])
        Y_train = pd.concat([Y[slice0], Y[slice1]])

    elif xslice == 'all':
        split0 = datetime(2016, 2, 10, 0, 15)
        X_train = X[split0:split1]
        Y_train = Y[split0:split1]

    X_valid = X[split1: split2]
    X_test = X[split2:]

    Y_valid = Y[split1: split2]
    Y_test = Y[split2:]

    X_train = X_train.interpolate(limit_direction='both').fillna(0)
    Y_train = Y_train.interpolate(limit_direction='both').fillna(0)
    X_valid = X_valid.interpolate(limit_direction='both').fillna(0)
    X_test = X_test.interpolate(limit_direction='both').fillna(0)

    train_params = {
        'params': {
            'num_threads': 12,
            'learning_rate': 0.02,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mae',
            'num_leaves': 45,
            'min_data_in_leaf': 60,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,
            'seed': 16,
            'verbosity': -1,
            'max_bins': 300,
            'num_iterations': 2000,
            'max_depth': max_depth
        },
        'callbacks': [
            lgb.log_evaluation(500),
            lgb.early_stopping(2000)
        ],
        # 'early_stopping_rounds': 2000
    }

    if model == 'lgb':
        train_set = lgb.Dataset(X_train, Y_train)
        valid_sets = [train_set, lgb.Dataset(X_valid, Y_valid)]
        model = lgb.train(train_set=train_set,
                          valid_sets=valid_sets, **train_params)
        if diff:
            Y_pred = pd.DataFrame({'load': model.predict(X_test)},
                                  index=Y_test.index) + Y_ori.shift(192)[Y_test.index].to_frame()
            Y_test = Y_ori[Y_test.index]
        else:
            Y_pred = pd.DataFrame({'load': model.predict(X_test)}, index=Y_test.index)

    elif model == 'linear':
        model = LinearRegression()
        model.fit(X_train, Y_train)
        Y_pred = pd.DataFrame({'load': model.predict(X_test)}, index=Y_test.index)

    elif model == 'evloss':
        train_set = lgb.Dataset(X_train, Y_train, free_raw_data=False)
        valid_sets = [train_set, lgb.Dataset(X_valid, Y_valid, free_raw_data=False)]
        model = lgb.train(train_set=train_set, valid_sets=valid_sets, **train_params,
                          fobj=my_closure_lgb1,
                          feval=my_closure_lgb2)
        Y_pred = pd.DataFrame({'load': model.predict(X_test)}, index=Y_test.index)

    Y_pred_train = pd.DataFrame({'load': model.predict(X_train)}, index=Y_train.index)
    #     plt.figure(figsize=(30,3))
    #     plt.plot(Y_pred_train)
    #     plt.plot(Y_train)
    #     plt.plot(Y_test)
    #     plt.plot(Y_pred)
    #     plt.legend(['pred', 'true', 'true', 'pred'])
    return Y_pred, Y_test


locs = ['Rat', 'Robin', 'Peacock']
rmses = []
mapes = []
for loc in locs:
    ## load data
    df = pd.read_csv('../../data/public_dataset/{}_20160101_20171231_15T.csv'.format(loc),
                     parse_dates=['dt']).set_index('dt')

    ## data preprocess
    x = df['load']
    persist_ad = PersistAD()
    anomalies = persist_ad.fit_detect(df['load'])
    df.loc[anomalies == True, 'load'] = np.nan
    x = x.interpolate()

    # target column = 'load'
    # get long-term trend (or yearly trend)
    dfyt = df.copy()
    dfyt['load'] = dfyt['load'].rolling(96 * 30 * 1, min_periods=1).mean().interpolate(limit_direction='both')
    res = (df['load'] - dfyt['load']).to_frame()

    # get short-term trend
    dft = df.rolling(96).mean().copy()
    dft['load'] = (res['load'].rolling(96).mean()).to_frame().interpolate(limit_direction='both')

    # get periodic
    dfs = df.copy()
    dfs['load'] = (res['load'] - dft['load']).to_frame().interpolate(limit_direction='both')

    # predict dfyt, dft, dfs individually
    Y_pred1, Y_test1 = pred_one_component(feature_ensembler, dfyt, feature_engi=True, diff=False, model='linear',
                                          xslice='all')

    Y_pred2, Y_test2 = pred_one_component(feature_ensembler2, dft.rolling(96).mean(), feature_engi=True,
                                          diff=True, model='evloss', xslice='all')  # , max_depth=1)

    Y_pred3, Y_test3 = pred_one_component(feature_ensembler, dfs, feature_engi=True,
                                          diff=True, model='lgb', xslice='all')

    Y_pred = Y_pred1.rename(columns={'load': 0}) + Y_pred2.rename(columns={'load': 0}) + Y_pred3.rename(
        columns={'load': 0})
    Y_test = Y_test1 + Y_test2[Y_test1.index] + Y_test3

    rmse = get_acc(Y_pred.values, Y_test.to_frame().values)
    mape = get_mape(Y_pred.values, Y_test.to_frame().values)
    rmses += [rmse]
    mapes += [mape]
    print('=' * 10 + ' results ' + '=' * 10)
    print('loc={}'.format(loc))
    print('daily_mean_rmse={}'.format(rmse))
    print('daily_mean_mape={}'.format(mape))

result = pd.DataFrame({
    'model': ['SaDI'] * len(locs),
    'dataset': locs,
    'rmse': rmses,
    'mape': mapes
})

result.to_csv('../../results/results.csv')


