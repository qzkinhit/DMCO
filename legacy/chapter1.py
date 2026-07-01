import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR, LinearSVC
from autosklearn import regression, metrics
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from AutoClean import AutoClean
import time
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor


def clean(X, indices, mask, file):
    clean_data = pd.read_csv(file)
    cnt = 0
    for i in range(len(X)):
        if mask[i]:
            X.iloc[i] = clean_data.iloc[indices[i]][:-1]
            cnt += 1
    # print(cnt)
    return X


# 计算均方误差
def compute_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2)


def sample_random(X, y, fraction):
    return np.random.choice(X.index, size=int(len(X) * fraction), replace=False)


def automl(X_train, y_train, families, r,
                             base_time=300, per_run=30, folds=5, seed=0):
    include = None if r == 0 else ({"regressor": families} if families else None)

    automl = AutoSklearnRegressor(
        time_left_for_this_task=base_time + int(0.2 * base_time) * r,
        include=include,
        metric=metrics.mean_squared_error,
        per_run_time_limit=per_run,
        seed=seed,
        # resampling_strategy='cv',
        # resampling_strategy_arguments={'folds': 5}
    )

    automl.fit(X_train, y_train)
    return automl


def fit_svm(X_train, y_train):
    model = SVR(C=1, max_iter=10000)
    model.fit(X_train, y_train)
    return model


def cal_metric(model, X_test, y_test):
    y_pred = model.predict(X_test)
    # metric = f1_score(y_test, y_pred)
    metric = mean_squared_error(y_test, y_pred)
    return metric


def total():
    # datasets = ["smartfactory", "cancer",  "adult", "skin"]
    datasets = ["nasa", "soilmoisture"]
    types = ["random_outliers", "random_missing", "system_outliers", "system_missing", "gauss", "white"]
    for type in types:
        # 2. 配置
        res = [[] for _ in range(len(datasets))]
        i = 0
        for dataset in datasets:
            subsample_fractions = [i*10 for i in range(1, 6)]  # 清洗比例
            print(type, dataset)

            # 3. 不同比例下循环
            for fraction in subsample_fractions:
                print(f"Processing subsample fraction: {fraction}")
                data = pd.read_csv('data/inject_all/' + dataset + '_' + type + '_' + str(fraction) + '.csv')
                X = data.iloc[:, :-1]
                y = data.iloc[:, -1]
                indices = np.arange(X.shape[0])
                X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.2,
                                                                                                 random_state=42)
                X_test = clean(X_test, test_indices, np.ones(len(test_indices), dtype=bool),
                               'data/' + dataset + '/' + dataset + '_data_vectorized.csv')

                # nn_model = fit_svm(X_train, y_train)
                # nn_metric = cal_metric(nn_model, X_test, y_test)
                # print(nn_metric)
                # res[i].append(nn_metric)

                # if len(X_train) > 1000:
                #     random_idx_model = sample_random(X_train.copy().reset_index(), y_train, 1000/len(X_train))
                #     X_random = X_train.copy().reset_index(drop=True).iloc[random_idx_model]
                #     y_random = y_train.copy().reset_index(drop=True).iloc[random_idx_model]
                #     ny_model = fit_svm(X_random, y_random)
                #     ny_metric = cal_metric(ny_model, X_test, y_test)
                #     print(ny_metric)
                #     res[i].append(ny_metric)
                # else:
                #     ny_model = fit_svm(X_train, y_train)
                #     ny_metric = cal_metric(ny_model, X_test, y_test)
                #     print(ny_metric)
                #     res[i].append(ny_metric)

                if len(X_train) > 10000:
                    random_idx_model = sample_random(X_train.copy().reset_index(), y_train, 10000/len(X_train))
                    X_random = X_train.copy().reset_index(drop=True).iloc[random_idx_model]
                    y_random = y_train.copy().reset_index(drop=True).iloc[random_idx_model]
                    ny_model = automl(X_random, y_random, None, 0)
                    ny_metric = cal_metric(ny_model, X_test, y_test)
                    print(ny_metric)
                    res[i].append(ny_metric)
                else:
                    ny_model = automl(X_train, y_train, None, 0)
                    ny_metric = cal_metric(ny_model, X_test, y_test)
                    print(ny_metric)
                    res[i].append(ny_metric)
            i+=1

        # 4. 绘图
        print(res)
        f = open('log/' + type + '_res.txt', 'w')
        f.write(str(res))
        f.close()

        # plt.figure(figsize=(8, 6))
        # plt.plot(subsample_fractions, res[0], marker='o', label=datasets[0])
        # plt.plot(subsample_fractions, res[1], marker='s', label=datasets[1])
        # plt.plot(subsample_fractions, res[2], marker='^', label=datasets[2])
        # plt.plot(subsample_fractions, res[3], marker='D', label=datasets[3])
        # # plt.plot(subsample_fractions, ours, marker='P', label='DMCO')
        # plt.xlabel('Error Rate (%)')
        # plt.ylabel('Model performance')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(type)
        # plt.show()


def total_mix(root_dir):
    res = []
    dataset = "cancer"
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            full_path = os.path.join(root, filename)
            data = pd.read_csv(full_path)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            indices = np.arange(X.shape[0])
            X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.2, random_state=42)
            X_test = clean(X_test, test_indices, np.ones(len(test_indices), dtype=bool),
                               'data/' + dataset + '/' + dataset + '_data_vectorized.csv')

            ny_model = automl(X_train, y_train, None, 0)
            ny_metric = cal_metric(ny_model, X_test, y_test)
            print(full_path, ":", ny_metric)
            res.append(ny_metric)


    print(res)
    f = open('data/cancer_mix/res.txt', 'w')
    f.write(str(res))
    f.close()


def total_mix_single(root_dir):
    res = []
    dataset = "cancer"
    errors = ["random_missing", "random_outliers", "system_missing", "system_outliers", "gauss", "white"]
    for error in errors:
        full_path = "data/inject_all/cancer_" + error + "_30.csv"
        data = pd.read_csv(full_path)
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        indices = np.arange(X.shape[0])
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.2, random_state=42)
        X_test = clean(X_test, test_indices, np.ones(len(test_indices), dtype=bool),
                                   'data/' + dataset + '/' + dataset + '_data_vectorized.csv')

        ny_model = automl(X_train, y_train, None, 0)
        ny_metric = cal_metric(ny_model, X_test, y_test)
        print(full_path, ":", ny_metric)
        res.append(ny_metric)


    print(res)
    f = open('data/cancer_mix/res_single.txt', 'w')
    f.write(str(res))
    f.close()
total()
# total_mix('data/cancer_mix')
# total_mix_single('')