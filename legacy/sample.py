import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
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


def sample_base_loss(X, y, fraction, random_state=0, beta=0.5):
    """
    无预处理、一步到位的带权随机采样：
    - 用 Huber 回归的绝对残差做“信息量权重”（越大越有用，但做截断防爆）
    - 用 beta 控制“信息采样 vs 纯随机”的权衡（0=纯随机，1=全按残差加权）
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import HuberRegressor

    # 保留原始索引，不改动特征
    if isinstance(X, pd.DataFrame):
        orig_idx = X.index.to_numpy()
        X_mat = X.to_numpy()
    else:
        X_mat = np.asarray(X)
        orig_idx = np.arange(len(X_mat))
    y_vec = np.asarray(y).reshape(-1)

    n = len(y_vec)
    k = max(1, min(int(np.ceil(fraction * n)), n))
    if k >= n:  # 取满就直接返回
        return orig_idx[:k]

    rng = np.random.RandomState(random_state)

    # 1) 单次鲁棒拟合（无标准化）
    try:
        hr = HuberRegressor(epsilon=1.35, alpha=1e-4, fit_intercept=True)
        hr.fit(X_mat, y_vec)
        resid = y_vec - hr.predict(X_mat)
    except Exception:
        # 万一拟合失败，退化成以 y 的偏差近似
        resid = y_vec - np.median(y_vec)

    # 2) 绝对残差 → 权重（截断95分位，避免被极端点主导）
    s = np.abs(resid).astype(float)
    cap = np.quantile(s, 0.95)
    s = np.minimum(s, cap)
    s = s + 1e-12  # 防止全0

    # 3) 混合权重： (1-beta)·均匀 + beta·(残差占比)
    p_info = s / s.sum()
    p = (1.0 - beta) * (np.ones(n) / n) + beta * p_info
    p = p / p.sum()

    # 4) 带权随机无放回抽样
    #   numpy 没有原生的加权无放回 -> 用 Gumbel-top-k 近似实现
    g = -np.log(-np.log(rng.uniform(1e-12, 1 - 1e-12, size=n)))
    score = np.log(p) + g
    pick = np.argsort(-score)[:k]

    return orig_idx[pick]


def sample_base_grad(X, y, fraction):
    model_svc = SVR()
    model_svc.fit(X, y)

    y_pred = model_svc.predict(X)
    res = y_pred - y
    # ε-不敏感损失的 dL/dŷ：|res|<=ε 时梯度为 0，其他为 sign(res)
    g_pred = np.where(np.abs(res) > model_svc.epsilon, np.sign(res), 0.0)

    # 近似每个样本对参数的梯度范数：|dL/dŷ| * ||x||_2
    g_norm = np.abs(g_pred) * np.linalg.norm(X, axis=1)

    selected_samples = np.argsort(-g_norm)[:int(fraction * len(g_norm))]
    # todo : duo yang xing
    return selected_samples


def preprocess(X):
    # 数据预处理
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    preprocessor.fit(X)
    X = preprocessor.fit_transform(X)
    return X


def pick_families_from_leaderboard(automl, max_fams=3):
    try:
        lb = automl.leaderboard(detailed=True)  # pandas.DataFrame, 已按分数排序
    except Exception:
        return None
    col = "pipeline" if "pipeline" in lb.columns else lb.columns[-1]
    vocab = [
        "random_forest", "gradient_boosting", "xgradient_boosting", "extra_trees",
        "sgd", "ridge_regression", "adaboost", "k_nearest_neighbors",
        "gaussian_process", "liblinear_svr"
    ]
    fams = []
    for s in lb[col].astype(str):
        for f in vocab:
            if f in s and f not in fams:
                fams.append(f)
        if len(fams) >= max_fams:
            break
    return fams or None


def automl(X_train, y_train, families, r,
                             base_time=300, per_run=30, folds=5, seed=0):
    include = None if r == 0 else ({"regressor": families} if families else None)

    automl = regression.AutoSklearnRegressor(
        time_left_for_this_task=base_time + int(0.2 * base_time) * r,
        include=include,
        metric=metrics.mean_squared_error,
        per_run_time_limit=per_run,
        seed=seed,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': 5}
    )

    automl.fit(X_train, y_train)

    # 从本轮结果提取赢家家族，供下一轮缩搜
    fam_next = pick_families_from_leaderboard(automl, max_fams=max(2, 3 - (r // 2)))
    if fam_next:
        families = fam_next
    return automl, families


def run():
    dataset = "nasa"
    data = pd.read_csv('data/'+dataset+'/nasa_25.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.2, random_state=4)
    X_test = clean(X_test, test_indices, np.ones(len(test_indices), dtype=bool), 'data/'+dataset+'/'+dataset+'_data_vectorized.csv')

    # 2. 配置
    subsample_fractions = [i/20 for i in range(0, 21)]  # 清洗比例
    # subsample_fractions = [0.0, 1]
    random_results, loss_results = [], []

    # 3. 不同比例下循环
    i = 0
    for fraction in subsample_fractions:
        # fraction /= 5
        # fraction += 0.8
        print(f"Processing subsample fraction: {fraction}")
        start_time = time.time()

        if fraction == 0.0:
            # 不进行清洗
            random_X_train_update = X_train.copy()
            loss_X_train_update = X_train.copy()
        else:
            random_idx = sample_random(X_train.copy().reset_index(), y_train, fraction)
            mask = np.zeros(len(train_indices), dtype=bool)
            mask[random_idx] = True
            random_X_train_update = clean(X_train.copy(), train_indices, mask,
                                          'data/' + dataset + '/' + dataset + '_data_vectorized.csv')

            loss_idx = sample_base_grad(X_train.copy().reset_index(), y_train, fraction)
            mask = np.zeros(len(train_indices), dtype=bool)
            mask[loss_idx] = True
            loss_X_train_update = clean(X_train.copy(), train_indices, mask,
                                        'data/'+dataset+'/'+dataset+'_data_vectorized.csv')

        # 拆开特征和标签
        random_X_train_cleaned = random_X_train_update
        loss_X_train_cleaned = loss_X_train_update



        # X_cleaned = preprocess(X_cleaned)

        model_svc = SVC()
        model_svc = SVR()
        model_svc.fit(random_X_train_cleaned, y_train)
        y_pred = model_svc.predict(X_test)
        acc_auto = mean_squared_error(y_test, y_pred)
        # acc_auto = f1_score(y_test, y_pred)
        random_results.append(acc_auto)
        print(acc_auto)
        model_svc = SVR()
        model_svc.fit(loss_X_train_cleaned, y_train)
        y_pred = model_svc.predict(X_test)
        acc_auto = mean_squared_error(y_test, y_pred)
        # acc_auto = f1_score(y_test, y_pred)
        loss_results.append(acc_auto)
        i += 1

    # 4. 绘图

    plt.figure(figsize=(8, 6))
    plt.plot(subsample_fractions, random_results, marker='o', label='Sample random')
    plt.plot(subsample_fractions, loss_results, marker='*', label='Sample on loss')
    plt.xlabel('Sample fraction (%)')
    plt.ylabel('Test mse')
    plt.title('Effect of sampling for cleaning on SVM Performance')
    plt.legend()
    plt.grid(True)
    plt.show()


def run_Auto():
    dataset = "nasa"
    data = pd.read_csv('data/' + dataset + '/nasa_25.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.2,
                                                                                     random_state=4)
    X_test = clean(X_test, test_indices, np.ones(len(test_indices), dtype=bool),
                   'data/' + dataset + '/' + dataset + '_data_vectorized.csv')

    # 2. 配置
    subsample_fractions = [i / 20 for i in range(0, 21)]  # 清洗比例
    # subsample_fractions = [0.0, 1]
    random_results, grad_results = [], []

    # 3. 不同比例下循环
    i = 0
    for fraction in subsample_fractions:
        # fraction /= 5
        # fraction += 0.8
        print(f"Processing subsample fraction: {fraction}")

        if fraction == 0.0:
            # 不进行清洗
            fraction = 0.01
        random_idx = sample_random(X_train.copy().reset_index(), y_train, fraction)
        X_random = X_train.copy().reset_index(drop=True).iloc[random_idx]
        y_random = y_train.copy().reset_index(drop=True).iloc[random_idx]
        automodel = automl(X_random, y_random)
        y_pred = automodel.predict(X_test)
        acc_auto = mean_squared_error(y_test, y_pred)
        random_results.append(acc_auto)
        print("random: ", acc_auto)

        grad_idx = sample_base_loss(X_train.copy().reset_index(), y_train.copy().reset_index(drop=True), fraction)
        X_grad = X_train.copy().reset_index(drop=True).iloc[grad_idx]
        y_grad = y_train.copy().reset_index(drop=True).iloc[grad_idx]
        automodel = automl(X_grad, y_grad)
        y_pred = automodel.predict(X_test)
        acc_auto = mean_squared_error(y_test, y_pred)
        grad_results.append(acc_auto)
        print("grad: ", acc_auto)

        i += 1

    # 4. 绘图

    plt.figure(figsize=(8, 6))
    x_a = [i * 100 for i in subsample_fractions]
    plt.plot(x_a, random_results, marker='o', label='Sample random')
    plt.plot(x_a, grad_results, marker='*', label='Sample on grad')
    plt.xlabel('Sample fraction (%)')
    plt.ylabel('Test acc')
    plt.title('Effect of sampling for AutoML on SVM Performance')
    plt.legend()
    plt.grid(True)
    plt.show()


def fit_svm(X_train, y_train):
    model = SVR()
    model.fit(X_train, y_train)
    return model


def cal_metric(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metric = mean_squared_error(y_test, y_pred)
    return metric


def total():
    dataset = "nasa"
    error = "_random_outliers_30"
    data = pd.read_csv('data/inject_all/' + dataset + error + '.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.2, random_state=4)
    X_test = clean(X_test, test_indices, np.ones(len(test_indices), dtype=bool), 'data/'+dataset+'/'+dataset+'_data_vectorized.csv')

    # 2. 配置
    subsample_fractions = [i*5 for i in range(1, 11)]  # 清洗比例
    nn_results, ny_results, yn_results, yy_results = [], [], [], []
    ours = []

    # 3. 不同比例下循环
    i = 0
    for fraction in subsample_fractions:
        print(f"Processing subsample fraction: {fraction}")
        start_time = time.time()

        if fraction == 0.0:
            # 不进行清洗
            random_X_train_update = X_train.copy()
            loss_X_train_update = X_train.copy()
        else:
            # # Dirty Data + Raw Model
            random_idx_model = sample_random(X_train.copy().reset_index(), y_train, fraction)
            X_random = X_train.copy().reset_index(drop=True).iloc[random_idx_model]
            y_random = y_train.copy().reset_index(drop=True).iloc[random_idx_model]
            nn_model = fit_svm(X_random, y_random)
            nn_metric = cal_metric(nn_model, X_test, y_test)
            print("Dirty + Raw : ", nn_metric)
            nn_results.append(nn_metric)
            #
            # Dirty Data + AutoML
            X_random = X_train.copy().reset_index(drop=True).iloc[random_idx_model]
            y_random = y_train.copy().reset_index(drop=True).iloc[random_idx_model]
            ny_model, families = automl(X_random, y_random, None, 0)
            ny_metric = cal_metric(ny_model, X_test, y_test)
            print("Dirty + Auto : ", ny_metric)
            ny_results.append(ny_metric)

            # Clean Data + Raw Model
            random_idx_clean = sample_random(X_train.copy().reset_index(), y_train, fraction)
            mask = np.zeros(len(train_indices), dtype=bool)
            mask[random_idx_clean] = True
            X_train_cleaned = clean(X_train.copy(), train_indices, mask, 'data/' + dataset + '/' + dataset + '_data_vectorized.csv')
            # random_idx = sample_random(X_train_cleaned.copy().reset_index(), y_train, fraction)
            X_random = X_train_cleaned.copy().reset_index(drop=True).iloc[random_idx_model]
            y_random = y_train.copy().reset_index(drop=True).iloc[random_idx_model]
            yn_model = fit_svm(X_random, y_random)
            yn_metric = cal_metric(yn_model, X_test, y_test)
            print("Clean + Raw : ", yn_metric)
            yn_results.append(yn_metric)

            # Clean Data + AutoML
            random_idx_clean_half = random_idx_clean[:int(fraction/2 * len(X_train))]
            random_idx_model_half = random_idx_model[int(fraction/2 * len(X_train)):]
            mask = np.zeros(len(train_indices), dtype=bool)
            mask[random_idx_clean_half] = True
            X_train_cleaned = clean(X_train.copy(), train_indices, mask, 'data/' + dataset + '/' + dataset + '_data_vectorized.csv')

            X_random = X_train_cleaned.copy().reset_index(drop=True).iloc[random_idx_model_half]
            y_random = y_train.copy().reset_index(drop=True).iloc[random_idx_model_half]
            yy_model, families = automl(X_random, y_random, None, 0)
            yy_metric = cal_metric(yy_model, X_test, y_test)
            print("Clean + Auto : ", yy_metric)
            yy_results.append(yy_metric)

            # # DMCO
            # grad_idx = sample_base_grad(X_train.copy().reset_index(), y_train, fraction/2)
            # mask = np.zeros(len(train_indices), dtype=bool)
            # mask[grad_idx] = True
            # X_train_cleaned = clean(X_train.copy(), train_indices, mask,
            #                         'data/' + dataset + '/' + dataset + '_data_vectorized.csv')
            # loss_idx = sample_base_loss(X_train_cleaned.copy().reset_index(), y_train, fraction/2)
            # X_loss = X_train_cleaned.copy().reset_index(drop=True).iloc[loss_idx]
            # y_loss = y_train.copy().reset_index(drop=True).iloc[loss_idx]
            # ours_model, families = automl(X_loss, y_loss, None, 0)
            # ours_metric = cal_metric(ours_model, X_test, y_test)
            # print("DMCO : ", ours_metric)
            # ours.append(ours_metric)

    # 4. 绘图
    f = open('log/' + dataset + '_ori_res.txt', 'w')
    f.write(str(nn_results))
    f.write(str(ny_results))
    f.write(str(yn_results))
    f.write(str(yy_results))
    f.close()

    plt.figure(figsize=(8, 6))
    plt.plot(subsample_fractions, nn_results, marker='o', label='Dirty Data + Raw Model')
    plt.plot(subsample_fractions, ny_results, marker='s', label='Dirty Data + AutoML')
    plt.plot(subsample_fractions, yn_results, marker='^', label='Clean Data + Raw Model')
    plt.plot(subsample_fractions, yy_results, marker='D', label='Clean Data + AutoML')
    plt.plot(subsample_fractions, ours, marker='P', label='DMCO')
    plt.xlabel('Sample fraction (%)')
    plt.ylabel('Test mse')
    plt.title('Effect of sampling for cleaning on SVM Performance')
    plt.legend()
    plt.grid(True)
    plt.show()


total()