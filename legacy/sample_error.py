import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from autosklearn.classification import AutoSklearnClassifier
# from autosklearn.regression import AutoSklearnRegressor


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


def automl(X, y, families, r,
                             base_time=300, per_run=15, folds=5, seed=0):
    include = None if r == 0 else ({"regressor": families} if families else None)

    if base_time < 30:
        base_time = 30
    if per_run >= base_time / 2:
        per_run = int(base_time / 2) - 1

    automl = AutoSklearnClassifier(
        time_left_for_this_task=base_time,
        per_run_time_limit=per_run,
        n_jobs=1,
        memory_limit=4096,  # 物理内存足够可设 None
        resampling_strategy="holdout",
        resampling_strategy_arguments={"train_size": 0.8},
        # include={"classifier": ["gaussian_nb", "lda", "libsvm_svc", "mlp", "qda", "sgd"]},  # 先锁死一个最稳的
        # ensemble_size=1, ensemble_nbest=2,  # 禁用集成，避免加载问题
        # initial_configurations_via_metalearning=0,  # 关掉金属学习，进一步简化
        # seed=seed,
        # delete_tmp_folder_after_terminate=False,
        # disable_evaluator_output=False,  # 子进程日志直接打印
    )
    try:
        automl.fit(X, y)
        # 从本轮结果提取赢家家族，供下一轮缩搜
        fam_next = pick_families_from_leaderboard(automl, max_fams=max(2, 3 - (r // 2)))
        if fam_next:
            families = fam_next
        return automl, families
    except Exception as e:
        # 3) 兜底：保证可用
        model = SVR()
        model.fit(X, y)# 从本轮结果提取赢家家族，供下一轮缩搜
        fam_next = pick_families_from_leaderboard(automl, max_fams=max(2, 3 - (r // 2)))
        if fam_next:
            families = fam_next
        return model, families


def fit_svm(X_train, y_train):
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model


def cal_metric(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metric = f1_score(y_test, y_pred)
    return metric


def total():
    dataset = "cancer"
    data = pd.read_csv('data/'+dataset+'/'+dataset+'_25.csv')
    es = [10, 20, 30, 40, 50]
    es_res = [[], [], [], [], []]
    for error_rate in es:
        index_err = es.index(error_rate)
        error = "_random_missing_" + str(error_rate)
        # data = pd.read_csv('data/' + dataset + '/' + dataset + '_25.csv')
        data = pd.read_csv('data/inject_all/' + dataset + error + '.csv')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        indices = np.arange(X.shape[0])
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.2,
                                                                                         random_state=4)
        X_test = clean(X_test, test_indices, np.ones(len(test_indices), dtype=bool),
                       'data/' + dataset + '/' + dataset + '_data_vectorized.csv')

        # 2. 配置
        subsample_fractions = [i * 60 for i in range(1, 6)]  # 清洗比例
        nn_results, ny_results, yn_results, yy_results, dh_results = [], [], [], [], []
        ng_results, two_results = [], []
        ours = []

        min_sample_num = 1000

        # 3. 不同比例下循环
        i = 0
        for fraction in subsample_fractions:
            print(f"Processing subsample fraction: {fraction}")
            clean_fraction = fraction / np.max(subsample_fractions)
            dmco_model_fraction = np.max([clean_fraction/2, min_sample_num * len(X_train)])

            if clean_fraction < 0.2:
                dmco_model_fraction = min_sample_num / len(X_train)

            if fraction == 0.0:
                # 不进行清洗
                random_X_train_update = X_train.copy()
                loss_X_train_update = X_train.copy()
            else:
                # # Dirty Data + Raw Model
                # nn_model = fit_svm(X_train.copy(), y_train)
                # nn_metric = cal_metric(nn_model, X_test, y_test)
                # print("Dirty + Raw : ", nn_metric)
                # nn_results.append(nn_metric)
                #
                # # Clean Data + Raw Model
                # random_idx_clean = sample_random(X_train.copy().reset_index(), y_train, clean_fraction)
                # mask = np.zeros(len(train_indices), dtype=bool)
                # mask[random_idx_clean] = True
                # X_train_cleaned = clean(X_train.copy(), train_indices, mask,
                #                         'data/' + dataset + '/' + dataset + '_data_vectorized.csv')
                # yn_model = fit_svm(X_train_cleaned, y_train)
                # yn_metric = cal_metric(yn_model, X_test, y_test)
                # print("Clean + Raw : ", yn_metric)
                # yn_results.append(yn_metric)
                # #
                # # Dirty Data + AutoML
                # ny_model, families = automl(X_train.copy(), y_train, None, 0, fraction)
                # ny_metric = cal_metric(ny_model, X_test, y_test)
                # print("Dirty + Auto : ", ny_metric)
                # ny_results.append(ny_metric)
                #
                # # Clean Data + AutoML
                # # random_idx_clean_half = random_idx_clean[:int(clean_fraction/2 * len(X_train)):]
                # # mask = np.zeros(len(train_indices), dtype=bool)
                # # mask[random_idx_clean_half] = True
                # mask = np.ones(len(train_indices), dtype=bool)
                # X_train_cleaned = clean(X_train.copy(), train_indices, mask,
                #                         'data/' + dataset + '/' + dataset + '_data_vectorized.csv')
                # yy_model, families = automl(X_train_cleaned.copy(), y_train, None, 0, fraction)
                # yy_metric = cal_metric(yy_model, X_test, y_test)
                # print("Clean + Auto : ", yy_metric)
                # yy_results.append(yy_metric)
                #
                # # no loss
                # grad_idx = sample_base_grad(X_train.copy().reset_index(), y_train, clean_fraction/3*2)
                # mask = np.zeros(len(train_indices), dtype=bool)
                # mask[grad_idx] = True
                # X_train_cleaned = clean(X_train.copy(), train_indices, mask,
                #                         'data/' + dataset + '/' + dataset + '_data_vectorized.csv')
                # dh_model, families = automl(X_train_cleaned.copy(), y_train, None, 0, int(fraction/3))
                # dh_metric = cal_metric(dh_model, X_test, y_test)
                # print("No loss : ", dh_metric)
                # dh_results.append(dh_metric)
                #
                # # no grad
                # random_idx_clean_half = random_idx_clean[:int(clean_fraction/3*2 * len(X_train))]
                # mask = np.zeros(len(train_indices), dtype=bool)
                # mask[random_idx_clean_half] = True
                # X_train_cleaned = clean(X_train.copy(), train_indices, mask,
                #                         'data/' + dataset + '/' + dataset + '_data_vectorized.csv')
                # loss_idx = sample_base_loss(X_train_cleaned.copy().reset_index(), y_train, dmco_model_fraction)
                # X_loss = X_train_cleaned.copy().reset_index(drop=True).iloc[loss_idx]
                # y_loss = y_train.copy().reset_index(drop=True).iloc[loss_idx]
                # ng_model, families = automl(X_loss, y_loss, None, 0, int(fraction / 3))
                # ng_model.refit(X_train_cleaned, y_train)
                # ng_metric = cal_metric(ng_model, X_test, y_test)
                # # print(time.time()-start)
                # print("No grad : ", ng_metric)
                # ng_results.append(ng_metric)
                #
                # # Two Stage
                # grad_idx = sample_base_grad(X_train.copy().reset_index(), y_train, clean_fraction / 2)
                # mask = np.zeros(len(train_indices), dtype=bool)
                # mask[grad_idx] = True
                # X_train_cleaned = clean(X_train.copy(), train_indices, mask,
                #                         'data/' + dataset + '/' + dataset + '_data_vectorized.csv')
                # loss_idx = sample_base_loss(X_train_cleaned.copy().reset_index(), y_train, dmco_model_fraction)
                # X_loss = X_train_cleaned.copy().reset_index(drop=True).iloc[loss_idx]
                # y_loss = y_train.copy().reset_index(drop=True).iloc[loss_idx]
                # two_model, families = automl(X_loss, y_loss, None, 0, int(fraction / 2))
                # two_model.refit(X_train_cleaned, y_train)
                # two_metric = cal_metric(two_model, X_test, y_test)
                # # print(time.time()-start)
                # print("Two Stage: ", two_metric)
                # two_results.append(two_metric)

                # DMCO
                grad_idx = sample_base_grad(X_train.copy().reset_index(), y_train, clean_fraction/3*2)
                mask = np.zeros(len(train_indices), dtype=bool)
                mask[grad_idx] = True
                X_train_cleaned = clean(X_train.copy(), train_indices, mask,
                                        'data/' + dataset + '/' + dataset + '_data_vectorized.csv')
                loss_idx = sample_base_loss(X_train_cleaned.copy().reset_index(), y_train, dmco_model_fraction)
                X_loss = X_train_cleaned.copy().reset_index(drop=True).iloc[loss_idx]
                y_loss = y_train.copy().reset_index(drop=True).iloc[loss_idx]
                ours_model, families = automl(X_loss, y_loss, None, 0, int(fraction/3))
                ours_model.refit(X_train_cleaned, y_train)
                ours_metric = cal_metric(ours_model, X_test, y_test)
                # print(time.time()-start)
                print("DMCO : ", ours_metric)
                es_res[index_err].append(ours_metric)

    # 4. 绘图
    f = open('error_log/' + dataset + error + '_error_res_1216.txt', 'w')
    f.write(str(es_res[0])+'\n')
    f.write(str(es_res[1])+'\n')
    f.write(str(es_res[2])+'\n')
    f.write(str(es_res[3])+'\n')
    f.write(str(es_res[4])+'\n')
    f.close()

    plt.figure(figsize=(8, 6))
    plt.plot(subsample_fractions, es_res[0], marker='o', label='10')
    plt.plot(subsample_fractions, es_res[1], marker='s', label='20')
    plt.plot(subsample_fractions, es_res[2], marker='^', label='30')
    plt.plot(subsample_fractions, es_res[3], marker='D', label='40')
    plt.plot(subsample_fractions, es_res[4], marker='P', label='50')
    plt.xlabel('Sample fraction (%)')
    plt.ylabel('Test mse')
    plt.title('Effect of sampling for cleaning on SVM Performance')
    plt.legend()
    plt.grid(True)
    plt.show()
total()