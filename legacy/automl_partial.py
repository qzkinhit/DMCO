# autosklearn_meta_krounds.py
from autosklearn import regression, metrics
from autosklearn.metrics import r2
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def clean(X, indices, mask, file):
    clean_data = pd.read_csv(file)
    cnt = 0
    for i in range(len(X)):
        if mask[i]:
            X.iloc[i] = clean_data.iloc[indices[i]][:-1]
            cnt += 1
    print(cnt)
    return X


# ------- 解析本轮赢家家族，用于下一轮 include 逐步缩搜 -------
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


# ------- 主流程：k 段逐轮（每轮只用一份数据），逐步缩小搜索空间 -------
def autosklearn_meta_krounds(X_train, y_train, X_test, y_test,
                             k=3, base_time=30, per_run=60, folds=5, seed=0):
    n = len(y_train)
    cuts = np.linspace(0, n, k + 1, dtype=int)
    parts = [(X_train[cuts[i]:cuts[i+1]], y_train[cuts[i]:cuts[i+1]]) for i in range(k)]

    best_model, best_score, best_round, families = None, -np.inf, None, None

    for r in range(k):
        Xr, yr = parts[r]

        # 第0轮：include=None（让内置元学习暖启）；之后各轮用赢家家族缩搜
        include = None if r == 0 else ({"regressor": families} if families else None)

        automl = regression.AutoSklearnRegressor(
            time_left_for_this_task=base_time + int(0.2 * base_time) * r,
            per_run_time_limit=per_run,
            resampling_strategy="cv",
            resampling_strategy_arguments={"folds": folds},
            include=include,
            ensemble_kwargs={"ensemble_size": 30},
            metric=metrics.r2,
            n_jobs=-1,
            seed=seed,
        )

        automl.fit(Xr, yr, dataset_name=f"kround_{r+1}")
        automl.refit(Xr, yr)

        score = automl.score(X_test, y_test)
        print(f"[Round {r+1}] include={include}  Test R2={score:.4f}")

        # 记录最佳轮
        if score > best_score:
            best_score, best_model, best_round = score, automl, r + 1

        # 从本轮结果提取赢家家族，供下一轮缩搜
        fam_next = pick_families_from_leaderboard(automl, max_fams=max(2, 3 - (r // 2)))
        if fam_next:
            families = fam_next
        # 若解析不到，则保持原 families 不变（或置 None 让下轮继续全空间）

    print(f"[Final] choose Round {best_round}  Best Test R2={best_score:.4f}")
    return best_model, best_score, best_round


# ------- 演示：按需替换为你的数据 -------
if __name__ == "__main__":
    dataset = "nasa"
    data = pd.read_csv('data/' + dataset + '/nasa_25.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.2,
                                                                                     random_state=4)
    X_test = clean(X_test, test_indices, np.ones(len(test_indices), dtype=bool),
                   'data/' + dataset + '/' + dataset + '_data_vectorized.csv')
    model, best_r2, best_round = autosklearn_meta_krounds(
        X_train, y_train, X_test, y_test,
        k=3,           # 训练集分成3份，逐轮搜索
        base_time=30, # 每轮总时间，后续轮适度增加
        per_run=30,    # 单模型最长训练时间
        folds=5,
        seed=0,
    )
    # 产出：best_round 对应轮次训练好的 automl 模型（不做全量重训）
    y_pred = model.predict(X_test)

    model_automl = AutoSklearnRegressor(
        time_left_for_this_task=30,
        seed=42
    )
    start_auto = time.time()
    model_automl.fit(X, y)
    time_auto = time.time() - start_auto
    print("AutoML time : ", time_auto)