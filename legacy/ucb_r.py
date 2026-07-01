import logging

import pandas as pd
import math
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error


from autosklearn.regression import AutoSklearnRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPRegressor
logging.getLogger("smac").setLevel(logging.ERROR)
logging.getLogger("AutoMLSMBO").setLevel(logging.ERROR)
logging.getLogger("Client-AutoMLSMBO").setLevel(logging.ERROR)

dmco_lib = [
        "gaussian_nb", "lda", "libsvm_svc", "mlp", "qda", "sgd"
    ]
dmco_lib = ['libsvm_svr', 'mlp', 'sgd', 'nn']

def refit_best_sklearn_model(automl,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             verbose: bool = True):
    """
    根据 AutoSklearnClassifier / AutoSklearnRegressor 的结果，
    选出 ensemble_weight 最大的子模型，并在外面对其“最终 estimator”
    重新构造 + 拟合一个纯 sklearn 模型。

    注意：
    - 这里只取最终分类器的类型 + 参数（EstimatorClass + get_params），
      不保留 auto-sklearn 自己加的预处理 / pipeline 壳子。
    - 适用于你后续在这个模型上做梯度 / 敏感度分析。

    返回:
      new_est: 重新拟合好的 sklearn 模型实例
    """
    models = automl.show_models()
    if not models:
        raise RuntimeError("AutoSklearn 里没有任何模型。")

    # 选 ensemble_weight 最大的模型
    best_info = max(models.values(), key=lambda m: m["ensemble_weight"])
    base_est = best_info["sklearn_regressor"]

    # 如果是 Pipeline，只取最后一层分类器
    if isinstance(base_est, Pipeline):
        final_est = base_est.steps[-1][1]
    else:
        final_est = base_est

    EstClass = final_est.__class__
    params = final_est.get_params()

    if verbose:
        print(f"[refit_best_sklearn_model] 选中的最优模型: {EstClass.__name__}")
        print(f"[refit_best_sklearn_model] 参数个数: {len(params)}")

    # 用相同类型 + 参数重新构造模型，并在外面用原始特征拟合
    new_est = EstClass(**params)
    new_est.fit(X_train, y_train)

    return new_est


def compute_gradient_scores(est,
                            X: np.ndarray,
                            y: np.ndarray,
                            mode: str = "auto",
                            eps: float = 1e-4,
                            max_samples: Optional[int] = None
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在给定的 sklearn 模型 est 上，对输入样本 X 计算“梯度/敏感度分数”。

    支持三种模式：
      - mode = "analytic" : 只用解析梯度（线性 + 概率）；
      - mode = "numeric"  : 仅用数值差分；
      - mode = "auto"     : 先尝试解析梯度，不行就退回数值差分。

    参数:
      est        : 训练好的 sklearn 模型（可含 Pipeline，但建议是纯 estimator）
      X, y       : 训练数据和标签（分类任务）
      mode       : "auto" / "analytic" / "numeric"
      eps        : 数值差分步长
      max_samples: 为了省时间，可限制最多对多少个样本算梯度。
                   若为 None，则对所有样本计算。

    返回:
      grads      : [m, d]，样本的输入梯度（m = 实际参与计算的样本数）
      grad_norm  : [m]，每个样本梯度的 L2 范数
      used_idx   : [m]，这些梯度对应的 X 中的全局行索引
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n, d = X.shape

    # 决定实际参与计算的样本
    if max_samples is not None and max_samples < n:
        used_idx = np.random.choice(n, size=max_samples, replace=False)
        X_use = X[used_idx]
        y_use = y[used_idx]
    else:
        used_idx = np.arange(n)
        X_use = X
        y_use = y

    # 解析梯度 + 数值梯度的内部函数
    def _is_analytic_linear_model(clf) -> bool:
        """判断 clf 是否是你的解析梯度公式可以安全使用的线性+概率模型"""
        # if not (hasattr(clf, "coef_") and hasattr(clf, "classes_")):
        #     return False
        # if not hasattr(clf, "predict_proba"):
        #     return False

        # 只在 LogisticRegression / LDA / log-loss SGD 上用解析梯度
        if isinstance(clf, MLPRegressor):
            return True
        if isinstance(clf, LinearDiscriminantAnalysis):
            return True
        if isinstance(clf, SGDRegressor):
            if getattr(clf, "loss", None) in ("log_loss", "log", "modified_huber"):
                return True
        # 其他线性模型理论上也可以，但容易踩坑，这里先保守一点
        return False

    def _get_clf_and_proba_estimator(est_):
        """
        把 est 拆成:
          - clf: 真正的最终分类器（拿 coef_/classes_ 用）
          - proba_est: 负责 predict_proba 的对象（Pipeline 或 clf 自身）
        """
        if isinstance(est_, Pipeline):
            clf_ = est_.steps[-1][1]
            proba_est_ = est_
        else:
            clf_ = est_
            proba_est_ = est_
        return clf_, proba_est_

    def _compute_analytic_grads(est_, X_, y_):
        """你的解析梯度公式版本"""
        clf_, proba_est_ = _get_clf_and_proba_estimator(est_)

        if not _is_analytic_linear_model(clf_):
            raise ValueError("当前模型不适合使用解析梯度公式。")

        proba = proba_est_.predict_proba(X_)        # [m, C]
        m, C = proba.shape
        classes = clf_.classes_

        y_ = y_.astype(classes.dtype)
        y_idx = np.searchsorted(classes, y_)

        one_hot = np.zeros_like(proba)
        one_hot[np.arange(m), y_idx] = 1.0

        diff = proba - one_hot                      # [m, C]

        W = clf_.coef_                              # [K, d]
        dd = W.shape[1]
        if dd != X_.shape[1]:
            raise ValueError(f"coef_ 维度 {dd} 与输入特征维度 {X_.shape[1]} 不一致。")

        if W.shape[0] == 1 and C == 2:
            W_full = np.vstack([np.zeros((1, dd)), W])  # [2, d]
        elif W.shape[0] == C:
            W_full = W
        else:
            raise ValueError(f"Unexpected coef_ shape {W.shape} for {C} classes.")

        grads_ = diff @ W_full                     # [m, d]
        return grads_

    # 单样本数值梯度（对任意模型都适用）
    def _sample_loss(est_, x_, y_):
        """定义单样本损失 L(x, y)，便于数值差分"""
        x_ = x_.reshape(1, -1)
        # 优先用概率
        if hasattr(est_, "predict_proba"):
            proba = est_.predict_proba(x_)[0]
            proba = np.clip(proba, 1e-12, 1.0)
            return -np.log(proba[int(y_)])
        # 有 decision_function，则构造 hinge 类损失
        if hasattr(est_, "decision_function"):
            scores = est_.decision_function(x_)
            scores = np.atleast_1d(scores)
            if scores.ndim == 1:  # 二分类
                sign = 1 if y_ == 1 else -1
                margin = sign * scores[0]
            else:
                correct_score = scores[0, int(y_)]
                other_max = np.max(np.delete(scores[0], int(y_)))
                margin = correct_score - other_max
            return max(0.0, 1.0 - margin)
        # 最退化：0-1 损失
        y_pred = est_.predict(x_)[0]
        return 0.0 if y_pred == y_ else 1.0

    def _numeric_gradient_one(est_, x_, y_, eps_):
        """对单个样本做数值差分"""
        x_ = x_.astype(float)
        d_ = x_.shape[0]
        g = np.zeros(d_, dtype=float)
        base = x_.copy()
        for j in range(d_):
            x_pos = base.copy()
            x_neg = base.copy()
            x_pos[j] += eps_
            x_neg[j] -= eps_
            L_pos = _sample_loss(est_, x_pos, y_)
            L_neg = _sample_loss(est_, x_neg, y_)
            g[j] = (L_pos - L_neg) / (2.0 * eps_)
        return g

    def _compute_numeric_grads(est_, X_, y_, eps_):
        m_ = X_.shape[0]
        d_ = X_.shape[1]
        grads_ = np.zeros((m_, d_), dtype=float)
        for i in range(m_):
            grads_[i] = _numeric_gradient_one(est_, X_[i], y_[i], eps_)
        return grads_

    # ======== 根据 mode 决定怎么算 ========
    if mode not in ("auto", "analytic", "numeric"):
        raise ValueError("mode 必须是 'auto' / 'analytic' / 'numeric' 之一。")

    grads = None

    if mode in ("analytic", "auto"):
        try:
            grads = _compute_analytic_grads(est, X_use, y_use)
            # 成功算出解析梯度
        except Exception as e:
            if mode == "analytic":
                raise
            # auto 模式下，解析失败则退回数值
            print(f"[compute_gradient_scores] 解析梯度失败，将退回数值差分。原因: {e}")

    if grads is None:
        grads = _compute_numeric_grads(est, X_use, y_use, eps)

    grad_norm = np.linalg.norm(grads, axis=1)

    return grads, grad_norm, used_idx


class CostAwareMAB:
    """
    代价感知多臂老虎机：
    arms = ["clean", "AutoML"]
    每轮：
      1) 用 choose() 选臂
      2) 执行对应策略
      3) 用 observe() 上报 metric_t, cost_t
    """
    def __init__(self,
                 arms=('clean', 'AutoML'),
                 gamma=0.9,      # 折扣因子
                 c=1.0,          # UCB 系数
                 H=5,            # 停滞阈值
                 boost=1.0,      # 探索提升因子
                 metric0=None):  # 初始 metric（可选）
        self.arms = list(arms)
        self.gamma = gamma
        self.c = c
        self.H = H
        self.boost = boost

        self.N = {a: 0.0 for a in self.arms}   # 折扣计数 N_a
        self.mu = {a: 0.0 for a in self.arms}  # 折扣和，对应折扣均值

        self.t = 0
        self.last_metric = metric0
        self.best_metric = metric0 if metric0 is not None else float('-inf')
        self.stall = 0  # 连续无进展轮数

    def _ucb_scores(self):
        """内部：计算每个臂的折扣 UCB 分数"""
        scores = {}
        t_log = math.log(self.t + 1) if self.t > 0 else 0.0
        for a in self.arms:
            mean = self.mu[a] / (self.N[a] + 1e-9) if self.N[a] > 0 else 0.0
            bonus = self.c * math.sqrt(t_log / (self.N[a] + 1))
            scores[a] = mean + bonus

        # 若连续 H 轮无进展，提升非当前最优臂的探索权重
        if self.stall >= self.H:
            best = max(scores, key=scores.get)
            for a in self.arms:
                if a != best:
                    scores[a] += self.boost
        return scores

    def choose(self):
        """选择当前轮要采用的臂（clean / AutoML）"""
        self.t += 1
        scores = self._ucb_scores()
        return max(scores, key=scores.get)

    def observe(self, arm, metric_t, cost_t):
        """
        上报本轮执行结果，更新折扣统计和停滞检测
        r_t = (metric_t - metric_{t-1}) / cost_t
        cost_t 为本轮消耗的时间等代价
        """
        if self.last_metric is None:
            r_t = 0.0
        else:
            r_t = (metric_t - self.last_metric) / max(cost_t, 1e-9)
        self.last_metric = metric_t

        # 折扣更新 N_a, μ_a
        self.N[arm] = self.gamma * self.N[arm] + 1.0
        self.mu[arm] = self.gamma * self.mu[arm] + r_t

        # 停滞检测：若 metric 没有刷新历史最好，则累计停滞轮数
        if self.best_metric is None or metric_t > self.best_metric:
            self.best_metric = metric_t
            self.stall = 0
        else:
            self.stall += 1

    def state(self):
        """返回当前内部状态，可用于调试或日志"""
        return {
            "t": self.t,
            "N": dict(self.N),
            "mu": dict(self.mu),
            "best_metric": self.best_metric,
            "stall": self.stall,
        }


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


def sample_base_grad(X, y, fraction, X_train_grad_norm, iter):
    print(iter*int(fraction * len(X_train_grad_norm)), (iter+1)*int(fraction * len(X_train_grad_norm)))
    selected_samples = np.argsort(-X_train_grad_norm)[iter*int(fraction * len(X)):(iter+1)*int(fraction * len(X))]
    return selected_samples


def pick_families_from_leaderboard(automl, max_fams=3):
    try:
        lb = automl.leaderboard(detailed=True)  # pandas.DataFrame, 已按分数排序
    except Exception:
        return None
    col = "type"
    vocab = [
         "adaboost", "bernoulli_nb", "decision_tree", "extra_trees",
        "gaussian_nb", "gradient_boosting", "k_nearest_neighbors",
        "lda", "libsvm_svc", "mlp", "multinomial_nb",
        "qda", "sgd"
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
    include = dmco_lib if r == 0 else (families if families else dmco_lib)

    if base_time < 30:
        base_time = 30
    if per_run >= base_time / 2:
        per_run = int(base_time / 2) - 1

    automl = AutoSklearnRegressor(
        time_left_for_this_task=base_time,
        per_run_time_limit=per_run,
        n_jobs=1,
        # include={
        #     'classifier': ["lda", "sgd"],
        # },
        include={
            'regressor': include,
        },
        memory_limit=4096,  # 物理内存足够可设 None
        resampling_strategy="holdout",
        resampling_strategy_arguments={"train_size": 0.8},
    )

    try:
        automl.fit(X, y)
        # 从本轮结果提取赢家家族，供下一轮缩搜
        best_est = refit_best_sklearn_model(automl, X, y)
        grads_X, grad_norm, used_idx = compute_gradient_scores(
            best_est,
            X,
            y,
            mode="auto",  # 也可以强制 "analytic" / "numeric"
            eps=1e-4,
            max_samples=None  # 或者比如 500，先对 500 个样本算
        )
        fam_next = pick_families_from_leaderboard(automl, max_fams=max(2, 3 - (r // 2)))

        if fam_next:
            families = fam_next
        return automl, families, grads_X, grad_norm
    except Exception as e:
        # 3) 兜底：保证可用
        model = SVR()
        model.fit(X, y) # 从本轮结果提取赢家家族，供下一轮缩搜
        fam_next = pick_families_from_leaderboard(automl, max_fams=max(2, 3 - (r // 2)))
        if fam_next:
            families = fam_next
        return model, families


def fit_svm(X_train, y_train):
    model = SVR()
    model.fit(X_train, y_train)
    return model


def cal_metric(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metric = mean_squared_error(y_test, y_pred)
    return metric


def total():
    dataset = "soilmoisture"
    error = "_system_outliers_30"
    # data = pd.read_csv('data/'+dataset+'/'+dataset+'_25.csv')
    data = pd.read_csv('data/inject_all/' + dataset + error + '.csv')
    data = pd.read_csv('data/' + dataset + '/' + dataset + '_25.csv')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices, test_size=0.2, random_state=4)
    X_test = clean(X_test, test_indices, np.ones(len(test_indices), dtype=bool), 'data/'+dataset+'/'+dataset+'_data_vectorized.csv')
    max_ml_num = 10000
    if len(X_train) < max_ml_num:
        max_ml_num = len(X_train)
        print("big~~")
        # 2. 配置
        times = [i * 30 for i in range(1, 11)]
        ours = []

        min_sample_num = 500
        for time in times:
            print(f"Time : {time}")
            clean_fraction = time / np.max(times)
            dmco_model_fraction = np.max([clean_fraction, min_sample_num / len(X_train)])
            print("dmco_model_fraction: ", dmco_model_fraction)
            big_data_fraction = max_ml_num / len(X_train)

            random_idx_model = sample_random(X_train.copy().reset_index(), y_train, big_data_fraction)

            metric = 0.6
            mab = CostAwareMAB(metric0=metric, gamma=0.9, c=1.0, H=3, boost=0.5)
            flag = True
            loss_idx = sample_base_loss(X_train.copy().reset_index(), y_train, dmco_model_fraction)
            X_loss = X_train.copy().reset_index(drop=True).iloc[loss_idx]
            y_loss = y_train.copy().reset_index(drop=True).iloc[loss_idx]
            dmco_model, families, X_train_grad, X_train_norm = automl(X_loss, y_loss, None, 0, int(time / 2))

            clean_iter = 0
            for t in range(1, 3):  # 模拟 20 轮自适应选择
                arm = mab.choose()
                # 这里用一个简单“环境”来模拟两种策略的效果和代价：
                if arm == 'AutoML':  # 'AutoML'
                    # AutoML：提升略保守，但代价较低
                    print(t, ": auto")
                    if flag:
                        loss_idx = sample_base_loss(X_train.copy().reset_index(), y_train, dmco_model_fraction)
                        X_loss = X_train.copy().reset_index(drop=True).iloc[loss_idx]
                        X_random = X_train.copy().reset_index(drop=True).iloc[random_idx_model]
                    else:
                        loss_idx = sample_base_loss(X_train_cleaned.copy().reset_index(), y_train, dmco_model_fraction)
                        X_loss = X_train_cleaned.copy().reset_index(drop=True).iloc[loss_idx]
                        X_random = X_train_cleaned.copy().reset_index(drop=True).iloc[random_idx_model]
                    y_loss = y_train.copy().reset_index(drop=True).iloc[loss_idx]
                    dmco_model, families, X_train_grad, X_train_norm = automl(X_loss, y_loss, None, 0, int(time / 2))
                    y_random = y_train.copy().reset_index(drop=True).iloc[random_idx_model]
                    dmco_model.refit(X_random, y_random)
                    dmco_metric = cal_metric(dmco_model, X_test, y_test)
                    # ours.append(dmco_metric)
                else:
                    # 清洗：效果提升稍大，但代价偏高
                    print(t, ": clean")
                    if flag:
                        grad_idx = sample_base_grad(X_train.copy().reset_index(), y_train, clean_fraction / 2, X_train_norm, 0)
                        mask = np.zeros(len(train_indices), dtype=bool)
                        mask[grad_idx] = True
                        X_train_cleaned = clean(X_train.copy(), train_indices, mask, 'data/' + dataset + '/' + dataset + '_data_vectorized.csv')
                    else:
                        grad_idx = sample_base_grad(X_train_cleaned.copy().reset_index(), y_train, clean_fraction / 2, X_train_norm, clean_iter)
                        mask = np.zeros(len(train_indices), dtype=bool)
                        mask[grad_idx] = True
                        X_train_cleaned = clean(X_train_cleaned.copy(), train_indices, mask, 'data/' + dataset + '/' + dataset + '_data_vectorized.csv')

                    clean_iter += 1
                    flag = False
                    dmco_model = fit_svm(X_train_cleaned, y_train)
                    dmco_metric = cal_metric(dmco_model, X_test, y_test)

                metric_t = metric + dmco_metric
                cost = 1.0
                mab.observe(arm, metric_t, cost)
                metric = metric_t

                print(f"Round {t:02d} | arm={arm:6s} | metric={metric:.4f} | cost={cost:.2f} | stall={mab.stall}")

            loss_idx = sample_base_loss(X_train_cleaned.copy().reset_index(), y_train, dmco_model_fraction)
            X_loss = X_train_cleaned.copy().reset_index(drop=True).iloc[loss_idx]
            y_loss = y_train.copy().reset_index(drop=True).iloc[loss_idx]
            dmco_model, families, X_train_grad, X_train_norm = automl(X_loss, y_loss, families, 1, int(time / 2))

            dmco_model.refit(X_train_cleaned, y_train)
            dmco_metric = cal_metric(dmco_model, X_test, y_test)
            ours.append(dmco_metric)
            print("dmco: ", dmco_metric)

        f = open('log/' + dataset + error + '_res.txt', 'w')
        f.write(str(ours))
        f.close()


total()


# if __name__ == "__main__":
#     # 假设当前验证集指标 metric 从 0.60 起步
#     metric = 0.60
#
#     mab = CostAwareMAB(metric0=metric,
#                        gamma=0.9,
#                        c=1.0,
#                        H=3,
#                        boost=0.5)
#
#     for t in range(1, 4):  # 模拟 20 轮自适应选择
#         arm = mab.choose()
#         # 这里用一个简单“环境”来模拟两种策略的效果和代价：
#         if arm == 'clean':
#             # 清洗：效果提升稍大，但代价偏高
#             delta = random.gauss(0.015, 0.01)  # metric 提升
#             cost = 2.0                         # 时间消耗
#         else:  # 'AutoML'
#             # AutoML：提升略保守，但代价较低
#             delta = random.gauss(0.010, 0.007)
#             cost = 1.0
#
#         metric_t = metric + delta
#         mab.observe(arm, metric_t, cost)
#         metric = metric_t
#
#         print(f"Round {t:02d} | arm={arm:6s} | metric={metric:.4f} | cost={cost:.2f} | stall={mab.stall}")
