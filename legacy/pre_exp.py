import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from AutoClean import AutoClean
import time
from autosklearn.classification import AutoSklearnClassifier

# 1. 加载 Adult 数据
adult = fetch_openml(name='adult', version=2, as_frame=True)
X = adult.data
y = adult.target

# 标签编码为0/1
y = (y == '>50K').astype(int)

# 合并特征和标签
data = X.copy()
data['target'] = y

# 2. 配置
subsample_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  # 清洗比例
ac = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
results = []
times = []


# 3. 不同比例下循环
i = 0
for fraction in subsample_fractions:
    print(f"Processing subsample fraction: {fraction}")
    start_time = time.time()

    if fraction == 0.0:
        # 不进行清洗
        data_update = data.copy()
    else:
        # 抽取子集并清洗
        subsample_idx = np.random.choice(data.index, size=int(len(data) * fraction), replace=False)
        subsample_data = data.loc[subsample_idx]

        ac = AutoClean(
            subsample_data,
            mode='auto'
        )
        cleaned_subsample = ac.output

        # 还原清洗后的子集
        data_update = data.copy()
        data_update.loc[cleaned_subsample.index] = cleaned_subsample

    # 拆开特征和标签
    X_final = data_update.drop('target', axis=1)
    y_final = data_update['target']

    # 数据预处理
    numeric_features = X_final.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_final.select_dtypes(include=['object']).columns.tolist()

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

    preprocessor.fit(X_final)
    X_final = preprocessor.fit_transform(X_final)

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
    times.append(time.time() - start_time)
    print("time :", time.time() - start_time)
    model_automl = AutoSklearnClassifier(
        time_left_for_this_task=int(300 - time.time()+start_time),
        seed=42
    )
    start_auto = time.time()
    model_automl.fit(X_train, y_train)
    pred_auto = model_automl.predict(X_test)
    acc_auto = accuracy_score(y_test, pred_auto)
    time_auto = time.time() - start_auto
    results.append(acc_auto)
    print("total time : ", time.time() - start_time)

    print("acc : ", acc_auto)
    i += 1

# 4. 绘图

plt.figure(figsize=(8, 6))
plt.plot(times, results, marker='o')
plt.xlabel('Time for Cleaning (%)')
plt.ylabel('Test acc')
plt.title('Effect of Cleaning and AutoML on SVM Performance')
plt.grid(True)
plt.show()
