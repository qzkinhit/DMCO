import time
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from autosklearn.classification import AutoSklearnClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. 加载 Adult 数据集
adult = fetch_openml(name='adult', version=2, as_frame=True)
X = adult.data
y = adult.target

# 2. 简单处理：将标签二分类
y = (y == '>50K').astype(int)  # '>50K'标记为1，其他为0

# 3. 区分数值型和类别型特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# 4. 建立预处理器
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

# 5. 建立整个Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

# Load dataset
# X, y = load_wine(return_X_y=True)
X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Finer-grained fractions
fractions = np.arange(0.1, 1.001, 0.03)
svc_accuracies, automl_accuracies = [], []
svc_times, automl_times = [], []

for frac in fractions:
    print(f"\n===== Sampling fraction: {frac:.2f} =====")
    n_samples = int(len(X_full_train) * frac)
    indices = np.random.choice(len(X_full_train), n_samples, replace=False)
    # X_train, y_train = X_full_train[indices], y_full_train[indices]
    X_train, y_train = X_full_train, y_full_train

    # --- SVC ---
    # start_svc = time.time()
    # model_svc = SVC()
    # model_svc.fit(X_train, y_train)
    # pred_svc = model_svc.predict(X_test)
    # acc_svc = accuracy_score(y_test, pred_svc)
    # time_svc = time.time() - start_svc
    # svc_accuracies.append(acc_svc)
    # svc_times.append(time_svc)
    start_svc = time.time()
    model.fit(X_train, y_train)
    pred_svc = model.predict(X_test)
    acc_svc = accuracy_score(y_test, pred_svc)
    time_svc = time.time() - start_svc
    svc_accuracies.append(acc_svc)
    svc_times.append(time_svc)

    # --- AutoML ---
    model_automl = AutoSklearnClassifier(
        time_left_for_this_task=int(300*frac),
        seed=42
    )
    start_auto = time.time()
    model_automl.fit(X_train, y_train)
    pred_auto = model_automl.predict(X_test)
    acc_auto = accuracy_score(y_test, pred_auto)
    time_auto = time.time() - start_auto
    automl_accuracies.append(acc_auto)
    automl_times.append(time_auto)

    print(f"SVC    - acc: {acc_svc:.4f}, time: {time_svc:.2f}s")
    print(f"AutoML - acc: {acc_auto:.4f}, time: {time_auto:.2f}s")

# Plot Accuracy
plt.figure(figsize=(10, 6))
fractions = [frac * 300 for frac in fractions]
plt.plot(fractions, svc_accuracies, label='SVC (default)', marker='o')
plt.plot(fractions, automl_accuracies, label='AutoML', marker='o')
plt.title('Accuracy vs. Time Cost')
plt.xlabel('Time Cost')
plt.ylabel('Test Accuracy')
plt.xticks(fractions)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot Time
plt.figure(figsize=(10, 6))
plt.plot(fractions, svc_times, label='SVC Time', marker='o')
plt.plot(fractions, automl_times, label='AutoML Time', marker='o')
plt.title('Training Time vs. Training Data Fraction')
plt.xlabel('Training Data Fraction')
plt.ylabel('Training Time (s)')
plt.xticks(fractions)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
