from autogluon.tabular import TabularDataset, TabularPredictor

# 加载数据集
train_data = TabularDataset('path_to_train.csv')
test_data = TabularDataset('path_to_test.csv')

# 训练模型
predictor = TabularPredictor(label='target_column', problem_type='regression')
predictor.fit(train_data)

# 评估模型
leaderboard = predictor.leaderboard(test_data)

# 使用模型进行预测
predictions = predictor.predict(test_data)