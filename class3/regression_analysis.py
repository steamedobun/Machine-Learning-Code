import findspark
import logging
import sys
import os
import time

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.sql import SparkSession
# 决策树 模块
from pyspark.ml.regression import DecisionTreeRegressor
# 梯度提升树 模块
from pyspark.ml.regression import GBTRegressor
# 随机森林 模块
from pyspark.ml.regression import RandomForestRegressor
import matplotlib.pyplot as plt

findspark.init()

# java home
os.environ['JAVA_HOME'] = "C:/Program Files/Java/jre1.8.0_331"

# sql
spark = SparkSession.builder.appName("big-mart-sales").getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

src_dir = r"./dataset/"
train_df = spark.read.csv(os.path.join(
    src_dir, 'train_model.csv'), header=True, inferSchema=True)
test_df = spark.read.csv(os.path.join(
    src_dir, 'test_model.csv'), header=True, inferSchema=True)

feature_cols = train_df.columns

# 移除target列索引
feature_cols.remove('Item_Outlet_Sales')
# 除target以外所有列索引转为特征向量，整合为feature列
vectorAssembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

# 根据特征向量feature列索引，提取train_data中相关列，并转为单向量列
train_df = vectorAssembler.transform(train_df)
train_df = train_df.select(['features', 'Item_Outlet_Sales'])
# 根据特征向量feature列索引，提取test_data中相关列，并转为单向量列
test_df = vectorAssembler.transform(test_df)
test_df = test_df.select(['features', 'Item_Outlet_Sales'])

"""线性回归模型"""
model_1 = LinearRegression(featuresCol='features',
                           labelCol='Item_Outlet_Sales')

model_1.getFeaturesCol(), model_1.getPredictionCol, model_1.getLabelCol()
# 默认最大迭代次、Ridge回归? ELasticNet回归?
model_1.getMaxIter(), model_1.getRegParam(), model_1.getElasticNetParam()

# 设置超参数
evaluator = RegressionEvaluator(
    predictionCol="prediction", labelCol="Item_Outlet_Sales", metricName="rmse")

paramGrid = ParamGridBuilder().addGrid(model_1.maxIter, [10, 50, 10]).build()

# 验证对象、验证方法 (网格搜索法)、参数估计器、5交叉验证 (默认3折)
cv = CrossValidator(estimator=model_1, estimatorParamMaps=paramGrid,
                    evaluator=evaluator, numFolds=5)

cv_model_1 = cv.fit(train_df)
print(cv_model_1.avgMetrics)

# 保存模型
# model_dir = os.path.join(src_dir, f'cv_model_1_{int(time.time())}')
# cv_model_1.write().save(model_dir)
# cv_model_1 = CrossValidatorModel.read().load(model_dir)

prediction_1 = cv_model_1.transform(test_df)
prediction_1.select(model_1.getFeaturesCol(),
                    model_1.getLabelCol(),
                    model_1.getPredictionCol()
                    ).show(n=5)

# 提取参数估计器的评估指标的值
model_1_rmse = evaluator.evaluate(prediction_1)
model_1_r2 = RegressionEvaluator(predictionCol="prediction",
                                 labelCol="Item_Outlet_Sales",
                                 metricName="r2").evaluate(prediction_1)
print(model_1_rmse, model_1_r2)

print("线性回归模型完成")

"""Ridge 回归模型"""
model_2 = LinearRegression(featuresCol='features', labelCol='Item_Outlet_Sales')
evaluator = RegressionEvaluator(predictionCol='prediction', labelCol='Item_Outlet_Sales',
                                metricName='rmse')
paramGrid = ParamGridBuilder() \
    .addGrid(model_2.regParam, [0.1, 0.3, 0.5, 0.8]) \
    .build()

# 交叉验证
cv = CrossValidator(estimator=model_2, estimatorParamMaps=paramGrid,
                    evaluator=evaluator, numFolds=5)
# 模型象数与好参数
print(model_2.getMaxIter(), model_2.getRegParam(), model_2.getElasticNetParam())
# 训练
cv_model_2 = cv.fit(train_df)
# 预测
prediction_2 = cv_model_2.transform(test_df)
# 评估
model_2_rmse = evaluator.evaluate(prediction_2)
model_2_r2 = RegressionEvaluator(predictionCol="prediction",
                                 labelCol="Item_Outlet_Sales",
                                 metricName="r2").evaluate(prediction_2)
print("Ridge回归模型完成")


"""Lasso回归模型"""
model_3 = LinearRegression(featuresCol='features',
                           labelCol='Item_Outlet_Sales',
                           elasticNetParam=1)

evaluator = RegressionEvaluator(predictionCol="prediction",
                                labelCol="Item_Outlet_Sales",
                                metricName="rmse")

paramGrid = ParamGridBuilder() \
    .addGrid(model_3.regParam, [0.1, 0.3, 0.5, 0.8]) \
    .build()

cv = CrossValidator(estimator=model_3,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)

cv_model_3 = cv.fit(train_df)
prediction_3 = cv_model_3.transform(test_df)
model_3_rmse = evaluator.evaluate(prediction_3)
model_3_r2 = RegressionEvaluator(predictionCol="prediction",
                                 labelCol="Item_Outlet_Sales",
                                 metricName="r2").evaluate(prediction_3)
print("Lasso回归模型完成")


"""Elastic Net 回归模型"""
model_4 = LinearRegression(featuresCol='features',
                           labelCol='Item_Outlet_Sales')

evaluator = RegressionEvaluator(predictionCol="prediction",
                                labelCol="Item_Outlet_Sales",
                                metricName="rmse")
# L1+L2正刚化: model_4.regParam + model_4.elasticNetParam
# .addGrid(model_4.maxIter. [0, 50, 100])
paramGrid = ParamGridBuilder() \
    .addGrid(model_4.regParam, [0.1, 0.3, 0.5, .8]) \
    .addGrid(model_4.elasticNetParam, [0.2, 0.4, 0.6, 0.8]) \
    .build()
cv = CrossValidator(estimator=model_4,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)

# 训练
cv_model_4 = cv.fit(train_df)
# 预测
prediction_4 = cv_model_4.transform(test_df)

model_4_rmse = evaluator.evaluate(prediction_4)
model_4_r2 = RegressionEvaluator(predictionCol="prediction",
                                 labelCol="Item_Outlet_Sales",
                                 metricName="r2").evaluate(prediction_4)
print("ElasticNet回归模型完成")


"""决策树回归模型"""
model_5 = DecisionTreeRegressor(featuresCol='features',
                                labelCol='Item_Outlet_Sales')
evaluator = RegressionEvaluator(predictionCol="prediction",
                                labelCol="Item_Outlet_Sales",
                                metricName="rmse")
paramGrid = ParamGridBuilder() \
    .addGrid(model_5.maxDepth, [5, 10, 15]) \
    .addGrid(model_5.minInstancesPerNode, [100, 150, 2001]) \
    .build()
cv = CrossValidator(estimator=model_5,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)

cv_model_5 = cv.fit(train_df)
prediction_5 = cv_model_5.transform(test_df)
model_5_rmse = evaluator.evaluate(prediction_5)
model_5_r2 = RegressionEvaluator(predictionCol="prediction",
                                 labelCol="Item_Outlet_Sales",
                                 metricName="r2").evaluate(prediction_5)
print("决策树回归模型完成")


"""梯度提升树回归模型"""
model_6 = GBTRegressor(featuresCol='features',
                       labelCol='Item_Outlet_Sales',
                       maxIter=10)
evaluator = RegressionEvaluator(predictionCol="prediction",
                                labelCol="Item_Outlet_Sales",
                                metricName="rmse")
paramGrid = ParamGridBuilder() \
    .addGrid(model_6.maxDepth, [5, 10, 15]) \
    .addGrid(model_6.minInstancesPerNode, [100, 150, 200]) \
    .build()
cv = CrossValidator(estimator=model_6,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)

cv_model_6 = cv.fit(train_df)
prediction_6 = cv_model_6.transform(test_df)

model_6_rmse = evaluator.evaluate(prediction_6)
model_6_r2 = RegressionEvaluator(predictionCol="prediction",
                                 labelCol="Item_Outlet_Sales",
                                 metricName="r2").evaluate(prediction_6)
print("梯度提升树回归模型完成")


"""随机森林回归模型"""
model_7 = RandomForestRegressor(featuresCol='features',
                                labelCol='Item_Outlet_Sales',
                                minInstancesPerNode=108)
evaluator = RegressionEvaluator(predictionCol="prediction",
                                labelCol="Item_Outlet_Sales",
                                metricName="rmse")
paramGrid = ParamGridBuilder() \
    .addGrid(model_7.maxDepth, [5, 8, 10]) \
    .addGrid(model_7.numTrees, [200, 400]) \
    .build()
cv = CrossValidator(estimator=model_7,
                    estimatorParamMaps=paramGrid,
                    evaluator=evaluator,
                    numFolds=5)
# 训练
cv_model_7 = cv.fit(train_df)
# 预测
prediction_7 = cv_model_7.transform(test_df)

model_7_rmse = evaluator.evaluate(prediction_7)
model_7_r2 = RegressionEvaluator(predictionCol="prediction",
                                 labelCol="Item_Outlet_Sales",
                                 metricName="r2").evaluate(prediction_7)
print("随机森林回归模型完成")


"""模型评估"""
x_labels = ['lr', 'rig', 'las', 'esc', 'dt', 'gbdt', 'rf']
rmse = [model_1_rmse, model_2_rmse, model_3_rmse, model_4_rmse, model_5_rmse, model_6_rmse, model_7_rmse]
r2 = [model_1_r2, model_2_r2, model_3_r2, model_4_r2, model_5_r2, model_6_r2, model_7_r2]
x1, x2 = [], []
width = 0.3
for i in range(len(x_labels)):
    x1.append(i)
    x2.append(i + width)

# 设置左侧Y轴对应的figure
fig, ax1 = plt.subplots()
ax1.set_ylabel('rmse')
ax1.bar(x1, rmse, width=width, color='green', align='edge', tick_label=x_labels, label='rmse')
ax1.set_xticklabels(ax1.get_xticklabels())  # 设置共用的x轴
ax1.set_ylim(min(rmse) - 10, max(rmse) + 10)
plt.legend(loc='upper left')

# 设当石侧Y输对应figure
ax2 = ax1.twinx()
ax2.set_ylabel('r^2')
ax2.bar(x2, r2, width=width, color='blue', align='edge', tick_label=x_labels, label='r^2')
ax2.set_ylim(min(r2) - 0.02, max(r2) + 0.02)

# 紧凑布局
plt.tight_layout()
plt.grid()
plt.legend(loc='upper right')
plt.savefig(os.path.join(src_dir, 'seven_model.png'))
plt.show()
