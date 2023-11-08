import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
# 多重共线性方差感胀因子
from statsmodels.stats.outliers_influence import variance_inflation_factor
# PCA 主成分分析法
from sklearn.decomposition import PCA

# 在pycharm中安装TensorFlow后，使用Tensorflow1版本的操作，屏蔽tf2的操作
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

# 解决方法
import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

"""
1.数据分析
"""


# 查看train与test的同分布
def kdeplot(train_data, test_data, save_flag):
    """
    绘制核密度估计图
    :param save_flag: str
    """
    plt.figure(figsize=(4 * 5, 4 * 8))
    for i in range(38):
        ax = plt.subplot(8, 5, i + 1)
        ax = sns.kdeplot(train_data['V' + str(i)], color="Red", label='train')  # shade=True
        ax = sns.kdeplot(test_data['V' + str(i)], color="Blue", label='test')  # shade=True
        plt.legend()
    plt.savefig(os.path.join(src_dir, f'train_Vi与test_Vi的特征分布-{save_flag}.png'))
    # plt.show()


# 查看Vi与target字段的线性关系
def regplot(train_data, save_flag):
    """
    （单变量）线性回归模型拟合
    :param train_data:
    :param save_flag:
    :return:
    """
    plt.figure(figsize=(4 * 5, 4 * 8))
    for i in range(38):
        ax = plt.subplot(8, 5, i + 1)
        sns.regplot(x='V' + str(i), y='target', data=train_data, ax=ax,
                    scatter_kws={'marker': '.', 's': 3, 'alpha': 0.3},
                    line_kws={'color': 'k'})
    plt.savefig(os.path.join(src_dir, f'train_Vi与test_Vi的线性关系-{save_flag}.png'))
    # plt.show()


# 查看Vi与Vj的相关性
def heatmap_1(train_data, save_flag):
    """
    画出相关性热力图
    :param train_data:
    :param save_flag:
    :return:
    """
    train_corr = train_data.corr()
    plt.figure(figsize=(20, 20))
    ax = plt.subplot(1, 1, 1)
    sns.heatmap(data=train_corr, vmax=.8, square=True, annot=True, fmt='0.2f')  # 画热力图，annot显示相关系数
    plt.savefig(os.path.join(src_dir, f'train_data_Vi与Vj的相关性-{save_flag}.png'))
    # plt.show()


def heatmap_2(train_data, save_flag):
    """
    画出相关程度热力图
    :param train_data:
    :param save_flag:
    :return:
    """
    cols = train_data.columns.tolist()  # 列表头
    # 相关系数矩阵，即给出了任意两个变量之间的相关系数
    train_corr = train_data[cols].corr(method="spearman")
    # 构造与corr同维数矩阵(bool型)
    mask = np.zeros_like(train_corr, dtype=np.bool8)
    mask[np.triu_indices_from(mask)] = True  # 角分线右侧为True ----> 以下三角矩阵显示
    plt.figure(figsize=(20, 20))  # 指定绘图对象宽度和高度
    cmap = sns.diverging_palette(h_neg=220, h_pos=0,
                                 as_cmap=True)
    # 返回matplotlib colormap对象
    sns.heatmap(data=train_corr, mask=mask, cmap=cmap,
                square=True, annot=True, fmt='0.2f')  # 热力图(看两两相似度)
    plt.savefig(os.path.join(src_dir, f'train_data_Vi与Vj的相关程度-{save_flag}.png'))
    # plt.show()


def heatmap_3(train_data, save_flag, K=10):
    """
    查找和target相关系数最大的 K 个特征变量(Vi)
    查找目标
    K=10
    :param train_data:
    :param save_flag:
    :param K: 10
    :return:
    """
    # 相关系数矩阵
    train_corr = train_data.corr().abs()  # V37为负相关,'V37'，'V4，'V12'，'V3'，'V16'
    # 寻找 K个最相关的特征信息
    cols = train_corr.nlargest(K + 1, 'target')['target'].index  # 剔除target字段
    plt.figure(figsize=(10, 10))
    sns.heatmap(train_data[cols].corr(), annot=True, square=True)
    plt.title(f'MAX {K}: ' + ','.join(cols))
    plt.savefig(os.path.join(src_dir, f'train_dataVi与tar的相关程度_MAX{K}-{save_flag}.png'))
    # plt.show()


def heatmap_4(train_data, save_flag, threshold=0.5):
    """
    查找和target相关系数 >threshold 的特征变量(Vi)
    :param train_data:
    :param save_flag:
    :param threshold:
    :return:
    """
    train_corr = train_data.corr().abs()
    cols = train_corr[train_corr["target"] > threshold].sort_values(
        by='target', ascending=False).index
    plt.figure(figsize=(10, 10))
    sns.heatmap(train_data[cols].corr(), annot=True, cmap="RdYlGn")
    plt.title(f'threshold={threshold}:' + ''.join(cols))
    plt.savefig(os.path.join(src_dir, f'train_data_Vi与tar的相关程度_threshold{threshold}-{save_flag}.png'))
    # plt.show()


"""
2.数据处理
"""


def set_abnormal(value, i, upper, lower):
    """
    均值替换异常值
    执行dropna()时会因没有空行而无影响
    只对新的箱型图产生一次性的、新的影响
    :param value:
    :param i:
    :param upper:
    :param lower:
    :return:
    """
    global mean  # 均值
    if lower[i] <= value <= upper[i]:
        return value
    else:
        # 检测出异常值，置为所在列的平均值
        # 后续遇到dropna()时，无影响
        return mean[i]


def process(data, upper, lower):
    """
    异常处理
    :param data:
    :param upper:
    :param lower:
    :return:
    """
    features = data.columns[:-1]
    # 检测前 (copy()为副本)
    bef = data.copy()
    # 检测中 (忽视target列)
    for i in range(len(features)):
        data.iloc[:, i] = data.iloc[:, i].apply(
            set_abnormal, i=i, upper=upper, lower=lower)
    # 检测率(保留4位小数)
    check_rate = round(sum(sum(bef.values != data.values)) /
                       (bef.shape[0] * bef.shape[1]), 4)
    # 处理异常 (删除空、删除重复、自更新)
    # test的target为空
    # data.dropna(inplace=True)
    data = data.loc[data[features].dropna().index, :]
    data.drop_duplicates(inplace=True)
    # 处理率(保留4 位小数)
    process_rate = round(1 - max(data.count()) / max(bef.count()), 4)

    data.to_csv(os.path.join(
        src_dir, f'train+test_data_aft.txt'), index=False)
    return check_rate, process_rate


def check_error_values(data):
    """
    3o原则统计法处理异常值
    :param data:
    :return:
    """
    global mean
    std = data.std().values
    upper = mean + std * 3
    lower = mean - std * 3
    print('check rate:{0}\tprocess rate:{1}'
          .format(*process(data, upper, lower)))


# 剔除异分布字段：第2个模型数据集
def kdeplot_2(train_data, test_data, drop_cols):
    """
    绘制核密度估计图
    :param train_data:
    :param test_data:
    :param drop_cols:
    :return:
    """
    plt.figure(figsize=(4 * 5, 4 * 2))
    for i in range(len(drop_cols)):
        ax = plt.subplot(2, 5, i + 1)
        ax = sns.kdeplot(train_data[drop_cols[i]], color="Red", label='train')
        ax = sns.kdeplot(test_data[drop_cols[i]], color="Blue", label='test')
        plt.legend()

    plt.savefig(os.path.join(src_dir, f'train_test_drop_cols的特征分布.png'))
    # plt.show()


"""
3.划分数据集
"""


def get_dataset(data):
    """
    按7：3划分train_data
    :param data:
    :return: train_data, test_data
    """
    data = data.sample(frac=1.0)
    cut_idx = int(round(0.7 * data.shape[0]))
    train_data, test_data = data.iloc[:cut_idx], data.iloc[cut_idx:]

    return train_data, test_data


def draw(rmse, r2, data_name, model_name, lr):
    """
    绘画双轴折线图
    :param rmse:
    :param r2:
    :param data_name:
    :param model_name:
    :param lr:
    :return:
    """
    # 画布
    fig, ax1 = plt.subplots()
    ax1.plot(list(range(len(rmse))), rmse, color='red', label='rmse', ls='-.')
    plt.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(list(range(len(r2))), r2, color='green', label='r^2', ls='--')
    plt.title(f'{data_name}_{model_name}_{lr}')
    plt.legend(loc="upper right")

    plt.savefig(os.path.join(src_dir, f'{data_name}_{model_name}_{lr}.png'))
    # plt.show()


"""
4.构建模型
"""


# 线性回归模型
def model_1(train_data, test_data, data_name, model_name, lr=0.005, run_times=1000):
    # 数据集
    features = train_data.columns[:-1]

    x_train = train_data.loc[:, features]
    y_train = train_data['target']
    x_test = test_data.loc[:, features]
    y_test = test_data['target']

    # 转型
    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    y_train = np.mat(y_train)

    # 定义W b

    # x_train 的形状是 (38,)，这意味着它是一个一维数组。
    # 在这种情况下，x_train.shape[1] 将会引发一个 IndexError，
    # 因为一维数组只有一个维度，索引 1 超出了范围。

    # print(x_train.shape[1])
    W = tf.Variable(tf.zeros([x_train.shape[1], 1]))
    b = tf.Variable(tf.zeros([1, 1]))

    # 线性回归模型 y = W * x + b
    y = tf.matmul(x_train, W) + b

    # 损失函数
    loss = tf.reduce_mean(tf.square(y - y_train))
    # 梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    # 训练
    rmse_list = []
    r2_list = []
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for step in range(run_times):
        sess.run(optimizer)
        if step % 100 == 0:
            y_pred = tf.matmul(x_test, W) + b
            y_pred = tf.cast(y_pred, tf.float64)
            y_test = tf.cast(y_test, tf.float64)
            y_pred = sess.run(y_pred).flatten()
            y_test = sess.run(y_test)
            length = len(y_pred)
            error_sum = tot = reg = 0
            mean = np.mean(y_test)
            for i in range(0, length):
                error_sum += (y_test[i] ** 2)
                tot += (y_test[i] - mean) ** 2
                reg += (y_pred[i] - mean) ** 2
            rmse = np.sqrt(error_sum / length)
            r2 = reg / tot
            rmse_list.append(rmse)
            r2_list.append(r2)

    draw(rmse_list, r2_list, data_name, model_name, lr)


# 逻辑回归模型
def model_3(train_data, test_data, data_name, model_name, lr=0.005, run_times=1000):
    # 数据集
    features = train_data.columns[:-1]

    x_train = train_data.loc[:, features]
    y_train = train_data['target']
    x_test = test_data.loc[:, features]
    y_test = test_data['target']
    # 转型
    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)
    y_train = np.mat(y_train)
    # 定义W b
    W = tf.Variable(tf.zeros([x_train.shape[1], 1]))
    b = tf.Variable(tf.zeros([1, 1]))

    # 逻辑回归模型 y = 1 / e ^ (-W * x + b)
    y = 1 / (1 + tf.exp(-tf.matmul(x_train, W) + b))
    # 损失函数
    loss = tf.reduce_mean(- y_train.reshape(-1, 1) * tf.log(y) - (1 - y_train.reshape(-1, 1)) * tf.log(1 - y))
    # 梯度下降优化器
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    # 训练
    rmse_list = []
    r2_list = []
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for step in range(run_times):
        sess.run(optimizer)
        if step % 100 == 0:
            y_pred = (1 / (1 + tf.exp(-tf.matmul(x_test, W) + b))) * (y_test.max() - y_test.min()) + y_test.min()
            y_pred = tf.cast(y_pred, tf.float64)
            y_test = tf.cast(y_test, tf.float64)
            y_pred = sess.run(y_pred).flatten()
            y_test = sess.run(y_test)
            length = len(y_pred)
            error_sum = tot = reg = 0
            mean = np.mean(y_test)
            for i in range(0, length):
                error_sum += (y_pred[i] - y_test[i]) ** 2
                tot += (y_test[i] - mean) ** 2
                reg += (y_pred[i] - mean) ** 2
            rmse = np.sqrt(error_sum / length)
            r2 = reg / tot
            rmse_list.append(rmse)
            r2_list.append(r2)
    draw(rmse_list, r2_list, data_name, model_name, lr)


# 神经网络模型
def model_4(train_data, test_data, data_name, model_name, lr=0.005, run_times=1000):
    def add_layer(inputs, in_size, out_size, activate_func=None):
        """
        网络层
        :param inputs:
        :param in_size:
        :param out_size:
        :param activate_func:
        :return:
        """
        Weights = tf.cast(tf.Variable(
            tf.random_normal([in_size, out_size])), tf.float64)
        biases = tf.cast(tf.Variable(tf.zeros([1, out_size]) + 0.1), tf.float64)
        outputs = tf.matmul(inputs, Weights) + biases
        if activate_func:
            return activate_func(outputs)
        return outputs

    # 数据集
    features = train_data.columns[:-1]

    x_train = train_data.loc[:, features]
    y_train = train_data['target']
    x_test = test_data.loc[:, features]
    y_test = test_data['target']
    # 转型
    y_train = np.mat(y_train).T

    # 传入层
    xs = tf.placeholder(tf.float64, [None, len(features)])
    ys = tf.placeholder(tf.float64, [None, 1])

    # 隐藏层
    layer_hidden = add_layer(xs, len(features), len(features) // 3 * 2,
                             activate_func=tf.nn.relu)
    # 输出层
    layer_pred = add_layer(layer_hidden, len(features) // 3 * 2, 1,
                           activate_func=None)
    # loss 表达式
    loss = tf.reduce_mean(
        tf.reduce_sum(tf.square(ys - layer_pred), reduction_indices=[1]))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    rmse_list = []
    r2_list = []
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(run_times):
        sess.run(optimizer, feed_dict={xs: x_train, ys: y_train})
        if i % 100 == 0:
            y_pred = sess.run(layer_pred, feed_dict={xs: x_test})
            y_pred = tf.cast(y_pred, tf.float64)
            y_test = tf.cast(y_test, tf.float64)
            y_pred = sess.run(y_pred).flatten()
            y_test = sess.run(y_test)

            length = len(y_pred)
            error_sum = reg = tot = 0
            mean = np.mean(y_test)
            for i in range(0, length):
                error_sum = error_sum + (y_pred[i] - y_test[i]) ** 2
                tot += (y_test[i] - mean) ** 2
                reg += (y_pred[i] - mean) ** 2
            rmse = np.sqrt(error_sum / length)
            r2 = reg / tot
            rmse_list.append(rmse)
            r2_list.append(r2)
    draw(rmse_list, r2_list, data_name, model_name, lr)


if __name__ == '__main__':
    # 数据路径
    src_dir = r"./dataset"

    # [实验 1]
    # 处理前
    train_data_bef = pd.read_csv(os.path.join(src_dir, 'train_withHeader.txt'))
    test_data_bef = pd.read_csv(os.path.join(src_dir, 'test_withHeader.txt'))
    # 处理后
    train_data = pd.read_csv(os.path.join(src_dir, 'train_data_trible_methed_set_abnormal.txt'))
    test_data = pd.read_csv(os.path.join(src_dir, 'test_data_trible_methed_set_abnormal.txt'))

    # 查看train与test的同分布
    kdeplot(train_data_bef, test_data_bef, 'bef')
    kdeplot(train_data, test_data, 'aft')

    # 查看Vi与target字段的线性关系
    regplot(train_data_bef, 'bef')
    regplot(train_data, 'aft')

    # 查看Vi与Vj的相关性
    heatmap_1(train_data_bef, 'bef')
    heatmap_1(train_data, 'aft')
    # 只显示下三角矩阵
    heatmap_2(train_data_bef, 'bef')
    heatmap_2(train_data, 'aft')

    heatmap_3(train_data_bef, 'bef')
    heatmap_3(train_data, 'aft')

    heatmap_4(train_data_bef, 'bef')
    heatmap_4(train_data, 'aft')

    # 拼接
    data = pd.concat([train_data_bef, test_data_bef], ignore_index=True)

    # 异常处理
    mean = data.mean().values
    check_error_values(data)

    # 对除target列以外的字段归一化
    features = data.columns[:-1]
    data[features] = (data[features] - data[features].min()) / \
                     (data[features].max() - data[features].min())

    # 重新提取train_data、test_data
    train_data_scaler = data.loc[data['target'].isnull().apply(
        lambda x: not x)]
    test_data_scaler = data.loc[data['target'].isnull()].drop(
        'target', axis=1)

    # 根“特征分布”筛选
    drop_cols = ['V5', 'V9', 'V17', 'V21', 'V22', 'V23', 'V35']
    # 确认train_data与test_data在drop_cols上的分布
    kdeplot_2(train_data_scaler, test_data_scaler, drop_cols)

    # 删除分布有异的字段
    train_data_drop = train_data_scaler.drop(drop_cols, axis=1)
    test_data_drop = test_data_scaler.drop(drop_cols, axis=1)

    # 筛选相关性 >0.5 的字段
    # 考虑到负相关的情况
    data_corr = data.corr().abs()
    slt_cols = data_corr[data_corr["target"] > 0.5].sort_values(
        by='target', ascending=False).index[1:]

    train_data_slt = train_data_scaler.loc[:, slt_cols]
    train_data_slt['target'] = train_data_scaler.loc[:, 'target']
    # test没有target
    test_data_slt = train_data_scaler.loc[:, slt_cols]

    train_data_slt.describe(), test_data_slt.describe()

    # 多重共线性方差感胀因子
    features = data.columns[:-1]
    X = np.matrix(data[features])
    # 方差膨胀因子(Variance Inflation Factor， VIF)
    vif_cols = [features[i] for i in range(X.shape[1]) if variance_inflation_factor(X, i) <= 20]
    train_data_vif = train_data_scaler.loc[:, vif_cols]
    train_data_vif['target'] = train_data_scaler.loc[:, 'target']
    test_data_vif = test_data_scaler.loc[:, vif_cols]
    print(train_data_vif.describe(), test_data_vif.describe())

    # 降低维度:第5个数据集
    # PCA处理降维
    pca = PCA(n_components=0.8)
    features = data.columns[:-1]
    pca.fit(data[features])

    train_data_pca = pd.DataFrame(pca.transform(train_data_scaler[features]))
    train_data_pca['target'] = train_data_scaler.loc[:, 'target']

    # test没有target
    test_data_pca = pd.DataFrame(pca.transform(test_data_scaler[features]))
    train_data_pca.describe(), train_data_pca.describe()

    print(train_data_scaler.shape, test_data_scaler.shape, train_data_pca.shape, test_data_pca.shape)
    print(train_data_pca.columns, test_data_pca.columns)

    # 对比分析模型效果
    for lr in [0.001, 0.005, 0.010, 0.050, 0.100]:
        train_tmp, test_tmp = get_dataset(train_data_scaler)
        model_1(train_tmp, test_tmp, 'train_data_scaler', 'linear', lr)  # 异常与归一
        model_3(train_tmp, test_tmp, 'train_data_scaler', 'logic', lr)
        model_4(train_tmp, test_tmp, 'train_data_scaler', 'network', lr)

        train_tmp, test_tmp = get_dataset(train_data_drop)
        model_1(train_tmp, test_tmp, 'train_data_drop', 'linear', lr)  # 删除异分布
        model_3(train_tmp, test_tmp, 'train_data_drop', 'logic', lr)
        model_4(train_tmp, test_tmp, 'train_data_drop', 'network', lr)

        train_tmp, test_tmp = get_dataset(train_data_slt)
        model_1(train_tmp, test_tmp, 'train_data_slt', 'linear', lr)  # 相关性>0.5
        model_3(train_tmp, test_tmp, 'train_data_slt', 'logic', lr)
        model_4(train_tmp, test_tmp, 'train_data_slt', 'network', lr)

        train_tmp, test_tmp = get_dataset(train_data_vif)
        model_1(train_tmp, test_tmp, 'train_data_vif', 'linear', lr)  # 共线性<20
        model_3(train_tmp, test_tmp, 'train_data_vif', 'network', lr)
        model_4(train_tmp, test_tmp, 'train_data_vif', 'network', lr)

        train_tmp, test_tmp = get_dataset(train_data_pca)
        model_1(train_tmp, test_tmp, 'train_data_pca', 'linear', lr)  # 特征降维0.8
        model_3(train_tmp, test_tmp, 'train_data_pca', 'logic', lr)
        model_4(train_tmp, test_tmp, 'train_data_pca', 'network', lr)
