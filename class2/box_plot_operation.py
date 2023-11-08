import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 联合可视化
import matplotlib.image as mpimg

"""
本案例中搜集了一个表，表中各列从左至右代表着锅炉的可调参数以及锅炉的工况等数据，
其中有些属性对预测蒸汽量不产生影响，还有一些噪声以及不够规范的数据，
因此需要进行数据预处理以获得高质量的数据。
"""


def del_abnormal(value, i, upper, lower):
    """替换异常值"""
    if lower[i] <= value <= upper[i]:
        return value
    else:
        # 异常值
        return np.nan


def set_abnormal(value, i, upper, lower):
    """替换异常值"""
    # 均值
    mean = train_data_with.mean().values

    if lower[i] <= value <= upper[i]:
        return value
    else:
        # 异常值（替换为均值）
        return mean[i]


def process_1(s, func):
    """模拟箱型图法处理异常"""
    q025 = train_data_with.quantile(0.25).values
    q075 = train_data_with.quantile(0.75).values
    q050 = train_data_with.quantile(0.50).values

    q_dis = q075 - q025
    upper_1 = q075 + q_dis * 1.5
    lower_1 = q025 - q_dis * 1.5

    print("处理前:", train_data_with.count())

    for i in range(len(train_data_with.columns) - 1):
        train_data_with.iloc[:, i] = train_data_with.iloc[:, i].apply(
            func, i=i, upper=upper_1, lower=lower_1
        )
    print("处理中:", train_data_with.count())
    train_data_with.dropna(inplace=True)
    print("处理后:", train_data_with.count())

    train_data_with.to_csv(os.path.join(
        src_dir, f'train_data_process_1_{s}.txt'), index=False
                           )

    plt.figure(figsize=(12, 9))
    # 去掉target列
    plt.boxplot(
        train_data_with.iloc[:, :-1],
        labels=["V" + str(i) for i in range(38)]
    )
    plt.title('box_method')
    plt.savefig(os.path.join(src_dir, 'train_data_box_method_{0}.jpg'.format(s)))
    plt.show()


def process_2(s, func):
    """3o原则统计法处理异常值"""
    mean = train_data_with.mean().values
    std = train_data_with.std().values
    upper_2 = mean + std * 3
    lower_2 = mean - std * 3

    # 异常值处理
    print("处理前:", train_data_with.count())

    for i in range(len(train_data_with.columns) - 1):
        train_data_with.iloc[:, i] = train_data_with.iloc[:, i].apply(
            func, i=i, upper=upper_2, lower=lower_2
        )
    print("处理中:", train_data_with.count())
    train_data_with.dropna(inplace=True)
    print("处理后:", train_data_with.count())

    train_data_with.to_csv(os.path.join(
        src_dir, f'train_data_process_2_{s}.txt'), index=False
                           )

    plt.figure(figsize=(12, 9))

    # 去掉target列
    plt.boxplot(
        train_data_with.iloc[:, :-1],
        labels=["V" + str(i) for i in range(38)]
    )
    plt.title('3o method')
    plt.savefig(os.path.join(src_dir, 'train_data_3o_method_{0}.jpg'.format(s)))
    plt.show()


# 联合绘图
def draw_4box():
    pictures = [mpimg.imread(os.path.join(src_dir, sub))
                for sub in os.listdir(src_dir) if sub.endswith('.jpg')]
    plt.figure(figsize=(24, 18))
    plt.subplot(221)
    plt.imshow(pictures[0])
    plt.subplot(222)
    plt.imshow(pictures[1])
    plt.subplot(223)
    plt.imshow(pictures[2])
    plt.subplot(224)
    plt.imshow(pictures[3])
    plt.savefig(os.path.join(src_dir, f'train_data_contrast.png'))
    plt.show()


if __name__ == '__main__':
    src_dir = r'./dataset'
    train_data = pd.read_csv(os.path.join(src_dir, 'train_noHeader.txt'))
    # print(train_data)
    #
    with open(os.path.join(src_dir, 'train_noHeader.txt'), 'r+') as file:
        content = f'{",".join(["V" + str(i) for i in range(38)] + ["target"])}\n{file.read()}'
    with open(os.path.join(src_dir, 'train_withHeader.txt'), 'w+') as file:
        file.write(content)
    train_data_with = pd.read_csv(os.path.join(src_dir, 'train_withHeader.txt'))

    print(train_data_with.shape)
    print(train_data_with.describe())
    print(train_data_with.head())

    print(train_data_with.count())
    # 数据被存储在DataFrame对象中，Pandas的dropna()方法对缺失数据进行处理
    train_data_with.dropna(inplace=True)
    print(train_data_with.count())

    """处理重复值"""
    train_data_with.drop_duplicates(inplace=True)
    print(train_data_with.count())

    """处理异常值"""
    # 箱型图可视化
    plt.figure(figsize=(12, 9))
    # 去掉target列
    plt.boxplot(train_data_with.iloc[:, :-1],
                labels=["V" + str(i) for i in range(38)])
    plt.title('original')
    plt.savefig(os.path.join(src_dir, 'train_data_box_original.png'))
    plt.show()

    """
    异常值处理的方法通常包括：
    a)	删除含有异常值的数据行；
    b)	将异常值视为缺失值，使用缺失值处理的方法来处理；
    c)	使用包含异常值的数据列的数据的平均值来填充异常值；
    d)	不做处理。
    """
    # 模拟箱型图方法（异常值设空）
    process_1('del', del_abnormal)
    # 3o原则统计法（异常值设空）
    process_2('del', del_abnormal)
    # 模拟箱型图方法（异常值均值重置）
    process_1('mean', set_abnormal)
    # 3o原则统计法（异常值均值重置）
    process_2('mean', set_abnormal)

    # 联合可视化
    draw_4box()
