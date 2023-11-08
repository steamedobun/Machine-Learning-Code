import os.path
import pandas as pd
# Scikit-learn机器学习库
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import datetime

if __name__ == '__main__':
    """数据源"""
    src_dir = r'./dataset'
    train_ds = os.path.join(src_dir, 'train.csv')
    test_ds = os.path.join(src_dir, 'test.csv')

    train_data = pd.read_csv(train_ds)
    test_data = pd.read_csv(test_ds)
    # 标记数据记录来源
    train_data['source'] = 'train'
    test_data['source'] = 'test'
    data = pd.concat([train_data, test_data], ignore_index=True)
    # 数据格式
    print(train_data.shape, test_data.shape, data.shape)
    # 头5行尾5行
    print(data.head(5))
    print(data.tail(5))
    # 只针对数值型
    print(data.describe())

    """数据探索"""
    # 每列中缺失值个数
    data.apply(lambda x: sum(x.isnull()))
    # 商品类型Item_Type有限种
    print(data['Item_Type'].drop_duplicates())
    # LF Low Fat reg Regular 统一值处理
    print(data['Item_Fat_Content'].drop_duplicates())
    # 商店面积（存在大量缺失值）
    print(data['Outlet_Size'].drop_duplicates())

    """商品重量<->同类商品平均值"""
    item_weight_isnull = data['Item_Weight'].isnull()
    # 同类商品平均值
    item_avg_weight = data.pivot_table(
        values='Item_Weight', index='Item_Identifier'
    )
    print(item_avg_weight.head(5))
    # 均值补全
    data.loc[item_weight_isnull, 'Item_Weight'] = \
        data.loc[item_weight_isnull, 'Item_Identifier'].apply(
            lambda x: item_avg_weight.loc[x]
        )
    # 验证
    sum(data['Item_Weight'].isnull())

    """商品面积<->商品类型"""
    Outlet_Size_isnull = data['Outlet_Size'].isnull()
    # 按商品类型分组，求众数
    outlet_size_mode = data.groupby('Outlet_Type')['Outlet_Size'].apply(
        lambda x: x.mode()[0]
    )
    print(outlet_size_mode)
    data.loc[Outlet_Size_isnull, 'Outlet_Size'] = \
        data.loc[Outlet_Size_isnull, 'Outlet_Type'].apply(
            lambda x: outlet_size_mode[x]
        )
    sum(data['Outlet_Size'].isnull())

    """修正异常值"""
    """曝光度<->错误值<->均值替代"""
    # 筛选含有异常值布尔索引
    item_visibility_iszero = (data['Item_Visibility'] == 0)
    # 同一商品曝光均值透视图
    item_visibility_avg = data.pivot_table(
        values='Item_Visibility', index='Item_Identifier'
    )
    # 均值替换：对应取出透视表的Item_Visibility均值
    data.loc[item_visibility_iszero, 'Item_Visibility'] = \
        data.loc[item_visibility_iszero, 'Item_Identifier'].apply(
            lambda x: item_visibility_avg.loc[x]
        )
    print(data['Item_Visibility'].describe())

    """商品脂肪含量<->缩写/简写<->统一标记"""
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(
        {'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'}
    )
    print(data['Item_Fat_Content'].unique())

    """商品平均曝光率"""
    """
    item_visibility_avg = data.pivot_table(
        values='Item_Visibility', index='Item_Identifier'
    )
    """
    # 按行处理
    data['Item_Visibility_MeanRatio'] = \
        data.apply(lambda x: x['Item_Visibility'] /
                             item_visibility_avg.loc[x['Item_Identifier']], axis=1)
    print(data.head(5)[['Item_Visibility', 'Item_Visibility_MeanRatio']])

    """商品合并分类"""
    data['Item_Type_Combined'] = \
        data['Item_Identifier'].apply(lambda x: x[0:2]).map(
            {'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'}
        )
    print(data.head(5)[['Item_Identifier', 'Item_Type_Combined']])

    """商品脂肪含量"""
    data.loc[data['Item_Type_Combined'] ==
             "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"

    print(data.head(5)[['Item_Fat_Content', 'Item_Type_Combined']])

    """商品运营年数"""
    data['Outlet_Years'] = datetime.datetime.today().year - \
                           data['Outlet_Establishment_Year']

    print(data.head(5)[['Outlet_Establishment_Year', 'Outlet_Years']])

    """字符串数据类型 <-> 独热编码"""
    # LabelEncoder 对 Outlet_Identifier 进行编码
    le = LabelEncoder()
    data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
    # 进行LabelEncoder编码
    le = LabelEncoder()
    var_mod = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size',
               'Item_Type_Combined', 'Outlet_Type', 'Outlet']
    for i in var_mod:
        data[i] = le.fit_transform(data[i])

    # 独立热编码
    data = pd.get_dummies(data, columns=var_mod)
    print(data.head(10))

    # 删除列
    data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)
    # 根据source分割train和test
    train_data = data.loc[data['source'] == "train"]
    test_data = data.loc[data['source'] == "test"]
    # 删除（辅助列）
    train_data.drop(columns=['source'], inplace=True)
    # 删除（空列、辅助列）
    test_data.drop(columns=['Item_Outlet_Sales', 'source'], inplace=True)

    # 将数据保存为CSV文件
    train_data.to_csv(os.path.join(src_dir, 'train_aft_hot_code.csv'), index=False)
    test_data.to_csv(os.path.join(src_dir, 'test_aft_hot_code.csv'), index=False)
    # 删除（商品ID）列
    train_data.drop(columns=['Item_Identifier'], inplace=True)

    # 扰乱train_data顺序
    train_data = train_data.sample(frac=1.0)

    # 分割test和train
    cut_idx = int(round(0.3 * train_data.shape[0]))
    test_data, train_data = train_data.iloc[:cut_idx], train_data.iloc[cut_idx:]

    # 将数据保存为CSV文件
    train_data.to_csv(os.path.join(src_dir, 'train_data_model.csv'), index=False)
    test_data.to_csv(os.path.join(src_dir, 'test_data_model.csv'), index=False)

    """数据可视化"""
    train_data = pd.read_csv(os.path.join(src_dir, 'train_aft_hot_code.csv'))
    y = train_data.groupby('Outlet_Identifier')['Item_Outlet_Sales'].aggregate(func=sum)
    x = y.index
    plt.bar(x, y, tick_label=[_[3:] for _ in x])
    plt.xlabel('Outlet_x')
    plt.ylabel('Item_Sales')
    plt.show()

    x = train_data.sort_values(by='Outlet_Years', ascending=False)[['Outlet_Years', 'Outlet_Identifier']]
    y = train_data.groupby('Outlet_Identifier')['Item_Outlet_Sales'].aggregate(func=sum)

    tmp = x.merge(y, on=['Outlet_Identifier'], how='left')
    x, y = tmp['Outlet_Identifier'], tmp['Item_Outlet_Sales']

    plt.bar(x, y, tick_label=[_[3:]for _ in x])
    plt.xlabel('Outlet_x')
    plt.ylabel('Item_Sales')
    plt.show()

    """"""