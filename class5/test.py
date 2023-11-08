import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

src_dir = r'4'
train_data = pd.read_csv(os.path.join(src_dir, '实验4_train_data.csv'))


def numeritative(data, cols):
    for col in cols:
        data[col] = pd.factorize(data[col])[0]
    return data


train_data = numeritative(train_data, ['Geography', 'Gender'])  # 调用


def discretizate(data, cols_dict):
    for col, value in cols_dict.items():
        for i in data.index:
            if data.loc[i, col] < value['low']:
                data.loc[i, col] = 0
            elif value['low'] <= data.loc[i, col] < value['middle']:
                data.loc[i, col] = 1
            elif value['middle'] <= data.loc[i, col] < value['high']: \
                    data.loc[i, col] = 2
            else:

                data.loc[i, col] = 3  # high
    return data


describe = train_data.describe()
discretizate_dict = {
    'CreditScore': {'low': describe.loc['25%', 'CreditScore'],
                    'middle': describe.loc['50%', 'CreditScore'],
                    'high': describe.loc['75%', 'CreditScore']},
    'Age': {'low': describe.loc['25%', 'Age'],
            'middle': describe.loc['50%', 'Age'],
            'high': describe.loc['75%', 'Age']},
    'Balance': {'low': describe.loc['25%', 'Balance'],
                'middle': describe.loc['50%', 'Balance'],
                'high': describe.loc['75%', 'Balance']},
    'EstimatedSalary': {'low': describe.loc['25%', 'EstimatedSalary'],
                        'middle': describe.loc['50%', 'EstimatedSalary'],
                        'high': describe.loc['75%', 'EstimatedSalary']}}
train_data = discretizate(train_data, discretizate_dict)

train_data = train_data[train_data.columns[3:-1]]


def select(data, target):
    ones = sum(data[target])  # 类别1
    zeros = len(data[target]) - ones  # 类别
    data.sample(frac=1.0, replace=True)  # 随机打乱
    data = pd.concat([data.loc[data[target] == 0].iloc[:min(ones, zeros)],
                      data.loc[data[target] == 1].iloc[:min(ones, zeros)]])
    return data.sample(frac=1.0)  # 再次打乱


train_data = select(train_data, 'Exited')
train_data.to_csv(os.path.join(src_dir, 'train data final.csv'), index=False)

from sklearn.model_selection import train_test_split

data = pd.read_csv(os.path.join(src_dir, '实验4_train_data_final.csv'))
# DecisionTreeClassifierg训练、评估过程需要MatrixLike  ArrayLike类型
data_array = np.array(data)
features = data_array[:, :-1]
target = data_array[:, -1]
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=10)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

from sklearn.tree import DecisionTreeClassifier

model_1 = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_split=100)
model_1.fit(x_train, y_train)
DecisionTreeClassifier(max_depth=6, min_samples_split=200)
y_pred = model_1.predict(x_test)
score = model_1.score(x_test, y_test)
scores = []
scores.append(score)

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=1)
plt.figure()
# figsize=(8，8)
plt.plot(fpr, tpr, linewidth=2, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Guess')
plt.xlabel('False Positive Rate', fontsize=12)
# 横坐标为假阳性率 (False Positive Rate，FPR)，FP是N个负
plt.ylabel('True Positive Rate', fontsize=12)
# 纵坐标为真阳性率 (True Positive Rate，TPR
plt.ylim(0, 1.01)  # 边界范围plt.xlim(0，1.01)#边界范围
plt.title("ROC Curve", fontsize=14)
plt.legend(loc=4, fontsize=12)
plt.savefig(os.path.join(src_dir, 'ROC Curve.png'))
plt.show()  # 显示作图结果

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
cms = []
cms.append(cm)


def draw(scores, cms):
    for i, (score, cm) in enumerate(zip(scores, cms)):
        plt.figure()  # figsize=(10，10)
        plt.matshow(cm, fignum=0, cmap=plt.cm.Oranges)
        plt.colorbar()  # 颜色标签
        for x in range(len(cm)):  # 数据标签
            for y in range(len(cm)):
                plt.annotate(cm[x, y], xy=(x, y), fontsize=12,
                             horizontalalignment='center',
                             verticalalignment='center')
                plt.xlabel('True Class', fontsize=12)
                plt.ylabel('Pred Class', fontsize=12)
        plt.title(f"Confusion Matrix (score={score:.2%})", fontsize=14)
        plt.savefig(os.path.join(src_dir, f'Confusion Matrix matshow model {i}.png'))
        plt.show()


draw(scores, cms)

import seaborn as sns

f, ax = plt.subplots()
sns.heatmap(cm, annot=True, ax=ax)
ax.set_xlabel('Pred Class', fontsize=12)
ax.set_ylabel('Ture Class', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.savefig(os.path.join(src_dir, f'Confusion Matrix_matshow_model_1.png'))
plt.show()

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
plt.plot(recall, precision)
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('P-R Curve', fontsize=14)
plt.savefig(os.path.join(src_dir, 'P-R Curve.png'))
plt.show()

from sklearn.model_selection import StratifiedKFold


def kfold_cv(ks):
    def draw(x, y, k, avg, max, min):
        plt.plot(x, y)
        plt.ylim(0.5, 1.0)
        plt.xlabel('k Rounds')
        plt.ylabel('True Rate')
        plt.title(
            f'KFold Cross Validation\n(k={k},tol={int(x_test.shape[0] / k)}, ' + f'avg={avg:.2%},max={max:.2%},min={min:.2%})')
        plt.savefig(os.path.join(src_dir, f'KFold Cross Validation_{k}.png'))
        plt.show()

    for k in ks:
        skfold = StratifiedKFold(n_splits=k, shuffle=False)
        model = DecisionTreeClassifier(
            criterion='gini', max_depth=6, min_samples_split=200
        )
        y = []
        m = []
        for train_index, test_index in skfold.split(features, target):
            skfold_x_train = features[train_index]
            skfold_y_train = target[train_index]
            model.fit(skfold_x_train, skfold_y_train)
            skfold_x_test = features[test_index]
            skfold_y_test = target[test_index]
            y.append(model.score(skfold_x_test, skfold_y_test))
            m.append(confusion_matrix(skfold_y_test,
                                      model.predict(skfold_x_test), labels=[0, 1]))
        avg, max, min = np.mean(y), np.max(y), np.min(y)
        draw(range(1, k + 1), y, k, avg, max, min)
        scores.append(avg)
        tp = []
        fn = []
        fp = []
        tn = []
        for tmp in m:
            tp.append(tmp[0][0])
            fn.append(tmp[0][1])
            fp.append(tmp[1][0])
            tn.append(tmp[1][1])
        cm = np.array([[int(np.mean(tp)), int(np.mean(fn))],
                       [int(np.mean(fp)), int(np.mean(tn))]])
        cms.append(cm)


from sklearn import svm

svc = svm.SVC()
svc.fit(x_train, y_train)
scores.append(svc.score(x_test, y_test))
cms.append(confusion_matrix(y_test, svc.predict(x_test)))
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 11), random_state=1)
mlp.fit(x_train, y_train)

ks = [5, 10, 15, 20]
kfold_cv(ks)
