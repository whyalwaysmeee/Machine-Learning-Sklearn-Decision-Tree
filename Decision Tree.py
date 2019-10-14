import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import graphviz


#初始化一个决策树分类器
#几个重要参数：
#criterion：分类依据，通常使用熵或者基尼系数
#random_state：设置分支中的随机模式参数，让模型稳定，在高维度数据集中效果明显
#splitter：也是用来控制随机选项，best会优先挑选更重要的特征进行分枝（即下面feature_importances_的结果），random会更加随机，会含有更多不必要信息，树也会更大
#剪枝参数：
#max_depth：最大深度限制,即树的深度不能超过设置值
#min_samples_leaf：叶子节点上的最小样本个数限制，即如果一个节点有一个子节点包含的样本个数达不到设置值，则不发生分枝
#Min_sample_split: 节点上的最小样本个数限制，即如果一个节点包含的样本个数达不到设置值，则不发生分枝
#max_features：特征个数限制，即超过限制个数的特征会被舍弃
#min_impurity_decrease：信息增益大小限制，即信息增益小于设置值时，不发生分枝
clf = tree.DecisionTreeClassifier(criterion="entropy",random_state = 30,splitter='random',
                                  max_depth = 3,
                                  min_samples_leaf = 10,
                                  min_samples_split = 50
                                  #min_impurity_decrease
                                  #max_features
                                  )
#加载数据集
wine = load_wine()
#查看特征
wine.feature_names
#查看标签
wine.target_names
#将数据集以图表形式呈现
#f = pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)
#将数据集分为训练集和测试集
xtrain,xtest,ytrain,ytest = train_test_split(wine.data,wine.target,test_size=0.3)
#将数据集放入分类器中并用fit匹配标签
clf = clf.fit(xtrain,ytrain)
#评估模型对测试集的拟合程度
score = clf.score(xtest,ytest)
print(score)
#评估模型对训练集的拟合程度
train_score = clf.score(xtrain,ytrain)
print(train_score)
#用中文标签替换
feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','花青素','颜 色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
#将生成的决策树以dot文件的形式输出，如果不加out_file也是默认输出dot文件但是不会保存下来，而是由下面的Source函数直接转化为gv文件
#使用Source及view方法打开生成的决策树不能够显示中文字符，所以我们采用out_file将dot文件保存，用记事本打开dot文件后修改器中的参数
#使其允许中文字符，最后在cmd中将dot文件转化为任意可以用作查看决策树的类型，如pdf、png等再来查看
dot_data = tree.export_graphviz(clf,out_file="tree5.dot"
                    ,feature_names=feature_name
                    ,class_names=["琴酒", "雪莉", "贝尔摩德"]
                    , filled=True
                    ,rounded=True)
#graph = graphviz.Source(dot_data)
#将生成的gv文件转化为默认的pdf文件并且打开
#graph.view()


#查看每个特征的重要程度，数值越大越重要
feature_importance = clf.feature_importances_

#确定最优的剪枝参数
#以max_depth为例
test = []
#以max_depth为1到10生成决策树
for i in range(10):
    clf=tree.DecisionTreeClassifier(max_depth=i+1
                                    ,criterion="entropy"
,random_state=30
                                    )
    clf=clf.fit(xtrain,ytrain)
    score_train = clf.score(xtrain, ytrain)
    #获取score并存入一个列表中
    test.append(score_train)
#绘图，以max_depth为横轴，score为纵轴
plt.plot(range(1,11), test, color='red', label='max_depth')
plt.legend()
plt.show()
#观察曲线，选取导数值变小幅度最大的一个点。因为从这个点以后score不会有很大的变化，而之前score变化显著