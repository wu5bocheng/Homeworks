from sklearn.linear_model import LogisticRegression
import seaborn as sns
from hyperopt import fmin, Trials, hp, anneal
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import random
import pandas as pd
import numpy as np

'''
纯自定义指标
'''
# 读取本地csv文件到dataframe
dfUser = pd.read_csv("data/user.csv", encoding='utf-8')
dfRel = pd.read_csv("data/relation.csv", encoding='utf-8')
# loc函数取targetUser==1的用户的id
targetUsers = dfUser.loc[dfUser['targetUser'] == 1, ['id']]

# 关联二度好友
df1 = pd.merge(targetUsers, dfRel, left_on=['id'], right_on=['userID'])
df2 = pd.merge(df1, dfRel, left_on=['friendID'], right_on=['userID'])
# 用group by计算共同好友数，形成分层索引；通过reset_index将series转为dataframe
# df3 = df2.groupby(['id','friendID_y'])['friendID_x'].count().reset_index()
friendNum = dfRel.groupby(['userID'])['friendID'].count().reset_index()
friendNum.rename(columns={'friendID': 'friendNum'}, inplace=True)
df3 = pd.merge(df2, friendNum, left_on='friendID_x', right_on='userID')
df3.drop(columns=['userID_x', 'userID_y', 'userID'], inplace=True)
# 将结算结果改名为userID, ffID, CN
df3.rename(columns={'id': 'userID', 'friendID_y': 'ffID',
           'friendID_x': 'CN_id', 'friendNum': 'CN_friendNum'}, inplace=True)
df3["AA_index"] = 1/df3["CN_friendNum"]*np.log(df3["CN_friendNum"])
df4 = df3.groupby(['userID', 'ffID'])['AA_index'].sum().reset_index()
# 关联好友表，用于标记二度好友是否为好友
df6 = pd.merge(df4, dfRel, how='left', left_on=[
               'userID', 'ffID'], right_on=['userID', 'friendID'])
# 将标记改为0,1，并改名为isFriend，数据类型改为int
df6.loc[df6['friendID'].isnull(), ['friendID']] = 0
df6.loc[df6['friendID'] > 0, ['friendID']] = 1
df6.rename(columns={'friendID': 'isFriend'}, inplace=True)
df6['isFriend'] = df6['isFriend'].astype(int)
# 构造输出结果
team = 'wbc_try_is_fake'
batch = 11
df7 = df6.loc[df6['isFriend'] == 0, ['userID', 'ffID', "AA_index"]]
df7 = df7[df7['userID'] != df7['ffID']]

# 选出前10推荐
df7['rn'] = df7.groupby('userID')['AA_index'].rank(
    ascending=False, method='first')
df7.loc[df7['rn'] <= 15].to_csv("main.csv", index=None)
df7 = df7.loc[df7['rn'] <= 10]
df7.insert(0, 'teamID', team)
df7.insert(1, 'batch', batch)
df7.rename(columns={'AA_index': 'sIndex'}, inplace=True)  # 结果矩阵

'''
抽样并预处理
'''

# 读取本地csv文件到dataframe
dfUser = pd.read_csv("data/user.csv")
dfRel = pd.read_csv("data/relation.csv")
dfRel["is_friend"] = 1
# -*- coding:utf-8 -*-
# 概率入样
target_num = 1926400  # 样本数量
user_list = list(dfUser["id"].values)
n = int(target_num/len(user_list))
for userID in user_list:
    not_friends = list(dfUser["id"].values)
    friend_list = list(dfRel[dfRel["userID"] == userID]["friendID"].values)
    friend_list.append(userID)
    for i in friend_list:
        try:
            not_friends.remove(i)
        except:
            print(userID, i)
    sample_list = random.sample(not_friends, n)
    temp = pd.DataFrame(sample_list, columns=["friendID"])
    temp["userID"] = userID
    temp["is_friend"] = 0
    dfRel = pd.concat([dfRel, temp], ignore_index=True)

# 和user表格拼接并处理
data = pd.merge(dfRel, dfUser, left_on=['userID'], right_on=['id'])
data = pd.merge(data, dfUser, left_on=['friendID'], right_on=['id'])
data["fName"] = (data["fName_x"] == data["fName_y"])
data["homeProvince"] = (data["homeProvince_x"] == data["homeProvince_y"])
data["homeCity"] = (data["homeCity_x"] == data["homeCity_y"])
data["location"] = (data["location_x"] == data["location_y"])
data["privacy_mixture"] = data["privacy_x"]*data["privacy_y"]
data["gender"] = (data["gender_x"] == data["gender_y"])
data["nVisits_mixture"] = data["nVisits_x"]*data["nVisits_y"]
data["nShortPosts_mixture"] = data["nShortPosts_x"]*data["nShortPosts_y"]
data["nAlbums_mixture"] = data["nAlbums_x"]*data["nAlbums_y"]
data["nShares_mixture"] = data["nShares_x"]*data["nShares_y"]
data["nPosts_mixture"] = data["nPosts_x"]*data["nPosts_y"]
data["nLike_mixture"] = data["nLike_x"]*data["nLike_y"]
data["mobileUser"] = (data["mobileUser_x"] == data["mobileUser_y"])
data["starUser_mixture"] = (data["starUser_x"]*data["starUser_y"])
data.drop(columns=['id_x', 'targetUser_x', 'fName_x', 'gender_x', 'starUser_x', 'mobileUser_x', 'homeProvince_x', 'homeCity_x', 'nFriends_x', 'nLike_x', 'nPosts_x', 'nShares_x', 'nAlbums_x', 'nShortPosts_x', 'nVisits_x', 'location_x', 'privacy_x', 'timeline_x',
          'id_y', 'targetUser_y', 'fName_y', 'gender_y', 'starUser_y', 'mobileUser_y', 'homeProvince_y', 'homeCity_y', 'nFriends_y', 'nLike_y', 'nPosts_y', 'nShares_y', 'nAlbums_y', 'nShortPosts_y', 'nVisits_y', 'location_y', 'privacy_y', 'timeline_y'], inplace=True)

# 计算结点相似度指标
dfUser = pd.read_csv("data/user.csv")
dfRel = pd.read_csv("data/relation.csv")
dfTruth = pd.read_csv("data/truth.csv")
dfX = pd.read_csv("result.csv")
friendNum = dfRel.groupby(['userID'])['friendID'].count().reset_index()
friendNum_dic = dict(zip(friendNum['userID'], friendNum['friendID']))
user_list = dfX.loc[:, "userID":"friendID"]
Rel_matrix = pd.merge(dfRel, dfRel, left_on=['friendID'], right_on=['userID'])
Rel_matrix = pd.merge(Rel_matrix, friendNum,
                      left_on='friendID_x', right_on='userID')
# Rel_matrix.drop(columns=['userID_y','userID'],inplace = True)
# 将结算结果改名为userID, ffID, CN
# Rel_matrix.rename(columns={'userID_x':'userID','friendID_y':'ffID','friendID_x':'CN_id','friendID':'CN_friendNum'}, inplace=True)
Rel_matrix["AA_index"] = 1/Rel_matrix["friendID"] * \
    np.log(Rel_matrix["friendID"])
Rel_matrix = Rel_matrix.groupby(['userID_x', 'friendID_y'])[
    'AA_index'].sum().reset_index()
Rel_matrix.rename(columns={'userID_x': 'userID',
                  'friendID_y': 'ffID'}, inplace=True)
Rel_matrix.to_csv("AA_index.csv", index=None)
dfX = pd.merge(dfX, Rel_matrix, left_on=["userID", "friendID"], right_on=[
               "userID", "ffID"], how="left")
dfX.to_csv("result.csv", index=None)

# 缺失值填补

# 预定义变量和进行全局设置
workpath = 'F:/python/f_python/py_sjwlywbfx_notes/competition1/'  # 工作路径
pd.set_option("display.max_columns", None)  # 让DataFrame对象在打印时可以显示所有列，以便预览
# 读取数据
df1 = pd.read_csv(workpath+'target_matrix.csv', sep=",", encoding='utf-8')
df1 = df1.astype('float64')  # 统一数据类型
print(df1.head(10))
print(df1.describe())
print(df1.isnull().agg('sum', axis=0))
del df1['nShares_mixture']  # nShares_mixture列的缺失值太多，因此直接删除这一列
# 填补缺失值
column_num = ['nVisits_mixture', 'nShortPosts_mixture', 'nAlbums_mixture', 'nPosts_mixture',
              'nLike_mixture']  # , 'nShares_mixture'
column_category = ['fName', 'homeProvince', 'homeCity', 'location',
                   'privacy_mixture', 'gender', 'mobileUser', 'starUser_mixture']
# KNN填补太慢，放弃
'''
imputer = KNNImputer(n_neighbors=5)
imputed = imputer.fit_transform(df1)
df1_imputed = pd.DataFrame(imputed, columns=df1.columns)
for column in list(column_num): # 数值型变量采用KNN方法填补
    print(column)
    df1[column] = df1_imputed[column]
'''
for column in list(column_num):  # 数值型变量采用中位数填补
    median_val = df1[column].median()
    df1[column].fillna(median_val, inplace=True)
for column in list(column_category):  # 类别型变量采用众数填补
    mode_val = df1[column].mode()[0]
    df1[column].fillna(mode_val, inplace=True)
print(df1.describe())
print(df1.isnull().agg('sum', axis=0))
df1.loc[:10, :].to_csv(
    workpath+'target_matrix_cleaned_first10.csv', encoding='utf-8-sig', index=False)
df1.to_csv(workpath+'target_matrix_cleaned.csv',
           encoding='utf-8-sig', index=False)
print('已成功保存预处理后的数据！')
print(df1.head(10))


'''
模型训练
'''

# 因为代码区使用了Prittier插件，所以换行逗号引号等都是标准格式，没有个人特色

# PLT中文支持
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class mnist_comparation:
    def __init__(
        self,
        model,
        params_GridSearchCV=None,
        params_hyperopt_space=None,
        test_size=50000,
        train_size=50000,
        random_state=1,
    ):
        self.params_GridSearchCV = params_GridSearchCV
        self.params_hyperopt_space = params_hyperopt_space
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_process()
        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("clf", model)]
        )  # 标准化后和选择的模型一起封装到pipeline中

    def data_process(self):
        mnist = pd.read_csv("../input/renren/X_matrix_cleaned.csv")
        X, y = mnist.loc[:, ['AA_index', 'fName', 'homeProvince', 'homeCity',
                             'location', 'privacy_mixture', 'gender', 'nVisits_mixture',
                             'nShortPosts_mixture', 'nAlbums_mixture', 'nPosts_mixture',
                             'nLike_mixture', 'mobileUser', 'starUser_mixture']], mnist["is_friend"]
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            train_size=self.train_size,
            stratify=y,
            random_state=self.random_state,
        )  # 分层抽样，保证每个数字都有足够多的训练样本

    def plot_confusion_matrix(self, title, name):  # 绘制混淆矩阵并用热图格式展示
        self.y_pred = self.pipeline.predict(self.X_test)
        conf_matrix = pd.DataFrame(
            confusion_matrix(self.y_test, self.y_pred),
            index=range(0, 2),
            columns=range(0, 2),
        )
        fig, ax = plt.subplots(figsize=(20, 15))
        sns.heatmap(conf_matrix, annot=True, annot_kws={
                    "size": 15}, cmap="Blues")
        plt.title("{}".format(title), fontsize=30)
        plt.ylabel("True label", fontsize=20)
        plt.xlabel("Predicted label", fontsize=20)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig("pic/{}.jpg".format(name), dpi=150, bbox_inches="tight")
        plt.show()

    # hyperopt参数调优目标函数

    def object(self, params):
        pipeline = self.pipeline
        pipeline.set_params(**params)
        res = np.mean(cross_val_score(
            pipeline, self.X_train, self.y_train, cv=5))
        return -res

    # hyperopt参数调优具体过程

    def hyperopt_train(self, max_evals=30, algo=anneal.suggest):
        trials = Trials()
        # max_evals是搜索次数
        best_params = fmin(
            fn=self.object,
            space=self.params_hyperopt_space,
            algo=algo,
            max_evals=max_evals,
            trials=trials,
        )
        print("hyperopt调参得到最优参数: \n", best_params)
        return best_params

    def gridseachCV_train(self, scoring="accuracy", cv=5, n_jobs=-1):
        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.params_GridSearchCV,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
        )
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_
        print("GridSeachCV调参得到最优参数: \n", best_params)
        return best_params

    def best_model_result(self, model_name, best_params=None):
        pipeline = self.pipeline
        if best_params != None:
            pipeline.set_params(**best_params)
        pipeline.fit(self.X_train, self.y_train)
        y_pred = pipeline.predict(self.X_test)
        print(model_name + "模型准确率: ", accuracy_score(self.y_test, y_pred))
#         print("最优参数下得到的混淆矩阵为: ")
#         self.plot_confusion_matrix(title=model_name, name=model_name)


if __name__ == "__main__":
    # 引入五个模型分别计算对比
    # Section 1 : 分类决策树模型
    # 由于组成了pipeline,所以变量名前面要加上组成clf__
    params_GridSearchCV = {
        "clf__C": [5],
        "clf__penalty": ["l2"],
        "clf__solver": ["saga"]
    }
    # 因为选用的hyperopt最小化的目标函数效果不理想, 得到的调参结果效果较差
    params_hyperopt_space = {
        "clf__max_depth": hp.choice("clf__max_depth", range(10, 21)),
        "clf__penalty": ["l1", "l2"],
    }
    logistic_regression = mnist_comparation(
        model=LogisticRegression(),
        params_GridSearchCV=params_GridSearchCV,
        params_hyperopt_space=params_hyperopt_space,
        train_size=3500000,
        test_size=100000
    )
    best_params = {"clf__C": 5, "clf__penalty": "l2"}
    grid_search_param = logistic_regression.gridseachCV_train()
#     hyperopt_param = logistic_regression.hyperopt_train()
    logistic_regression.best_model_result(
        "logistic Regression", best_params=best_params)
