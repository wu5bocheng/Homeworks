# %% [markdown]
# # 网络总体特征

# %%
from sklearn.cluster import FeatureAgglomeration
from networkx.algorithms import community
from warnings import simplefilter
from sklearn.cluster import KMeans
from matplotlib import cm  # 节点颜色
import time
from numpy import random
from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
# 读取图
G = nx.DiGraph()
nodes = pd.read_csv("data/fb_nodes.csv")
edges = pd.read_csv("data/fb-pages-government.csv")
G.add_nodes_from(nodes["Id"])
G.add_edges_from(edges.apply(lambda x: tuple(x), axis=1).values.tolist())
# 转化无向图
UG = G.to_undirected()
UG.remove_edges_from(nx.selfloop_edges(UG))

# %%
# 计算网络拓扑结构
# 判断是否是连通图
print('FB-GOVERN网络图是否是弱连通图:', nx.is_weakly_connected(G))
print('FB-GOVERN网络图是否是强连通图:', nx.is_strongly_connected(G))
# 计算密度
print('FB-GOVERN网络图的密度:', nx.density(G))
# 富人俱乐部系数(转化成无向图)
# print('FB-GOVERN网络图的富人俱乐部系数:',nx.rich_club_coefficient(UG))

# 平均路径长度
print('FB-GOVERN网络图的平均路径长度:', nx.average_shortest_path_length(G))
# 直径（无向图，因为图不是强连通，所以有向图的直径为正无穷）
print('FB-GOVERN网络图的直径:', nx.diameter(UG))
# 平均聚类系数
print('FB-GOVERN网络图的平均聚类系数:', nx.average_clustering(G))
# 平均度
degree = pd.DataFrame(G.degree(), columns=["nodes", "degree"])
print('FB-GOVERN网络图的平均度:', np.mean(degree["degree"].values)/2)
# 度分布图
degree = nx.degree_histogram(G)
x = range(len(degree))  # 生成X轴序列，从1到最大度
y = [z/float(sum(degree))for z in degree]  # 将频次转化为频率，利用列表内涵
plt.figure(figsize=(7, 7))
plt.scatter(x, y, s=1, color=(0, 0, 0))  # 在双对坐标轴上绘制度分布曲线
plt.xlabel("Degree")
plt.ylabel("Density")
plt.savefig("degree_distribution.png", dpi=300)  # 显示图表

# 度同配系数
print('FB-GOVERN网络图的度同配系数:', nx.degree_assortativity_coefficient(G))

# %% [markdown]
# # 节点重要性度量

# %%
G.remove_edges_from(nx.selfloop_edges(G))  # 删除自连接边
DC = nx.degree(G)  # 度中心性
BC = nx.degree_centrality(G)  # 介数中心性
CC = nx.closeness_centrality(G)  # 接近中心性
KC = nx.core_number(G)  # k核值(k-core)
EC = nx.eigenvector_centrality(G, max_iter=500)  # 特征向量中心性
PR = nx.pagerank(G)  # PageRank算法

# %%


def get_top(dic, num, name):  # 通过name算法,获取的dic中value最大的num个(结点,中心度)对
    sorted_dic = sorted(dict(dic).items(), key=lambda x: x[1], reverse=True)[
        :num]  # 将dic强制转化为字典格式并按照value由大到小排序，取出前num个
    data = pd.DataFrame(sorted_dic, columns=['nodes', name])  # 转为dataframe格式
    return data.set_index(["nodes"])  # 将结点编号作为index


result = pd.concat([get_top(DC, 3, "DC"), get_top(BC, 3, "BC"), get_top(
    CC, 3, "CC"), get_top(KC, 3, "KC"), get_top(EC, 3, "EC"), get_top(PR, 3, "PR")])
result

# %% [markdown]
# # 好友推荐

# %%
# 构建无向图的边列表
data = pd.DataFrame(UG.edges, columns=["Source", "Target"])
Uedges = pd.concat([data, pd.DataFrame(
    data.loc[:, ["Target", "Source"]].values, columns=["Source", "Target"])], axis=0)

# %%
random.seed(1)
targetUsers = pd.DataFrame(random.choice(nodes["Id"], 30), columns=[
                           "Id"])  # 篇幅限制，只给30个随机用户推荐好友
# 关联二度好友
df1 = pd.merge(targetUsers, Uedges, left_on=['Id'], right_on=['Source'])
df2 = pd.merge(df1, Uedges, left_on=['Target'], right_on=['Source'])
friendNum = Uedges.groupby(['Source'])['Target'].count().reset_index()
friendNum.rename(columns={'Target': 'friendNum'}, inplace=True)
df3 = pd.merge(df2, friendNum, left_on='Target_x', right_on='Source')
df3.drop(columns=['Source_x', 'Source_y', 'Source'], inplace=True)
# 将结算结果改名为userID, ffID, CN
df3.rename(columns={'Id': 'userID', 'Target_y': 'ffID',
           'Target_x': 'CN_id', 'friendNum': 'CN_friendNum'}, inplace=True)
# df3包含了二度好友表和共同好友的好友数

# 计算AA_index
df3["AA_index"] = 1/np.log(df3["CN_friendNum"])
df4 = df3.groupby(['userID', 'ffID'])['AA_index'].sum().reset_index()
# 关联好友表，用于标记二度好友是否为好友,已经是好友的删除，推荐自己的删除
df6 = pd.merge(df4, Uedges, how='left', left_on=[
               'userID', 'ffID'], right_on=['Source', 'Target'])
df6 = df6.loc[df6['Target'].isnull(), ['userID', 'ffID', "AA_index"]]
df6 = df6[df6['userID'] != df6['ffID']]
#  按照AAindex排序推荐
df6['rn'] = df6.groupby('userID')['AA_index'].rank(
    ascending=False, method='first')
result = df6.loc[df6['rn'] == 1]
result.drop(columns=['rn'], inplace=True)

# %%
result = pd.DataFrame(result.iloc[:24, :].values.reshape(12, 6), columns=[
                      'userID', 'ffID', 'AA_index', 'userID', 'ffID', 'AA_index'])
print(result.to_latex(index=None, float_format="%.3f", bold_rows=True))

# %% [markdown]
# ### 自行设计

# %%
random.seed(1)
targetUsers = pd.DataFrame(random.choice(nodes["Id"], 30), columns=[
                           "Id"])  # 篇幅限制，只给30个随机用户推荐好友
# 关联二度好友
df1 = pd.merge(targetUsers, Uedges, left_on=['Id'], right_on=['Source'])
df2 = pd.merge(df1, Uedges, left_on=['Target'], right_on=['Source'])
friendNum = Uedges.groupby(['Source'])['Target'].count().reset_index()
friendNum.rename(columns={'Target': 'friendNum'}, inplace=True)
df3 = pd.merge(df2, friendNum, left_on='Target_x', right_on='Source')
df3.drop(columns=['Source_x', 'Source_y', 'Source'], inplace=True)
# 将结算结果改名为userID, ffID, CN
df3.rename(columns={'Id': 'userID', 'Target_y': 'ffID',
           'Target_x': 'CN_id', 'friendNum': 'CN_friendNum'}, inplace=True)
# df3包含了二度好友表和共同好友的好友数

# 计算WBC_index
df3["WBC_index"] = 1/df3["CN_friendNum"]*np.log(df3["CN_friendNum"])
df4 = df3.groupby(['userID', 'ffID'])['WBC_index'].sum().reset_index()
# 关联好友表，用于标记二度好友是否为好友,已经是好友的删除，推荐自己的删除
df6 = pd.merge(df4, Uedges, how='left', left_on=[
               'userID', 'ffID'], right_on=['Source', 'Target'])
df6 = df6.loc[df6['Target'].isnull(), ['userID', 'ffID', "WBC_index"]]
df6 = df6[df6['userID'] != df6['ffID']]
#  按照WBCindex排序推荐
df6['rn'] = df6.groupby('userID')['WBC_index'].rank(
    ascending=False, method='first')
result = df6.loc[df6['rn'] == 1]
result.drop(columns=['rn'], inplace=True)

# %%
result = pd.DataFrame(result.iloc[:24, :].values.reshape(12, 6), columns=[
                      'userID', 'ffID', 'WBC-index', 'userID', 'ffID', 'WBC-index'])
print(result.to_latex(index=None, float_format="%.3f", bold_rows=True))

# %% [markdown]
# # 社团识别

# %%

# 抽样获得小数据集
G_small = nx.Graph()
nodes = pd.read_csv("data/fb_nodes.csv")
edges = pd.read_csv("data/fb-pages-government.csv")
df1 = pd.merge(nodes, edges, left_on="Id", right_on="Source")
df2 = df1.groupby(['Source'])['Target'].count().reset_index()
df2 = pd.DataFrame(df2.values, columns=["Id", "friendNum"])
df2["rn"] = df2["friendNum"].rank(ascending=False, method='first')
# nodes = df2.loc[df2["rn"]<=100,["Id"]]
nodes = pd.DataFrame(np.random.choice(df2["Id"], 150, replace=False, p=(
    1/df2["rn"])/sum(1/df2["rn"])), columns=["Id"])
edges = pd.merge(nodes, edges, left_on="Id", right_on="Source",
                 how="inner").loc[:, ["Source", "Target"]]
edges = pd.merge(nodes, edges, left_on="Id", right_on="Target",
                 how="inner").loc[:, ["Source", "Target"]]
G_small.add_nodes_from(nodes["Id"])
G_small.add_edges_from(edges.apply(lambda x: tuple(x), axis=1).values.tolist())
# 删除自连接
G_small.remove_edges_from(nx.selfloop_edges(G_small))
# 找出最大连同子图
largest = max(nx.connected_components(G_small), key=len)
G_small = G_small.subgraph(largest)
print("新社交网络节点数", G_small.number_of_nodes())
print("新社交网络边数", G_small.number_of_edges())

# %%


def draw_plot(G, y, title):
    i = 0
    colors = []
    for node in G.nodes:
        colors.append(cm.Set1(y[i]))
        i += 1
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=nx.spring_layout(G, k=0.3, seed=1, iterations=50), node_size=100,
            node_color=colors, width=0.3, edge_color="grey", with_labels=False)
    plt.savefig("{title}.png".format(title=title), dpi=200)


# %%
simplefilter(action='ignore', category=FutureWarning)
data = nx.adjacency_matrix(G_small).todense()
# Kmeans聚类
start = time.time()
kmeans = KMeans(n_clusters=5, random_state=1).fit(data)
y = kmeans.predict(data)
end = time.time()
draw_plot(G_small, y, "Kmeans")
print("划分的社团数", len(set(y)))
print("Kmeans聚类耗时:%.3f s" % (end-start))

# %%
# GN算法
start = time.time()
comp = community.girvan_newman(G_small)  # 显示不同划分层级的节点
y = []
for keys, values in dict(enumerate(next(comp))).items():
    for value in values:
        y.append(keys)
end = time.time()
draw_plot(G_small, y, "GN_model")
print("划分的社团数", len(set(y)))
print("GN算法耗时:%.3f s" % (end-start))

# %%
# 基于Q函数的贪婪算法
start = time.time()
comp = community.greedy_modularity_communities(G_small)  # 显示不同划分层级的节点
y = []
for keys, values in dict(enumerate(comp)).items():
    for value in values:
        y.append(keys)
end = time.time()
draw_plot(G_small, y, "Q_model")
print("划分的社团数", len(set(y)))
print("基于Q函数的贪婪算法耗时:%.3f s" % (end-start))

# %%
# LPA社区发现算法
start = time.time()
comp = community.asyn_lpa_communities(G_small)
y = []
for keys, values in dict(enumerate(comp)).items():
    for value in values:
        y.append(keys)
end = time.time()
draw_plot(G_small, y, "LPA")
print("划分的社团数", len(set(y)))
print("LPA社区发现算法耗时:%.3f s" % (end-start))

# %%
# 分层聚类
start = time.time()
agg = FeatureAgglomeration(n_clusters=5).fit(data)
y = agg.labels_
end = time.time()
draw_plot(G_small, y, "Feature")
print("划分的社团数", len(set(y)))
print("分层聚类耗时:%.3f s" % (end-start))
