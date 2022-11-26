from numpy import random
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
