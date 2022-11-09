import pymysql
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

db = pymysql.connect(host="cdb-r2g8flnu.bj.tencentcdb.com", port=10209, user="dase2020", password="dase2020", database="dase_intro_2020")
cursor = db.cursor()
sql = "SELECT * FROM SH_Grade;"
cursor.execute(sql)
result = cursor.fetchall()
stu_class = []
with open('SH_Grade.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'StuId', 'Sex', 'CHI611', 'MATH611', 'ENG611', 'CHI612', 'MATH612', 'ENG612', 'CHI621', 'MATH621', 'ENG621', 'CHI622', 'MATH622', 'ENG622', 'CHI711', 'MATH711', 'ENG711', 'CHI712', 'MATH712', 'ENG712', 'CHI721', 'MATH721', 'ENG721', 'CHI722', 'MATH722', 'ENG722', 'CHI811', 'MATH811', 'ENG811', 'PHY811', 'CHI812', 'MATH812', 'ENG812', 'PHY812', 'CHI821', 'MATH821', 'ENG821', 'PHY821', 'CHI822', 'MATH822', 'ENG822', 'PHY822', 'CHI911', 'MATH911', 'ENG911', 'PHY911', 'CHE911', 'CHI912', 'MATH912', 'ENG912', 'PHY912', 'CHE912', 'CHI921', 'MATH921', 'ENG921', 'PHY921', 'CHE921'])
    for i in result:
        writer.writerow(i)
with open('SH_Grade.csv', 'r') as file:
    file.readline()
    for j in file:
        j = j.strip()
        elements = j.split(',')
        stu_class.append(elements[1][0])
file1 = open('SH_Grade.csv', 'r')
data1 = pd.read_csv(file1)
data1.insert(2, 'Class', stu_class)
data1.to_csv('SH_Grade.csv', index=False)
file1.close()

#数据预处理1：剔除重复数据
file2 = open('SH_Grade.csv', 'r')
data2 = pd.read_csv(file2)
print('处理前:', data2.shape[0])
change1 = data2.drop_duplicates(subset=['StuId'])
print('处理后:', change1.shape[0])
change1.to_csv('SH_Grade.csv', index=False)
file2.close()

#数据预处理2：剔除大于等于12个字段为空的数据行
file3 = open('SH_Grade.csv', 'r')
data3 = pd.read_csv(file3)
print('处理前:', data3.shape[0])
change2 = data3.dropna(thresh=49)
print('处理后:', change2.shape[0])
change2.to_csv('SH_Grade.csv', index=False)
file3.close()

#数据预处理3：性别用上一条数据行的性别填充，成绩用该次该门考试的中位数填充。
file4 = open('SH_Grade.csv', 'r')
data4 = pd.read_csv(file4)
index = ['CHI611', 'MATH611', 'ENG611', 'CHI612', 'MATH612', 'ENG612', 'CHI621', 'MATH621', 'ENG621', 'CHI622', 'MATH622', 'ENG622', 'CHI711', 'MATH711', 'ENG711', 'CHI712', 'MATH712', 'ENG712', 'CHI721', 'MATH721', 'ENG721', 'CHI722', 'MATH722', 'ENG722', 'CHI811', 'MATH811', 'ENG811', 'PHY811', 'CHI812', 'MATH812', 'ENG812', 'PHY812', 'CHI821', 'MATH821', 'ENG821', 'PHY821', 'CHI822', 'MATH822', 'ENG822', 'PHY822', 'CHI911', 'MATH911', 'ENG911', 'PHY911', 'CHE911', 'CHI912', 'MATH912', 'ENG912', 'PHY912', 'CHE912', 'CHI921', 'MATH921', 'ENG921', 'PHY921', 'CHE921']
for j in index:
    data4[j] = data4[j].fillna(data4[j].median())
data4['Sex'] = data4['Sex'].fillna(method='ffill')
data4.to_csv('SH_Grade.csv', index=False)
file4.close()

#数据预处理4：将部分非百分制计分的列转换为百分制计分
file5 = open('SH_Grade.csv', 'r')
data5 = pd.read_csv(file5)
index = ['CHI611', 'MATH611', 'ENG611', 'CHI612', 'MATH612', 'ENG612', 'CHI621', 'MATH621', 'ENG621', 'CHI622', 'MATH622', 'ENG622', 'CHI711', 'MATH711', 'ENG711', 'CHI712', 'MATH712', 'ENG712', 'CHI721', 'MATH721', 'ENG721', 'CHI722', 'MATH722', 'ENG722', 'CHI811', 'MATH811', 'ENG811', 'PHY811', 'CHI812', 'MATH812', 'ENG812', 'PHY812', 'CHI821', 'MATH821', 'ENG821', 'PHY821', 'CHI822', 'MATH822', 'ENG822', 'PHY822', 'CHI911', 'MATH911', 'ENG911', 'PHY911', 'CHE911', 'CHI912', 'MATH912', 'ENG912', 'PHY912', 'CHE912', 'CHI921', 'MATH921', 'ENG921', 'PHY921', 'CHE921']
for j in index:
    if data5[j].max() > 100:
        if data5[j][3] == '8' and data5[j][4] == '2' and data5[j][5] == '2':
            data5[j] = data5[j] * 100 / 120
        else:
            data5[j] = data5[j] * 100 / 150
    elif data5[j].max() <= 90 and data5[j][2] == 'Y':#物理
            data5[j] = data5[j] * 100 / 90
    elif data5[j].max() <= 60 and data5[j][2] == 'E':#化学
            data5[j] = data5[j] * 100 / 60
data5.to_csv('SH_Grade.csv', index=False)
file5.close()

#数据预处理5：绘制各班男女人数的叠加条形图。
file6 = open('SH_Grade.csv', 'r')
graph_1 = pd.read_csv(file6)
graph_male = graph_1.loc[graph_1['Sex'] == 'M']
graph_female = graph_1.loc[graph_1['Sex'] == 'F']
graph_male_A = graph_male.loc[graph_male['Class'] == 'A']
graph_male_B = graph_male.loc[graph_male['Class'] == 'B']
graph_male_C = graph_male.loc[graph_male['Class'] == 'C']
graph_male_D = graph_male.loc[graph_male['Class'] == 'D']
graph_male_E = graph_male.loc[graph_male['Class'] == 'E']
graph_male_F = graph_male.loc[graph_male['Class'] == 'F']
graph_male_G = graph_male.loc[graph_male['Class'] == 'G']
graph_female_A = graph_female.loc[graph_female['Class'] == 'A']
graph_female_B = graph_female.loc[graph_female['Class'] == 'B']
graph_female_C = graph_female.loc[graph_female['Class'] == 'C']
graph_female_D = graph_female.loc[graph_female['Class'] == 'D']
graph_female_E = graph_female.loc[graph_female['Class'] == 'E']
graph_female_F = graph_female.loc[graph_female['Class'] == 'F']
graph_female_G = graph_female.loc[graph_female['Class'] == 'G']

y_1 = np.array([graph_male_A.shape[0] - 1, graph_male_B.shape[0] - 1, graph_male_C.shape[0] - 1, graph_male_D.shape[0] - 1, graph_male_E.shape[0] - 1, graph_male_F.shape[0] - 1, graph_male_G.shape[0] - 1])
y_2 = np.array([graph_female_A.shape[0] - 1, graph_female_B.shape[0] - 1, graph_female_C.shape[0] - 1, graph_female_D.shape[0] - 1, graph_female_E.shape[0] - 1, graph_female_F.shape[0] - 1, graph_female_G.shape[0] - 1])
plt.bar(['A', 'B', 'C', 'D', 'E', 'F', 'G'], y_1, tick_label=['A', 'B', 'C', 'D', 'E', 'F', 'G'], label='Male')
plt.bar(['A', 'B', 'C', 'D', 'E', 'F', 'G'], y_2, tick_label=['A', 'B', 'C', 'D', 'E', 'F', 'G'], bottom=y_1, label='Female')
plt.legend()
plt.show()

#数据预处理6：在一张图表中分别绘制学生代码为A13和A15的学生每次语文考试成绩走势折线图。
file7 = open('SH_Grade.csv', 'r')
graph_2 = pd.read_csv(file7)
graph_2_A13 = graph_2.loc[graph_2['StuId'] == 'A13']
graph_2_A13 = graph_2_A13.drop(['id', 'StuId', 'Class', 'Sex', 'MATH611', 'ENG611', 'MATH612', 'ENG612', 'MATH621', 'ENG621', 'MATH622', 'ENG622', 'MATH711', 'ENG711', 'MATH712', 'ENG712', 'MATH721', 'ENG721', 'MATH722', 'ENG722', 'CHI811', 'MATH811', 'ENG811', 'PHY811', 'MATH812', 'ENG812', 'PHY812', 'MATH821', 'ENG821', 'PHY821', 'MATH822', 'ENG822', 'PHY822', 'MATH911', 'ENG911', 'PHY911', 'CHE911', 'MATH912', 'ENG912', 'PHY912', 'CHE912', 'MATH921', 'ENG921', 'PHY921', 'CHE921'], axis=1)
graph_2_A15 = graph_2.loc[graph_2['StuId'] == 'A15']
graph_2_A15 = graph_2_A15.drop(['id', 'StuId', 'Class', 'Sex', 'MATH611', 'ENG611', 'MATH612', 'ENG612', 'MATH621', 'ENG621', 'MATH622', 'ENG622', 'MATH711', 'ENG711', 'MATH712', 'ENG712', 'MATH721', 'ENG721', 'MATH722', 'ENG722', 'CHI811', 'MATH811', 'ENG811', 'PHY811', 'MATH812', 'ENG812', 'PHY812', 'MATH821', 'ENG821', 'PHY821', 'MATH822', 'ENG822', 'PHY822', 'MATH911', 'ENG911', 'PHY911', 'CHE911', 'MATH912', 'ENG912', 'PHY912', 'CHE912', 'MATH921', 'ENG921', 'PHY921', 'CHE921'], axis=1)
y_A13 = graph_2_A13.values.T
x_A13 = graph_2_A13.columns
y_A15 = graph_2_A15.values.T
x_A15 = graph_2_A15.columns
# print(y_A13)
# print(x_A13)
plt.plot(x_A13, y_A13, label='A13', color='red')
plt.plot(x_A15, y_A15, label='A15', color='green')
plt.xticks([])
plt.legend()
plt.show()

#数据预处理7：输出7年级第2学期期中考试中英语成绩小于60分或语文成绩小于60分的学生编号、班级、英语成绩、语文成绩
file8 = open('SH_Grade.csv', 'r')
data5 = pd.read_csv(file8)
data5 = data5.drop(['id', 'Sex', 'CHI611', 'MATH611', 'ENG611', 'CHI612', 'MATH612', 'ENG612', 'CHI621', 'MATH621', 'ENG621', 'CHI622', 'MATH622', 'ENG622', 'CHI711', 'MATH711', 'ENG711', 'CHI712', 'MATH712', 'ENG712', 'MATH721', 'CHI722', 'MATH722', 'ENG722', 'CHI811', 'MATH811', 'ENG811', 'PHY811', 'CHI812', 'MATH812', 'ENG812', 'PHY812', 'CHI821', 'MATH821', 'ENG821', 'PHY821', 'CHI822', 'MATH822', 'ENG822', 'PHY822', 'CHI911', 'MATH911', 'ENG911', 'PHY911', 'CHE911', 'CHI912', 'MATH912', 'ENG912', 'PHY912', 'CHE912', 'CHI921', 'MATH921', 'ENG921', 'PHY921', 'CHE921'],axis=1)
change3 = data5.loc[(data5['ENG721'] < 60) | (data5['CHI721'] < 60)]
print(change3)
file8.close()

#数据预处理8：输出A班和C班6年级第2学期期末考试中各个科目的均值和方差，然后用文字（程序注释）简要比较两个班级各科目的表现情况
file9 = open('SH_Grade.csv', 'r')
data6 = pd.read_csv(file9)
Index = ['CHI622', 'MATH622', 'ENG622']
A = data6.loc[data6['Class'] == 'A']
C = data6.loc[data6['Class'] == 'C']
data7 = pd.DataFrame([[A['CHI622'].mean(), C['CHI622'].mean()], [A['MATH622'].mean(), C['MATH622'].mean()], [A['ENG622'].mean(), C['ENG622'].mean()], [A['CHI622'].var(), C['CHI622'].var()], [A['MATH622'].var(), C['MATH622'].var()], [A['ENG622'].var(), C['ENG622'].var()]], index=['CHI622_mean', 'MATH622_mean', 'ENG622_mean', 'CHI622_var', 'MATH622_var', 'ENG622_var'], columns=['A', 'C'])
print(data7)
file9.close()
#A班的语文和数学平均成绩均高于C班，但英语平均成绩低于C班，语文成绩的方差略高于C班，其他均小于C班，说明A班的成绩更为集中，C班的成绩可能出现高分低分人都很多的情况

#数据预处理9：将任务8产生的DataFrame对象生成CSV文件，文件名为task8.csvdata7 = pd.DataFrame([[A['CHI622'].mean(), C['CHI622'].mean()], [A['MATH622'].mean(), C['MATH622'].mean()], [A['ENG622'].mean(), C['ENG622'].mean()], [A['CHI622'].var(), C['CHI622'].var()], [A['MATH622'].var(), C['MATH622'].var()], [A['ENG622'].var(), C['ENG622'].var()]], index=['CHI622_mean', 'MATH622_mean', 'ENG622_mean', 'CHI622_var', 'MATH622_var', 'ENG622_var'], columns=['A', 'C'])
data7.to_csv('task8.csv')