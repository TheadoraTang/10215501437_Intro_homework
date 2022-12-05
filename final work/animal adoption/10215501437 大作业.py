import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

animal = pd.read_csv('animal-data-origin.csv')
# animal.info() #了解数据
change = animal.drop(columns=['sheltercode', 'animalname', 'identichipnumber', 'istrial']) #删除无关字段，这些内容对数据统计不会产生影响
# change.info() #检验是否成功
change.to_csv('animal-data-1.csv', index=0) #将处理过的数据备份
data1 = change.drop_duplicates(subset=['id']) #处理重复数据的行数
print("处理之后的行数：", data1.shape[0])

#将空缺数据填满
data1 = data1.copy()
data1.fillna('missing', inplace=True)
data1.info()

year = data1['intakedate'] #宠物被带来收容所的日期
year = year.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
Y = year.map(lambda x: x.year) #提取宠物被带来的年份
Y = Y.values
data1.insert(0, 'year_take', Y)
data1['year_take'].unique()
newchart = data1.groupby('year_take').count()
Y = newchart['id'].values
X = data1['year_take'].unique()
X = [str(number) for number in X] #将int转换为str，方便柱状图x轴的绘制

#统计每年收容的宠物数量
fig = plt.figure(figsize=(10, 8))
plt.pie(Y,  labels=X, autopct='%1.2f%%')
plt.tight_layout()

data1 = data1[data1['year_take'] > 2016] #将2016年之前的少量数据删除
data1.to_csv('animal-data-1.csv', index=0) #备份，data1是去除重复id且删除了2017年之前的数据,增加了year_take的表格

#统计数据中宠物的种类
sp_count = pd.DataFrame(data1.groupby(['speciesname'], as_index=False)['id'].count())
data2 = pd.DataFrame({'speciesname': sp_count.speciesname, 'count': sp_count.id})
data2 = data2.sort_values(by=['count'], ascending=False)

plt.figure(figsize=(10, 8))
ax = sns.barplot(x=data2['count'], y=data2['speciesname'], palette='Set3')

plt.ylabel('Animal species')
plt.xlabel('count')
plt.title('the number of animals has been to this shelter by species')

#统计动物在收容所的结局
plt.figure(figsize=(15, 8))
sns.countplot(y=data1['intakereason'], palette='Set3')

plt.xticks(rotation=90)
plt.xlabel('intakereason', fontsize=15)
plt.ylabel('count', fontsize=15)
plt.title('Why the animal end up at the shelter', fontsize=20)

#不同动物的不同进入收容所的原因
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

sns.countplot(data=data1, y='intakereason', hue='speciesname', ax=ax1,
              palette='Set2', alpha=0.6)
sns.countplot(data=data1, y='speciesname', hue='intakereason', ax=ax2,
              palette='Set2', alpha=0.6)

ax1.set_title('Intakereason and Speciesname')

#只留下猫狗
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
x_cat = data1.loc[(data1['speciesname'] == 'Cat')]
x_dog = data1.loc[(data1['speciesname'] == 'Dog')]
data3 = pd.concat([x_cat, x_dog])

sns.countplot(data=data3, y='intakereason', hue='speciesname', ax=ax1,
              palette='Set2', alpha=0.6)
sns.countplot(data=data3, y='speciesname', hue='intakereason', ax=ax2,
              palette='Set2', alpha=0.6)

ax1.set_title('Intakereason and Speciesname')

# 去向和性别之间的联系
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
sns.countplot(data=data1, x='movementtype', hue='sexname', ax=ax1,
              palette='Set2', alpha=0.6)
sns.countplot(data=data1, x='sexname', hue='movementtype', ax=ax2,
              palette='Set2', alpha=0.6)

ax1.set_title('Movementtype and Sexname', fontsize=20)

# 去向和品种之间的联系
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
data4 = pd.concat([x_cat, x_dog])

sns.countplot(data=data3, x='movementtype', hue='speciesname', ax=ax1,
              palette='Set2', alpha=0.6)
sns.countplot(data=data3, x='speciesname', hue='movementtype', ax=ax2,
              palette='Set2', alpha=0.6)
ax1.set_title('Movementtype and Speciesname', fontsize=20)

# #统计抛弃的动物的年龄
data1['age'] = data1['animalage'].str.split(' ', expand=True)[0] #为了防止图表过于乱，只选择年
plt.figure(figsize=(15, 8), dpi=80)
data1['age'] = [int(element) for element in data1['age']]
ax1 = sns.histplot(data1.sort_values(by=['age'])['age'])
plt.title('The distribution of Age of the animals')

#只将猫狗的数据留下来
data1 = data1.query('speciesname == "Cat" or speciesname == "Dog"')
data1 = data1[data1['age'] < 30]
data1['age'] = data1['animalage'].str.split(' ', expand=True)[0] #为了防止图表过于乱，只选择年
data1.to_csv('test.csv', index=0)
plt.figure(figsize=(15, 8), dpi=80)
data1['age'] = [int(element) for element in data1['age']]
ax2 = sns.histplot(data1.sort_values(by=['age'])['age'])
plt.title('The distribution of Age of cats and dogs')

#按照年龄对宠物进行幼年，成年，老年的区分
category = []
for ages in data1['age']:
    if ages < 3:
        category.append("young")
    elif ages < 5:
        category.append("young adult")
    elif ages < 10:
        category.append("adult")
    else:
         category.append("old")
data1['AgeCategory'] = pd.DataFrame(category)
plt.figure(figsize=(10, 4))
ax3 = sns.countplot(x=data1['AgeCategory'], palette='Set3')
plt.xlabel('Age type')
plt.ylabel('count')

#转移原因和年龄区段的联系
f, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 8))
sns.countplot(data=data1, x='movementtype', hue='AgeCategory', ax=ax1,
             palette='Set1', alpha=0.6)
sns.countplot(data=data1, x='AgeCategory', hue='movementtype', ax=ax2,
             palette='Set1', alpha=0.6)

ax1.set_title('Movementtype and Age', fontsize=15)

# 转移原因和颜色的关系
adopt = data1.loc[data1['movementtype'] == 'Adoption']
reclaim = data1.loc[data1['movementtype'] == 'Reclaimed']
not_adopt = data1.loc[data1['movementtype'] != 'Adoption']
plt.figure(figsize=(10, 45))
sns.countplot(data=data1, y='basecolour', hue='movementtype', palette='Set1')
plt.xlabel('Color type')
plt.ylabel('count')
plt.grid(alpha=0.6)

#每个颜色所占的比例
fig = plt.figure(figsize=(10, 8))
data_x = data1['basecolour'].value_counts()
color_key = data_x.keys().tolist()
# print(color_key)
# print(len(color_key)) #可知一共有78个颜色
plt.pie(data_x, labels=color_key, autopct='%1.2f%%')
plt.tight_layout()

#猫的领养与颜色的关系
cat_data = data1.copy()
cat_data = data1.loc[data1['speciesname'] == 'Cat']
fig = plt.figure(figsize=(10, 8))
cat_value = cat_data['basecolour'].value_counts()
cat_color_key = cat_value.keys().tolist()
for sub_count in range(0, 5):
    plt.subplot(1, 5, sub_count + 1)
    use_x = (cat_data['basecolour'] == cat_color_key[sub_count]).value_counts().tolist()
    use_y = ['Adopt', "Not Adopt"]
    plt.pie(x=use_x, labels=use_y, autopct='%1.2f%%', colors=['lightskyblue', 'gold'])
    plt.xlabel(cat_color_key[sub_count])
plt.tight_layout()

#狗的领养与颜色的关系
dog_data = data1.copy()
dog_data = data1.loc[data1['speciesname'] == 'Dog']
fig = plt.figure(figsize=(10, 8))
dog_value = dog_data['basecolour'].value_counts()
dog_color_key = dog_value.keys().tolist()
for sub_count in range(0, 5):
    plt.subplot(1, 5, sub_count + 1)
    use_x = (dog_data['basecolour'] == dog_color_key[sub_count]).value_counts().tolist()
    use_y = ['Adopt', "Not Adopt"]
    plt.pie(x=use_x, labels=use_y, autopct='%1.2f%%', colors=['pink', 'green'])
    plt.xlabel(cat_color_key[sub_count])
plt.tight_layout()

# 宠物被带离开收容所与月份之间的关系
month_take = data1['movementdate'] #宠物被带来收容所的日期
month_take = month_take.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
month = month_take.map(lambda x: x.month)
month = month.values
data1.insert(0, 'month_take', month)
data1['month_take'].unique()
newchart1 = data1.groupby('month_take').count()

group_year = data1.groupby(['month_take', 'year_take'], as_index=False).count()
plt.figure(figsize=(10, 4))
sns.lineplot(x="month_take", y='id', hue='year_take', data=group_year)
plt.grid()

#宠物转运走的原因与日期之间的关系
data1['movementdate'] = data1['movementdate'].str.split(' ', expand=True)[0]
plt.figure(figsize=(10, 4))
data1['movementtype'].groupby(data1['movementdate']).count().plot(kind="line", alpha=.7)
plt.grid()
# plt.show()

#详细数据
test = pd.read_csv(r'C:\Users\唐小卉\PycharmProjects\pythonProject3\Intro\10215501437_Intro_homework\final work\changeddata.csv')
test = test.copy()
adopt = 0
not_adopt = 0
test1 = test[test['season'] == 0]
for i in test1['movementtype']:
    if i == 'Adoption':
        adopt += 1
    else:
        not_adopt += 1
print("春天：", adopt / (adopt + not_adopt))

adopt = 0
not_adopt = 0
test2 = test[test['season'] == 1]
for i in test2['movementtype']:
    if i == 'Adoption':
        adopt += 1
    else:
        not_adopt += 1
print("夏天：", adopt / (adopt + not_adopt))

adopt = 0
not_adopt = 0
test2 = test[test['season'] == 2]
for i in test2['movementtype']:
    if i == 'Adoption':
        adopt += 1
    else:
        not_adopt += 1
print("秋天：", adopt / (adopt + not_adopt))

adopt = 0
not_adopt = 0
test2 = test[test['season'] == 3]
for i in test2['movementtype']:
    if i == 'Adoption':
        adopt += 1
    else:
        not_adopt += 1
print("冬天：", adopt / (adopt + not_adopt))


monthGroup = data1['movementdate'].groupby(data1['movementtype'])
plt.subplots(7, 1, figsize=(15, 25), sharex=True)
plt.subplots_adjust(hspace=0.7)
colors = list('rgbcmyk')
for i, (_, g) in enumerate(monthGroup):
    plt.subplot(7, 1, i+1)
    plt.title(_)
    g.groupby(data1["movementdate"]).count().plot(kind="line", color=colors[i], grid=True, alpha=.5)

count = 0
count1 = 0
count2 = 0
for i in data1['movementtype']:
    if i == "Escaped":
        count = count + 1
    elif i == "Released To Wild":
        count1 = count1 + 1
    elif i == "Stolen":
        count2 = count2 + 1
print("Escaped:", count)
print("Released To Wild:", count1)
print("Stolen:", count2)

# data1.info()
end = data1['movementtype'].value_counts()
new_column = []
for i in data1['movementtype']:
    if i == 'Adoption':
        new_column.append(1)
    else:
        new_column.append(0)
new_column = pd.DataFrame(new_column)
data1.insert(23, 'whether_adopt', new_column)

#猫狗是否有关
new_column_species = []

for j in data1['speciesname']:
    if j == 'Cat':
        new_column_species.append(0)
    else:
        new_column_species.append(1)
new_column_species = pd.DataFrame(new_column_species)
data1.insert(22, 'cat_or_dog', new_column_species)

#季节是否有关
new_column_season = []
for j in data1['month_take']:
    if j >= 2 and j <= 4:
        new_column_season.append(0)
    elif j > 4 and j <= 7:
        new_column_season.append(1)
    elif j > 7 and j <= 10:
        new_column_season.append(2)
    else:
        new_column_season.append(3)
new_column_season = pd.DataFrame(new_column_season)
data1.insert(22, 'season', new_column_season)

data1.to_csv('changeddata.csv', index=0)

df_heat = data1.drop(columns=['intakedate', 'movementdate', 'id',
                           'animalage', 'returndate', 'returnedreason',
                           'deceaseddate', 'deceasedreason',
                           'year_take', 'movementtype', 'istransfer', 'month_take',
                           'diedoffshelter', 'isdoa', 'puttosleep'])

le_sp = preprocessing.LabelEncoder()
df_heat.cat_or_dog = le_sp.fit_transform(df_heat.cat_or_dog)

le_take = preprocessing.LabelEncoder()
df_heat.intakereason = le_take.fit_transform(df_heat.intakereason)

le_AgeCategory = preprocessing.LabelEncoder()
df_heat.AgeCategory = le_take.fit_transform(df_heat.AgeCategory)

le_breed = preprocessing.LabelEncoder()
df_heat.breedname = le_breed.fit_transform(df_heat.breedname)

le_location = preprocessing.LabelEncoder()
df_heat.location = le_breed.fit_transform(df_heat.location)

le_intakereason = preprocessing.LabelEncoder()
df_heat.intakereason = le_breed.fit_transform(df_heat.intakereason)

le_color = preprocessing.LabelEncoder()
df_heat.basecolour = le_color.fit_transform(df_heat.basecolour)

le_adopt = preprocessing.LabelEncoder()
df_heat.whether_adopt = le_adopt.fit_transform(df_heat.whether_adopt)

le_season = preprocessing.LabelEncoder()
df_heat.season = le_season.fit_transform(df_heat.season)

le_sexname = preprocessing.LabelEncoder()
df_heat.sexname = le_season.fit_transform(df_heat.sexname)

le_age = preprocessing.LabelEncoder()
df_heat.age = le_age.fit_transform(df_heat.age)

df_heat.info()
df_heat.to_csv('test.csv', index=0)
corr = df_heat.select_dtypes(include=['int64', 'int32']).iloc[:, 1:].corr()

cor_dict = corr['whether_adopt'].to_dict()
del cor_dict['whether_adopt']
for ele in sorted(cor_dict.items(), key=lambda x: -abs(x[1])):
    print("{0}: {1}".format(*ele))

# 相关矩阵热图
corrmat = df_heat.corr()
plt.figure(figsize=(10, 15))

k = 50000
cols = corrmat.nlargest(k, 'whether_adopt')['whether_adopt'].index
cm = np.corrcoef(df_heat[cols].values.T)

mask = np.zeros_like(cm, dtype=bool)
mask[np.triu_indices_from(mask)] = True

sns.set(font_scale=1)
sns.heatmap(cm, mask=mask, cbar=True, annot=True, square=True,
            fmt='.2f', annot_kws={'size': 12}, yticklabels=cols.values,
            xticklabels=cols.values, cmap='PuBu', lw=.1)
# plt.show()

lastchart = df_heat.drop(columns=['intakereason', 'age', 'sexname', 'breedname', 'speciesname', 'location', 'basecolour'])
lastchart.to_csv('lastchart.csv')
data = pd.read_csv('lastchart.csv')

#逻辑回归算法
# 确定特征值,目标值
x = data.iloc[1:4218, 1:4]
print(x.head())
y = data["whether_adopt"].loc[1:4217]
print(y.head())
# 分割数据
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
# 特征工程(标准化)
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 机器学习(逻辑回归)
estimator = LogisticRegression()
estimator.fit(x_train, y_train)
# 模型评估
y_predict = estimator.predict(x_test)
print("预测值为：", y_predict.tolist())
print("准确率为：", estimator.score(x_test, y_test))

# #调用scikit-learn中集成的决策树tree
# from sklearn import tree
# clf = tree.DecisionTreeClassifier(max_depth=5)
# clf = clf.fit(X_train, y_train)
#
# #展示决策树，定义函数
# with open("No-Yes.dot", 'w') as f:
#      f = tree.export_graphviz(clf,
#                               out_file=f,
#                               max_depth = 5,   #最优5层
#                               impurity = True,
#                               feature_names = list(X_train),
#                               class_names = ["Not Adopt", "Adopt"],
#                               rounded = True,
#                               filled= True )
#
# from subprocess import check_call
# check_call(['C:/Program Files/Graphviz 2.44.1/bin/dot','-Tpng','No-Yes.dot','-o','No-Yes.png'])
#
# from IPython.display import Image as PImage
# from PIL import Image, ImageDraw, ImageFont
# img = Image.open("No-Yes.png")
# draw = ImageDraw.Draw(img)
# img.save('output.png')
# PImage("output.png")