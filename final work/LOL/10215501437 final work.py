import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import missingno as msno
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import pydotplus
import warnings
warnings.filterwarnings("ignore")
sns.set_style('whitegrid')

data_champ = pd.read_json("champion_info.json")
data_champ2 = pd.read_json("champion_info_2.json")
champInfo = pd.read_json((data_champ2['data']).to_json(), orient='index')
champInfo2 = pd.read_json((data_champ["data"]).to_json(), orient="index")
data_spell = pd.read_json("summoner_spell_info.json")
data_spell_info = pd.read_json((data_spell['data']).to_json(), orient='index')
data_game = pd.read_csv("games.csv")
data_game_2 = pd.read_csv('games.csv')
winner = data_game["winner"]
winner = winner.T
name_dict = pd.Series(champInfo.key.values, index=champInfo.id).to_dict()


data_game.columns[data_game.isnull().any()]
print("Empty data:", data_game.isnull().sum())

#数据清洗，将对结果无影响的数据删除
data_game.drop(["gameId"], axis=1, inplace=True)
data_game.drop(["creationTime"], axis=1, inplace=True)
data_game.drop(["seasonId"], axis=1, inplace=True)
data_game = data_game.loc[data_game['firstInhibitor'] != 0]

#检查是否有空白数据
msno.bar(data_game, fontsize=6)
plt.show()
print(data_game.shape[1])
print(data_game.info())

champs = data_game[["t1_champ1id", "t1_champ2id",
                    "t1_champ3id", "t1_champ4id", "t1_champ5id", "t2_champ1id",
                    "t2_champ2id", "t2_champ3id", "t2_champ4id", "t2_champ5id", "t1_ban1", "t1_ban2", "t1_ban3",
                    "t1_ban4", "t1_ban5", "t2_ban1", "t2_ban2", "t2_ban3", "t2_ban4", "t2_ban5"]]
data_game.drop(["t1_champ1id", "t1_champ2id",
                "t1_champ3id", "t1_champ4id", "t1_champ5id", "t2_champ1id",
                "t2_champ2id", "t2_champ3id", "t2_champ4id", "t2_champ5id", "t1_ban1", "t1_ban2", "t1_ban3",
                "t1_ban4", "t1_ban5", "t2_ban1", "t2_ban2", "t2_ban3", "t2_ban4", "t2_ban5", "gameDuration",
                "seasonId"], axis=1, inplace=True)

data_game["t1_total_kills"] = data_game["t1_champ1_sum1"] + data_game["t1_champ1_sum2"] + data_game["t1_champ2_sum1"] + \
                              data_game["t1_champ2_sum2"] + data_game["t1_champ3_sum1"] + data_game["t1_champ3_sum2"] + \
                              data_game["t1_champ4_sum1"] + data_game["t1_champ4_sum2"] + data_game["t1_champ5_sum1"] + \
                              data_game["t1_champ5_sum2"]
data_game["t2_total_kills"] = data_game["t2_champ1_sum1"] + data_game["t2_champ1_sum2"] + data_game["t2_champ2_sum1"] + \
                              data_game["t2_champ2_sum2"] + data_game["t2_champ3_sum1"] + data_game["t2_champ3_sum2"] + \
                              data_game["t2_champ4_sum1"] + data_game["t2_champ4_sum2"] + data_game["t2_champ5_sum1"] + \
                              data_game["t2_champ5_sum2"]

data_game_2 = data_game_2[["winner","firstBlood","firstTower","firstInhibitor","firstBaron","firstDragon","firstRiftHerald","t1_towerKills","t1_inhibitorKills",'t1_baronKills',
       't1_dragonKills','t2_towerKills','t2_inhibitorKills', 't2_baronKills', 't2_dragonKills']]

#一水晶，一塔，一血，一先锋，一男爵，一龙魂
Inhibitor = data_game[["firstInhibitor", "winner"]].groupby(["firstInhibitor"], as_index=False).mean().sort_values(by="winner", ascending=False)
Tower = data_game[["firstTower", "winner"]].groupby(["firstTower"], as_index=False).mean().sort_values(by="winner", ascending=False)
Blood = data_game[["firstBlood", "winner"]].groupby(["firstBlood"], as_index=False).mean().sort_values(by="winner", ascending=False)
Dragon = data_game[["firstDragon", "winner"]].groupby(["firstDragon"], as_index=False).mean().sort_values(by="winner", ascending=False)
Rift = data_game[["firstRiftHerald", "winner"]].groupby(["firstRiftHerald"], as_index=False).mean().sort_values(by="winner", ascending=False)
Baron = data_game[["firstBaron", "winner"]].groupby(["firstBaron"], as_index=False).mean().sort_values(by="winner", ascending=False)
print(Inhibitor, '\n', Tower, '\n', Blood, '\n', Dragon, '\n', Rift, '\n', Baron)

winner_1 = data_game.loc[data_game['winner'] == 1]
winner_1.to_csv('winner1.csv', index=0)
winner_2 = data_game.loc[data_game['winner'] == 2]
factor = ['firstInhibitor', 'firstTower', 'firstBlood', 'firstDragon', 'firstRiftHerald', 'firstBaron']
fig = plt.figure(figsize=(20, 8))
for i in range(0, 6):
    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.5, hspace=0.2, top=0.9)
    plt.subplot(2, 6, i + 1)
    sns.countplot(data=winner_1, x='winner', hue=factor[i], palette='Set3', alpha=0.6)
    plt.xlabel(factor[i])
    plt.legend(fontsize=8)
for i in range(0, 6):
    plt.subplot(2, 6, i + 7)
    sns.countplot(data=winner_2, x='winner', hue=factor[i], palette='Set3', alpha=0.6)
    plt.xlabel(factor[i])
    plt.legend(fontsize=8)
plt.show()

g = sns.scatterplot(x="firstBlood", y="winner", data=data_game)
g = sns.scatterplot(x="firstInhibitor", y="winner", data=data_game)
g = sns.scatterplot(x="firstBaron", y="winner", data=data_game)
g = sns.scatterplot(x="firstTower", y="winner", data=data_game)
g = sns.scatterplot(x="firstRiftHerald", y="winner", data=data_game)
sns.heatmap(data_game[["firstInhibitor", "firstBlood", "firstTower", "firstBaron", "firstRiftHerald", "winner"]].corr(), annot=True)
plt.show()

factor = ['t1_towerKills', 't1_inhibitorKills', 't1_baronKills', 't1_dragonKills']
factor_ = ['t2_towerKills', 't2_inhibitorKills', 't2_baronKills', 't2_dragonKills']

f = data_game.groupby("winner").mean()
f.insert(0, 'winner', [1, 2])
f.to_csv('winner1.csv', index=0)
fig = plt.figure(figsize=(15, 8))
for i in range(0, 4):
    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.5, hspace=0.2, top=0.9)
    plt.subplot(2, 4, i + 1)
    sns.barplot(x=f['winner'], y=f[factor[i]], palette='Set3')
for i in range(0, 4):
    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.2, hspace=0.2, top=0.9)
    plt.subplot(2, 4, i + 5)
    sns.barplot(x=f['winner'], y=f[factor_[i]], palette='Set3')
plt.show()

data1 = data_game[['winner', 'firstBlood', 't1_towerKills', 't1_inhibitorKills', 't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills']]
data1.replace({'winner': {2: 0}}, inplace=True)
data1['firstBlood'].replace(2, 0, inplace=True)
graph = plt.figure(figsize=(7, 7))
sns.heatmap(data1.corr(), annot=True, square=True)
plt.show()

data2 = data_game[['winner', 'firstBlood','t2_towerKills', 't2_inhibitorKills', 't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']]
data2.replace({'winner': {1: 0}}, inplace=True)
data2.replace({'winner': {2: 1}}, inplace=True)
data2.replace({'firstBlood': {1: 0}}, inplace=True)
data2.replace({'firstBlood': {2: 1}}, inplace=True)
graph = plt.figure(figsize=(7, 7))
sns.heatmap(data2.corr(), annot=True, square=True)
plt.show()

champInfo = pd.read_json((data_champ2['data']).to_json(), orient= 'index')
Spellinfo= pd.read_json((data_spell['data']).to_json(), orient='index')
champCols = ['t1_champ1id', 't1_champ2id', 't1_champ3id', 't1_champ4id', 't1_champ5id',
             't2_champ1id', 't2_champ2id', 't2_champ3id', 't2_champ4id', 't2_champ5id']
banCols = ['t1_ban1', 't1_ban2', 't1_ban3', 't1_ban4', 't1_ban5',
             't2_ban1', 't2_ban2', 't2_ban3', 't2_ban4', 't2_ban5']
sumSpellsCols = ['t1_champ1_sum1', 't1_champ1_sum2', 't1_champ2_sum1', 't1_champ2_sum2', 't1_champ3_sum1', 't1_champ3_sum2',
                 't1_champ4_sum1', 't1_champ4_sum2', 't1_champ5_sum1', 't1_champ5_sum2', 't2_champ1_sum1', 't2_champ1_sum2',
                 't2_champ2_sum1', 't2_champ2_sum2', 't2_champ3_sum1', 't2_champ3_sum2', 't2_champ4_sum1', 't2_champ4_sum2',
                 't2_champ5_sum1', 't2_champ5_sum2']

champs = champInfo[['id', 'name']]
champ_dict = dict(zip(champs['id'], champs['name']))
for c in champCols:
    pick = data_game[c].replace(champ_dict, inplace=True)
for b in banCols:
    ban = data_game[b].replace(champ_dict, inplace=True)

spell = Spellinfo[['id', 'name']]
spell_dict = dict(zip(spell['id'],spell['name']))
for s in sumSpellsCols:
    spell = data_game[s].replace(spell_dict, inplace=True)



Picks = data_game.loc[:,['t1_champ1id','t1_champ2id','t1_champ3id','t1_champ4id','t1_champ5id',
             't2_champ1id', 't2_champ2id', 't2_champ3id', 't2_champ4id', 't2_champ5id']]
picksum = Picks.apply(pd.Series.value_counts)
Top_Pick = picksum.sum(axis=1).sort_values(ascending = False).head(10)
Top_Pick = Top_Pick.to_frame()

# Top picked champions
sns.barplot(x=Top_Pick.index, y=Top_Pick[0], palette='rocket')
plt.title('Most Picked Champions')
# plt.show()

Bans = data_game.loc[:,['t1_ban1','t1_ban2','t1_ban3','t1_ban4','t1_ban5',
             't2_ban1','t2_ban2','t2_ban3','t2_ban4','t2_ban5']]
Bansum = Bans.apply(pd.Series.value_counts)
Top_Ban = Bansum.sum(axis=1).sort_values(ascending=False).head(10)
Top_Ban = Top_Ban.to_frame()
sns.barplot(x= Top_Ban.index, y=Top_Ban[0], palette='Set2')
plt.title('Most Banned Champions')
# plt.show()

Sums = data_game.loc[:, ['t1_champ1_sum1','t1_champ1_sum2','t1_champ2_sum1','t1_champ2_sum2','t1_champ3_sum1','t1_champ3_sum2',
                 't1_champ4_sum1','t1_champ4_sum2','t1_champ5_sum1','t1_champ5_sum2','t2_champ1_sum1','t2_champ1_sum2',
                 't2_champ2_sum1','t2_champ2_sum2','t2_champ3_sum1','t2_champ3_sum2','t2_champ4_sum1','t2_champ4_sum2',
                 't2_champ5_sum1','t2_champ5_sum2']]
Sumsum= Sums.apply(pd.Series.value_counts)
Top_Sum = Sumsum.sum(axis=1).sort_values(ascending=False).head(10)
Top_Sum = Top_Sum.to_frame()
sns.barplot(x=Top_Sum.index, y=Top_Sum[0], palette='Set2')
plt.title('Most Used Summoner Spells')
# plt.show()

OP = pd.DataFrame()
OP['gameDuration'] = data_game['gameDuration'].astype(int)
def which_team(t):
    if (t['t1_champ1id'] == 'Tristana') or (t['t1_champ2id'] == 'Tristana') or (t['t1_champ3id'] == 'Tristana')\
            or (t['t1_champ4id'] == 'Tristana') or (t['t1_champ5id'] == 'Tristana') or (t['t1_champ1id'] == 'Thresh')\
            or (t['t1_champ2id'] == 'Thresh') or (t['t1_champ3id'] == 'Thresh')\
            or (t['t1_champ4id'] == 'Thresh') or (t['t1_champ5id'] == 'Thresh')\
            or (t['t1_champ1id'] == 'Yasuo') or (t['t1_champ2id'] == 'Yasuo') or (t['t1_champ3id'] == 'Yasuo')\
            or (t['t1_champ4id'] == 'Yasuo') or (t['t1_champ5id'] == 'Yasuo'):
        return 1
    else:
        return 2
he = data_game.apply(which_team, axis=1)
data_game['Team'] = he

def victory(t):
    if t['Team'] == t['winner']:
        return '1'
    else:
        return '0'
win = data_game.apply(victory, axis=1)
OP['victory'] = win

def blood(t):
    if t['Team'] == t['firstBlood']:
        return '1'
    else:
        return '0'
fb = data_game.apply(blood, axis=1)
OP['FirstBlood'] = fb

def drag(t):
    if t['Team'] == 2:
        return t['t2_dragonKills']
    else:
        return t['t1_dragonKills']
dragon = data_game.apply(drag, axis=1)
OP['Dragon'] = dragon

def bar(t):
    if t['Team'] == 2:
        return t['t2_baronKills']
    else:
        return t['t1_baronKills']
baron= data_game.apply(bar,axis=1)
OP['Baron'] = baron

def tow(t):
    if t['Team'] == 2:
        return t['t2_towerKills']
    else:
        return t['t1_towerKills']
tower = data_game.apply(tow, axis=1)
OP['Tower'] = tower

def inhib(t):
    if t['Team'] == 2:
        return t['t2_inhibitorKills']
    else:
        return t['t1_inhibitorKills']
inhibitor = data_game.apply(inhib, axis=1)
OP['Inhibitor'] = inhibitor


data_feature = OP[['gameDuration', 'FirstBlood', 'Dragon', 'Baron', 'Tower', 'Inhibitor']].values
data_target = OP[['victory']].values

X_train, X_test, Y_train, Y_test = train_test_split(
    data_feature, data_target, test_size=0.33, random_state=21, stratify=data_target)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
prediction = knn.predict(X_test)
print("KNN accuracy: ", knn.score(X_test, Y_test))

lg = LinearRegression()
lg.fit(X_train, Y_train)
print("LinearRegression accuracy: ", lg.score(X_test, Y_test))

dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
print("DecisionTree accuracy: ", dt.score(X_test, Y_test))

nb = GaussianNB()
nb.fit(X_train, Y_train)
print("Gaussian accuracy: ", nb.score(X_test, Y_test))

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
print("RandomForest accuracy: ", rf.score(X_test, Y_test))

lr = LogisticRegression()
lr.fit(X_train, Y_train)
print("LogisticRegression accuracy",  lr.score(X_test, Y_test))

svm = SVC(random_state=1)
svm.fit(X_train, Y_train)
prediction = svm.predict(X_test)
print("SVM accuracy: ", svm.score(X_test, Y_test), '\n')

willwin = np.array([[1145,1,2,0,5,1],[1324,0,1,0,2,1],[2568,0,1,0,6,1]])
print(lr.predict(willwin))

x1 = [[0,0,0,0,0,0]]
c = lr.predict_proba(x1).reshape(-1, 1)
print("winner is :", lr.predict(x1))
print("first team win probability is: ", list(c[0]*100), "% \nsecond team win probability is:", list(c[1]*100), "%")

y = data_game_2["winner"].values
x = data_game_2.drop(["winner"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.3,random_state=1)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("KNN accuracy: ", knn.score(x_test, y_test))

lg = LinearRegression()
lg.fit(x_train, y_train)
print("LinearRegression accuracy: ", lg.score(x_test, y_test))

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
print("DecisionTree accuracy: ", dt.score(x_test, y_test))

nb = GaussianNB()
nb.fit(x_train, y_train)
print("Gaussian accuracy: ", nb.score(x_test, y_test))

rf = RandomForestClassifier()
rf.fit(x_train, y_train)
print("RandomForest accuracy: ", rf.score(x_test, y_test))

lr = LogisticRegression()
lr.fit(x_train, y_train)
print("LogisticRegression accuracy",  lr.score(x_test, y_test))

x1 = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
c = lr.predict_proba(x1).reshape(-1,1)
print("winner is :" , lr.predict(x1)-1)
print("first team win probability is % ", list(c[0]*100),"\nsecond team win probability is %:",list(c[1]*100))

willwin = np.array([[1145,1,2,0,5,1],[1324,0,1,0,2,1],[2568,0,1,0,6,1]])
print(lr.predict(willwin))

for k in range(2, 10):
    model = DecisionTreeClassifier(max_depth=k)
    scores = cross_val_score(model, X_train, Y_train)
    print(k, scores.mean())

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train,Y_train)
y_pre = model.predict(X_test)
acc_score = accuracy_score(Y_test, y_pre)
print(acc_score)

dot_data = tree.export_graphviz(model, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("LOL.pdf")

data_feature = OP[['gameDuration', 'FirstBlood', 'Dragon', 'Baron', 'Tower', 'Inhibitor']].values
data_target = OP[['victory']].values

X_train, X_test, Y_train, Y_test = train_test_split(
    data_feature, data_target, test_size=0.33, random_state=21, stratify=data_target)

clf = tree.DecisionTreeClassifier()
clf.fit(OP[['gameDuration', 'FirstBlood', 'Dragon', 'Baron', 'Tower', 'Inhibitor']], OP['victory'])

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=['gameDuration', 'FirstBlood', 'Dragon', 'Baron', 'Tower', 'Inhibitor'],
                                class_names=['winner1', 'winner2'],
                                filled=True, rounded=True,
                                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('LOL.pdf')


plt.scatter(x['firstInhibitor'],x['firstBaron'])
plt.xlabel('firstInhibitor')
plt.ylabel('firstBaron')
plt.show()

plt.scatter(x['t1_towerKills'],x['t2_towerKills'])
plt.xlabel('t1_towerKills')
plt.ylabel('t2_towerKills')
plt.show()

plt.scatter(x['t1_towerKills'],x['t2_towerKills'])
plt.xlabel('t1_towerKills')
plt.ylabel('t2_towerKills')
plt.show()

wcss = []
for k in range(1,15):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(x.drop(['t1_champ1_sum1', 't1_champ1_sum2',
       't1_champ2_sum1', 't1_champ2_sum2', 't1_champ3_sum1', 't1_champ3_sum2',
       't1_champ4_sum1', 't1_champ4_sum2', 't1_champ5_sum1', 't1_champ5_sum2'],axis = 1))
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15),wcss)
plt.xlabel("?")
plt.ylabel("?")
plt.show()