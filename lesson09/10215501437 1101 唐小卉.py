import pymysql
import numpy as np
db = pymysql.connect(host = "cdb-r2g8flnu.bj.tencentcdb.com", port = 10209, user = "dase2020", password = "dase2020", database = "dase_intro_2020")
cursor = db.cursor()  # 使用 cursor() 方法创建一个游标对象 cursor,执行SQL语句都是通过游标对象实现

sql = "SELECT * FROM bicycle_train LIMIT 17, 5;"  # 该SQL语句返回MySQL的安装版本，用以确定是否成功连接服务器

cursor.execute(sql)  # 执行SQL语句
result = cursor.fetchall()  # 获取单条数据
print('编号|城市|小时|工作日|天气温度|体感温度|天气|风俗|单车租借量')
for i in result:
    for j in i:
        print("%3d" % j, end='|')
    print()

sql = "SELECT DISTINCT wind from bicycle_train ORDER BY wind;"
cursor.execute(sql)
result = cursor.fetchall()
print('风速最小值为%d 最大值为%d' % (result[0][0], result[-1][0]))

sql = "SELECT AVG(temp_air) FROM bicycle_train WHERE city=0 AND hour=10 AND weather=1 AND wind BETWEEN 0 AND 1 AND y>=100;"
cursor.execute(sql)
mean = cursor.fetchone()[0]
print('平均温度为 %.1f 摄氏度' % mean)

sql = "SELECT temp_air FROM bicycle_train WHERE city=0 AND hour=10 AND weather=1 AND wind BETWEEN 0 AND 1 AND y>=100;"
cursor.execute(sql)
result = cursor.fetchall()
temp_list = []
for record in result:
    temp_list.append(record[0])
temp_arr = np.array(temp_list)
temp_var = np.sum((temp_arr-mean)**2) / len(temp_arr)
print('平均温度的方差为 %.2f' % temp_var)

city_dict = {0:'北京', 1:'上海'}
sql = "SELECT city,SUM(y) FROM bicycle_train WHERE is_workday=1 AND weather=3 GROUP BY city ORDER BY SUM(y) DESC;"
cursor.execute(sql)
result = cursor.fetchall()
for record in result:
    print('%s:%d'%(city_dict[record[0]], record[1]))

sql = "SELECT hour,AVG(y) FROM bicycle_train WHERE hour BETWEEN 17 AND 19 AND city=1 AND is_workday=1 AND temp_body<=10 GROUP BY hour ORDER BY AVG(y);"
cursor.execute(sql)
result = cursor.fetchall()
for record in result:
    print('%d时：%d辆' % (record[0], round(record[1])))
