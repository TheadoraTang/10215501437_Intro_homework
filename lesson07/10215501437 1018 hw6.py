#实验1#
f1 = open('hw6.txt', 'r', encoding='utf-8')
s1 = f1.read()
print(s1)
f1.close()

#实验2#
f2 = open('stuGrade.csv', 'r', encoding='utf-8')
s2 = f2.read()
print(s2)
number = int(input())
average = []
sum_up = 0
for i in range(0, number):
    for j in range(0, 3):
        a = int(input())
        sum_up = sum_up + a
    average.append(sum_up / 3)
for k in range(0, len(average)):
    print("%.2f" % average[k], end=' ')
f2.close()

#实验3#
import time

results = [81.00, 171.00, 254.33, 351.33, 409.00]
f3 = open('my.txt', 'w', encoding='utf-8')
for grade in results:
    f3.write(str("%.2f" %grade))
    f3.write('\n')
t1 = time.strftime("%Y/%m/%d %H:%M:%S", )
f3.write(t1)
f3.write('\n')
time.sleep(3)  # 命令线程延迟运行3秒
t2 = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
f3.write(t2)
f3.close()
