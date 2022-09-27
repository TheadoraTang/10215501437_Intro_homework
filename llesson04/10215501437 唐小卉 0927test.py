#遍历练习#
sum_up = 1
for i in range(1, 101, 2):
    if i < 50:
        sum_up = sum_up * i
    print(i)
print(sum_up)

#循环练习#
list1 = [1, 2, 3, 4, 5]
i = len(list1) - 1
while i >= 0:
    print(list1[i], end =' ')
    i = i - 1
for j in range(len(list1) - 1, -1, -1):
    print(list1[j], end=' ')

#字符串#
s = str(input())
def longest_repetition(chars):
    if len(chars) == 0 or len(chars) == 1:
        return len(chars)
    result = [1] * len(chars)
    for left in range(len(chars) - 1):
        for right in range(left + 1, len(chars)):
            if chars[left] == chars[right]:
                result[left] += 1
            else:
                break
    return max(result)
print(longest_repetition(s))

#去掉空格#
s = str(input())
string = ""
for i in range(0, len(s)):
    if s[i] == " ":
        i = i + 1
    else:
        string += s[i]
print(string)

#蒙特卡洛
import random
import math
S = 2.0
N = 10000000
C = 0
for i in range(N):
    x = random.uniform(0.0, 1.0)
    y = random.uniform(0.0, 2.0)
    if y <= math.pow(x, 3) + math.pow(x, 2):
        C += 1
I = C / N * S
print(I)

#平方根#
def square_root_1(num):
    i = 0
    c = num
    m_max = c
    m_min = 0
    g = (m_min + m_max) / 2
    while (abs(g * g - c) > 0.00000000001):
        if (g * g < c):
            m_min = g
        else:
            m_max = g
        g = (m_min + m_max) / 2
        i = i + 1
        print("%d:g - %.13f" % (i, g))
def square_root_2(num):
    c = num
    i = 0
    g = 0
    for j in range(0, c + 1):
        if(j * j > c and g == 0):
            g = j - 1
    while(abs(g * g - c) > 0.0001):
        g = g + 0.00001
        i = i + 1
        print("%d:g - %.5f" % (i, g))
def square_root_3(num):
    c = num
    g = c / 2
    i = 0
    while(abs(g * g - c) > 0.0000000001):
        g = (g + c / g) / 2
        i = i + 1
        print("%d:%.13f" % (i, g))
print(square_root_2(2))
