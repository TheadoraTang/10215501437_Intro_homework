import numpy as np
import matplotlib.pyplot as plt

month = np.arange(1, 10)
day = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30])
sun_sum = np.zeros(9)
sun_average = np.empty(9)

with open('daily_KP_SUN_2020.csv', 'r', encoding='ISO-8859-1') as file:
    for i in range(3):
        file.readline()
    for j in file:
        j = j.strip()
        elements = j.split(',')
        sun_sum[int(elements[1]) - 1] += float(elements[3])
sun_average = sun_sum / day
plt.subplot(121)
plt.title("Sum Month")
plt.bar(month, sun_sum)

plt.subplot(122)
plt.title("Day Average")
plt.bar(month, sun_average)
plt.show()

point = {0: 'Sepal Length', 1: 'Sepal width', 2: 'Petal length', 3: 'Petal width'}
colors = {0: 'green', 2: 'red', 3: 'yellow'}
dictionary = {"setosa": [], "versicolor": [], "virginica": []}
with open('iris.csv', 'r',  encoding='utf-8') as file:
    file.readline()
    for i in file:
       i = i.strip()
       elements = i.split(',')
       dictionary[elements[4]].append(elements[0:4])
subplot_count = 1
plt.figure(dpi=100)
for type_1 in range(3):
    for type_2 in range(type_1 + 1, 4):
        plt.subplot(2, 3, subplot_count)
        for species, values in dictionary.items():
            values = np.array(values)
            plt.scatter(values[:, type_1], values[:, type_2], s=3, label=species)
        plt.xlabel(point[type_1])
        plt.ylabel(point[type_2])
        plt.xticks([])
        plt.yticks([])
        subplot_count += 1
        plt.legend(fontsize=5)
plt.show()
