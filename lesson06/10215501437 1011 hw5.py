#实验1#
import random
import time
#
# def bubble_sort(lst):
#     for i in range(1, len(lst)):
#         for j in range(0, len(lst)-i):
#             if lst[j] > lst[j+1]:
#                 a = lst[j+1]
#                 lst[j+1] = lst[j]
#                 lst[j] = a
#     return lst
#
# list1 = [5,9,8,6,5,2,4,1]
# bubble_sort(list1)
# print(list1)

#实验2#
#选择排序#
def selection_sort(arr):
    for i in range(len(arr) - 1):  # 如果前n-1个元素已经有序，则整体也已经有序，否则内部循环变量j会越界
        minIndex = i
        for j in range(i + 1, len(arr)):
            if arr[minIndex] > arr[j]:
                minIndex = j
        if i != minIndex:
            arr[i], arr[minIndex] = arr[minIndex], arr[i]  # 将后未排序元素中的最小值移动至已排序元素的末尾

#快速排序#
def quick_sort(arr, left, right):
    if left >= right:
        return
    pivot = arr[left]
    i = left
    j = right
    while i < j:
        while i < j and arr[j] >= pivot:
            j -= 1
        arr[i] = arr[j]
        while i < j and arr[i] < pivot:
            i += 1
        arr[j] = arr[i]
    arr[i] = pivot
    quick_sort(arr, left, i-1)
    quick_sort(arr, i+1, right)

list2 = []
list3 = []
a = int(input())
for i in range(0, pow(10, a)):
    list2.append(random.randint(0, 10000000))
    list3.append(random.randint(0, 10000000))
t1 = time.perf_counter()
quick_sort(list2, 0, len(list2) - 1)
t2 = time.perf_counter()
print(t2 - t1)
t3 = time.perf_counter()
selection_sort(list3)
t4 = time.perf_counter()
print(t4 - t3)
