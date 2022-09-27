###实验1
n = int(input())
answer = 1
for i in range(1, n+1):
    answer *= i
print(answer)

###实验2
import math
right = 27
left = 1
answer = 0
while right - left > 0:
    middle = float((right + left) / 2)
    if math.pow(middle, 3) > 27:
        right = middle
    elif math.pow(middle, 3) < 27:
        left = middle
    else:
        answer = math.fabs(middle)
        break
print(answer)

#实验3

def weightcal(coins, start, end):
    sum = 0
    for i in range(start,end+1):
        sum += coins[i]
    return sum

def findFalseCoin(coins, start, n):
    if(start == n):
        print("Fake coin:{}".format(start)); return
    if((n - start + 1) % 2 == 0):
        weight1 = weightcal(coins, start, n//2)
        weight2 = weightcal(coins, n//2 + 1, n)
        if(weight1 == weight2):
            print("Fake coin is not found"); return
        elif weight1<weight2:findFalseCoin(coins, start, n//2)
        else:findFalseCoin(coins, n//2 + 1, n)
    else:
        min = coins[start]
        flag = 0
        for i in range(start, n+1):
            if coins[i] < min:
                flag = 1
                min = i
                break
            elif coins[i] > min:
                flag = 1
                min = start
                break
        if flag == 1:
            print(min)


coins = [2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2]
findFalseCoin(coins, 0, len(coins)-1)



#实验4
import random
import math
S = 21.0
N = 10000000
C = 0
for i in range(N):
    x = random.uniform(2.0, 3.0)
    y = random.uniform(0.0, 21.0)
    if y <= math.pow(x, 2) + 4 * x * math.sin(x):
        C += 1
I = C / N * S
print(I)
