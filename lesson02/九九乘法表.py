for i in range(1, 10):
    for j in range(1, 10):
        if i >= j:
            answer = i * j
            print(f"{j} x {i} = {answer} \t", end = '')
        if j == 9:
            print("\t")
