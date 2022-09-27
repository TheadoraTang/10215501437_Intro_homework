height = float(input("请输入你的身高（m）"))
weight = float(input("请输入你的体重（kg）"))
BMI = weight / height**2
if BMI < 18.5:
    print("肥胖")
elif 18.5 <= BMI < 24:
    print("正常")
elif 24 <= BMI < 28:
    print("超重")
elif BMI >= 28:
    print("肥胖")
