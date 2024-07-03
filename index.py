import random


a = [1, 2, 3]
b = {"a": 1, "b": 2, "c": 3}
print(a, b, "要么出众，要么出局")

if 1 in a:
    print("i in a")
else:
    print("i not in a")

score = 77
if 90 < score <= 100:
    print("本次考试等级为 A")
elif 80 < score <= 90:
    print("本次考试等级为 B")
elif 70 < score <= 80:
    print("本次考试等级为 C")
elif 60 < score <= 70:
    print("本次考试等级为 D")
else:  # 这一行也可以写成：elif 0 <= score <= 60:
    print("本次考试等级为 E")


x = random.randint(0, 2)  # 随机生成 0、1、2 中的一个数字，赋值给变量 x
print(x)

str = "abcdefghijklmn"
print(str.find("abc"))

print(str.startswith("abc"))
print(str.endswith("mn"))


fruits = ["banana", "apple", "mango"]
for fruit in fruits:  # 第二个实例
    print("当前水果: %s" % fruit)

list1 = []  ## 空列表
list1.append("Google")  ## 使用 append() 添加元素
list1.append("Runoob")
print(list1)
del list1[1]
print(list1)

list2 = ["Google", "Runoob"]


class Parent:  # 定义父类
    parentAttr = 100

    def __init__(self):
        print("调用父类构造函数")

    def parentMethod(self):
        """
        调用父类方法的示例函数。

        参数:
        self - 表示实例本身的引用，允许访问类的属性和方法。

        返回值:
        无
        """
        print("调用父类方法")

    def setAttr(self, attr: int):
        Parent.parentAttr = attr

    def getAttr(self):
        print("父类属性 :", Parent.parentAttr)


class Child(Parent):  # 定义子类
    def __init__(self):
        print("调用子类构造方法")

    def childMethod(self):
        print("调用子类方法")


c = Child()  # 实例化子类
c.childMethod()  # 调用子类的方法
c.parentMethod()  # 调用父类方法
c.setAttr(200)  # 再次调用父类的方法 - 设置属性值
c.getAttr()  # 再次调用父类的方法 - 获取属性值


def multiple(**args):
    print(args)


def multiple2(**args):
    for key in args:
        print(key, args[key])

    multiple(**args)


if __name__ == "__main__":
    multiple2(name="Amy", age=12, single=23)
