#实验1#
def conv_to_binary(a):
    b = int(a);
    ans = []
    while b > 0:
        ans.append(b % 2)
        b = b // 2
    while len(ans) < 8:
        ans.append(0)
    ans.reverse()
    return ans

def convert_dec_to_ip_address(ip):
    ip_list = ip.split(".")
    sublist = []
    for i in range(4):
        sublist.append(conv_to_binary(ip_list[i]))
    for i in range(4):
        for j in range(8):
            print(sublist[i][j], end="")
    return 0
convert_dec_to_ip_address("203.179.25.37")

#实验2#
class Quene():
    def __init__(self):
        self.__items = []
    def print_element(self):
        for i in range(s.size()):
            print(self.__items[i], end="")
        print("\n")
    def size(self):  # 返回队列长度
        return len(self.__items)
    def isempty(self):  # 返回队列是否为空
        if len(self.__items) == 0:
            return True
        else:
            return False
    def push(self,element):  # 压入堆栈
        self.__items.append(element)
    def pop(self):  # 出队，注意需要处理队列为空的情况
        try:
            return self.__items.pop(0)
        except:
            print('ERROR: Quene is empty now!')
    def peek(self):  # 返回队首元素，注意需要处理队列为空的情况
        try:
            return self.__items[0]
        except:
            print('ERROR: Quene is empty now!')
s = Quene()
s.push(1)
s.push(2)   # 队列目前为 [1，2]
s.print_element()
print(s.pop())
print(s.pop())
print(s.pop())# 队列目前为空

print('**********')

s.push(3.5)  # 队列目前为[3.5]
s.push(2.7)  # 队列目前为[3.5, 2.7]
print(s.peek())
print(s.size())
print(s.isempty())

#实验3#
import queue
class BinaryTree:
    def __init__(self,data=None,left=None,right=None):  # 如果创建节点对象时left或right参数为空，则默认该节点没有左或右子树
        self.data = data
        self.left = left
        self.right = right
    def layerorder(self):#层序遍历
        q = queue.Queue()
        if self != None:
            q.put(self)
        else:
            return 0
        while q:
            #print(r.data, end=" ")
            node = q.get(0)# 弹出第一个值
            print(node.data, end="")
            if node.left:  # 左子树判断
                q.put(node.left)
            if node.right:  # 右子树判断
                q.put(node.right)

layer3_2 = BinaryTree(2,BinaryTree(7),BinaryTree(4))
layer2_5 = BinaryTree(5,BinaryTree(6),layer3_2)
layer2_1 = BinaryTree(1,BinaryTree(0),BinaryTree(8))
layer1_3 = BinaryTree(3,layer2_5,layer2_1)
layer1_3.layerorder()

#实验4#
class BinaryTree:
    def __init__(self,data=None,left=None,right=None):  # 如果创建节点对象时left或right参数为空，则默认该节点没有左或右子树
        self.data=data
        self.left=left
        self.right=right
    def preorder(self):  # 前序遍历
        if self.left == None and self.right == None:
            print(self.data,end=' ')
        if self.left != None:
            self.left.preorder()
        if self.right != None:
            self.right.preorder()
layer3_2 = BinaryTree(2,BinaryTree(7),BinaryTree(4))
layer2_5 = BinaryTree(5,BinaryTree(6),layer3_2)
layer2_1 = BinaryTree(1,BinaryTree(0),BinaryTree(8))
layer1_3 = BinaryTree(3,layer2_5,layer2_1)
layer1_3.preorder()