import sys
input = sys.stdin.readline

n = int(input())
tree = {}

for _ in range(n):
    parent, left, right = input().split()
    tree[parent] = [left, right]

def preorder(node):
    if node != '.':
        print(node, end='')
        preorder(tree[node][0])
        preorder(tree[node][1])

def inorder(node):
    if node != '.':
        inorder(tree[node][0])
        print(node, end = '')
        inorder(tree[node][1])

def postorder(node):
    if node != '.':
        postorder(tree[node][0])
        postorder(tree[node][1])
        print(node, end ='')

preorder('A')
print()
inorder('A')
print()
postorder('A')