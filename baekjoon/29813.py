import sys
input = sys.stdin.readline
from collections import deque

T = int(input())
dic = {}
q = deque([])
for _ in range(T):
    name, num = input().split()
    num = int(num)
    dic.update({name:num})
    q.append(name)

while len(q) != 1:
    finder = q.popleft()
    k = dic.get(finder)
    q.rotate(1-k)
    q.popleft()

print(*q)
