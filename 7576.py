import sys
input = sys.stdin.readline
from collections import deque

n, m = map(int, input().split())
tomato = [[] for _ in range(m)]
for i in range(m):
    tomato[i] = list(map(int, input().split()))

def bfs(a, d):
    q = deque([])
    