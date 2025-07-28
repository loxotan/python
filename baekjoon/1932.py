import sys
input = sys.stdin.readline
sys.setrecursionlimit(10000)
from collections import deque

n = int(input())
tri = [list(map(int, input().split())) for _ in range(n)]
sums = []
nums = [tri[0][0]]
#backtrack
def bkt(d, k):
    if d == n-1:
        sums.append(sum(nums))
        return
    
    for next in (k, k+1):
        nums.append(tri[d+1][next])
        bkt(d+1, next)
        nums.pop()
        
bkt(0,0)
print(max(sums))