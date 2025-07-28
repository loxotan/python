import sys
input = sys.stdin.readline

n = int(input())
liquids = list(map(int, input().split()))

left, right = 0, n-1
best_sum = sys.maxsize
best_pair = (0,0)

while left<right:
    mix = liquids[left] + liquids[right]
    
    if abs(mix) < abs(best_sum):
        best_sum = mix
        best_pair = (liquids[left], liquids[right])
        
    if mix<0:
        left += 1
    else:
        right -= 1

print(best_pair[0], best_pair[1])