N = int(input())
X = input().rstrip()
import copy
song = copy.copy(X)

ans = 1
for i in range(1, N):
    if X[i:min(i+N, len(X))] == song[:-i]:
        X += '???'
        ans += 1
        
print(ans)