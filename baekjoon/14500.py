import sys
input = sys.stdin.readline

n, m = map(int, input().split())
arr = []
for i in range(n):
    arr.append(list(map(int, input().split())))

move = [(1,0), (0,1), (-1,0), (0,-1)]
maxValue = 0

def dfs(ans, i, j, d):
    global maxValue
    if d == 3:
        maxValue = max(maxValue, ans)
        return
    
    for x, y in move:
        if 0<=i+x<n and 0<=j+y<m and not vst[i+x][j+y]:
            vst[i+x][j+y] = True
            ans += arr[i+x][j+y]
            dfs(ans,i+x,j+y,d+1)
            vst[i+x][j+y] = False
            ans -= arr[i+x][j+y]
            
def shape(i, j):
    global maxValue
    for a in range(4):
        ans = arr[i][j]
        for b in range(3):
            t = (a+b)%4
            x = i+move[t][0]
            y = j+move[t][1]
            
            if not(0<=x<n and 0<=y<m):
                ans = 0
                break
            ans += arr[x][y]
        maxValue = max(maxValue, ans)
                
vst = [[False]*m for _ in range(n)]

for i in range(n):
    for j in range(m):
        vst[i][j] = True
        dfs(arr[i][j], i, j, 0)
        vst[i][j] =False
        shape(i,j)

        
    
print(maxValue)