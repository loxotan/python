def check(a, b, op):
    if op == '>':
        if a<b: return False
    if op == '<':
        if a>b: return False
    return True
    
def dfs(cnt, num):
    if cnt == n+1:
        nums.append(num)
        return
    
    for i in range(10):
        if visited[i]: continue
        
        if cnt == 0 or check(num[cnt-1], str(i), budungho[cnt-1]):
            visited[i] = 1
            dfs(cnt+1, num+str(i))
            visited[i] = 0
            
n = int(input())
budungho = list(input().split())
nums = []           
visited = [0]*10
dfs(0, '')
nums.sort()
print(nums[-1])
print(nums[0])
        
    