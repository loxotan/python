import sys
input = sys.stdin.readline
sys.setrecursionlimit(10000)

K = int(input())

def dfs(v, a):
    global vst, result
    vst[v] = True
    next = stu[v]
    if next == a:
        result = True
    if not vst[next]:
        dfs(next, a)
        
for _ in range(K):
    n = int(input())
    stu = [0] + list(map(int, input().split()))
    
    check = [False] * (n+1)
    for i in range(1, n+1):
        if check[i] == False:
            vst = [False] * (n+1)
            result = False
            dfs(i, i)
            if result == True:
                for j in range(1, n+1):
                    if vst[j]:
                        check[j] = True

    cnt = 0
    for bool in check[1:]:
        if bool == False:
            cnt += 1
            
    print(cnt)