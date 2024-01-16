import sys
input = sys.stdin.readline

N = int(input())
A = list(map(int, input().split()))
setA = list(set(A))
dic = {}
stack = []
ans = [-1]*N

#dictionary 셋업
for i in range(len(setA)):
    dic.update({setA[i]:0})

#dictionary에 등장 횟수 기입
for j in range(N):
    k = dic[A[j]]
    dic.update({A[j]:k+1})

#오등큰수 정렬
for k in range(N):
    if not stack:
        stack.append(k)
    else:
        while stack and dic[A[stack[-1]]] < dic[A[k]]:
            ans[stack.pop()] = A[k]
        stack.append(k)
        
print(*ans)