N = int(input())
A = list(map(int, input().split()))

# i번째에서 끝나는 감소하는 수열

d = [1]*N
for i in range(N-1, -1, -1):
    for j in range(i+1, N):
        if A[i] > A[j] and d[i] < d[j] +1:
            d[i] = d[j] +1
            
print(max(d))
