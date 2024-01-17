nums = [False] * 2 + [True] * 97
primes = [[] for _ in range(10)]
for i in range(2, 98):
    if nums[i]:
        for j in range(i*i, 98, i):
            nums[j] = False
        if i >10:
            X = str(i)
            primes[int(X[0])].append(X[1])
            
a, b, n = map(int, input().split())
bb = str(b)[0]
cnt = 1
if bb in ['2', '4', '5', '6', '8']:
    print(-1)
    cnt = 0
    
if cnt == 1:    
    X = str(a)
    ans = [X[0], X[1]]
    while len(ans) != (n-2):
        ans.append(primes[int(ans[-1])][0])
    print(''.join(ans)+str(b))