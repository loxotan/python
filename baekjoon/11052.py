n = int(input())
pack = list(map(int, input().split()))
num = [i+1 for i in range(len(pack)+1)]
dic = dict(zip(num, pack))

dp = [0] * (n+1)
# dp[n] = n 개의 카드를 사는데 쓰는 최댓값
dp[1] = dic[1]
if n> 1:
    for i in range(2, n+1):
        for j in range(i+1):
            dp[i] = max(dp[i], dp[i-j]+dic.get(j,0))
            
print(dp[n])