import sys
input = sys.stdin.readline

k, n = map(int, input().split())
lan = []
for _ in range(k):
    lan.append(int(input()))

#대략적인 답
gili = sum(lan)//n
while True:
    sum = 0
    for x in lan: #대략적인 답으로 구한 길이로 몇개까지 가능한지?
        sum += x//gili
    if sum > n: #더 많이 자를 수 있다면 길이가 너무 짧으므로,
        gili += 1
    elif sum < n: #더 적게 잘라진다면 길이가 너무 긺으로,
        gili -= 1
    else:
        break

print(gili)