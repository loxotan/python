import sys
input = sys.stdin.readline

N = int(input())
before = list(map(int, input().split()))
after = list(map(int, input().split()))

chai = []
for i in range(N):
    chai.append(before[i] - after[i])

cnt = 0
if chai:
    chai_pos = []
    for i in range(N):
        if chai[i] > 0:
            chai_pos.append(chai[i])
    cnt = sum(chai_pos)
    
print(cnt)