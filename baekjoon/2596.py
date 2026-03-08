import sys 
input = sys.stdin.readline

T = int(input())
m = str(input().rstrip())

l = []
for i in range(T):
    l.append(m[i*6:(i+1)*6])

def decode(a, s):
    cnt = 0
    for i in range(6):
        if a[i] != s[i]:
            cnt += 1
    if cnt >= 2:
        return False
    return True

ans = [('A', '000000'), ('B', '001111'), ('C', '010011'), ('D', '011100'), ('E', '100110'), ('F', '101001'), ('G', '110101'), ('H', '111010')]
output = ''
for i in range(T):
    for a, b in ans:
        if decode(b, l[i]):
            output += a
            break
    if len(output) != i+1:
        print(i+1)
        break
if len(output) == T:
    print(output)